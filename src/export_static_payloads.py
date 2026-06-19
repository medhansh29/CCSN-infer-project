import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.fetch_successive_jsons import JSONFetcher
from src.ztf_client import ZTFClient

# Cardelli (1989) extinction coefficients relative to A_v, for R_v = 3.1
# A_filter = EXT_COEFF[filter] * A_v
EXT_COEFF = {'g': 1.161, 'r': 0.843, 'i': 0.633}


def get_distance_modulus(redshift: float) -> float:
    """
    Compute distance modulus for a given redshift using Planck18 cosmology.
    μ = 5 * log10(d_L / 10 pc)
    """
    from astropy.cosmology import Planck18
    import astropy.units as u

    if redshift is None or redshift <= 0:
        return 0.0

    d_L_pc = Planck18.luminosity_distance(redshift).to(u.pc).value
    return 5.0 * np.log10(d_L_pc / 10.0)


def export_static_payloads(
    convergence_csv: str = 'data/convergence_metrics.csv',
    outliers_csv: str = 'data/scatter_outliers.csv',
    tns_cache: str = '.tns_cache.json',
    output_dir: str = 'data/static_payloads'
):
    print(f"\n{'='*70}")
    print("STEP 5: Generating Static Payloads")
    print(f"{'-'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load convergence metrics
    if not os.path.exists(convergence_csv):
        print(f"❌ Error: {convergence_csv} not found.")
        return
        
    conv_df = pd.read_csv(convergence_csv)
    
    # Load outliers if they exist
    outliers_df = pd.DataFrame()
    if os.path.exists(outliers_csv):
        outliers_df = pd.read_csv(outliers_csv)
        
    # Load TNS cache for redshift
    tns_data = {}
    if os.path.exists(tns_cache):
        try:
            with open(tns_cache, 'r') as f:
                tns_data = json.load(f)
        except Exception as e:
            print(f"Warning: could not load tns cache: {e}")
            
    # Initialize JSONFetcher and ZTFClient
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    ztf_client = ZTFClient()
    
    summary_index = []
    
    for idx, row in conv_df.iterrows():
        obj_id = row['object_id']
        
        # Gather TNS Data
        tns_info = tns_data.get(obj_id, {})
        redshift = tns_info.get('redshift')
        
        # Parse basic_info
        basic_info = {
            "discovery_date": row.get('first_run'),
            "redshift": float(redshift) if redshift and pd.notna(redshift) else None,
            "plateau_duration_days": float(row.get('plateau_duration_days')) if pd.notna(row.get('plateau_duration_days')) else None
        }
        
        # Parse inferred_parameters
        inferred_parameters = {}
        for param in ['zams', 'k_energy', 'mloss_rate', 'beta', '56Ni', 'texp', 'A_v', 'logZ']:
            key_in_output = param
            if param == '56Ni':
                key_in_output = 'ni56'
            elif param == 'texp':
                key_in_output = 't_exp'
                
            val = row.get(f'{param}_final')
            inferred_parameters[key_in_output] = float(val) if pd.notna(val) else None
            
        # Parse anomalies
        # 1. Morphology
        morphology = {
            "early_rise_excess": bool(row.get('early_rise_excess_flag', False) == True),
            "arrested_cooling": bool(row.get('arrested_cooling_flag', False) == True),
            "plateau_extension": False, # Placeholder if not in df
            "plateau_rebrightening": bool(row.get('rebrightening_flag', False) == True),
            "precursor_detection": bool(row.get('precursor_flag', False) == True)
        }
        
        # 2. Composition
        composition = {
            "peak_brightness_excess": False, # Placeholder
            "nickel_overabundance": False # Placeholder
        }
        if pd.notna(row.get('is_anomaly')) and row.get('is_anomaly') == -1:
            # We could refine composition flags if we wanted
            pass
            
        # 3. Outliers mapping
        bivariate_outliers = {
            "mloss_ek_magnetar": False,
            "ek_ni_pair_instability": False,
            "texp_beta_diffusion": False,
            "logz_av_dust": False
        }
        
        multivariate_clusters_3d = {
            "energy_engine": False,
            "progenitor_evolution": False,
            "modeling_degeneracy": False,
            "ejecta_efficiency": False,
            "lc_morphology": False
        }
        
        if not outliers_df.empty:
            obj_outliers = outliers_df[outliers_df['object_id'] == obj_id]
            for _, out_row in obj_outliers.iterrows():
                out_type = out_row.get('outlier_type')
                if out_type == 'Mloss-Ek':
                    bivariate_outliers['mloss_ek_magnetar'] = True
                elif out_type == 'Ek-Ni':
                    bivariate_outliers['ek_ni_pair_instability'] = True
                elif out_type == 'Texp-Beta':
                    bivariate_outliers['texp_beta_diffusion'] = True
                elif out_type == 'logZ-Av':
                    bivariate_outliers['logz_av_dust'] = True
                elif out_type == 'Energy Engine':
                    multivariate_clusters_3d['energy_engine'] = True
                elif out_type == 'Progenitor Evolution':
                    multivariate_clusters_3d['progenitor_evolution'] = True
                elif out_type == 'Modeling Degeneracy':
                    multivariate_clusters_3d['modeling_degeneracy'] = True
                elif out_type == 'Ejecta Efficiency':
                    multivariate_clusters_3d['ejecta_efficiency'] = True
                elif out_type == 'LC Morphology':
                    multivariate_clusters_3d['lc_morphology'] = True

        anomalies = {
            "morphology": morphology,
            "composition": composition,
            "bivariate_outliers": bivariate_outliers,
            "multivariate_clusters_3d": multivariate_clusters_3d
        }
        
        summary_entry = {
            "object_id": obj_id,
            "basic_info": basic_info,
            "inferred_parameters": inferred_parameters,
            "anomalies": anomalies
        }
        summary_index.append(summary_entry)
        
        # --- Build LC Payload ---
        observations = []
        redshift = basic_info.get('redshift')
        a_v = float(row.get('A_v_final', 0)) if pd.notna(row.get('A_v_final')) else 0.0
        
        # Calculate distance modulus once per object
        mu = get_distance_modulus(redshift)

        raw_df = ztf_client.fetch_lightcurve(obj_id)
        if raw_df is not None and not raw_df.empty:
            for _, obs_row in raw_df.iterrows():
                band = str(obs_row['filter'])
                app_mag    = float(obs_row['mag'])
                app_magerr = float(obs_row['magerr'])
                
                a_filter = EXT_COEFF.get(band, 0.0) * a_v
                abs_mag = app_mag - mu - a_filter
                
                observations.append({
                    "mjd":    float(obs_row['mjd']),
                    "mag":    round(abs_mag, 6),
                    "magerr": round(app_magerr, 6),
                    "filter": band
                })
                
        # Model Fit
        model_fit = {}
        # Fetch the most recent JSONs for each filter
        if obj_id in fetcher.object_index:
            obj_files = fetcher.object_index[obj_id]
            # obj_files are sorted by date
            latest_by_filter = {}
            for (date_str, filter_band, filepath) in obj_files:
                latest_by_filter[filter_band] = filepath
                
            for filter_band, filepath in latest_by_filter.items():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    mjd_arr = data.get('mjd_arr', [])
                    mag_arr = data.get('mag_arr', [])
                    if len(mjd_arr) > 0:
                        a_filter = EXT_COEFF.get(filter_band, 0.0) * a_v
                        total_correction = mu + a_filter
                        
                        if len(mag_arr) == 3:
                            # Apply distance modulus + extinction
                            median = (np.array(mag_arr[0]) - total_correction).tolist()
                            upper_16th = (np.array(mag_arr[1]) - total_correction).tolist()
                            lower_84th = (np.array(mag_arr[2]) - total_correction).tolist()
                        elif len(mag_arr) > 3:
                            # Apply distance modulus + extinction
                            mag_arr_np = np.array(mag_arr) - total_correction
                            median = np.percentile(mag_arr_np, 50, axis=0).tolist()
                            upper_16th = np.percentile(mag_arr_np, 16, axis=0).tolist()
                            lower_84th = np.percentile(mag_arr_np, 84, axis=0).tolist()
                        else:
                            continue
                            
                        model_fit[f"{filter_band}_band"] = {
                            "mjd": mjd_arr,
                            "median": median,
                            "upper_16th": upper_16th,
                            "lower_84th": lower_84th
                        }
                except Exception as e:
                    print(f"Error loading model fit for {obj_id} {filter_band}: {e}")
                    
        lc_payload = {
            "object_id": obj_id,
            "observations": observations,
            "model_fit": model_fit
        }
        
        with open(os.path.join(output_dir, f"{obj_id}_lc.json"), 'w') as f:
            json.dump(lc_payload, f, indent=2)
            
    # Save summary index
    with open(os.path.join(output_dir, "summary_index.json"), 'w') as f:
        json.dump(summary_index, f, indent=2)
        
    print(f"✅ Exported {len(summary_index)} object profiles to {output_dir}/")
    
    # Automatically sync to the user portal if it exists
    this_dir = os.path.dirname(os.path.abspath(__file__))
    portal_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "REFITT-User-Portal"))
    if os.path.exists(portal_dir):
        import shutil
        print(f"Syncing payloads to user portal at {portal_dir}...")
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.json'):
                src = os.path.join(output_dir, file_name)
                dst = os.path.join(portal_dir, file_name)
                shutil.copy2(src, dst)
        print("✅ Synced to portal successfully.")

if __name__ == "__main__":
    export_static_payloads()
