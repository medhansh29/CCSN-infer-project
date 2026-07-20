import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from src.fetch_successive_jsons import JSONFetcher
from src.ztf_client import ZTFClient
from src.alerce_client import AlerceClient

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
    output_dir: str = 'data/static_payloads',
    use_alerce: bool = False
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
    alerce_client = AlerceClient()
    
    # Calculate global statistics for inferred parameters
    sample_stats = {}
    for param in ['zams', 'k_energy', 'mloss_rate', 'beta', '56Ni', 'texp', 'A_v', 'logZ']:
        col_name = f"{param}_final"
        if col_name in conv_df.columns:
            sample_stats[param] = {
                'mean': conv_df[col_name].mean(),
                'std': conv_df[col_name].std()
            }
    
    summary_index = []
    
    # Load ALeRCE meta cache
    alerce_meta_cache_file = "data/alerce_meta_cache.json"
    alerce_meta_cache = {}
    if os.path.exists(alerce_meta_cache_file):
        try:
            with open(alerce_meta_cache_file, "r") as f:
                alerce_meta_cache = json.load(f)
        except Exception:
            pass
    
    for idx, row in conv_df.iterrows():
        obj_id = row['object_id']
        
        # Gather TNS Data
        tns_info = tns_data.get(obj_id, {})
        redshift = tns_info.get('redshift')
        
        # Fetch Coordinates from ALeRCE (with cache)
        ra, dec = None, None
        if obj_id in alerce_meta_cache:
            ra = alerce_meta_cache[obj_id].get("meanra")
            dec = alerce_meta_cache[obj_id].get("meandec")
        else:
            try:
                resp = requests.get(f"https://api.alerce.online/ztf/v1/objects/{obj_id}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    ra = data.get("meanra")
                    dec = data.get("meandec")
                    alerce_meta_cache[obj_id] = {"meanra": ra, "meandec": dec}
            except Exception as e:
                print(f"Warning: could not fetch coordinates for {obj_id}: {e}")

        # Parse basic_info
        basic_info = {
            "discovery_date": row.get('first_run'),
            "redshift": float(redshift) if redshift and pd.notna(redshift) else None,
            "plateau_duration_days": float(row.get('plateau_duration_days')) if pd.notna(row.get('plateau_duration_days')) else None,
            "ra": float(ra) if ra is not None else None,
            "dec": float(dec) if dec is not None else None
        }
        
        # Gather JSON params for uncertainties
        json_params = {}
        if obj_id in fetcher.object_index:
            obj_files = fetcher.object_index[obj_id]
            if len(obj_files) > 0:
                latest_file = obj_files[-1][2]
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                        if 'parameters' in data:
                            json_params = data['parameters']
                except Exception as e:
                    print(f"Error reading params from {latest_file}: {e}")

        
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
            
            # calculate difference from sample in terms of standard deviations
            z_score = None
            if pd.notna(val) and param in sample_stats:
                mean = sample_stats[param]['mean']
                std = sample_stats[param]['std']
                if pd.notna(std) and std > 0:
                    z_score = float((val - mean) / std)
            inferred_parameters[f"{key_in_output}_zscore"] = z_score
            
            # calculate percentage uncertainty
            pct_unc = None
            pct_plus = None
            pct_minus = None
            if param in json_params and val and pd.notna(val):
                p_data = json_params[param]
                if len(p_data) == 3:
                    median_val, err_plus, err_minus = p_data
                    if abs(median_val) > 1e-6:
                        pct_unc = ((err_plus + err_minus) / 2.0 / abs(median_val)) * 100.0
                        pct_plus = (err_plus / abs(median_val)) * 100.0
                        pct_minus = (err_minus / abs(median_val)) * 100.0
            inferred_parameters[f"{key_in_output}_pct_uncertainty"] = pct_unc
            inferred_parameters[f"{key_in_output}_pct_plus"] = pct_plus
            inferred_parameters[f"{key_in_output}_pct_minus"] = pct_minus

            
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
        
        # --- Build LC Payload ---
        observations = []
        redshift = basic_info.get('redshift')
        a_v = float(row.get('A_v_final', 0)) if pd.notna(row.get('A_v_final')) else 0.0
        
        # Calculate distance modulus once per object
        mu = get_distance_modulus(redshift)

        raw_df = ztf_client.fetch_lightcurve(obj_id)
        if raw_df is None or raw_df.empty:
            raw_df = alerce_client.fetch_lightcurve(obj_id)
            
        if raw_df is not None and not raw_df.empty:
            for _, obs_row in raw_df.iterrows():
                band = str(obs_row['filter'])
                app_mag    = float(obs_row['mag'])
                is_ul = bool(obs_row.get('is_upperlimit', False))
                app_magerr = obs_row['magerr']
                
                a_filter = EXT_COEFF.get(band, 0.0) * a_v
                abs_mag = app_mag - mu - a_filter
                
                obs_dict = {
                    "mjd":    float(obs_row['mjd']),
                    "mag":    round(abs_mag, 6),
                    "magerr": round(float(app_magerr), 6) if pd.notna(app_magerr) else None,
                    "filter": band,
                    "is_upperlimit": is_ul
                }
                
                observations.append(obs_dict)
                
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
        
        # Calculate reduced chi^2 using observations and the model's 3 measurements (median, upper, lower)
        chi2 = 0.0
        n_obs = 0
        for obs in observations:
            band = obs['filter']
            if f"{band}_band" in model_fit:
                mf = model_fit[f"{band}_band"]
                mjd_arr = mf['mjd']
                median_arr = mf['median']
                upper_arr = mf['upper_16th']
                lower_arr = mf['lower_84th']
                
                obs_mjd = obs['mjd']
                obs_mag = obs['mag']
                obs_magerr = obs['magerr']
                
                if obs_magerr is None or obs.get('is_upperlimit', False):
                    continue
                
                mod_median = np.interp(obs_mjd, mjd_arr, median_arr)
                mod_upper = np.interp(obs_mjd, mjd_arr, upper_arr)
                mod_lower = np.interp(obs_mjd, mjd_arr, lower_arr)
                
                mod_err = abs(mod_lower - mod_upper) / 2.0
                total_err = np.sqrt(obs_magerr**2 + mod_err**2)
                
                if total_err > 0:
                    chi2 += ((obs_mag - mod_median) / total_err)**2
                    n_obs += 1
                    
        n_params = 8
        dof = max(1, n_obs - n_params)
        reduced_chi2 = chi2 / dof if n_obs > n_params else None
        inferred_parameters["reduced_chi2"] = reduced_chi2

        # Extract Parameter History
        parameter_history = []
        if obj_id in fetcher.object_index:
            for (date_str, filter_band, filepath) in fetcher.object_index[obj_id]:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if 'parameters' in data:
                        params = {}
                        for p_name, p_data in data['parameters'].items():
                            if isinstance(p_data, list) and len(p_data) == 3:
                                median_val, err_plus, err_minus = p_data
                                pct_plus = (err_plus / abs(median_val)) * 100 if median_val else 0
                                pct_minus = (err_minus / abs(median_val)) * 100 if median_val else 0
                                out_name = p_name
                                if p_name == '56Ni': out_name = 'ni56'
                                elif p_name == 'texp': out_name = 't_exp'
                                params[out_name] = {
                                    "val": median_val,
                                    "plus": err_plus,
                                    "minus": err_minus,
                                    "pct_plus": pct_plus,
                                    "pct_minus": pct_minus
                                }
                        phase = data['parameters'].get('Phase', 0)
                        if type(phase) is list and len(phase) > 0: phase = phase[0]
                        parameter_history.append({
                            "date": date_str,
                            "filter": filter_band,
                            "phase": float(phase),
                            "parameters": params
                        })
                except Exception as e:
                    pass

        
        has_lc = len(observations) > 0 or len(model_fit) > 0
        
        summary_entry = {
            "object_id": obj_id,
            "basic_info": basic_info,
            "inferred_parameters": inferred_parameters,
            "anomalies": anomalies,
            "has_light_curve": has_lc,
            "lightcurve": lc_payload,
            "parameter_history": parameter_history
        }
        summary_index.append(summary_entry)
        
        with open(os.path.join(output_dir, f"{obj_id}_lc.json"), 'w') as f:
            json.dump(lc_payload, f, indent=2)
            
    # Save summary index
    with open(os.path.join(output_dir, "summary_index.json"), 'w') as f:
        json.dump(summary_index, f, indent=2)
        
    # Save ALeRCE meta cache
    with open(alerce_meta_cache_file, "w") as f:
        json.dump(alerce_meta_cache, f, indent=2)
        
    print(f"✅ Exported {len(summary_index)} object profiles to {output_dir}/")
    
    # Automatically sync to the user portal if it exists
    this_dir = os.path.dirname(os.path.abspath(__file__))
    portal_root = os.path.abspath(os.path.join(this_dir, "..", "..", "REFITT-User-Portal"))
    
    if os.path.exists(portal_root):
        # Target the Vite public folder if Frontend exists, else fallback
        frontend_dir = os.path.join(portal_root, "Frontend")
        if os.path.exists(frontend_dir):
            portal_data_dir = os.path.join(frontend_dir, "public", "data")
        else:
            portal_data_dir = os.path.join(portal_root, "data")
            
        os.makedirs(portal_data_dir, exist_ok=True)
        import shutil
        print(f"Syncing summary payload to user portal at {portal_data_dir}...")
        src = os.path.join(output_dir, "summary_index.json")
        dst = os.path.join(portal_data_dir, "summary_index.json")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        print("✅ Synced summary to portal successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fallback-alerce', action='store_true', default=True, help="Use ALeRCE by default when running standalone")
    args = parser.parse_args()
    export_static_payloads(use_alerce=args.fallback_alerce)
