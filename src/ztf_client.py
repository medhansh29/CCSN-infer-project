import pandas as pd
import os
import re
import requests
from dotenv import load_dotenv
from typing import Optional

class ZTFClient:
    def __init__(self, data_dir: str = 'data/ztf_forced_photometry'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_lightcurve(self, ztf_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch and parse the ZTF forced photometry lightcurve for a given OID.
        Resolution order:
          1. Registry (data/zfps_registry.json) — handles batch-named files
          2. Conventional names in data_dir ({ztf_id}.txt / .csv)
        Returns a DataFrame with columns: mjd, mag, magerr, filter, isdiffpos
        """
        from src.zfps_registry import load_registry

        file_path = None

        # 1. Check registry first — ZFPS files are named by batch request ID, not OID
        registry = load_registry()
        reg_jd_start = None
        reg_jd_end = None
        
        if ztf_id in registry and registry[ztf_id].get('file_path'):
            reg_jd_start = registry[ztf_id].get('jd_start')
            reg_jd_end = registry[ztf_id].get('jd_end')
            candidate = registry[ztf_id]['file_path']
            if os.path.exists(candidate):
                file_path = candidate

        # 2. Fall back to conventional naming patterns
        if not file_path:
            for p in [
                os.path.join(self.data_dir, f"{ztf_id}.txt"),
                os.path.join(self.data_dir, f"forcedphotometry_{ztf_id}.txt"),
                os.path.join(self.data_dir, f"{ztf_id}.csv"),
                os.path.join(self.data_dir, f"forcedphotometry_{ztf_id}.csv"),
            ]:
                if os.path.exists(p):
                    file_path = p
                    break

        if not file_path:
            return None

        try:
            if file_path.endswith('.txt'):
                # ZFPS .txt format: header row uses ", " separators, data rows use spaces.
                # Read header and data separately then merge.
                header_line = None
                req_jd_end = None
                with open(file_path) as f:
                    for line in f:
                        stripped = line.strip()
                        if line.startswith('# Requested JD end = '):
                            import re
                            match = re.search(r'= ([\d\.]+) days', line)
                            if match: req_jd_end = float(match.group(1))
                            
                        if stripped.startswith('#') or not stripped:
                            continue
                        header_line = stripped
                        break
                        
                # If the file's requested end date misses our SN's actual end date, return None
                if req_jd_end and reg_jd_end and reg_jd_end > req_jd_end:
                    print(f"⚠️ [ZTFClient] {ztf_id} file misses tail-end of lightcurve. Falling back to ALeRCE.")
                    return None

                if header_line is None:
                    return None

                # Strip trailing commas from each column name
                col_names = [c.strip().rstrip(',') for c in header_line.split(',')]

                # Read data rows (skip comment lines and the header line itself)
                df = pd.read_csv(
                    file_path,
                    sep=r'\s+',
                    comment='#',
                    header=None,
                    names=col_names,
                    skiprows=lambda i: False  # header handled manually
                )
                # Drop the first non-comment row which is the header we already parsed
                df = df[~df['jd'].astype(str).str.contains('[a-zA-Z]', na=False)]
                # Preserve string columns (filter) before coercing everything else to numeric
                filter_col = df['filter'].copy() if 'filter' in df.columns else None
                df = df.apply(pd.to_numeric, errors='coerce')
                if filter_col is not None:
                    df['filter'] = filter_col.values

            else:
                df = pd.read_csv(file_path)
            
            # If the CSV has the standard ZTF forced photometry format, map it
            # Standard ZTF forced photometry columns: jd, filter, forcediffimflux, forcediffimfluxunc, etc.
            # We will map them to mjd, mag, magerr, filter, isdiffpos if they don't already exist.
            
            # Simple fallback if already mapped:
            if 'mjd' in df.columns and 'mag' in df.columns and 'magerr' in df.columns and 'filter' in df.columns:
                return df.dropna(subset=['mag', 'magerr']).sort_values(by="mjd").reset_index(drop=True)
                
            # If standard ZFPS format:
            records = []
            import numpy as np
            
            if 'jd' in df.columns and 'forcediffimflux' in df.columns:
                # Filter out sentinel null values (-99999.000)
                df = df[df['forcediffimflux'] > -90000].copy()
                
                if 'diffimgstatus' in df.columns:
                    df = df[df.get('diffimgstatus', 1) != 0].copy()
                
                if 'procstatus' in df.columns:
                    # '0' indicates nominal processing, anything else is a warning/error code
                    df = df[df['procstatus'].astype(str).str.strip() == '0'].copy()
                
                # We need to process per filter
                records = []
                import numpy as np
                for fid, group in df.groupby('filter'):
                    band = "unknown"
                    if fid == 'ZTF_g': band = "g"
                    elif fid == 'ZTF_r': band = "r"
                    elif fid == 'ZTF_i': band = "i"
                    elif type(fid) == str and 'g' in fid.lower(): band = 'g'
                    elif type(fid) == str and 'r' in fid.lower(): band = 'r'
                    elif type(fid) == str and 'i' in fid.lower(): band = 'i'

                    group = group.copy()
                    
                    # 1. Rescale uncertainties by sqrt(mean(chisq))
                    if 'forcediffimchisq' in group.columns:
                        chisq_mean = group['forcediffimchisq'].mean()
                        if pd.notna(chisq_mean) and chisq_mean > 0:
                            group['forcediffimfluxunc'] = group['forcediffimfluxunc'] * np.sqrt(chisq_mean)
                            
                    # 2. Baseline Calculation & Subtraction
                    baseline_flux = 0.0
                    rms_snr = 1.0
                    
                    if reg_jd_start:
                        # Baseline epochs are those before the SN start
                        baseline_mask = group['jd'] < reg_jd_start
                        if baseline_mask.sum() > 0:
                            baseline_flux = group.loc[baseline_mask, 'forcediffimflux'].median()
                            if pd.isna(baseline_flux):
                                baseline_flux = 0.0
                                
                            # Subtract baseline
                            group['flux_corrected'] = group['forcediffimflux'] - baseline_flux
                            
                            # 3. Temporal Systematics Rescaling (RMS of S/N)
                            snr_baseline = group.loc[baseline_mask, 'flux_corrected'] / group.loc[baseline_mask, 'forcediffimfluxunc']
                            snr_baseline = snr_baseline.dropna()
                            if len(snr_baseline) > 1:
                                p16 = np.percentile(snr_baseline, 16)
                                p84 = np.percentile(snr_baseline, 84)
                                rms_snr = 0.5 * (p84 - p16)
                                if rms_snr <= 0 or pd.isna(rms_snr):
                                    rms_snr = 1.0
                                    
                                # Multiply uncertainties by this RMS
                                group['forcediffimfluxunc'] = group['forcediffimfluxunc'] * rms_snr
                        else:
                            group['flux_corrected'] = group['forcediffimflux']
                    else:
                        group['flux_corrected'] = group['forcediffimflux']
                        
                    # 4. Magnitude Calculation & Non-Detection Flagging
                    for _, row in group.iterrows():
                        flux_corr = row['flux_corrected']
                        fluxerr = row['forcediffimfluxunc']
                        zpdiff = row.get('zpdiff', 22.5)
                        
                        if pd.isna(flux_corr) or pd.isna(fluxerr) or fluxerr <= 0:
                            continue
                            
                        snr = flux_corr / fluxerr
                        
                        if snr >= 3:
                            try:
                                mag = zpdiff - 2.5 * np.log10(flux_corr)
                                magerr = 1.0857 * (fluxerr / flux_corr)
                                is_ul = False
                            except:
                                continue
                        else:
                            try:
                                mag = zpdiff - 2.5 * np.log10(5 * fluxerr)
                                magerr = None
                                is_ul = True
                            except:
                                continue
                                
                        records.append({
                            "mjd": row['jd'] - 2400000.5,
                            "mag": mag,
                            "magerr": magerr,
                            "filter": band,
                            "isdiffpos": 1 if flux_corr > 0 else 0,
                            "is_upperlimit": is_ul
                        })
                        
                if records:
                    mapped_df = pd.DataFrame(records)
                    
                    # Trim data using registry bounds to avoid pre-explosion junk data
                    if reg_jd_start and reg_jd_end:
                        # mapped_df has 'mjd', registry has 'jd'. Convert for comparison.
                        mjd_start = reg_jd_start - 2400000.5
                        mjd_end = reg_jd_end - 2400000.5
                        mapped_df = mapped_df[(mapped_df['mjd'] >= mjd_start) & (mapped_df['mjd'] <= mjd_end)]

                    mapped_df = mapped_df.sort_values(by="mjd").reset_index(drop=True)
                    return mapped_df
            
            # Return raw if we couldn't parse it but it exists
            return df
            
        except Exception as e:
            print(f"[ZTFClient] Error reading forced photometry for {ztf_id}: {e}")
            return None

    def _query_zfps_database(self):
        """
        Query ZTF's getBatchForcedPhotometryRequests endpoint.
        Returns (status_code, response_text, lightcurve_paths_list).
        Mirrors ZTF's published check_status.py logic exactly.
        """
        load_dotenv()
        email    = os.getenv('ZTF_EMAIL')
        userpass = os.getenv('ZTF_USERPASS')

        if not email or not userpass:
            print("[ZTFClient] ❌ ZTF_EMAIL and ZTF_USERPASS must be set in .env")
            return None, None, []

        settings = {'email': email, 'userpass': userpass,
                    'option': 'All recent jobs', 'action': 'Query Database'}

        try:
            r = requests.get(
                'https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi',
                auth=('ztffps', 'dontgocrazy!'),
                params=settings
            )
        except Exception as e:
            print(f"[ZTFClient] ❌ Network error: {e}")
            return None, None, []

        lightcurves = []
        if r.status_code == 200:
            lightcurves = re.findall(r'/ztf/ops.+?lc\.txt\b', r.text)

        return r.status_code, r.text, lightcurves

    def check_job_status(self):
        """
        Query ZTF's database and print job status — exactly what ZTF's published
        check_status.py does. Shows wget lines for completed jobs, or a pending
        message if jobs are still queued. Does NOT download anything.
        """
        print("Script executed normally and queried the ZTF Batch Forced Photometry database.\n")

        status_code, text, lightcurves = self._query_zfps_database()

        if status_code == 200:
            wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O'
            wget_url    = 'https://ztfweb.ipac.caltech.edu'

            if lightcurves:
                print(f"Found {len(lightcurves)} completed lightcurve(s):\n")
                for lc in lightcurves:
                    p = re.match(r'.+/(.+)', lc)
                    fileonly = p.group(1)
                    print(f'{wget_prefix} {fileonly} "{wget_url}{lc}"')
            else:
                print("No lightcurves found. Your jobs might still be pending.")
        else:
            print(f"Status_code= {status_code} ; Jobs either queued or abnormal execution.")

    def download_pending_lightcurves(self):
        """
        Check status and automatically download any completed lightcurves
        from ZTF Batch Forced Photometry requests into data_dir.
        """
        status_code, text, lightcurves = self._query_zfps_database()

        if status_code != 200:
            print(f"[ZTFClient] Status_code= {status_code} ; Jobs either queued or abnormal execution.")
            return False

        print("[ZTFClient] Queried ZTF database.\n")

        if not lightcurves:
            print("[ZTFClient] No lightcurves found. Your jobs might still be pending.")
            return False

        curl_base = 'curl --silent --show-error --user ztffps:dontgocrazy! -o'
        wget_url  = 'https://ztfweb.ipac.caltech.edu'

        print(f"[ZTFClient] Found {len(lightcurves)} lightcurves. Downloading...\n")
        os.makedirs(self.data_dir, exist_ok=True)

        downloaded_paths = []
        ignore_file = os.path.join(self.data_dir, "ignore_list.txt")
        ignore_set = set()
        if os.path.exists(ignore_file):
            with open(ignore_file, 'r') as f:
                ignore_set = set(line.strip() for line in f)

        for lc in lightcurves:
            p = re.match(r'.+/(.+)', lc)
            fileonly = p.group(1)
            
            if fileonly in ignore_set:
                continue
                
            out_path = os.path.join(self.data_dir, fileonly)
            
            # Skip download if the file already exists and is non-empty
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                downloaded_paths.append(out_path)
                continue
                
            command  = f'{curl_base} "{out_path}" "{wget_url}{lc}"'
            print(f"[ZTFClient] Downloading {fileonly}...")
            exit_code = os.system(command)
            if exit_code == 0 and os.path.exists(out_path):
                downloaded_paths.append(out_path)
                print(f"[ZTFClient] ✅ Saved to {out_path}")
            else:
                print(f"[ZTFClient] ❌ Failed to download {fileonly} (exit code {exit_code})")

        n = len(downloaded_paths)
        print(f"\n[ZTFClient] {n}/{len(lightcurves)} files downloaded successfully.")
        return downloaded_paths  # caller can verify before updating registry

