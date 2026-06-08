import pandas as pd
import os
from typing import Optional

class ZTFClient:
    def __init__(self, data_dir: str = 'data/ztf_forced_photometry'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_lightcurve(self, ztf_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch the raw ZTF forced photometry lightcurve for a given object ID.
        Looks for a file named {ztf_id}.csv or forcedphotometry_{ztf_id}.csv in the data_dir.
        Returns a DataFrame with columns: mjd, mag, magerr, filter, isdiffpos
        """
        possible_paths = [
            os.path.join(self.data_dir, f"{ztf_id}.csv"),
            os.path.join(self.data_dir, f"forcedphotometry_{ztf_id}.csv")
        ]
        
        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break
                
        if not file_path:
            return None
            
        try:
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
                for _, row in df.iterrows():
                    flux = row.get('forcediffimflux')
                    fluxerr = row.get('forcediffimfluxunc')
                    if pd.isna(flux) or pd.isna(fluxerr) or flux <= 0:
                        continue
                        
                    # Calculate AB mag: mag = 22.5 - 2.5 * log10(flux)
                    try:
                        mag = 22.5 - 2.5 * np.log10(flux)
                        # magerr = 1.0857 * fluxerr / flux
                        magerr = 1.0857 * (fluxerr / flux)
                    except:
                        continue
                        
                    fid = row.get('filter')
                    band = "unknown"
                    if fid == 'ZTF_g': band = "g"
                    elif fid == 'ZTF_r': band = "r"
                    elif fid == 'ZTF_i': band = "i"
                    elif type(fid) == str and 'g' in fid.lower(): band = 'g'
                    elif type(fid) == str and 'r' in fid.lower(): band = 'r'
                    elif type(fid) == str and 'i' in fid.lower(): band = 'i'
                    
                    records.append({
                        "mjd": row['jd'] - 2400000.5,
                        "mag": mag,
                        "magerr": magerr,
                        "filter": band,
                        "isdiffpos": 1 if flux > 0 else 0
                    })
                    
                if records:
                    mapped_df = pd.DataFrame(records)
                    mapped_df = mapped_df.sort_values(by="mjd").reset_index(drop=True)
                    return mapped_df
            
            # Return raw if we couldn't parse it but it exists
            return df
            
        except Exception as e:
            print(f"[ZTFClient] Error reading forced photometry for {ztf_id}: {e}")
            return None
