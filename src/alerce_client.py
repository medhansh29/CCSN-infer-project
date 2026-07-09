import requests
import pandas as pd
import os
from typing import Optional
import time

class AlerceClient:
    def __init__(self, cache_dir: str = 'data/alerce_cache'):
        self.base_url = "https://api.alerce.online/ztf/v1/objects"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch_lightcurve(self, ztf_id: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch the raw ZTF lightcurve for a given object ID from ALeRCE.
        Checks cache_dir first. Returns a DataFrame with columns: mjd, mag, magerr, filter, isdiffpos
        """
        cache_path = os.path.join(self.cache_dir, f"{ztf_id}.csv")
        
        # 1. Try Loading from Cache
        if os.path.exists(cache_path):
            if os.path.getsize(cache_path) == 0:
                return None # Marker for previously checked, no data found
            try:
                df = pd.read_csv(cache_path)
                return df if not df.empty else None
            except pd.errors.EmptyDataError:
                return None
            except Exception as e:
                print(f"[ALeRCE] Error reading cache for {ztf_id}: {e}")
        
        # 2. Fetch from API
        url = f"{self.base_url}/{ztf_id}/lightcurve"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if "detections" not in data or not data["detections"]:
                    open(cache_path, 'w').close() # Create empty file to cache "no data"
                    return None
                    
                detections = data["detections"]
                records = []
                for det in detections:
                    # We generally rely on primary ZTF bands (1=g, 2=r, 3=i)
                    fid = det.get("fid")
                    if fid == 1:
                        band = "g"
                    elif fid == 2:
                        band = "r"
                    elif fid == 3:
                        band = "i"
                    else:
                        continue
                        
                    records.append({
                        "mjd": det.get("mjd"),
                        "mag": det.get("magpsf", det.get("magpsf_corr")),
                        "magerr": det.get("sigmapsf", det.get("sigmapsf_corr")),
                        "filter": band,
                        "isdiffpos": det.get("isdiffpos")
                    })
                    
                if not records:
                    open(cache_path, 'w').close()
                    return None
                    
                df = pd.DataFrame(records)
                # Filter for valid measurements
                df = df.dropna(subset=["mag", "magerr"])
                df = df.sort_values(by="mjd").reset_index(drop=True)
                
                # Save to Cache
                df.to_csv(cache_path, index=False)
                return df
                
            except requests.Timeout:
                print(f"[ALeRCE] Timeout fetching {ztf_id} (Attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            except requests.HTTPError as e:
                if response.status_code == 404:
                    open(cache_path, 'w').close()
                    return None
                print(f"[ALeRCE] HTTP Error fetching {ztf_id}: {e}")
                break
            except requests.RequestException as e:
                print(f"[ALeRCE] Error fetching {ztf_id}: {e}")
                break
                
        return None
