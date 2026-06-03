import os
import json
import numpy as np
from pathlib import Path

def fix_json_parameters(base_dir='.'):
    """
    Temporary wrapper function to fix the parameter misalignment in JSON files.
    It reads the corresponding {ZTFID}_samples.txt file, extracts the correct parameter 
    medians and errors, and overwrites the 'parameters' dictionary in the JSONs.
    
    This function should be removed once the upstream fix is implemented.
    """
    print("\n" + "-"*70)
    print("TEMPORARY FIX: Correcting misaligned JSON parameters from samples.txt")
    print("-"*70)
    
    # Define the order of parameters as they appear in the .txt files
    labels = ['zams', 'k_energy', 'mloss_rate', 'beta', '56Ni', 'csm_radius', 'texp', 'A_v']
    
    base_path = Path(base_dir)
    date_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.count('-') == 2])
    
    fixed_count = 0
    missing_txt_count = 0
    
    for date_dir in date_dirs:
        # Find all JSON files
        json_files = list(date_dir.glob("*_nn.json"))
        for json_file in json_files:
            parts = json_file.stem.split('_')
            if len(parts) >= 3:
                ztf_id = parts[0]
                
                # Check for corresponding samples.txt
                samples_txt = date_dir / f"{ztf_id}_samples.txt"
                if not samples_txt.exists():
                    missing_txt_count += 1
                    continue
                    
                try:
                    # Read chains from samples.txt
                    chains = np.loadtxt(samples_txt, skiprows=0).T
                    pct = np.percentile(chains, [16, 50, 84], axis=1)
                    
                    # Compute median, lower error, and upper error
                    extracted_data = {
                        labels[i]: {
                            'median': pct[1, i], 
                            'lower_err': pct[1, i] - pct[0, i], 
                            'upper_err': pct[2, i] - pct[1, i]
                        } for i in range(len(labels))
                    }
                    
                    # Update the JSON file
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'parameters' not in data:
                        data['parameters'] = {}
                        
                    # Update parameters in the format [median, upper_err, lower_err]
                    for label in labels:
                        med = float(extracted_data[label]['median'])
                        upper = float(extracted_data[label]['upper_err'])
                        lower = float(extracted_data[label]['lower_err'])
                        data['parameters'][label] = [med, upper, lower]
                    
                    with open(json_file, 'w') as f:
                        json.dump(data, f)
                        
                    fixed_count += 1
                except Exception as e:
                    print(f"  [!] Error processing {samples_txt}: {e}")
                    
    print(f"✅ Fixed parameters in {fixed_count} JSON files.")
    if missing_txt_count > 0:
        print(f"⚠️ Skipped {missing_txt_count} JSON files due to missing samples.txt.")
    print("-"*70 + "\n")
