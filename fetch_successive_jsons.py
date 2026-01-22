#!/usr/bin/env python3
"""
Phase 1: Fetch Successive JSONs for CCSN Inference Analysis

This script organizes JSON observation files across date directories,
grouping them by object ID and arranging chronologically to enable
temporal analysis of model parameter convergence.

Based on TL_DR CCSN Infer.md strategy for measuring model performance
over time using successive 5-day observation windows.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd


class JSONFetcher:
    """Fetches and organizes successive JSON observations by object ID."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.object_index = defaultdict(list)
        
    def scan_directories(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Scan all date directories and index JSON files by object ID.
        
        Returns:
            Dictionary mapping object_id -> list of (date, filter, filepath) tuples
        """
        print("Scanning directories for JSON files...")
        
        # Find all date directories (YYYY-MM-DD format)
        date_dirs = sorted([
            d for d in self.base_dir.iterdir() 
            if d.is_dir() and d.name.count('-') == 2
        ])
        
        for date_dir in date_dirs:
            date_str = date_dir.name
            
            # Find all JSON files in this date directory
            json_files = list(date_dir.glob("*_nn.json"))
            
            for json_file in json_files:
                # Parse filename: ZTF25absklbq_g_nn.json -> (ZTF25absklbq, g)
                parts = json_file.stem.split('_')
                if len(parts) >= 3:
                    object_id = parts[0]
                    filter_band = parts[1]  # 'g' or 'r'
                    
                    # Store as (date, filter, filepath)
                    self.object_index[object_id].append(
                        (date_str, filter_band, str(json_file))
                    )
        
        # Sort observations for each object chronologically
        for object_id in self.object_index:
            self.object_index[object_id].sort(key=lambda x: x[0])
        
        return dict(self.object_index)
    
    def load_json_data(self, filepath: str) -> dict:
        """Load and parse a single JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_object_timeline(self, object_id: str) -> pd.DataFrame:
        """
        Get chronological timeline of observations for a specific object.
        
        Returns:
            DataFrame with columns: date, filter, filepath, Phase, zams, mloss_rate, 56Ni
        """
        if object_id not in self.object_index:
            raise ValueError(f"Object {object_id} not found in index")
        
        timeline_data = []
        
        for date, filter_band, filepath in self.object_index[object_id]:
            data = self.load_json_data(filepath)
            
            # Extract key parameters
            params = data.get('parameters', {})
            
            timeline_data.append({
                'date': date,
                'filter': filter_band,
                'filepath': filepath,
                'mjd': data.get('mjd'),
                'Phase': params.get('Phase'),
                'zams': params.get('zams', [None])[0],  # Mean value
                'zams_std': params.get('zams', [None, None])[1] if len(params.get('zams', [])) > 1 else None,
                'mloss_rate': params.get('mloss_rate', [None])[0],
                'mloss_rate_std': params.get('mloss_rate', [None, None])[1] if len(params.get('mloss_rate', [])) > 1 else None,
                '56Ni': params.get('56Ni', [None])[0],
                '56Ni_std': params.get('56Ni', [None, None])[1] if len(params.get('56Ni', [])) > 1 else None,
                'k_energy': params.get('k_energy', [None])[0],
                'A_v': params.get('A_v', [None])[0],
            })
        
        return pd.DataFrame(timeline_data)
    
    def get_all_object_ids(self) -> List[str]:
        """Get list of all object IDs with observations."""
        return sorted(self.object_index.keys())
    
    def get_objects_with_multiple_obs(self, min_obs: int = 2) -> Dict[str, int]:
        """
        Get objects that have multiple observations across dates.
        
        Args:
            min_obs: Minimum number of observations required
            
        Returns:
            Dictionary mapping object_id -> number of observations
        """
        return {
            obj_id: len(observations)
            for obj_id, observations in self.object_index.items()
            if len(observations) >= min_obs
        }
    
    def print_summary(self):
        """Print summary statistics of the dataset."""
        total_objects = len(self.object_index)
        multi_obs = self.get_objects_with_multiple_obs()
        
        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total unique objects: {total_objects}")
        print(f"Objects with multiple observations: {len(multi_obs)}")
        print(f"Total JSON files indexed: {sum(len(obs) for obs in self.object_index.values())}")
        
        if multi_obs:
            max_obs_id = max(multi_obs, key=multi_obs.get)
            print(f"\nMost observed object: {max_obs_id} ({multi_obs[max_obs_id]} observations)")
            
            # Show distribution
            print(f"\nObservation count distribution:")
            obs_counts = defaultdict(int)
            for count in multi_obs.values():
                obs_counts[count] += 1
            
            for count in sorted(obs_counts.keys(), reverse=True)[:10]:
                print(f"  {count} observations: {obs_counts[count]} objects")
        
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    
    # Initialize fetcher
    fetcher = JSONFetcher()
    
    # Scan and index all JSON files
    object_index = fetcher.scan_directories()
    
    # Print summary
    fetcher.print_summary()
    
    # Get objects with multiple observations
    multi_obs_objects = fetcher.get_objects_with_multiple_obs(min_obs=5)
    
    print(f"Found {len(multi_obs_objects)} objects with 5+ observations")
    print("\nTop 10 most observed objects:")
    sorted_objects = sorted(multi_obs_objects.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for obj_id, count in sorted_objects:
        print(f"  {obj_id}: {count} observations")
    
    # Example: Show timeline for the most observed object
    if sorted_objects:
        example_id = sorted_objects[0][0]
        print(f"\n{'='*60}")
        print(f"Example timeline for {example_id}")
        print(f"{'='*60}")
        
        timeline = fetcher.get_object_timeline(example_id)
        print(timeline[['date', 'filter', 'Phase', 'zams', 'mloss_rate', '56Ni']].to_string(index=False))
        
        # Save timeline to CSV
        output_file = f"timeline_{example_id}.csv"
        timeline.to_csv(output_file, index=False)
        print(f"\nTimeline saved to: {output_file}")
    
    # Save complete object index summary
    summary_data = []
    for obj_id, observations in object_index.items():
        dates = [obs[0] for obs in observations]
        filters = set(obs[1] for obs in observations)
        
        summary_data.append({
            'object_id': obj_id,
            'num_observations': len(observations),
            'first_date': min(dates),
            'last_date': max(dates),
            'filters': ','.join(sorted(filters)),
            'date_span_days': (datetime.strptime(max(dates), '%Y-%m-%d') - 
                              datetime.strptime(min(dates), '%Y-%m-%d')).days
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('num_observations', ascending=False)
    summary_df.to_csv('object_index_summary.csv', index=False)
    print(f"\nObject index summary saved to: object_index_summary.csv")
    
    return fetcher


if __name__ == "__main__":
    main()
