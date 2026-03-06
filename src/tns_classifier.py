#!/usr/bin/env python3
"""
TNS Classification Filter

Queries the Transient Name Server (TNS) API for official spectroscopic
classifications and flags objects that are NOT Type II supernovae.

The REFITT model is trained on Type IIP only, so non-II objects
produce unreliable results and should be excluded from analysis.

Requires TNS_API_KEY, TNS_BOT_ID, and TNS_BOT_NAME in a .env file.
"""

import json
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

# Load credentials from .env
load_dotenv()

TNS_API_KEY = os.getenv('TNS_API_KEY', '')
TNS_BOT_ID = int(os.getenv('TNS_BOT_ID', '0'))
TNS_BOT_NAME = os.getenv('TNS_BOT_NAME', '')

TNS_BASE_URL = 'https://www.wis-tns.org/api/get'

# Accepted SN types — anything else gets flagged
ACCEPTED_TYPES = {'SN II', 'SN IIP'}

# Cache file to avoid repeated API calls
CACHE_FILE = '.tns_cache.json'


class TNSClassifier:
    """Classifies ZTF objects using official TNS spectroscopic data."""

    def __init__(self, cache_file: str = CACHE_FILE,
                 api_delay: float = 1.0):
        """
        Args:
            cache_file: Path to JSON cache for API results.
            api_delay: Seconds between API calls (TNS rate limit: 1 req/s).
        """
        if not TNS_API_KEY or not TNS_BOT_ID:
            raise ValueError(
                'TNS credentials not found. Add TNS_API_KEY, TNS_BOT_ID, '
                'and TNS_BOT_NAME to your .env file.'
            )
        self.headers = self._build_headers()
        self.cache_file = Path(cache_file)
        self.api_delay = api_delay
        self.cache = self._load_cache()

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_headers() -> dict:
        tns_marker = json.dumps({
            'tns_id': TNS_BOT_ID,
            'type': 'bot',
            'name': TNS_BOT_NAME,
        })
        return {'User-Agent': f'tns_marker{tns_marker}'}

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    # ------------------------------------------------------------------ #
    #  TNS API calls                                                      #
    # ------------------------------------------------------------------ #

    def _search_by_internal_name(self, ztf_id: str) -> dict | None:
        """Search TNS for a ZTF internal name → return first match."""
        resp = requests.post(
            f'{TNS_BASE_URL}/search',
            data={
                'api_key': TNS_API_KEY,
                'data': json.dumps({'internal_name': ztf_id}),
            },
            headers=self.headers,
        )
        if resp.status_code != 200:
            return None

        results = resp.json().get('data', [])
        return results[0] if results else None

    def _get_object(self, objname: str) -> dict | None:
        """Get full TNS object details by name."""
        resp = requests.post(
            f'{TNS_BASE_URL}/object',
            data={
                'api_key': TNS_API_KEY,
                'data': json.dumps({
                    'objname': objname,
                    'photometry': '0',
                    'spectra': '0',
                }),
            },
            headers=self.headers,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get('data', {})

    # ------------------------------------------------------------------ #
    #  Single-object classification                                       #
    # ------------------------------------------------------------------ #

    def classify_object(self, ztf_id: str) -> dict:
        """
        Get TNS classification for one ZTF object.

        Returns a dict with:
            object_id      – ZTF ID
            tns_name       – official TNS name (e.g. "SN 2025ryv")
            tns_type       – spectroscopic type (e.g. "SN II")
            redshift       – TNS redshift
            is_flagged     – True if tns_type is not in ACCEPTED_TYPES
            flag_reason    – human-readable reason (or empty)
        """
        if ztf_id in self.cache:
            return self.cache[ztf_id]

        result = self._query_tns(ztf_id)
        self.cache[ztf_id] = result
        return result

    def _query_tns(self, ztf_id: str) -> dict:
        """Query TNS for a ZTF object's classification."""
        result = {
            'object_id': ztf_id,
            'tns_name': '',
            'tns_type': '',
            'redshift': None,
            'is_flagged': False,
            'flag_reason': '',
        }

        # Step 1: Search for the object
        match = self._search_by_internal_name(ztf_id)
        time.sleep(self.api_delay)

        if not match:
            result['flag_reason'] = 'Not found on TNS'
            return result

        objname = match.get('objname', '')
        prefix = match.get('prefix', '')
        result['tns_name'] = f'{prefix} {objname}'.strip()

        # Step 2: Get full object details
        obj = self._get_object(objname)
        time.sleep(self.api_delay)

        if not obj:
            result['flag_reason'] = 'Could not retrieve TNS object details'
            return result

        # Extract classification
        obj_type = obj.get('object_type', {})
        tns_type = obj_type.get('name', '') if isinstance(obj_type, dict) else ''
        result['tns_type'] = tns_type
        result['redshift'] = obj.get('redshift')

        # Flag decision: only based on official TNS type
        if not tns_type:
            # No classification yet — don't flag (keep in analysis)
            result['flag_reason'] = 'No TNS classification yet'
        elif tns_type not in ACCEPTED_TYPES:
            result['is_flagged'] = True
            result['flag_reason'] = f'TNS type is {tns_type}, not SN II/IIP'

        return result

    # ------------------------------------------------------------------ #
    #  Batch classification                                               #
    # ------------------------------------------------------------------ #

    def classify_batch(self, object_ids: List[str],
                       verbose: bool = True) -> pd.DataFrame:
        """
        Classify a list of objects and return a DataFrame.

        Args:
            object_ids: List of ZTF object IDs.
            verbose: Print progress.

        Returns:
            DataFrame with one row per object.
        """
        rows = []
        new_queries = 0

        for i, oid in enumerate(sorted(object_ids)):
            was_cached = oid in self.cache
            info = self.classify_object(oid)
            rows.append(info)

            if not was_cached:
                new_queries += 1

            if verbose and (i + 1) % 10 == 0:
                print(f'  Classified {i + 1}/{len(object_ids)} objects '
                      f'({new_queries} new API calls)')

        # Persist cache after batch
        if new_queries > 0:
            self._save_cache()
            if verbose:
                print(f'  💾 Cache saved ({len(self.cache)} objects)')

        return pd.DataFrame(rows)

    def get_flagged_ids(self, object_ids: List[str]) -> set:
        """Return set of object IDs that should be excluded."""
        return {oid for oid in object_ids
                if self.classify_object(oid).get('is_flagged', False)}

    def get_clean_ids(self, object_ids: List[str]) -> list:
        """Return list of object IDs safe for IIP analysis."""
        return [oid for oid in object_ids
                if not self.classify_object(oid).get('is_flagged', False)]


def generate_flagged_csv(object_ids: List[str],
                         convergence_csv: str = 'data/convergence_metrics.csv',
                         output_csv: str = 'data/flagged_non_iip_objects.csv'):
    """
    Classify all objects via TNS, merge with convergence metrics,
    and save a CSV of flagged (non-IIP) objects.
    """
    print(f'🔍 Classifying {len(object_ids)} objects via TNS...')
    clf = TNSClassifier()
    class_df = clf.classify_batch(object_ids)

    # Merge with convergence metrics if available
    conv_path = Path(convergence_csv)
    if conv_path.exists():
        conv_df = pd.read_csv(conv_path)
        metric_cols = ['object_id']
        for col in conv_df.columns:
            if any(k in col for k in ['rel_uncertainty', 'rmse',
                                       'converged', 'phase_span',
                                       'num_observations']):
                metric_cols.append(col)
        conv_subset = conv_df[
            [c for c in metric_cols if c in conv_df.columns]
        ]
        class_df = class_df.merge(conv_subset, on='object_id', how='left')

    # Split into flagged and clean
    flagged = class_df[class_df['is_flagged']]
    clean = class_df[~class_df['is_flagged']]

    # Save flagged objects CSV
    flagged.to_csv(output_csv, index=False)

    print(f'\n📊 TNS Classification Results:')
    print(f'  ✅ Clean (SN II/IIP): {len(clean)} objects')
    print(f'  🚩 Flagged (non-II): {len(flagged)} objects')

    if len(flagged) > 0:
        print(f'\n🚩 Flagged objects:')
        for _, row in flagged.iterrows():
            print(f'  {row["object_id"]} ({row["tns_name"]}): {row["flag_reason"]}')

    print(f'\n💾 Flagged objects saved to: {output_csv}')
    return class_df


# ------------------------------------------------------------------ #
#  CLI entrypoint                                                     #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Classify ZTF objects via TNS and flag non-II types'
    )
    parser.add_argument('--output', type=str,
                        default='data/flagged_non_iip_objects.csv',
                        help='Output CSV for flagged objects')
    args = parser.parse_args()

    # Gather object IDs from the JSON fetcher
    from src.fetch_successive_jsons import JSONFetcher
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    all_ids = fetcher.get_all_object_ids()

    generate_flagged_csv(all_ids, output_csv=args.output)
