#!/usr/local/bin/python3
"""
ZTF Forced Photometry Service (ZFPS) Submission Script

Submits batch forced photometry requests for all OIDs found in the local
inference JSON outputs. Submission is registry-aware:

  - New OIDs are always submitted.
  - Previously submitted OIDs are skipped until their requested JD window
    has elapsed, at which point they are re-submitted with an updated range.

ZTF limits: 1500 sky positions per request.
ZTF guidance: include ≥30 day buffer before/after the transient window
              to allow baseline corrections and uncertainty rescaling.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add project root to path
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.fetch_successive_jsons import JSONFetcher
from src.zfps_registry import (
    load_registry, save_registry,
    needs_submission, record_submission,
    print_registry_summary
)

ZFPS_SUBMIT_URL = 'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit'
ZFPS_AUTH = ('ztffps', 'dontgocrazy!')
MJD_TO_JD = 2400000.5
BATCH_SIZE = 1500  # ZTF hard limit per request


def fetch_alerce_data(oid: str):
    """Fetch meanra, meandec, firstmjd, and lastmjd from ALeRCE for a given OID."""
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return (
                data.get('meanra'),
                data.get('meandec'),
                data.get('firstmjd'),
                data.get('lastmjd')
            )
        print(f"⚠️  ALeRCE {r.status_code} for {oid}")
        return None, None, None, None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  ALeRCE error for {oid}: {e}")
        return None, None, None, None


def submit_batch(ra_list, dec_list, jd_start, jd_end, email, userpass, dry_run=False):
    """POST one batch of up to 1500 targets to ZFPS."""
    payload = {
        'ra':      json.dumps(ra_list),
        'dec':     json.dumps(dec_list),
        'jdstart': json.dumps(jd_start),
        'jdend':   json.dumps(jd_end),
        'email':   email,
        'userpass': userpass
    }

    if dry_run:
        print(f"  [DRY RUN] Would POST {len(ra_list)} targets  JD {jd_start:.4f} → {jd_end:.4f}")
        return True

    try:
        r = requests.post(ZFPS_SUBMIT_URL, auth=ZFPS_AUTH, data=payload, timeout=30)
        print(f"  ZFPS response: HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"  Response body: {r.text[:300]}")
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Submission failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Registry-aware ZFPS batch submitter. "
                    "Submits new OIDs and re-submits when their JD window has elapsed."
    )
    parser.add_argument('--oid', type=str, default=None,
                        help="Target a single OID (e.g. for testing)")
    parser.add_argument('--buffer', type=float, default=100.0,
                        help="Days to pad before/after the observation window (default: 100)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what would be submitted without actually POSTing")
    parser.add_argument('--status', action='store_true',
                        help="Print registry summary and exit")
    args = parser.parse_args()

    # Load credentials
    load_dotenv()
    email    = os.getenv('ZTF_EMAIL')
    userpass = os.getenv('ZTF_USERPASS')
    if not email or not userpass:
        print("❌ ZTF_EMAIL and ZTF_USERPASS must be set in .env")
        sys.exit(1)

    # Load registry
    registry = load_registry()

    if args.status:
        print_registry_summary(registry)
        return

    print("=" * 60)
    print("ZFPS REGISTRY-AWARE SUBMITTER")
    print("=" * 60)

    # ── 1. Gather OIDs ────────────────────────────────────────────────
    fetcher = JSONFetcher()
    object_index = fetcher.scan_directories()

    if args.oid:
        if args.oid not in object_index:
            print(f"❌ {args.oid} not found in local JSON files.")
            sys.exit(1)
        candidate_oids = [args.oid]
        print(f"\n⚡ Single-OID mode: {args.oid}")
    else:
        candidate_oids = list(object_index.keys())
        print(f"\nFound {len(candidate_oids)} OIDs in JSON index.")

    # ── 2. Filter via registry ─────────────────────────────────────────
    to_submit = []
    skipped   = []

    for oid in candidate_oids:
        should, reason = needs_submission(oid, registry)
        if should:
            to_submit.append(oid)
            print(f"  ✅ {oid:<22} → SUBMIT  ({reason})")
        else:
            skipped.append(oid)
            print(f"  ⏭️  {oid:<22} → SKIP    ({reason})")

    print(f"\n{len(to_submit)} to submit, {len(skipped)} skipped.")

    if not to_submit:
        print("Nothing to do.")
        return

    # ── 3. Compute per-OID JD ranges and fetch coordinates ────────────
    print(f"\nFetching RA/Dec from ALeRCE and computing JD windows...")

    submit_targets = []   # list of (oid, ra, dec, jd_start, jd_end)

    for oid in tqdm(to_submit):
        timeline = fetcher.get_object_timeline(oid)

        if 'mjd' not in timeline.columns or timeline['mjd'].isna().all():
            print(f"  ⚠️  {oid}: no MJD data, skipping.")
            continue

        ra, dec, firstmjd, lastmjd = fetch_alerce_data(oid)
        if ra is None or dec is None or firstmjd is None or lastmjd is None:
            print(f"  ⚠️  {oid}: could not fetch complete ALeRCE data, skipping.")
            continue

        jd_start = firstmjd + MJD_TO_JD - args.buffer
        jd_end   = lastmjd + MJD_TO_JD + args.buffer

        submit_targets.append((
            oid,
            float('%.7f' % ra),
            float('%.7f' % dec),
            jd_start,
            jd_end
        ))

    if not submit_targets:
        print("❌ No valid targets to submit after coordinate lookup.")
        return

    # ── 4. Batch and submit (≤1500 per request) ────────────────────────
    # ZFPS requires a single jd_start/jd_end per batch, so we use the
    # global min/max across all targets in the batch.
    print(f"\nSubmitting {len(submit_targets)} targets in batches of ≤{BATCH_SIZE}...")

    for batch_start in range(0, len(submit_targets), BATCH_SIZE):
        batch = submit_targets[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(submit_targets) + BATCH_SIZE - 1) // BATCH_SIZE

        ra_list  = [t[1] for t in batch]
        dec_list = [t[2] for t in batch]
        jd_start = min(t[3] for t in batch)
        jd_end   = max(t[4] for t in batch)

        print(f"\nBatch {batch_num}/{total_batches}: {len(batch)} targets  "
              f"JD {jd_start:.2f} → {jd_end:.2f}")

        success = submit_batch(ra_list, dec_list, jd_start, jd_end,
                               email, userpass, dry_run=args.dry_run)

        if success:
            # Write each OID's individual window to the registry
            for oid, ra, dec, js, je in batch:
                record_submission(oid, js, je, registry)

    save_registry(registry)
    print(f"\n✅ Registry updated → {len(submit_targets)} OIDs recorded.")

    if args.dry_run:
        print("(Dry run — nothing was actually submitted to ZTF.)")


if __name__ == "__main__":
    main()
