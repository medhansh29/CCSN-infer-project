#!/usr/local/bin/python3
"""
ZFPS Pipeline
Standalone entry point for the ZTF Forced Photometry Service workflow.
NOT integrated into main.py to avoid spamming ZTF on every pipeline run.

Usage:
  python3 zfps_pipeline.py --submit              # submit new/eligible OIDs
  python3 zfps_pipeline.py --download            # download completed jobs
  python3 zfps_pipeline.py --submit --download   # both in sequence
  python3 zfps_pipeline.py --status              # show registry state

Typical cron schedule (runs daily):
  0 23 * * *  /usr/local/bin/python3 /path/to/zfps_pipeline.py --submit
  0 11 * * *  /usr/local/bin/python3 /path/to/zfps_pipeline.py --download
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.ztf_client import ZTFClient
from src.zfps_registry import (
    load_registry, save_registry,
    record_download, get_pending_downloads,
    print_registry_summary
)


def run_submit(extra_args: list):
    """Delegate to zfps_submit.py which handles all registry-aware submission logic."""
    submit_script = Path(__file__).resolve().parent / "zfps_submit.py"
    cmd = [sys.executable, str(submit_script)] + extra_args
    print("\n" + "=" * 60)
    print("STEP 1: SUBMITTING TO ZFPS")
    print("=" * 60)
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_ra_dec_from_zfps_file(filepath):
    """Extract RA and Dec from the header of a ZFPS file."""
    ra, dec = None, None
    try:
        with open(filepath, 'r') as f:
            for _ in range(20):
                line = f.readline()
                if not line: break
                if "Requested input R.A." in line:
                    try: ra = float(line.split('=')[1].split('degrees')[0].strip())
                    except: pass
                elif "Requested input Dec." in line:
                    try: dec = float(line.split('=')[1].split('degrees')[0].strip())
                    except: pass
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return ra, dec

def run_download():
    """Download completed ZFPS jobs and update the registry by matching coordinates."""
    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOADING COMPLETED JOBS")
    print("=" * 60)

    client = ZTFClient()
    client.download_pending_lightcurves() # Best effort download

    import glob
    downloaded_paths = glob.glob("data/ztf_forced_photometry/batchfp_req*_lc.txt")

    if not downloaded_paths:
        print("No ZTF batch request files found locally.")
        return

    registry = load_registry()
    pending  = get_pending_downloads(registry)

    if not pending:
        print("No OIDs pending in registry — nothing to update.")
        return

    print(f"\nMatching {len(downloaded_paths)} downloaded files to {len(pending)} pending OIDs by coordinates...")
    
    # Pre-parse RA/Dec for all downloaded files
    file_coords = {}
    for fp in downloaded_paths:
        ra, dec = get_ra_dec_from_zfps_file(fp)
        if ra is not None and dec is not None:
            file_coords[fp] = (ra, dec)

    matched_count = 0
    for oid in pending:
        entry = registry.get(oid, {})
        obj_ra = entry.get("ra")
        obj_dec = entry.get("dec")
        
        if obj_ra is None or obj_dec is None:
            print(f"  ⚠️  {oid}: Missing RA/Dec in registry, cannot match.")
            continue
            
        candidate_files = []
        for fp, (file_ra, file_dec) in file_coords.items():
            dist = (obj_ra - file_ra)**2 + (obj_dec - file_dec)**2
            if dist < 1e-6:
                candidate_files.append((fp, dist))
                
        import re
        def extract_req_id(filepath):
            m = re.search(r'req(\d+)', filepath)
            return int(m.group(1)) if m else 0

        if candidate_files:
            # Sort candidates by request ID descending (newest first)
            candidate_files.sort(key=lambda x: extract_req_id(x[0]), reverse=True)
            best_file = candidate_files[0][0]
            best_dist = candidate_files[0][1]
            
            old_file_path = entry.get("file_path")
            
            # Clean up the old redundant batch file to save disk space
            if old_file_path and old_file_path != best_file and os.path.exists(old_file_path):
                try:
                    os.remove(old_file_path)
                    print(f"  🗑️  Deleted redundant old file: {os.path.basename(old_file_path)}")
                except Exception as e:
                    pass

            record_download(oid, best_file, registry)
            print(f"  ✅ {oid:<15} → {os.path.basename(best_file)} (dist: {best_dist**0.5:.6f} deg)")
            matched_count += 1
            # Remove from pool so we don't assign it twice if there are very close objects
            del file_coords[best_file]
        else:
            print(f"  ⏳ {oid:<14} : no matching download file found")

    # Clean up unmatched files to keep folder clean and add them to ignore list
    assigned_files = set()
    for obj, data in registry.items():
        if data.get("file_path"):
            assigned_files.add(data["file_path"])

    ignore_list_path = "data/ztf_forced_photometry/ignore_list.txt"
    cleaned_count = 0
    with open(ignore_list_path, 'a') as ignore_f:
        for f in downloaded_paths:
            if f not in assigned_files:
                basename = os.path.basename(f)
                ignore_f.write(basename + "\n")
                if os.path.exists(f):
                    os.remove(f)
                cleaned_count += 1
                
    if cleaned_count > 0:
        print(f"\nCleaned up {cleaned_count} unmatched files and added to ignore list.")

    print(f"\nRegistry updated: {matched_count} new files matched.")
    save_registry(registry)
    print("\n✅ ZFPS pipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="ZFPS Pipeline — submit new OIDs and/or download completed jobs."
    )
    parser.add_argument('--submit',   action='store_true', help="Submit new/eligible OIDs to ZFPS")
    parser.add_argument('--download', action='store_true', help="Download completed ZFPS jobs")
    parser.add_argument('--check',    action='store_true', help="Query ZTF server status and print wget lines (no download)")
    parser.add_argument('--status',   action='store_true', help="Show local registry summary and exit")
    parser.add_argument('--oid',      type=str, default=None, help="Target a single OID")
    parser.add_argument('--buffer',   type=float, default=30.0, help="JD buffer days (default: 30)")
    parser.add_argument('--dry-run',  action='store_true', help="Dry run: don't actually POST to ZTF")
    args = parser.parse_args()

    if args.status:
        registry = load_registry()
        print_registry_summary(registry)
        return

    if args.check:
        print("\n" + "=" * 60)
        print("ZTF SERVER STATUS CHECK")
        print("=" * 60 + "\n")
        client = ZTFClient()
        client.check_job_status()
        return

    if not args.submit and not args.download:
        parser.print_help()
        print("\n⚠️  Specify at least one of --submit, --download, or --check.")
        sys.exit(1)

    # Build extra args to pass through to zfps_submit.py
    submit_args = []
    if args.oid:      submit_args += ['--oid', args.oid]
    if args.buffer:   submit_args += ['--buffer', str(args.buffer)]
    if args.dry_run:  submit_args += ['--dry-run']

    if args.submit:
        run_submit(submit_args)

    if args.download:
        if args.dry_run:
            print("\n[DRY RUN] Skipping download step.")
        else:
            run_download()

    print("\n✅ ZFPS pipeline complete.")


if __name__ == "__main__":
    main()
