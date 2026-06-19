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


def run_download():
    """Download completed ZFPS jobs and update the registry."""
    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOADING COMPLETED JOBS")
    print("=" * 60)

    client = ZTFClient()
    downloaded_paths = client.download_pending_lightcurves()

    # download_pending_lightcurves() returns [] or False when nothing to do
    if not downloaded_paths:
        print("No files were downloaded.")
        return

    # Update registry — ZFPS filenames use batch request IDs (e.g. batchfp_req0004921283_lc.txt),
    # not OID names, so match by order: each downloaded file covers the pending OIDs from
    # the same submission batch. Mark pending OIDs as downloaded with the batch file path.
    registry = load_registry()
    pending  = get_pending_downloads(registry)

    if not pending:
        print("No OIDs pending in registry — nothing to update.")
        return

    # Associate downloaded files with pending OIDs.
    # Because ZTF keeps old batch jobs around, `downloaded_paths` may contain
    # older files. The ZTF batch request IDs are chronological, so sorting
    # the paths guarantees the newest files are at the end. We take the last N files.
    downloaded_paths.sort()
    recent_paths = downloaded_paths[-len(pending):] if len(downloaded_paths) >= len(pending) else downloaded_paths

    for i, oid in enumerate(pending):
        if i < len(recent_paths):
            file_path = recent_paths[i]
            old_file_path = registry.get(oid, {}).get("file_path")
            
            # Clean up the old redundant batch file to save disk space
            if old_file_path and old_file_path != file_path and os.path.exists(old_file_path):
                try:
                    os.remove(old_file_path)
                    print(f"  🗑️  Deleted redundant old file: {os.path.basename(old_file_path)}")
                except Exception as e:
                    print(f"  ⚠️  Failed to delete old file {old_file_path}: {e}")

            record_download(oid, file_path, registry)
            print(f"  ✅ {oid} → {os.path.basename(file_path)}")
        else:
            print(f"  ⚠️  {oid}: no corresponding download file found")

    save_registry(registry)
    print("\nRegistry updated with download status.")


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
