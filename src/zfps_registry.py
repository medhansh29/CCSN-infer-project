#!/usr/local/bin/python3
"""
ZFPS Registry
Tracks the submission and download status of every OID sent to the ZTF
Forced Photometry Service, and determines when re-submission is appropriate.

Re-submission logic (per ZTF guidance):
  - Never re-submit if jd_end of the last request has not yet passed.
    (ZTF is still collecting / computing within that window.)
  - Re-submit once jd_end has elapsed, because new epochs now exist
    beyond the previously requested window.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

REGISTRY_PATH = os.path.join("data", "zfps_registry.json")
MJD_TO_JD = 2400000.5


def _current_jd() -> float:
    """Return the current time as a Julian Date."""
    from astropy.time import Time
    return Time.now().jd


def load_registry() -> dict:
    """Load the registry from disk. Returns empty dict if it doesn't exist yet."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    """Persist the registry to disk."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def needs_submission(oid: str, registry: dict) -> tuple[bool, str]:
    """
    Determine whether an OID should be submitted (or re-submitted) to ZFPS.

    Returns:
        (should_submit: bool, reason: str)
    """
    if oid not in registry:
        return True, "new object"

    entry = registry[oid]
    jd_end = entry.get("jd_end")

    if jd_end is None:
        return True, "no jd_end recorded in registry"

    current_jd = _current_jd()

    if current_jd < jd_end:
        days_remaining = jd_end - current_jd
        return False, f"window not elapsed ({days_remaining:.1f} days remaining until JD {jd_end:.2f})"

    # jd_end has passed — re-submit to capture new epochs
    return True, f"window elapsed (jd_end={jd_end:.2f} passed {current_jd - jd_end:.1f} days ago)"


def record_submission(oid: str, jd_start: float, jd_end: float, registry: dict) -> dict:
    """
    Record a successful ZFPS submission for an OID in the registry.
    Resets download state since this is a new/updated request.
    """
    registry[oid] = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "jd_start": round(jd_start, 4),
        "jd_end": round(jd_end, 4),
        "downloaded": False,
        "file_path": None
    }
    return registry


def record_download(oid: str, file_path: str, registry: dict) -> dict:
    """Mark an OID's forced photometry as successfully downloaded."""
    if oid not in registry:
        registry[oid] = {}
    registry[oid]["downloaded"] = True
    registry[oid]["file_path"] = file_path
    return registry


def get_pending_downloads(registry: dict) -> list[str]:
    """Return OIDs that have been submitted but not yet downloaded."""
    return [
        oid for oid, entry in registry.items()
        if not entry.get("downloaded", False)
    ]


def print_registry_summary(registry: dict) -> None:
    """Print a human-readable summary of the registry state."""
    if not registry:
        print("Registry is empty — no OIDs have been submitted yet.")
        return

    current_jd = _current_jd()
    total = len(registry)
    downloaded = sum(1 for e in registry.values() if e.get("downloaded"))
    pending = total - downloaded

    print(f"\n{'='*60}")
    print("ZFPS REGISTRY SUMMARY")
    print(f"{'='*60}")
    print(f"  Total OIDs tracked : {total}")
    print(f"  Downloaded         : {downloaded}")
    print(f"  Pending download   : {pending}")
    print(f"{'='*60}\n")

    for oid, entry in registry.items():
        jd_end = entry.get("jd_end", "?")
        submitted = entry.get("submitted_at", "?")[:10]
        dl = "✅" if entry.get("downloaded") else "⏳"
        window_status = ""
        if isinstance(jd_end, float):
            if current_jd < jd_end:
                window_status = f"(window open, {jd_end - current_jd:.0f}d left)"
            else:
                window_status = f"(window elapsed {current_jd - jd_end:.0f}d ago)"
        print(f"  {dl} {oid:<22} submitted={submitted}  jd_end={jd_end}  {window_status}")
    print()
