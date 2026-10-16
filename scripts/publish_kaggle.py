"""Script: Safe, audited Kaggle publication tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def verify_release_checksums(pkg_dir: Path) -> bool:
    """Verify SHA256 checksums of packaged release files."""
    chk_file = pkg_dir / "checksums.sha256"
    if not chk_file.exists():
        print("[ERROR] checksums.sha256 not found.")
        return False

    all_ok = True
    with open(chk_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            expected_hash, fname = parts
            target = pkg_dir / fname
            if not target.exists():
                # Check data subdirectory
                target = pkg_dir / "data" / fname
            if not target.exists():
                print(f"[FAIL] Missing file: {fname}")
                all_ok = False
                continue

            h = hashlib.sha256()
            with open(target, "rb") as bf:
                while chunk := bf.read(1024 * 1024):
                    h.update(chunk)
            actual_hash = h.hexdigest().upper()
            if actual_hash == expected_hash.upper():
                print(f"[PASS] {fname}")
            else:
                print(f"[FAIL] Checksum mismatch for {fname}")
                all_ok = False

    return all_ok


def check_kaggle_auth() -> bool:
    """Check whether Kaggle CLI is authenticated."""
    res = subprocess.run(["python", "-m", "kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
    if "Authentication required" in res.stderr or "Authentication required" in res.stdout:
        return False
    return res.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish OpenDiffraction dataset to Kaggle.")
    parser.add_argument("--release-dir", default="./kaggle_release", help="Path to release directory")
    parser.add_argument("--execute", action="store_true", help="Perform actual remote publication")
    parser.add_argument("--public", action="store_true", help="Make dataset public upon creation")
    args = parser.parse_args()

    pkg_p = Path(args.release_dir)
    if not pkg_p.exists():
        print(f"[ERROR] Release directory {pkg_p} does not exist.")
        sys.exit(1)

    print("=== OpenDiffraction Pre-Publication Check ===")
    meta_json = pkg_p / "dataset-metadata.json"
    if not meta_json.exists():
        print("[ERROR] dataset-metadata.json is missing.")
        sys.exit(1)

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Target Dataset ID: {meta.get('id')}")
    print(f"Title: {meta.get('title')}")

    print("\n1. Verifying SHA-256 Checksums...")
    if not verify_release_checksums(pkg_p):
        print("[ERROR] Checksum verification failed. Aborting.")
        sys.exit(1)

    print("\n2. Checking Kaggle API Authentication...")
    is_auth = check_kaggle_auth()
    if not is_auth:
        print("[WARNING] Kaggle credentials not detected.")
        print("To authenticate, please run:")
        print("  kaggle auth login")
        print("or set KAGGLE_API_TOKEN in your environment.")

    if not args.execute:
        print("\n[DRY RUN COMPLETE] To publish after authenticating, run:")
        print(f"  kaggle datasets create -p {pkg_p} {'--public' if args.public else ''}")
        return

    if not is_auth:
        print("[ERROR] Cannot execute publication without valid Kaggle credentials.")
        sys.exit(1)

    print("\nExecuting publication command...")
    cmd = ["kaggle", "datasets", "create", "-p", str(pkg_p)]
    if args.public:
        cmd.append("--public")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
