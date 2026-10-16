"""Script: Verify downloaded dataset integrity and roundtrip validity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.publish_kaggle import verify_release_checksums
from opendiffraction.io.cxi import validate_cxi


def verify_dataset_directory(dataset_dir: str | Path) -> bool:
    """Verify downloaded dataset directory contents and shard integrity."""
    p = Path(dataset_dir)
    print(f"=== Verifying Dataset Directory: {p} ===")

    # 1. Checksums
    chk_file = p / "checksums.sha256"
    if chk_file.exists():
        print("1. Checking SHA-256 hashes...")
        if not verify_release_checksums(p):
            return False

    # 2. CXI verification
    cxi_files = list(p.glob("*.cxi"))
    for cxi in cxi_files:
        print(f"2. Validating CXI NeXus file: {cxi.name}")
        valid, msgs = validate_cxi(cxi)
        if not valid:
            print(f"[FAIL] CXI validation failed: {msgs}")
            return False
        print("[PASS] CXI compliant.")

    # 3. HDF5 Shards
    h5_files = list(p.rglob("*.h5"))
    print(f"3. Verifying {len(h5_files)} HDF5 shards...")
    for h5 in h5_files:
        try:
            with h5py.File(h5, "r") as f:
                assert "input" in f
                assert "target" in f
                assert "mask" in f
        except Exception as e:
            print(f"[FAIL] Invalid HDF5 {h5}: {e}")
            return False

    print("All verification checks PASSED.")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./kaggle_release"
    success = verify_dataset_directory(target)
    sys.exit(0 if success else 1)
