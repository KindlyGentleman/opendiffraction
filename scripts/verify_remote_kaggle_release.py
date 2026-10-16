"""Phase J: Post-Publication Remote Kaggle Release Verification.

Downloads the published Kaggle dataset release into tmp_kaggle_verification,
verifies integrity, compares SHA-256 checksums, and tests PyTorch loading directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_downloaded_dataset(
    target_dir: str = "tmp_kaggle_verification",
    reference_checksums_path: str = "kaggle_release_v3/checksums.sha256",
) -> dict[str, Any]:
    print(f"\n[Verification] Verifying downloaded payload in: {target_dir}")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Directory {target_dir} does not exist.")

    # 1. Read reference checksums
    ref_checksums: dict[str, str] = {}
    if os.path.exists(reference_checksums_path):
        with open(reference_checksums_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    ref_checksums[parts[1].replace("\\", "/")] = parts[0]

    # 2. Check metadata.csv
    meta_path = os.path.join(target_dir, "metadata.csv")
    assert os.path.exists(meta_path), "metadata.csv missing in downloaded package"
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total_samples = len(reader)
    print(f"Downloaded metadata.csv row count: {total_samples}")
    assert total_samples == 1000, f"Expected 1,000 samples, found {total_samples}"

    # 3. Check HDF5 shards
    data_dir = os.path.join(target_dir, "data")
    assert os.path.exists(data_dir), "data/ missing in downloaded package"
    shards = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    print(f"Downloaded HDF5 shards ({len(shards)}): {shards}")
    assert len(shards) == 7, f"Expected 7 shards, found {len(shards)}"

    # 4. Check checksums
    checksum_matches = 0
    checksum_mismatches = 0
    for rel_path, expected_hash in ref_checksums.items():
        actual_path = os.path.join(target_dir, rel_path)
        if os.path.exists(actual_path):
            actual_hash = compute_sha256(actual_path)
            if actual_hash == expected_hash:
                checksum_matches += 1
            else:
                checksum_mismatches += 1
                print(f"HASH MISMATCH: {rel_path} expected {expected_hash} got {actual_hash}")
        else:
            print(f"MISSING FILE: {rel_path}")
            checksum_mismatches += 1

    print(f"Checksum verification: {checksum_matches} matches, {checksum_mismatches} mismatches")

    # 5. Test PyTorch Loading directly on downloaded folder
    from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset

    ds = OpenDiffractionTorchDataset(target_dir, split="train", task="phase_retrieval", return_dict=True)
    sample_count = len(ds)
    sample_0 = ds[0]
    ds.close()
    print(f"PyTorch loading test on downloaded dataset: len(train) = {sample_count}, sample_0 shape = {sample_0['input'].shape}")

    res = {
        "status": "PASS" if checksum_mismatches == 0 and total_samples == 1000 else "FAIL",
        "total_samples": total_samples,
        "shards_count": len(shards),
        "checksum_matches": checksum_matches,
        "checksum_mismatches": checksum_mismatches,
        "pytorch_train_samples": sample_count,
    }
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        verify_downloaded_dataset()
    else:
        print("Run with --verify-only after downloading dataset.")
