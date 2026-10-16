"""Phase 1 staged verification suite: Stage A (48), Stage B (480), and Stage C (1000).

Validates end-to-end pipeline integrity, group leakage, sharding, throughput,
and resume behavior prior to full 10,000-sample production generation.
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from typing import Any
import h5py
import numpy as np
import pandas as pd

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset


def verify_stage_a(staging_dir: str) -> dict[str, Any]:
    """Stage A: 48 samples verification."""
    print("\n--- Running Stage A (48 samples) ---")
    stage_dir = os.path.join(staging_dir, "stage_a_48")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)

    t0 = time.time()
    res = generate_dataset(
        output_dir=stage_dir,
        total_samples=48,
        profiles_mode="48",
        storage_mode="compact",
        compression="gzip",
        compression_opts=4,
        shard_size=48,
        num_workers=4,
        resume=False,
        overwrite=True,
    )
    wall_time = time.time() - t0
    throughput = 48 / wall_time

    # Verification checks
    meta_path = os.path.join(stage_dir, "metadata.csv")
    assert os.path.exists(meta_path), "metadata.csv missing in Stage A"
    df = pd.read_csv(meta_path)
    assert len(df) == 48, f"Expected 48 rows, found {len(df)}"

    # Check HDF5 contents
    data_dir = os.path.join(stage_dir, "data")
    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    assert len(h5_files) > 0, "No HDF5 files created in Stage A"

    total_bytes = sum(os.path.getsize(os.path.join(data_dir, f)) for f in h5_files)
    avg_bytes = total_bytes / 48

    # Verify tensor shapes and finite values using TorchDataset
    ds = OpenDiffractionTorchDataset(stage_dir, split=None, task="phase_retrieval", return_dict=True)
    sample_0 = ds[0]
    assert sample_0["input"].shape == (1, 512, 512), f"Input shape wrong: {sample_0['input'].shape}"
    assert sample_0["target"].shape == (1, 64, 64), f"Target shape wrong: {sample_0['target'].shape}"
    assert sample_0["mask"].shape == (1, 512, 512), f"Mask shape wrong: {sample_0['mask'].shape}"
    assert np.isfinite(sample_0["input"].numpy()).all(), "Non-finite input detected"
    assert np.isfinite(sample_0["target"].numpy()).all(), "Non-finite target detected"
    ds.close()

    print(f"Stage A Passed: 48 samples in {wall_time:.2f}s ({throughput:.1f} samples/s), {avg_bytes/1024:.1f} KB/sample")
    return {
        "stage": "Stage A (48)",
        "samples": 48,
        "wall_time_s": wall_time,
        "throughput_sps": throughput,
        "total_bytes": total_bytes,
        "avg_bytes_per_sample": avg_bytes,
        "status": "PASS",
    }


def verify_stage_b(staging_dir: str) -> dict[str, Any]:
    """Stage B: 480 samples verification (sharding and zero group leakage)."""
    print("\n--- Running Stage B (480 samples) ---")
    stage_dir = os.path.join(staging_dir, "stage_b_480")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)

    t0 = time.time()
    res = generate_dataset(
        output_dir=stage_dir,
        total_samples=480,
        profiles_mode="48",
        storage_mode="compact",
        compression="gzip",
        compression_opts=4,
        shard_size=240,
        num_workers=4,
        resume=False,
        overwrite=True,
    )
    wall_time = time.time() - t0
    throughput = 480 / wall_time

    meta_path = os.path.join(stage_dir, "metadata.csv")
    df = pd.read_csv(meta_path)
    assert len(df) == 480, f"Expected 480 rows, found {len(df)}"

    # Zero group leakage check
    group_splits = df.groupby("group_id")["split"].nunique()
    leaked_groups = group_splits[group_splits > 1]
    assert len(leaked_groups) == 0, f"Group leakage detected in {len(leaked_groups)} groups!"

    # Multi-acquisition check: 120 base objects x 4 acquisitions = 480
    assert df["base_object_id"].nunique() == 120, f"Expected 120 base objects, got {df['base_object_id'].nunique()}"
    acq_counts = df.groupby("base_object_id")["acquisition_id"].count()
    assert (acq_counts == 4).all(), "Some base objects do not have exactly 4 acquisitions"

    # Splits representation
    splits_present = set(df["split"].unique())
    assert {"train", "valid", "test", "test_ood"}.issubset(splits_present), f"Missing splits: {splits_present}"

    data_dir = os.path.join(stage_dir, "data")
    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    total_bytes = sum(os.path.getsize(os.path.join(data_dir, f)) for f in h5_files)
    avg_bytes = total_bytes / 480

    print(f"Stage B Passed: 480 samples in {wall_time:.2f}s ({throughput:.1f} samples/s), Zero Leakage Verified")
    return {
        "stage": "Stage B (480)",
        "samples": 480,
        "wall_time_s": wall_time,
        "throughput_sps": throughput,
        "total_bytes": total_bytes,
        "avg_bytes_per_sample": avg_bytes,
        "status": "PASS",
    }


def verify_stage_c(staging_dir: str) -> dict[str, Any]:
    """Stage C: 1,000 samples verification (throughput and resume safety)."""
    print("\n--- Running Stage C (1,000 samples) ---")
    stage_dir = os.path.join(staging_dir, "stage_c_1000")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)

    # First run: simulate interruption after 1 shard
    t0 = time.time()
    generate_dataset(
        output_dir=stage_dir,
        total_samples=1000,
        profiles_mode="48",
        storage_mode="compact",
        compression="gzip",
        compression_opts=4,
        shard_size=400,
        num_workers=4,
        resume=False,
        overwrite=True,
        max_shards_to_write=1,
    )

    data_dir = os.path.join(stage_dir, "data")
    initial_shards = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    assert len(initial_shards) == 1, f"Expected 1 shard before resume, found {len(initial_shards)}"

    # Second run: resume to completion
    res_resume = generate_dataset(
        output_dir=stage_dir,
        total_samples=1000,
        profiles_mode="48",
        storage_mode="compact",
        compression="gzip",
        compression_opts=4,
        shard_size=400,
        num_workers=4,
        resume=True,
        overwrite=False,
    )
    wall_time = time.time() - t0
    throughput = 1000 / wall_time

    assert res_resume["skipped_shards"] == 1, "Resume failed to recognize and skip the existing valid shard"

    meta_path = os.path.join(stage_dir, "metadata.csv")
    df = pd.read_csv(meta_path)
    assert len(df) == 1000, f"Expected 1000 rows after resume, found {len(df)}"
    assert df["sample_id"].is_unique, "Duplicate sample IDs detected after resume"

    # Verify no group leakage
    group_splits = df.groupby("group_id")["split"].nunique()
    assert (group_splits == 1).all(), "Group leakage detected in Stage C"

    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    total_bytes = sum(os.path.getsize(os.path.join(data_dir, f)) for f in h5_files)
    avg_bytes = total_bytes / 1000

    print(f"Stage C Passed: 1,000 samples in {wall_time:.2f}s ({throughput:.1f} samples/s), Resume Verified")
    return {
        "stage": "Stage C (1,000)",
        "samples": 1000,
        "wall_time_s": wall_time,
        "throughput_sps": throughput,
        "total_bytes": total_bytes,
        "avg_bytes_per_sample": avg_bytes,
        "status": "PASS",
    }


def main() -> None:
    staging_dir = "d:/03_Proyek/Diffraction_Dataset/output_staging"
    os.makedirs(staging_dir, exist_ok=True)

    stage_a = verify_stage_a(staging_dir)
    stage_b = verify_stage_b(staging_dir)
    stage_c = verify_stage_c(staging_dir)

    print("\n================ STAGE VERIFICATION SUMMARY ================")
    for res in [stage_a, stage_b, stage_c]:
        print(
            f"{res['stage']}: Status={res['status']}, Time={res['wall_time_s']:.2f}s, "
            f"Throughput={res['throughput_sps']:.1f} samp/s, Size={res['total_bytes']/(1024*1024):.1f} MB, "
            f"Avg={res['avg_bytes_per_sample']/1024:.1f} KB/sample"
        )
    print("============================================================")

    # Clean up staging directory to free disk space for Stage D (10,000 samples)
    shutil.rmtree(staging_dir)
    print("Cleaned up staging directory to preserve disk budget for Stage D production.")


if __name__ == "__main__":
    main()
