"""Automated fault-injection and verification suite for crash-safe, idempotent resume."""

from __future__ import annotations

import os
import shutil
import tempfile
import h5py
import numpy as np
import pandas as pd

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.generator.storage import DatasetWriter
from opendiffraction.io.loader import OpenDiffractionDataset


def test_resume_interruption_fault_injection() -> bool:
    """Simulate an interrupted run after 1 shard, inject corruption, and verify resume."""
    print("\n--- Test 1: Fault Injection and Interrupted Resume ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Total 24 samples with shard_size=8 -> exactly 3 shards
        total_samples = 24
        shard_size = 8

        # Step 1: Generate initial run of only 8 samples (simulating crash after shard 0)
        print("1. Running initial partial batch (interrupted after 1 shard)...")
        res1 = generate_dataset(
            output_dir=tmpdir,
            preset="quick",
            total_samples=total_samples,
            shard_size=shard_size,
            seed=42,
            resume=True,
            max_shards_to_write=1,
        )
        assert res1["written_samples"] == shard_size

        data_dir = os.path.join(tmpdir, "data")
        shards_after_step1 = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
        assert len(shards_after_step1) >= 1
        first_shard = shards_after_step1[0]
        first_shard_path = os.path.join(data_dir, first_shard)

        # Record first shard content before interruption
        with h5py.File(first_shard_path, "r") as f:
            first_shard_input = np.copy(f["input"][:])
            first_shard_count = len(f["input"])

        # Step 2: Inject orphaned .tmp file to simulate crash during write
        print("2. Injecting orphaned .tmp file to simulate crash...")
        corrupt_tmp = os.path.join(data_dir, "train_001.h5.tmp")
        with open(corrupt_tmp, "w") as f:
            f.write("INCOMPLETE_ORPHANED_DATA_BYTES")

        # Step 3: Resume generation for full 24 samples
        print("3. Resuming generation for all 24 samples...")
        res2 = generate_dataset(
            output_dir=tmpdir,
            preset="quick",
            total_samples=total_samples,
            shard_size=shard_size,
            seed=42,
            resume=True,
        )
        print(f"Resume result: {res2}")

        # Step 4: Verify that first shard was NOT overwritten
        with h5py.File(first_shard_path, "r") as f:
            resumed_shard_input = f["input"][:]
            assert np.array_equal(first_shard_input, resumed_shard_input), (
                "CRITICAL ERROR: First shard was overwritten during resume!"
            )

        # Step 5: Verify total dataset integrity and row alignment
        ds = OpenDiffractionDataset(tmpdir)
        assert len(ds) == total_samples, f"Expected {total_samples} samples, got {len(ds)}"

        # Check metadata uniqueness
        df_meta = pd.read_csv(os.path.join(tmpdir, "metadata.csv"))
        assert len(df_meta) == total_samples
        assert df_meta["sample_id"].nunique() == total_samples, "Duplicate sample IDs in metadata!"

        # Step 6: Verify row indices point to valid HDF5 items
        for idx in range(total_samples):
            sample = ds[idx]
            assert sample.input.shape == (1, 512, 512)
            assert np.isfinite(sample.input).all()

        print("PASS: Fault injection and resume test succeeded without data loss or corruption.")
        return True


def test_no_resume_policy() -> bool:
    """Verify that --no-resume refuses to silently clobber existing data without --overwrite."""
    print("\n--- Test 2: Safe Overwrite Policy Verification ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate small dataset
        generate_dataset(output_dir=tmpdir, total_samples=8, shard_size=8, seed=42)

        # Attempt to run with resume=False and overwrite=False
        try:
            generate_dataset(
                output_dir=tmpdir,
                total_samples=8,
                shard_size=8,
                seed=42,
                resume=False,
                overwrite=False,
            )
            raise AssertionError("Expected RuntimeError when running without resume or overwrite!")
        except RuntimeError as e:
            print(f"Caught expected error: {e}")

        # Run with overwrite=True: should succeed
        res = generate_dataset(
            output_dir=tmpdir,
            total_samples=8,
            shard_size=8,
            seed=42,
            resume=False,
            overwrite=True,
        )
        assert res["status"] == "success"
        print("PASS: Safe overwrite policy verified.")
        return True


def test_deterministic_replay() -> bool:
    """Verify that two independent runs with identical seeds produce bit-identical data."""
    print("\n--- Test 3: Deterministic Replay Verification ---")
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        generate_dataset(output_dir=dir1, total_samples=16, shard_size=8, seed=123)
        generate_dataset(output_dir=dir2, total_samples=16, shard_size=8, seed=123)

        df1 = pd.read_csv(os.path.join(dir1, "metadata.csv"))
        df2 = pd.read_csv(os.path.join(dir2, "metadata.csv"))
        pd.testing.assert_frame_equal(df1, df2)

        ds1 = OpenDiffractionDataset(dir1)
        ds2 = OpenDiffractionDataset(dir2)

        for i in range(16):
            s1 = ds1[i]
            s2 = ds2[i]
            assert np.array_equal(s1.input, s2.input), f"Mismatch in sample {i} input"
            assert np.array_equal(s1.target, s2.target), f"Mismatch in sample {i} target"

        print("PASS: Deterministic replay verified bit-for-bit.")
        return True


if __name__ == "__main__":
    t1 = test_resume_interruption_fault_injection()
    t2 = test_no_resume_policy()
    t3 = test_deterministic_replay()
    if t1 and t2 and t3:
        print("\nALL RESUME AND INTEGRITY VERIFICATION CHECKS PASSED SUCCESSFULLY.")
