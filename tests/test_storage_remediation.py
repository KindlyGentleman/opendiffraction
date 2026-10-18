"""Regression test suite for OD-008 and OD-009: Bounded-memory streaming and resume safety."""

from __future__ import annotations

import os
import tempfile
import h5py
import numpy as np
import pytest

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.generator.storage import (
    DatasetWriter,
    IncrementalShardWriter,
    compute_config_fingerprint,
)


def test_incremental_shard_write_bounded_memory() -> None:
    """IncrementalShardWriter must write samples sequentially without buffering in RAM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, "shard_test.h5")
        writer = IncrementalShardWriter(
            shard_path=shard_path,
            total_samples=3,
            storage_mode="compact",
            config_fingerprint="test_fp_123",
        )

        for i in range(3):
            sample = {
                "target_dpr": np.ones((64, 64), dtype=np.float32) * float(i),
                "intensity_observed": np.ones((512, 512), dtype=np.float32) * float(i * 10),
                "valid_mask": np.ones((512, 512), dtype=bool),
            }
            writer.write_sample(i, sample)

        checksum = writer.finalize()
        assert os.path.exists(shard_path)
        assert len(checksum) == 64

        with h5py.File(shard_path, "r") as f:
            assert f.attrs["config_fingerprint"] == "test_fp_123"
            assert f.attrs["num_samples"] == 3
            assert f["input"].shape == (3, 1, 512, 512)
            assert f["target"].shape == (3, 1, 64, 64)
            np.testing.assert_allclose(f["input"][1, 0], 10.0)
            np.testing.assert_allclose(f["target"][2, 0], 2.0)


def test_resume_rejects_incompatible_config_hash() -> None:
    """Resume verification must reject shards generated with different configuration fingerprints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DatasetWriter(tmpdir, storage_mode="compact")
        shard_path = os.path.join(tmpdir, "shard_000.h5")

        inc = writer.create_incremental_shard_writer(
            shard_path, total_samples=2, config_fingerprint="config_version_A"
        )
        for i in range(2):
            sample = {
                "target_dpr": np.ones((64, 64), dtype=np.float32),
                "intensity_observed": np.ones((512, 512), dtype=np.float32),
                "valid_mask": np.ones((512, 512), dtype=bool),
            }
            inc.write_sample(i, sample)
        inc.finalize()

        # Matching configuration hash must pass
        assert writer.verify_shard(shard_path, 2, expected_config_hash="config_version_A")

        # Incompatible configuration hash must be rejected
        assert not writer.verify_shard(shard_path, 2, expected_config_hash="config_version_B")


def test_resume_rejects_corrupted_shard() -> None:
    """verify_shard must return False when encountering a truncated or corrupted file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DatasetWriter(tmpdir)
        corrupted_path = os.path.join(tmpdir, "corrupt.h5")
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_VALID_HDF5_FILE_TRUNCATED_BYTES")

        assert not writer.verify_shard(corrupted_path, 5)


def test_generate_dataset_incremental_execution_roundtrip() -> None:
    """End-to-end dataset generation must write incremental shards with config fingerprint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res1 = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=12, seed=777)
        assert res1["status"] == "success"
        assert res1["written_samples"] == 12

        # Check that written shard has config_fingerprint attribute
        shard_file = os.path.join(tmpdir, "data", "train_000.h5")
        with h5py.File(shard_file, "r") as f:
            assert "config_fingerprint" in f.attrs
            assert len(f.attrs["config_fingerprint"]) == 64

        # Resume with identical settings must recognize completion
        res2 = generate_dataset(
            output_dir=tmpdir, preset="quick", total_samples=12, seed=777, resume=True
        )
        assert res2["status"] == "already_completed"
