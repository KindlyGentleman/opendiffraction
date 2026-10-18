"""Tests for resume integrity, fail-closed fingerprint verification, and bounded crash recovery."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import h5py
import numpy as np
import pytest

from opendiffraction.generator.storage import (
    DatasetWriter,
    IncrementalShardWriter,
    compute_file_sha256,
)
from opendiffraction.generator.pipeline import generate_dataset


def test_empty_or_missing_config_fingerprint_fails_closed() -> None:
    """A shard with missing or empty config_fingerprint attribute must fail verification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = DatasetWriter(tmp_dir, shard_size=10, storage_mode="extended")
        shard_path = os.path.join(tmp_dir, "data", "shard_test.h5")

        # Create shard without config_fingerprint attribute
        with h5py.File(shard_path, "w") as f:
            f.create_dataset("input", data=np.zeros((5, 1, 512, 512), dtype=np.float32))
            f.create_dataset("target", data=np.zeros((5, 1, 64, 64), dtype=np.float32))
            f.create_dataset("mask", data=np.ones((5, 1, 512, 512), dtype=bool))
            f.create_dataset("object_amplitude", data=np.zeros((5, 1, 64, 64), dtype=np.float32))
            f.create_dataset("intensity_measured", data=np.zeros((5, 1, 512, 512), dtype=np.float32))

        # Missing fingerprint:
        assert not writer.verify_shard(
            shard_path, expected_count=5, expected_config_hash="target_fingerprint_123"
        ), "Shard with missing config_fingerprint must return False when fingerprint is expected"

        # Explicitly empty fingerprint:
        with h5py.File(shard_path, "a") as f:
            f.attrs["config_fingerprint"] = ""

        assert not writer.verify_shard(
            shard_path, expected_count=5, expected_config_hash="target_fingerprint_123"
        ), "Shard with empty config_fingerprint must return False when fingerprint is expected"

        # Matching fingerprint succeeds:
        with h5py.File(shard_path, "a") as f:
            f.attrs["config_fingerprint"] = "target_fingerprint_123"

        assert writer.verify_shard(
            shard_path, expected_count=5, expected_config_hash="target_fingerprint_123"
        ), "Shard with matching config_fingerprint must pass verification"


def test_corrupted_shard_checksum_fails_verification() -> None:
    """Tampering with an existing shard must fail checksum integrity verification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = DatasetWriter(tmp_dir, shard_size=5, storage_mode="extended")
        shard_path = os.path.join(tmp_dir, "data", "shard_corrupt.h5")

        inc_w = writer.create_incremental_shard_writer(
            shard_path, total_samples=2, config_fingerprint="cfg_hash_456"
        )
        sample = {
            "target_dpr": np.zeros((64, 64), dtype=np.float32),
            "valid_mask": np.ones((512, 512), dtype=bool),
            "object_amplitude": np.zeros((64, 64), dtype=np.float32),
            "intensity_observed": np.zeros((512, 512), dtype=np.float32),
        }
        inc_w.write_sample(0, sample)
        inc_w.write_sample(1, sample)
        sha = inc_w.finalize()

        writer.record_shard_integrity(
            shard_name="shard_corrupt.h5",
            sha256_hash=sha,
            sample_count=2,
            config_fingerprint="cfg_hash_456",
        )

        assert writer.verify_shard(
            shard_path, expected_count=2, expected_config_hash="cfg_hash_456", verify_checksum=True
        )

        # Corrupt file by appending garbage bytes to end of file
        with open(shard_path, "ab") as f:
            f.write(b"\x00\xff\xee\xdd")

        assert not writer.verify_shard(
            shard_path, expected_count=2, expected_config_hash="cfg_hash_456", verify_checksum=True
        ), "Corrupted shard file must fail SHA-256 verification"


def test_resume_does_not_skip_shard_if_manifest_is_incomplete() -> None:
    """If generator crashed before metadata.csv write, resume must not skip the shard."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = DatasetWriter(tmp_dir, shard_size=5, storage_mode="extended")
        shard_path = os.path.join(tmp_dir, "data", "train_shard_000.h5")

        # Finalize shard
        inc_w = writer.create_incremental_shard_writer(
            shard_path, total_samples=2, config_fingerprint="cfg_hash_789"
        )
        sample = {
            "target_dpr": np.zeros((64, 64), dtype=np.float32),
            "valid_mask": np.ones((512, 512), dtype=bool),
            "object_amplitude": np.zeros((64, 64), dtype=np.float32),
            "intensity_observed": np.zeros((512, 512), dtype=np.float32),
        }
        inc_w.write_sample(0, sample)
        inc_w.write_sample(1, sample)
        sha = inc_w.finalize()
        writer.record_shard_integrity("train_shard_000.h5", sha, 2, "cfg_hash_789")

        # But DO NOT write sample_002 to metadata.csv
        writer.append_manifest_rows([{"sample_id": "sample_001"}])
        writer.update_progress(["sample_001"])

        # has_completed_manifest_records must return False for ["sample_001", "sample_002"]
        assert not writer.has_completed_manifest_records(["sample_001", "sample_002"])


def test_bounded_parallel_generation_and_integrity_manifest() -> None:
    """End-to-end small generation run with 2 workers verifies bounded pipeline and integrity manifest."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = generate_dataset(
            output_dir=tmp_dir,
            total_samples=4,
            shard_size=2,
            profiles_mode="12",
            num_workers=2,
            resume=True,
            overwrite=True,
        )
        assert res["status"] == "success"
        assert res["written_samples"] == 4

        # Verify integrity manifest was written
        integ_file = os.path.join(tmp_dir, "shard_integrity_manifest.json")
        assert os.path.exists(integ_file)
        with open(integ_file, "r", encoding="utf-8") as f:
            integ_data = json.load(f)
        assert sum(v["sample_count"] for v in integ_data.values()) == 4

        # Verify second run with resume skips all shards cleanly
        res2 = generate_dataset(
            output_dir=tmp_dir,
            total_samples=4,
            shard_size=2,
            profiles_mode="12",
            num_workers=2,
            resume=True,
            overwrite=False,
        )
        assert res2["status"] == "already_completed"
        assert res2["skipped_shards"] == len(integ_data)
        assert res2["written_samples"] == 0
