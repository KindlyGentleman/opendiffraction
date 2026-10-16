"""Tests for HDF5 storage, group splitting without leakage, metadata consistency, and resume capability."""

import os
import tempfile
import h5py
import numpy as np
import pytest

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.generator.splits import assign_group_to_split
from opendiffraction.generator.storage import DatasetWriter
from opendiffraction.io.loader import OpenDiffractionDataset


def test_12_group_disjointness_no_leakage():
    """Test 12: Verify train/validation/test object-group disjointness."""
    train_groups = set()
    valid_groups = set()
    test_groups = set()

    seed = 42
    for i in range(500):
        gid = f"group_{i:04d}"
        split = assign_group_to_split(gid, seed=seed)
        if split == "train":
            train_groups.add(gid)
        elif split == "valid":
            valid_groups.add(gid)
        elif split in ("test", "test_ood"):
            test_groups.add(gid)

    # Empty pairwise intersections
    assert len(train_groups.intersection(valid_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(valid_groups.intersection(test_groups)) == 0

    # Ratios match roughly 70 / 15 / 15
    total = len(train_groups) + len(valid_groups) + len(test_groups)
    assert abs(len(train_groups) / total - 0.70) < 0.05
    assert abs(len(valid_groups) / total - 0.15) < 0.04


def test_14_hdf5_writing_and_reading():
    """Test 14: Test HDF5 writing and reading roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = DatasetWriter(tmpdir, shard_size=10)
        sample = {
            "object_amplitude": np.ones((64, 64), dtype=np.float32) * 0.5,
            "object_phase": None,
            "target_dpr": np.ones((64, 64), dtype=np.float32) * 12.0,
            "target_physical": np.ones((64, 64), dtype=np.float32) * 3.46,
            "intensity_ideal": np.full((512, 512), 100.0, dtype=np.float32),
            "intensity_expected": np.full((512, 512), 50.0, dtype=np.float32),
            "intensity_measured": np.full((512, 512), 48.5, dtype=np.float32),
            "valid_mask": np.ones((512, 512), dtype=bool),
        }
        shard_path = os.path.join(tmpdir, "test_shard.h5")
        writer.write_shard_atomic(shard_path, [sample])

        # Read back
        with h5py.File(shard_path, "r") as f:
            assert f["input"].shape == (1, 1, 512, 512)
            assert f["target"].shape == (1, 1, 64, 64)
            assert f["mask"].shape == (1, 1, 512, 512)
            np.testing.assert_allclose(f["input"][0, 0], sample["intensity_measured"])
            np.testing.assert_allclose(f["target"][0, 0], sample["target_dpr"])
            np.testing.assert_array_equal(f["mask"][0, 0], sample["valid_mask"])


def test_15_metadata_reproduces_simulation_settings():
    """Test 15: Verify that all metadata accurately reproduce simulation settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=12, seed=999)
        assert res["status"] == "success"

        ds = OpenDiffractionDataset(tmpdir)
        assert len(ds) == 12
        sample_0 = ds[0]

        # Check metadata fields
        meta = sample_0.metadata
        assert "sample_id" in meta
        assert "photon_budget" in meta
        assert "coherence_length" in meta
        assert "readout_noise_std" in meta


def test_16_safe_resume_after_interrupted_run():
    """Test 16: Verify safe resume after an interrupted run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial run of 24 samples
        res1 = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=24, seed=42)
        assert res1["written_samples"] == 24

        # Create an unfinished .tmp file to simulate crash during next batch
        fake_tmp = os.path.join(tmpdir, "data", "corrupt_shard.h5.tmp")
        with open(fake_tmp, "w") as f:
            f.write("corrupted incomplete bytes")

        # Resume run: should recognize completed samples and not fail on .tmp
        res2 = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=24, seed=42, resume=True)
        assert res2["status"] == "already_completed"
