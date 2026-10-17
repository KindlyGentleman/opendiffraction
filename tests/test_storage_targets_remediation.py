"""Regression tests for OD-004: Target integrity and missing clean target error handling."""

from __future__ import annotations

import os
import tempfile
import h5py
import numpy as np
import pytest

from opendiffraction.generator.storage import DatasetWriter
from opendiffraction.io.loader import OpenDiffractionDataset


def test_missing_clean_target_raises_error_in_torch_dataset() -> None:
    """Supervised denoising or inpainting on compact shards must raise KeyError if clean target is absent."""
    try:
        import torch
        from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
    except ImportError:
        pytest.skip("PyTorch not installed; skipping torch dataset test.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = DatasetWriter(tmp_dir, shard_size=10, storage_mode="compact")
        sample = {
            "object_amplitude": np.ones((64, 64), dtype=np.float32),
            "object_phase": None,
            "target_dpr": np.ones((64, 64), dtype=np.float32),
            "target_physical": np.ones((64, 64), dtype=np.float32),
            "intensity_ideal": np.ones((512, 512), dtype=np.float32),
            "intensity_expected": np.ones((512, 512), dtype=np.float32),
            "intensity_raw": np.ones((512, 512), dtype=np.float32),
            "intensity_observed": np.ones((512, 512), dtype=np.float32),
            "intensity_measured": np.ones((512, 512), dtype=np.float32),
            "valid_mask": np.ones((512, 512), dtype=bool),
        }
        shard_path = os.path.join(tmp_dir, "data", "train_000.h5")
        writer.write_shard_atomic(shard_path, [sample])
        writer.append_manifest_rows(
            [
                {
                    "sample_id": "test_0",
                    "shard_name": "train_000.h5",
                    "row_index": 0,
                    "split": "train",
                    "profile_id": "V01",
                }
            ]
        )

        # In compact mode, intensity_expected and intensity_ideal are omitted
        # Denoising task must raise KeyError when clean target is missing
        with OpenDiffractionTorchDataset(tmp_dir, split="train", task="denoising") as ds:
            with pytest.raises(KeyError, match="requires clean ground-truth targets"):
                _ = ds[0]

        # Explicit self_supervised=True should succeed
        with OpenDiffractionTorchDataset(
            tmp_dir, split="train", task="denoising", self_supervised=True
        ) as ds_self:
            inp, tgt = ds_self[0]
            assert inp.shape == (1, 512, 512)
            assert tgt.shape == (1, 512, 512)


def test_hdf5_extended_storage_roundtrip() -> None:
    """Extended mode must store intensity_raw, intensity_observed, and object_amplitude."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = DatasetWriter(tmp_dir, shard_size=10, storage_mode="extended")
        raw_val = np.full((512, 512), 42.0, dtype=np.float32)
        obs_val = np.zeros((512, 512), dtype=np.float32)
        obs_val[100:200, 100:200] = 42.0
        mask = np.zeros((512, 512), dtype=bool)
        mask[100:200, 100:200] = True

        sample = {
            "object_amplitude": np.ones((64, 64), dtype=np.float32) * 3.14,
            "object_phase": None,
            "target_dpr": np.ones((64, 64), dtype=np.float32),
            "target_physical": np.ones((64, 64), dtype=np.float32),
            "intensity_ideal": np.ones((512, 512), dtype=np.float32),
            "intensity_expected": np.ones((512, 512), dtype=np.float32),
            "intensity_raw": raw_val,
            "intensity_observed": obs_val,
            "intensity_measured": obs_val,
            "valid_mask": mask,
        }
        shard_path = os.path.join(tmp_dir, "data", "train_000.h5")
        writer.write_shard_atomic(shard_path, [sample])
        writer.append_manifest_rows(
            [
                {
                    "sample_id": "test_ext_0",
                    "shard_name": "train_000.h5",
                    "row_index": 0,
                    "split": "train",
                    "profile_id": "V01",
                }
            ]
        )

        dataset = OpenDiffractionDataset(tmp_dir, split="train")
        assert len(dataset) == 1
        item = dataset[0]
        assert np.allclose(item.object_amplitude, 3.14)
        assert np.all(item.input[0, ~mask] == 0.0)
        assert item.intensity_raw is not None
        assert np.allclose(item.intensity_raw, 42.0)
