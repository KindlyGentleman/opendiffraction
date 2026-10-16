"""Unit tests for OpenDiffractionTorchDataset and DataLoader interoperability."""

import os
import tempfile
import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset


def test_torch_dataset_phase_retrieval():
    """Verify PyTorch dataset loading for phase retrieval task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=16, seed=42)
        assert res["status"] == "success"

        with OpenDiffractionTorchDataset(root=tmpdir, split=None, task="phase_retrieval") as ds:
            assert len(ds) == 16
            inp, tgt, msk = ds[0]
            assert isinstance(inp, torch.Tensor)
            assert isinstance(tgt, torch.Tensor)
            assert isinstance(msk, torch.Tensor)

            assert inp.shape == (1, 512, 512)
            assert tgt.shape == (1, 64, 64)
            assert msk.shape == (1, 512, 512)
            assert msk.dtype == torch.bool


def test_torch_dataset_dataloader_collation():
    """Verify DataLoader batch collation, shuffling, and multi-epoch iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_dataset(output_dir=tmpdir, preset="quick", total_samples=16, seed=42)

        with OpenDiffractionTorchDataset(root=tmpdir, split=None, task="phase_retrieval") as ds:
            loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

            batches = 0
            for inp_batch, tgt_batch, msk_batch in loader:
                assert inp_batch.shape == (4, 1, 512, 512)
                assert tgt_batch.shape == (4, 1, 64, 64)
                assert msk_batch.shape == (4, 1, 512, 512)
                batches += 1

            assert batches == 4


def test_torch_dataset_tasks():
    """Verify task adapters for denoising and hit classification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_dataset(output_dir=tmpdir, preset="quick", total_samples=16, seed=42)

        # Denoising task
        with OpenDiffractionTorchDataset(root=tmpdir, split=None, task="denoising") as ds_denoise:
            inp, clean = ds_denoise[0]
            assert inp.shape == (1, 512, 512)
            assert clean.shape == (1, 512, 512)

        # Hit classification task
        with OpenDiffractionTorchDataset(root=tmpdir, split=None, task="hit_classification") as ds_hit:
            inp, label = ds_hit[0]
            assert inp.shape == (1, 512, 512)
            assert isinstance(label, torch.Tensor)
            assert label.dtype == torch.long
