"""Tests for DPR compatibility: CustomDataset loader and forward model numerical agreement."""

import os
import sys
import tempfile
import numpy as np
import pytest
import torch

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction

# Add DPR_upstream to path if present
DPR_PATH = r"d:\03_Proyek\DPR_upstream"
if os.path.exists(DPR_PATH) and DPR_PATH not in sys.path:
    sys.path.insert(0, DPR_PATH)


def test_17_dpr_custom_dataset_loading():
    """Test 17: Check that DPR-compatible outputs load through deeppr.CustomDataset."""
    try:
        from deeppr import CustomDataset
    except ImportError:
        pytest.skip("deeppr package not installed/available for direct import.")

    with tempfile.TemporaryDirectory() as tmpdir:
        res = generate_dataset(output_dir=tmpdir, preset="quick", total_samples=12, seed=123)
        assert res["status"] == "success"

        # Find first generated shard
        data_dir = os.path.join(tmpdir, "data")
        shards = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
        assert len(shards) > 0
        first_shard = os.path.join(data_dir, shards[0])

        # Load with deeppr.CustomDataset
        ds = CustomDataset(first_shard)
        assert len(ds) > 0

        inp, tgt, msk = ds[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)
        assert isinstance(msk, torch.Tensor)

        assert inp.shape == (1, 512, 512)
        assert tgt.shape == (1, 64, 64)
        assert msk.shape == (1, 512, 512)
        assert msk.dtype == torch.bool


def test_18_numerical_agreement_with_dpr_forward_model():
    """Test 18: Check numerical agreement with the current DPR forward model for an equivalent supported configuration."""
    try:
        from deeppr import GenerateDiffraction
    except ImportError:
        pytest.skip("deeppr package not installed/available for direct import.")

    # 1. Deterministic clean object
    rng = np.random.default_rng(42)
    obj_np = rng.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)

    # Convert to PyTorch tensor matching DPR GenerateDiffraction input shape (N, 1, 64, 64)
    obj_torch = torch.from_numpy(obj_np)[None, None, ...].clone()

    # In clean coherent mode with zero noise and fixed flux:
    # OpenDiffraction:
    cfg = SimulationConfig(
        object_size=(64, 64),
        detector_size=(512, 512),
        photon_budget=1e7,
        coherence_length=float("inf"),
        readout_std=0.0,
        false_scale=True,
    )
    # OpenDiffraction ideal intensity before noise:
    from opendiffraction.physics.forward import compute_coherent_diffraction, pad_object_to_detector
    frame = pad_object_to_detector(obj_np, detector_size=(512, 512))
    _, inten_opendiff = compute_coherent_diffraction(frame)

    # DPR ideal intensity:
    import torch.nn.functional as F
    from deeppr.network import _fft2
    padded_torch = F.pad(obj_torch, (224, 224, 224, 224))
    inten_dpr = torch.abs(_fft2(padded_torch)) ** 2
    inten_dpr_np = inten_dpr[0, 0].cpu().numpy()

    # Compare unnormalized coherent intensity: should agree within floating point precision (float32 tolerance)
    np.testing.assert_allclose(inten_opendiff, inten_dpr_np, rtol=1e-4, atol=1e-2)
