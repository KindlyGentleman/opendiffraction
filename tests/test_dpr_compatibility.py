"""Tests for DPR compatibility: CustomDataset loader and forward model numerical agreement."""

import os
import sys
import tempfile
import numpy as np
import pytest

torch = pytest.importorskip("torch")


from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction

# Dynamic discovery of DPR upstream directory
def _discover_dpr_path() -> str | None:
    candidates = [
        os.environ.get("DPR_PATH"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DPR_upstream")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DPR")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DPR_upstream")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DPR")),
        r"d:\03_Proyek\DPR_upstream",
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "deeppr", "__init__.py")):
            return cand
    return None

_dpr_dir = _discover_dpr_path()
if _dpr_dir and _dpr_dir not in sys.path:
    sys.path.insert(0, _dpr_dir)


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


def test_19_gsm_coherence_decay_and_energy_conservation():
    """Test 19: Check GSM partial coherence kernel decay, zero lag, and energy conservation."""
    from opendiffraction.physics.coherence import apply_partial_coherence, create_coherence_kernel

    shape = (512, 512)
    l_coh = 60.0

    # 1. Kernel properties
    kernel = create_coherence_kernel(l_coh=l_coh, shape=shape, legacy=False)
    # DC / zero lag is at index (0, 0) for ifft2 convention
    assert np.isclose(kernel[0, 0], 1.0, atol=1e-6)

    # Check value at lag = l_coh along y-axis: r = l_coh -> exp(-0.5) ~ 0.60653
    # In ifft2 frequency indexing, y = 60 corresponds to row index 60
    assert np.isclose(kernel[int(l_coh), 0], np.exp(-0.5), rtol=1e-3)

    # Infinite coherence length returns identically 1.0 everywhere
    kernel_inf = create_coherence_kernel(l_coh=float("inf"), shape=shape)
    assert np.all(kernel_inf == 1.0)

    # 2. Energy conservation under partial coherence
    rng = np.random.default_rng(123)
    # Simulated smooth diffraction intensity
    kx = np.linspace(-10, 10, 512)
    ky = np.linspace(-10, 10, 512)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    intensity_ideal = np.exp(-(kyy**2 + kxx**2) / 20.0).astype(np.float64)

    intensity_part = apply_partial_coherence(intensity_ideal, l_coh=l_coh)
    # Total photon energy must be conserved
    sum_ideal = np.sum(intensity_ideal)
    sum_part = np.sum(intensity_part)
    assert np.isclose(sum_ideal, sum_part, rtol=1e-5)

    # Fully coherent should return identical intensity
    intensity_coh = apply_partial_coherence(intensity_ideal, l_coh=float("inf"))
    np.testing.assert_allclose(intensity_coh, intensity_ideal, rtol=1e-6)


def test_20_dpr_target_scaling_and_physical_scaling():
    """Test 20: Check DPR false_scale=True (target=obj*scale) vs physical scaling (target=obj*sqrt(scale))."""
    rng = np.random.default_rng(777)
    obj = rng.uniform(0.1, 0.8, size=(64, 64)).astype(np.float32)

    cfg = SimulationConfig(
        object_size=(64, 64),
        detector_size=(512, 512),
        photon_budget=5e6,
        coherence_length=150.0,
        false_scale=True,
    )
    result = simulate_diffraction(obj, config=cfg, rng=rng)

    scale = result.scale_factor
    # DPR false_scale convention: target = obj * scale
    expected_dpr = (obj * scale).astype(np.float32)
    np.testing.assert_allclose(result.target_dpr, expected_dpr, rtol=1e-5)

    # Physical convention: target = obj * sqrt(scale)
    expected_phys = (obj * np.sqrt(scale)).astype(np.float32)
    np.testing.assert_allclose(result.target_physical, expected_phys, rtol=1e-5)

    # Ratio between DPR target and physical target must be sqrt(scale)
    ratio = result.target_dpr / (result.target_physical + 1e-12)
    np.testing.assert_allclose(ratio, np.sqrt(scale), rtol=1e-4)


def test_21_detector_masks_and_masked_observations():
    """Test 21: Check beam stop, module gaps, and observed masked integrity."""
    from opendiffraction.physics.masks import composite_detector_mask

    rng = np.random.default_rng(999)
    mask = composite_detector_mask(
        shape=(512, 512),
        include_beam_stop=True,
        beam_stop_hw=16,
        beam_stop_hh=16,
        include_module_gaps=True,
        gap_width=4,
        rng=rng,
    )

    # Beam stop at center (256, 256): indices [240:272, 240:272] must be False (0)
    center_y, center_x = 256, 256
    beam_stop_region = mask[center_y - 16 : center_y + 16, center_x - 16 : center_x + 16]
    assert not np.any(beam_stop_region), "Beam stop area should be masked (False)"

    # Full simulation check: observed_masked is strictly 0 wherever mask is False
    obj = rng.uniform(0.2, 0.8, size=(64, 64)).astype(np.float32)
    cfg = SimulationConfig(
        include_beam_stop=True,
        beam_stop_hw=16,
        beam_stop_hh=16,
        include_module_gaps=True,
        gap_width=4,
    )
    res = simulate_diffraction(obj, config=cfg, rng=rng)
    assert np.all(res.observed_masked[~res.valid_mask] == 0.0)
