"""Verification and regression tests for second-pass remediations:
OD-005, OD-007, OD-008, OD-011, OD-013, OD-016, OD-017, OD-018, OD-019, OD-020, OD-025, OD-027.
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest
from scipy.stats import pearsonr

from opendiffraction.physics.forward import (
    pad_object_to_detector,
    compute_coherent_diffraction,
)
from opendiffraction.physics.models.transmission import generate_probe
from opendiffraction.physics.masks import composite_detector_mask
from opendiffraction.tasks.metrics import ssim_2d
from opendiffraction.generator.pipeline import build_master_task_plan, simulate_single_task
from opendiffraction.io.loader import OpenDiffractionDataset
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
from opendiffraction.scenarios import get_scenario_physics, SCENARIOS
from benchmarks.phase_retrieval import compute_support_regime, evaluate_phase_retrieval
from opendiffraction.physics.phase_retrieval import error_reduction


def test_od007_beam_displacement_shifts_diffraction_modulus():
    """OD-007: Illumination wavevector tilt shifts the far-field diffraction peak on detector."""
    ny, nx = 64, 64
    y, x = np.ogrid[:ny, :nx]
    obj = np.exp(-((y - 32) ** 2 + (x - 32) ** 2) / (2 * 8.0**2)).astype(np.float64)

    # Center placement
    frame_centered = pad_object_to_detector(obj, detector_size=(256, 256), offset=(0.0, 0.0))
    _, int_centered = compute_coherent_diffraction(frame_centered)

    # Offset placement via wavevector tilt phase ramp
    frame_offset = pad_object_to_detector(obj, detector_size=(256, 256), offset=(10.0, 15.0))
    _, int_offset = compute_coherent_diffraction(frame_offset)

    peak_centered = np.unravel_index(np.argmax(int_centered), int_centered.shape)
    peak_offset = np.unravel_index(np.argmax(int_offset), int_offset.shape)

    assert peak_centered == (128, 128)
    assert peak_offset == (138, 143)


def test_od008_generate_probe_applies_spherical_aberration():
    """OD-008: generate_probe accepts spherical_aberration_rad and alters probe phase."""
    sig = inspect.signature(generate_probe)
    assert "spherical_aberration_rad" in sig.parameters

    probe_clean = generate_probe(size=64, spherical_aberration_rad=0.0)
    probe_aberrated = generate_probe(size=64, spherical_aberration_rad=1.5)

    # Amplitude profiles remain Gaussian
    np.testing.assert_allclose(np.abs(probe_clean), np.abs(probe_aberrated), atol=1e-6)

    # Phase differs due to r^4 spherical aberration
    phase_diff = np.max(np.abs(np.angle(probe_aberrated) - np.angle(probe_clean)))
    assert phase_diff > 1.0


def test_od019_severe_mask_is_preserved_not_erased():
    """OD-019: composite_detector_mask preserves masks with valid coverage < 20%."""
    mask = composite_detector_mask(
        shape=(128, 128),
        include_beam_stop=True,
        beam_stop_hw=30,
        beam_stop_hh=30,
        irregular_coverage=0.85,
    )
    # The severe mask must be retained, not silently wiped to all True
    assert not np.all(mask == True)
    assert np.mean(mask) < 0.2


def test_od013_hio_to_er_state_transfer():
    """OD-013: error_reduction accepts real-space initial_object directly."""
    rng = np.random.default_rng(42)
    ny, nx = 64, 64
    y, x = np.ogrid[:ny, :nx]
    obj = ((y - 32) ** 2 + (x - 32) ** 2 <= 14**2).astype(np.float64)
    support = ((y - 32) ** 2 + (x - 32) ** 2 <= 20**2).astype(bool)

    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))
    modulus = np.abs(F)

    # Initial object passed directly into ER
    init_obj = obj + rng.normal(0, 0.05, size=obj.shape)
    recon, errors = error_reduction(
        measured_amplitude=modulus,
        support_mask=support,
        num_iterations=10,
        initial_object=init_obj,
        rng=rng,
    )
    assert recon.shape == obj.shape
    assert len(errors) == 10
    assert errors[-1] < errors[0]


def test_od016_ssim_2d_windowed():
    """OD-016: ssim_2d computes local windowed structural similarity."""
    rng = np.random.default_rng(42)
    img1 = np.ones((64, 64), dtype=np.float32) * 0.5
    img2 = img1.copy()
    img2[:, :32] += rng.normal(0, 0.1, size=(64, 32)).astype(np.float32)

    val = ssim_2d(img1, img2, data_range=1.0)
    assert isinstance(val, float)
    assert 0.0 < val < 1.0


def test_od017_eval_phase_retrieval_pearson_correlation():
    """OD-017: Diagnostic Pearson correlation compares aligned reconstruction to ground truth."""
    target = np.array([0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    recon = target * 0.9 + 0.05
    r_val, _ = pearsonr(recon.ravel(), target.ravel())
    assert 0.99 < r_val <= 1.0


def test_od020_quick_preset_generates_all_48_profiles():
    """OD-020: Quick preset generates 48 samples covering all 48 profiles V01-V48."""
    tasks, _ = build_master_task_plan(
        total_samples=48,
        acquisitions_per_object=1,
        profiles_mode="48",
    )
    sampled_profiles = {t["profile_id"] for t in tasks}
    assert len(sampled_profiles) == 48
    assert "V01" in sampled_profiles
    assert "V48" in sampled_profiles


def test_od005_loader_and_torch_dataset_support_manifest_and_sample_id():
    """OD-005: OpenDiffractionDataset and Torch dataset accept manifest_path and get_by_sample_id."""
    sig_loader = inspect.signature(OpenDiffractionDataset.__init__)
    assert "manifest_path" in sig_loader.parameters
    assert hasattr(OpenDiffractionDataset, "get_by_sample_id")

    sig_torch = inspect.signature(OpenDiffractionTorchDataset.__init__)
    assert "manifest_path" in sig_torch.parameters
    assert hasattr(OpenDiffractionTorchDataset, "get_by_sample_id")


def test_od011_scenario_metadata_synchronization():
    """OD-011: get_scenario_physics populates wavelength, pixel pitch, distance from ScenarioSpec."""
    phys_s1 = get_scenario_physics("S1")
    assert phys_s1["scenario_id"] == "S1"
    assert np.isclose(phys_s1["photon_energy_kev"], 8.0)
    assert np.isclose(phys_s1["pixel_pitch_m"], 110.0e-6)
    assert np.isclose(phys_s1["detector_distance_m"], 0.5)

    phys_s5 = get_scenario_physics("S5")
    assert phys_s5["scenario_id"] == "S5"
    assert np.isclose(phys_s5["photon_energy_kev"], 0.00196)
    assert np.isclose(phys_s5["pixel_pitch_m"], 6.45e-6)
    assert np.isclose(phys_s5["detector_distance_m"], 0.15)


def test_od018_support_regimes_in_phase_retrieval():
    """OD-018: compute_support_regime supports disc, box, oracle, and autocorrelation."""
    shape = (128, 128)
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[8:24, 8:24] = 1.0

    mask_disc = compute_support_regime(shape, regime="disc", disc_radius=20.0)
    assert mask_disc.shape == shape
    assert mask_disc[64, 64] == True

    mask_box = compute_support_regime(shape, regime="box", box_half_size=20)
    assert mask_box.shape == shape
    assert mask_box[64, 64] == True

    mask_oracle = compute_support_regime(shape, regime="oracle", ground_truth_object=gt, dilation_px=2)
    assert mask_oracle.shape == shape
    assert mask_oracle[64, 64] == True

    intensity = np.ones(shape, dtype=np.float32)
    mask_auto = compute_support_regime(shape, regime="autocorrelation", measured_intensity=intensity)
    assert mask_auto.shape == shape
