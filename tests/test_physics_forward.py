"""Tests for forward physical simulation: FFT, Parseval, nonnegativity, Friedel symmetry, and masking."""

import numpy as np
import pytest
from scipy.fft import fft2 as scipy_fft2

from opendiffraction.objects.procedural import generate_procedural_object
from opendiffraction.physics.forward import (
    SimulationConfig,
    compute_coherent_diffraction,
    pad_object_to_detector,
    simulate_diffraction,
)


def test_01_dimensions_shapes_and_mask_semantics():
    """Test 1: Verify image dimensions, shape, dtype, finite values, and mask semantics."""
    rng = np.random.default_rng(42)
    obj, _ = generate_procedural_object("basic_geometric", "uniform", size=64, rng=rng)

    cfg = SimulationConfig(
        object_size=(64, 64),
        detector_size=(512, 512),
        photon_budget=1e6,
        include_beam_stop=True,
    )
    sim = simulate_diffraction(obj, cfg, rng=rng)

    # Shapes
    assert sim.intensity_ideal.shape == (512, 512)
    assert sim.intensity_measured.shape == (512, 512)
    assert sim.valid_mask.shape == (512, 512)
    assert sim.target_dpr.shape == (64, 64)
    assert sim.target_physical.shape == (64, 64)

    # Dtypes
    assert sim.intensity_measured.dtype == np.float32
    assert sim.valid_mask.dtype == bool
    assert sim.target_dpr.dtype == np.float32

    # Finite values
    assert np.all(np.isfinite(sim.intensity_measured))
    assert np.all(np.isfinite(sim.target_dpr))

    # Mask semantics: beam stop center must be masked out (False)
    cy, cx = 256, 256
    assert sim.valid_mask[cy, cx] == False
    # Outer corners must be valid (True)
    assert sim.valid_mask[10, 10] == True


def test_03_independent_fft_verification():
    """Test 3: Verify coherent intensity against an independent FFT implementation (SciPy)."""
    rng = np.random.default_rng(123)
    frame = rng.standard_normal((512, 512))

    # OpenDiffraction implementation
    _, inten_numpy = compute_coherent_diffraction(frame)

    # SciPy independent implementation
    scipy_field = np.fft.fftshift(scipy_fft2(frame))
    scipy_inten = np.abs(scipy_field) ** 2

    np.testing.assert_allclose(inten_numpy, scipy_inten, rtol=1e-12, atol=1e-12)


def test_04_parseval_theorem():
    """Test 4: Check Parseval's theorem under the declared FFT normalization."""
    rng = np.random.default_rng(456)
    frame = rng.standard_normal((512, 512))

    # Under standard unnormalized FFT: sum(|FFT|^2) = N_y * N_x * sum(|frame|^2)
    real_space_energy = np.sum(np.abs(frame) ** 2)
    field = np.fft.fft2(frame)
    fourier_space_energy = np.sum(np.abs(field) ** 2) / (512 * 512)

    rel_diff = abs(real_space_energy - fourier_space_energy) / real_space_energy
    assert rel_diff < 1e-10, f"Parseval discrepancy: {rel_diff}"


def test_05_nonnegative_ideal_intensity():
    """Test 5: Verify nonnegative ideal intensity."""
    rng = np.random.default_rng(789)
    obj, _ = generate_procedural_object("particle_blob", "textured", size=64, rng=rng)
    cfg = SimulationConfig(photon_budget=1e6, coherence_length=200.0)
    sim = simulate_diffraction(obj, cfg, rng=rng)

    assert np.all(sim.intensity_ideal >= 0.0)
    assert np.all(sim.intensity_expected >= 0.0)


def test_10_clean_friedel_symmetry():
    """Test 10: Check clean real-object Friedel symmetry: I(-q) = I(q)."""
    rng = np.random.default_rng(101)
    obj, _ = generate_procedural_object("basic_geometric", "uniform", size=64, rng=rng)
    # Ensure real-valued object with no beam center shift
    cfg = SimulationConfig(
        photon_budget=1e7,
        coherence_length=float("inf"),
        readout_std=0.0,
        beam_center_offset=(0.0, 0.0),
    )
    sim = simulate_diffraction(obj, cfg, rng=rng)

    I = sim.intensity_ideal
    # Centered DC is at (256, 256). Inversion: I[256+dy, 256+dx] == I[256-dy, 256-dx]
    # For array of size 512, flip along both axes:
    # Note: DC at 256 means index 0 is Nyquist frequency; indices 1..511 are centered around 256.
    sub_I = I[1:, 1:]  # (511, 511) centered exactly at (255, 255)
    sub_I_flipped = np.flip(sub_I)

    rel_err = np.max(np.abs(sub_I - sub_I_flipped)) / np.max(sub_I)
    assert rel_err < 1e-5, f"Friedel symmetry violation: {rel_err}"


def test_11_masking_preserves_unmasked_measurement():
    """Test 11: Verify masking preserves unmasked raw measurement while masking observed measurement."""
    rng = np.random.default_rng(202)
    obj, _ = generate_procedural_object("porous_heterogeneous", "binary", size=64, rng=rng)
    cfg = SimulationConfig(include_beam_stop=True, include_module_gaps=True)
    sim = simulate_diffraction(obj, cfg, rng=rng)

    # Raw intensity must have values under the beam stop (unmasked detector readout)
    beamstop_pixels = ~sim.valid_mask
    assert np.any(beamstop_pixels)
    # The stored intensity_raw has non-zero noise readouts under the beam stop
    assert np.any(sim.intensity_raw[beamstop_pixels] != 0.0)
    # But intensity_observed and intensity_measured have exactly 0 under the mask
    assert np.all(sim.intensity_observed[beamstop_pixels] == 0.0)
    assert np.all(sim.intensity_measured[beamstop_pixels] == 0.0)
    assert np.all(sim.observed_masked[beamstop_pixels] == 0.0)
