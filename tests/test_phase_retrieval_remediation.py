"""Regression test suite for OD-005: Phase retrieval algorithms and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from opendiffraction.objects.specimens import make_virus_particle
from opendiffraction.physics.forward import pad_object_to_detector, compute_coherent_diffraction
from opendiffraction.physics.masks import create_centered_beam_stop
from opendiffraction.physics.phase_retrieval import error_reduction, hybrid_input_output
from opendiffraction.tasks.metrics import (
    align_twin_and_translation,
    aligned_mse,
    aligned_ssim,
    compute_r_factor,
    ssim_2d,
)
from benchmarks.phase_retrieval import evaluate_phase_retrieval


def test_hio_reconstructs_noise_free_phantom() -> None:
    """HIO + ER must accurately reconstruct a compact noise-free phantom."""
    rng = np.random.default_rng(42)
    obj = make_virus_particle(64, rng=rng)
    pad = pad_object_to_detector(obj, (512, 512))
    _, intensity = compute_coherent_diffraction(pad)
    amp = np.sqrt(intensity)

    # Support enclosing the object
    supp = np.zeros((512, 512), dtype=bool)
    supp[256 - 32 : 256 + 32, 256 - 32 : 256 + 32] = (obj > 0)

    # 80 HIO + 40 ER iterations
    hio_rec, _ = hybrid_input_output(amp, supp, num_iterations=80, beta=0.8, rng=rng)
    init_phase = np.angle(np.fft.fft2(np.fft.ifftshift(hio_rec)))
    er_rec, er_errors = error_reduction(
        amp, supp, num_iterations=40, initial_phase=init_phase, rng=rng
    )

    crop = er_rec[256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
    crop_norm = crop / (np.max(crop) + 1e-12)
    gt_norm = obj / (np.max(obj) + 1e-12)

    ssim_val = aligned_ssim(crop_norm, gt_norm)
    final_r = er_errors[-1]

    # Must achieve high fidelity reconstruction on clean data
    assert ssim_val > 0.90, f"Expected SSIM > 0.90, got {ssim_val:.4f}"
    assert final_r < 0.03, f"Expected R-factor < 0.03, got {final_r:.6f}"


def test_hio_reconstructs_with_central_beamstop_unconstrained() -> None:
    """Unconstrained Fourier projection under central beamstop must dramatically outperform forced zero."""
    rng = np.random.default_rng(123)
    obj = make_virus_particle(64, rng=rng)
    pad = pad_object_to_detector(obj, (512, 512))
    _, intensity = compute_coherent_diffraction(pad)
    amp = np.sqrt(intensity)

    # Circular central beamstop blocking low-q
    v_mask = create_centered_beam_stop((512, 512), half_width=16, half_height=16, circular=True)
    masked_amp = amp * v_mask

    supp = np.zeros((512, 512), dtype=bool)
    supp[256 - 32 : 256 + 32, 256 - 32 : 256 + 32] = (obj > 0)

    # 1. Remediated: mask-aware unconstrained Fourier update
    hio_unconstrained, _ = hybrid_input_output(
        masked_amp, supp, valid_mask=v_mask, num_iterations=80, beta=0.8, rng=rng
    )
    init_phase = np.angle(np.fft.fft2(np.fft.ifftshift(hio_unconstrained)))
    er_unconstrained, _ = error_reduction(
        masked_amp, supp, valid_mask=v_mask, num_iterations=40, initial_phase=init_phase, rng=rng
    )
    crop_unconstrained = er_unconstrained[256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
    crop_unconstrained_norm = crop_unconstrained / (np.max(crop_unconstrained) + 1e-12)
    gt_norm = obj / (np.max(obj) + 1e-12)

    ssim_unconstrained = aligned_ssim(crop_unconstrained_norm, gt_norm)

    # 2. Old buggy behavior: forcing masked pixels to zero (valid_mask=None)
    hio_zero, _ = hybrid_input_output(
        masked_amp, supp, valid_mask=None, num_iterations=80, beta=0.8, rng=rng
    )
    init_phase_zero = np.angle(np.fft.fft2(np.fft.ifftshift(hio_zero)))
    er_zero, _ = error_reduction(
        masked_amp, supp, valid_mask=None, num_iterations=40, initial_phase=init_phase_zero, rng=rng
    )
    crop_zero = er_zero[256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
    crop_zero_norm = crop_zero / (np.max(crop_zero) + 1e-12)

    ssim_zero = aligned_ssim(crop_zero_norm, gt_norm)

    # Unconstrained interpolation must recover low-q frequencies
    assert ssim_unconstrained > 0.60, f"Unconstrained SSIM was too low: {ssim_unconstrained:.4f}"
    # Forced-zero should show severe degradation (e.g. SSIM < 0.05)
    assert ssim_unconstrained > ssim_zero + 0.50, (
        f"Expected unconstrained ({ssim_unconstrained:.4f}) to strongly outperform "
        f"forced-zero ({ssim_zero:.4f})"
    )


def test_crystallographic_r_factor_definition() -> None:
    """R-factor must evaluate sum(|I_meas^0.5 - s * I_sim^0.5|) / sum(I_meas^0.5) over valid pixels."""
    i_meas = np.array([100.0, 400.0, 900.0, 0.0], dtype=np.float32)
    i_sim = np.array([100.0, 400.0, 900.0, 500.0], dtype=np.float32)
    mask = np.array([True, True, True, False])  # 4th pixel is masked

    # On valid pixels, amplitudes are 10, 20, 30. Perfectly matched with scale=1.
    r_perfect = compute_r_factor(i_sim, i_meas, mask=mask)
    assert np.isclose(r_perfect, 0.0, atol=1e-6)

    # Perturbed amplitude
    i_sim_perturbed = np.array([121.0, 400.0, 900.0, 0.0], dtype=np.float32)
    r_perturbed = compute_r_factor(i_sim_perturbed, i_meas, mask=mask)
    assert 0.0 < r_perturbed < 0.10


def test_alignment_resolves_twin_image_and_translation() -> None:
    """Alignment utility must resolve 180-deg twin image reflection and spatial translation."""
    from scipy.ndimage import shift

    rng = np.random.default_rng(999)
    obj = make_virus_particle(64, rng=rng)
    obj_norm = obj / (np.max(obj) + 1e-12)

    # 1. Integer shift and twin reflection: exact recovery
    twin = np.rot90(obj_norm, 2)
    shifted_twin = shift(twin, shift=(3.0, -2.0), mode="wrap", order=1)
    aligned = align_twin_and_translation(shifted_twin, obj_norm)
    ssim_val = ssim_2d(aligned, obj_norm, data_range=1.0)
    assert ssim_val > 0.99, f"Alignment failed on integer shift: SSIM={ssim_val:.4f}"

    # 2. Sub-pixel shift: robust recovery
    sub_shifted_twin = shift(twin, shift=(3.5, -2.5), mode="wrap", order=1)
    aligned_sub = align_twin_and_translation(sub_shifted_twin, obj_norm)
    ssim_sub = ssim_2d(aligned_sub, obj_norm, data_range=1.0)
    assert ssim_sub > 0.90, f"Alignment failed on subpixel shift: SSIM={ssim_sub:.4f}"
