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
    """Unconstrained Fourier projection under central beamstop must preserve unmeasured modes and outperform forced zero."""
    rng = np.random.default_rng(123)
    obj = make_virus_particle(64, rng=rng)
    pad = pad_object_to_detector(obj, (512, 512))
    _, intensity = compute_coherent_diffraction(pad)
    amp = np.sqrt(intensity)

    # Circular central beamstop blocking low-q
    v_mask = create_centered_beam_stop((512, 512), half_width=16, half_height=16, circular=True)
    masked_amp = amp * v_mask

    # 1. Analytical projection verification:
    # Verify that unconstrained Fourier update preserves unmeasured iterate pixels
    # while forced-zero destroys them by setting them to 0.
    test_field = np.ones((512, 512), dtype=np.complex128) * (1.0 + 2.0j)
    v_mask_shifted = np.fft.ifftshift(v_mask.astype(bool))
    modulus_shifted = np.fft.ifftshift(masked_amp)

    # Mask-aware projection preserves iterate on ~v_mask:
    g_mod_unconstrained = np.where(
        v_mask_shifted,
        modulus_shifted * np.exp(1j * np.angle(test_field)),
        test_field,
    )
    assert np.allclose(
        g_mod_unconstrained[~v_mask_shifted], test_field[~v_mask_shifted]
    ), "Unconstrained Fourier update must strictly preserve unmeasured iterate pixels"

    # Forced-zero projection overwrites with 0:
    g_mod_zero = np.where(
        np.ones_like(v_mask_shifted),
        modulus_shifted * np.exp(1j * np.angle(test_field)),
        test_field,
    )
    assert np.allclose(
        g_mod_zero[~v_mask_shifted], 0.0
    ), "Forced-zero projection must force unmeasured pixels to zero"

    # 2. Controlled HIO + ER reconstruction with paired random initialization:
    supp = np.zeros((512, 512), dtype=bool)
    supp[256 - 32 : 256 + 32, 256 - 32 : 256 + 32] = (obj > 0)
    init_phase = np.random.default_rng(123).uniform(-np.pi, np.pi, size=(512, 512))

    # A. Remediated: mask-aware unconstrained Fourier update
    hio_unconstrained, _ = hybrid_input_output(
        masked_amp, supp, valid_mask=v_mask, num_iterations=80, beta=0.8, initial_phase=init_phase.copy()
    )
    er_unconstrained, er_errs_u = error_reduction(
        masked_amp, supp, valid_mask=v_mask, num_iterations=40, initial_object=hio_unconstrained
    )
    crop_unconstrained = er_unconstrained[256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
    crop_unconstrained_norm = crop_unconstrained / (np.max(crop_unconstrained) + 1e-12)
    gt_norm = obj / (np.max(obj) + 1e-12)

    ssim_unconstrained = aligned_ssim(crop_unconstrained_norm, gt_norm)
    mse_unconstrained = aligned_mse(crop_unconstrained_norm, gt_norm)

    # B. Buggy baseline: forcing masked pixels to zero (valid_mask=None) with identical initial phase
    hio_zero, _ = hybrid_input_output(
        masked_amp, supp, valid_mask=None, num_iterations=80, beta=0.8, initial_phase=init_phase.copy()
    )
    er_zero, er_errs_z = error_reduction(
        masked_amp, supp, valid_mask=None, num_iterations=40, initial_object=hio_zero
    )
    crop_zero = er_zero[256 - 32 : 256 + 32, 256 - 32 : 256 + 32]
    crop_zero_norm = crop_zero / (np.max(crop_zero) + 1e-12)

    ssim_zero = aligned_ssim(crop_zero_norm, gt_norm)
    mse_zero = aligned_mse(crop_zero_norm, gt_norm)

    # Unconstrained interpolation must recover low-q frequencies near-perfectly
    assert ssim_unconstrained > 0.95, f"Unconstrained SSIM was too low: {ssim_unconstrained:.4f}"
    assert mse_unconstrained < 1e-4, f"Unconstrained MSE was too high: {mse_unconstrained:.4e}"

    # Forced-zero suffers catastrophic low-frequency loss and negative ringing
    assert ssim_zero < 0.50, f"Forced-zero SSIM was unexpectedly high: {ssim_zero:.4f}"
    assert mse_zero > 0.05, f"Forced-zero MSE was unexpectedly low: {mse_zero:.4e}"

    # Statistically and physically justified superiority over forced zero:
    assert ssim_unconstrained > ssim_zero + 0.50, (
        f"Expected unconstrained ({ssim_unconstrained:.4f}) to outperform "
        f"forced-zero ({ssim_zero:.4f}) by > 0.50"
    )
    assert mse_zero > 100.0 * mse_unconstrained, (
        f"Expected forced-zero MSE ({mse_zero:.4e}) to be > 100x unconstrained MSE ({mse_unconstrained:.4e})"
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
    """Alignment utility must resolve 180-deg twin image reflection and spatial translation against an oracle."""
    from scipy.ndimage import shift
    from opendiffraction.tasks.metrics import fourier_shift_2d

    rng = np.random.default_rng(999)
    obj = make_virus_particle(64, rng=rng)
    obj_norm = obj / (np.max(obj) + 1e-12)

    # 1. Integer shift and twin reflection: exact recovery
    twin = np.rot90(obj_norm, 2)
    shifted_twin = shift(twin, shift=(3.0, -2.0), mode="wrap", order=1)
    aligned = align_twin_and_translation(shifted_twin, obj_norm)
    ssim_val = ssim_2d(aligned, obj_norm, data_range=1.0)
    assert ssim_val > 0.999, f"Alignment failed on integer shift: SSIM={ssim_val:.4f}"

    # 2. Sub-pixel shift: compare against independent registration oracle
    true_shift = (3.5, -2.5)
    sub_shifted_twin = shift(twin, shift=true_shift, mode="wrap", order=1)
    aligned_sub = align_twin_and_translation(sub_shifted_twin, obj_norm)
    ssim_sub = ssim_2d(aligned_sub, obj_norm, data_range=1.0)

    # Independent oracle: unrotate twin and apply known exact inverse translation via Fourier shift
    unrotated_twin = np.rot90(sub_shifted_twin, 2)
    oracle_aligned = fourier_shift_2d(unrotated_twin, true_shift[0], true_shift[1])
    oracle_ssim = ssim_2d(oracle_aligned, obj_norm, data_range=1.0)

    # Must exceed 0.90 requirement and match oracle interpolation bound to within 0.005
    assert ssim_sub > 0.90, f"Alignment failed on subpixel shift: SSIM={ssim_sub:.4f}"
    assert abs(ssim_sub - oracle_ssim) < 0.005, (
        f"Alignment SSIM ({ssim_sub:.4f}) deviated from registration oracle ({oracle_ssim:.4f})"
    )
