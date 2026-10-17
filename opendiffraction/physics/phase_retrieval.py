"""Classical phase retrieval algorithms: Error Reduction (ER) and Hybrid Input-Output (HIO).

Used as physics baselines and benchmark reconstructors.
"""

from __future__ import annotations

import numpy as np


def error_reduction(
    measured_amplitude: np.ndarray,
    support_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    num_iterations: int = 50,
    initial_phase: np.ndarray | None = None,
    initial_object: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    is_complex: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Classic Error Reduction (ER / Gerchberg-Saxton) phase retrieval.

    Parameters
    ----------
    measured_amplitude : np.ndarray
        Fourier modulus |F| of shape (N, N), centered (fftshifted).
    support_mask : np.ndarray
        Real-space support constraint (bool) of shape (N, N).
    valid_mask : np.ndarray or None
        Detector valid pixel mask (bool) of shape (N, N).
        True for measured pixels, False for beamstops, module gaps, or dead pixels.
        If None, all pixels are assumed measured.
    num_iterations : int
        Number of iterations.
    initial_phase : np.ndarray or None
        Initial phase in radians. If None, randomized uniformly in [-pi, pi].
    initial_object : np.ndarray or None
        Direct real-space object initialization (centered) from prior iterates (e.g. HIO).
    rng : np.random.Generator or None
        Random generator for initial phase.
    is_complex : bool
        If True, permits complex-valued objects; if False, enforces non-negativity.

    Returns
    -------
    recon_object : np.ndarray
        Reconstructed real-space object of shape (N, N).
    errors : list of float
        Fourier space crystallographic R-factor error per iteration evaluated over valid pixels.
    """
    if rng is None:
        rng = np.random.default_rng()

    modulus = np.fft.ifftshift(measured_amplitude)
    if valid_mask is not None:
        v_mask = np.fft.ifftshift(valid_mask.astype(bool))
    else:
        v_mask = np.ones_like(modulus, dtype=bool)

    if initial_object is not None:
        curr_obj = np.fft.ifftshift(initial_object)
        if not is_complex:
            curr_obj = np.maximum(np.real(curr_obj), 0.0)
    else:
        if initial_phase is None:
            initial_phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
        curr_phase = np.fft.ifftshift(initial_phase)
        init_modulus = np.where(v_mask, modulus, np.maximum(modulus, 1.0))
        curr_field = init_modulus * np.exp(1j * curr_phase)
        curr_obj = np.fft.ifft2(curr_field)
        if not is_complex:
            curr_obj = np.real(curr_obj)

    supp = np.fft.ifftshift(support_mask.astype(bool))
    errors: list[float] = []

    for _ in range(num_iterations):
        # 1. Fourier projection
        G = np.fft.fft2(curr_obj)
        G_amp = np.abs(G)
        G_phase = np.angle(G)

        # Compute crystallographic R-factor on valid measured pixels
        if np.any(v_mask):
            m_valid = modulus[v_mask]
            g_valid = G_amp[v_mask]
            scale_denom = np.sum(g_valid**2)
            scale = float(np.sum(m_valid * g_valid) / (scale_denom + 1e-12)) if scale_denom > 0 else 1.0
            r_num = float(np.sum(np.abs(scale * g_valid - m_valid)))
            r_den = float(max(np.sum(m_valid), 1e-12))
            errors.append(r_num / r_den)
        else:
            errors.append(0.0)

        # Mask-aware Fourier constraint:
        # Replace amplitude on valid pixels; leave unmeasured pixels unconstrained
        G_mod = np.where(v_mask, modulus * np.exp(1j * G_phase), G)

        # 2. Real-space projection
        g_prime = np.fft.ifft2(G_mod)
        if not is_complex:
            g_prime = np.real(g_prime)
            curr_obj = np.where(supp & (g_prime >= 0), g_prime, 0.0)
        else:
            curr_obj = np.where(supp, g_prime, 0.0)

    if not is_complex:
        recon = np.fft.fftshift(np.maximum(np.real(curr_obj), 0.0))
    else:
        recon = np.fft.fftshift(curr_obj)
    return recon, errors


def hybrid_input_output(
    measured_amplitude: np.ndarray,
    support_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    num_iterations: int = 50,
    beta: float = 0.8,
    initial_phase: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    is_complex: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Hybrid Input-Output (HIO) algorithm for phase retrieval.

    Parameters
    ----------
    measured_amplitude : np.ndarray
        Fourier modulus |F| of shape (N, N), centered (fftshifted).
    support_mask : np.ndarray
        Real-space support constraint (bool) of shape (N, N).
    valid_mask : np.ndarray or None
        Detector valid pixel mask (bool) of shape (N, N).
        True for measured pixels, False for beamstops, module gaps, or dead pixels.
        If None, all pixels are assumed measured.
    num_iterations : int
        Number of iterations.
    beta : float
        Feedback parameter (default 0.8).
    initial_phase : np.ndarray or None
        Initial phase in radians.
    rng : np.random.Generator or None
        Random generator for initial phase.
    is_complex : bool
        If True, permits complex-valued objects; if False, enforces non-negativity.

    Returns
    -------
    recon_object : np.ndarray
        Reconstructed real-space object.
    errors : list of float
        Fourier space crystallographic R-factor error per iteration evaluated over valid pixels.
    """
    if rng is None:
        rng = np.random.default_rng()

    modulus = np.fft.ifftshift(measured_amplitude)
    if valid_mask is not None:
        v_mask = np.fft.ifftshift(valid_mask.astype(bool))
    else:
        v_mask = np.ones_like(modulus, dtype=bool)

    if initial_phase is None:
        initial_phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)

    curr_phase = np.fft.ifftshift(initial_phase)
    init_modulus = np.where(v_mask, modulus, np.maximum(modulus, 1.0))
    curr_field = init_modulus * np.exp(1j * curr_phase)
    curr_obj = np.fft.ifft2(curr_field)
    if not is_complex:
        curr_obj = np.real(curr_obj)

    supp = np.fft.ifftshift(support_mask.astype(bool))
    errors: list[float] = []

    for _ in range(num_iterations):
        # 1. Fourier projection
        G = np.fft.fft2(curr_obj)
        G_amp = np.abs(G)
        G_phase = np.angle(G)

        # Compute crystallographic R-factor on valid measured pixels
        if np.any(v_mask):
            m_valid = modulus[v_mask]
            g_valid = G_amp[v_mask]
            scale_denom = np.sum(g_valid**2)
            scale = float(np.sum(m_valid * g_valid) / (scale_denom + 1e-12)) if scale_denom > 0 else 1.0
            r_num = float(np.sum(np.abs(scale * g_valid - m_valid)))
            r_den = float(max(np.sum(m_valid), 1e-12))
            errors.append(r_num / r_den)
        else:
            errors.append(0.0)

        # Mask-aware Fourier constraint:
        # Replace amplitude on valid pixels; leave unmeasured pixels unconstrained
        G_mod = np.where(v_mask, modulus * np.exp(1j * G_phase), G)
        g_prime = np.fft.ifft2(G_mod)
        if not is_complex:
            g_prime = np.real(g_prime)
            valid_region = supp & (g_prime >= 0)
        else:
            valid_region = supp

        # 2. HIO real-space update
        curr_obj = np.where(valid_region, g_prime, curr_obj - beta * g_prime)

    if not is_complex:
        recon = np.fft.fftshift(np.maximum(np.real(curr_obj), 0.0))
    else:
        recon = np.fft.fftshift(curr_obj)
    return recon, errors


def reconstruct_hio(
    intensity: np.ndarray,
    support_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    max_iterations: int = 50,
    beta: float = 0.8,
    rng: np.random.Generator | None = None,
    is_complex: bool = False,
) -> np.ndarray:
    """Convenience wrapper for HIO phase retrieval from intensity."""
    amp = np.sqrt(np.maximum(intensity, 0.0))
    if support_mask is None:
        ny, nx = intensity.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        support_mask = (y - cy) ** 2 + (x - cx) ** 2 <= 30.0**2

    recon, _ = hybrid_input_output(
        amp,
        support_mask,
        valid_mask=valid_mask,
        num_iterations=max_iterations,
        beta=beta,
        rng=rng,
        is_complex=is_complex,
    )
    ny, nx = recon.shape
    cy, cx = ny // 2, nx // 2
    cropped = recon[cy - 32 : cy + 32, cx - 32 : cx + 32]
    p_max = np.max(cropped)
    if p_max > 0:
        cropped = cropped / p_max
    return cropped.astype(np.float32)
