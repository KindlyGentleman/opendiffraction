"""Classical phase retrieval algorithms: Error Reduction (ER) and Hybrid Input-Output (HIO).

Used as physics baselines and benchmark reconstructors.
"""

from __future__ import annotations

import numpy as np


def error_reduction(
    measured_amplitude: np.ndarray,
    support_mask: np.ndarray,
    num_iterations: int = 50,
    initial_phase: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Classic Error Reduction (ER / Gerchberg-Saxton) phase retrieval.

    Parameters
    ----------
    measured_amplitude : np.ndarray
        Fourier modulus |F| of shape (N, N), centered (fftshifted).
    support_mask : np.ndarray
        Real-space support constraint (bool) of shape (N, N).
    num_iterations : int
        Number of iterations.
    initial_phase : np.ndarray or None
        Initial phase in radians. If None, randomized.
    rng : np.random.Generator or None
        Random generator for initial phase.

    Returns
    -------
    recon_object : np.ndarray
        Reconstructed real-space object.
    errors : list of float
        Fourier space R-factor error per iteration.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = measured_amplitude.shape[-1]
    if initial_phase is None:
        initial_phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)

    # Convert centered amplitude to non-centered for standard FFT
    modulus = np.fft.ifftshift(measured_amplitude)
    curr_field = modulus * np.exp(1j * np.fft.ifftshift(initial_phase))
    curr_obj = np.real(np.fft.ifft2(curr_field))

    # Shift support to match non-centered real-space FFT output if centered
    supp = np.fft.ifftshift(support_mask)
    errors = []

    for _ in range(num_iterations):
        # 1. Fourier projection
        G = np.fft.fft2(curr_obj)
        error = float(np.sum(np.abs(np.abs(G) - modulus) ** 2) / max(np.sum(modulus**2), 1e-8))
        errors.append(error)

        G_mod = modulus * np.exp(1j * np.angle(G))

        # 2. Real-space projection
        g_prime = np.real(np.fft.ifft2(G_mod))

        # Support & nonnegativity constraint
        curr_obj = np.where(supp & (g_prime >= 0), g_prime, 0.0)

    recon = np.fft.fftshift(curr_obj)
    return recon, errors


def hybrid_input_output(
    measured_amplitude: np.ndarray,
    support_mask: np.ndarray,
    num_iterations: int = 50,
    beta: float = 0.8,
    initial_phase: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Hybrid Input-Output (HIO) algorithm for phase retrieval.

    Parameters
    ----------
    measured_amplitude : np.ndarray
        Fourier modulus |F| of shape (N, N), centered (fftshifted).
    support_mask : np.ndarray
        Real-space support constraint (bool) of shape (N, N).
    num_iterations : int
        Number of iterations.
    beta : float
        Feedback parameter (default 0.8).
    initial_phase : np.ndarray or None
        Initial phase in radians.
    rng : np.random.Generator or None
        Random generator for initial phase.

    Returns
    -------
    recon_object : np.ndarray
        Reconstructed real-space object.
    errors : list of float
        Fourier space R-factor error per iteration.
    """
    if rng is None:
        rng = np.random.default_rng()

    modulus = np.fft.ifftshift(measured_amplitude)
    if initial_phase is None:
        initial_phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)

    curr_field = modulus * np.exp(1j * np.fft.ifftshift(initial_phase))
    curr_obj = np.real(np.fft.ifft2(curr_field))
    supp = np.fft.ifftshift(support_mask)

    errors = []

    for _ in range(num_iterations):
        # 1. Fourier projection
        G = np.fft.fft2(curr_obj)
        error = float(np.sum(np.abs(np.abs(G) - modulus) ** 2) / max(np.sum(modulus**2), 1e-8))
        errors.append(error)

        G_mod = modulus * np.exp(1j * np.angle(G))
        g_prime = np.real(np.fft.ifft2(G_mod))

        # 2. HIO real-space update
        valid_region = supp & (g_prime >= 0)
        curr_obj = np.where(valid_region, g_prime, curr_obj - beta * g_prime)

    recon = np.fft.fftshift(np.maximum(curr_obj, 0.0))
    return recon, errors


def reconstruct_hio(
    intensity: np.ndarray,
    support_mask: np.ndarray | None = None,
    max_iterations: int = 50,
    beta: float = 0.8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convenience wrapper for HIO phase retrieval from intensity."""
    amp = np.sqrt(np.maximum(intensity, 0.0))
    if support_mask is None:
        ny, nx = intensity.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        support_mask = (y - cy)**2 + (x - cx)**2 <= 20.0**2
    recon, _ = hybrid_input_output(amp, support_mask, num_iterations=max_iterations, beta=beta, rng=rng)
    ny, nx = recon.shape
    cy, cx = ny // 2, nx // 2
    cropped = recon[cy - 32 : cy + 32, cx - 32 : cx + 32]
    p_max = np.max(cropped)
    if p_max > 0:
        cropped = cropped / p_max
    return cropped.astype(np.float32)

