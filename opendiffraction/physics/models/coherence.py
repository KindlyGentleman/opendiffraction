"""Model F2: Partial-Coherence Diffraction.

Supports:
1. Gaussian Schell-model (GSM) autocorrelation damping centered at zero spatial lag.
2. Coherent-mode decomposition (incoherent sum of orthogonal spatial modes).
"""

from __future__ import annotations

import numpy as np
from opendiffraction.physics.coherence import apply_partial_coherence, create_coherence_kernel
from .fraunhofer import simulate_fraunhofer_coherent


def simulate_partial_coherence_gsm(
    intensity: np.ndarray,
    l_coh: float,
    legacy: bool = False,
) -> np.ndarray:
    """Apply Gaussian Schell-model spatial coherence damping.

    Parameters
    ----------
    intensity : np.ndarray
        Diffraction intensity with centered zero-frequency DC.
    l_coh : float
        Coherence length in pixels.
    legacy : bool
        If True, reproduces DPR v1.0-legacy unshifted kernel. Default False (corrected zero-lag).

    Returns
    -------
    np.ndarray
        Partially coherent intensity.
    """
    return apply_partial_coherence(intensity, l_coh=l_coh, legacy=legacy)


def simulate_multimode_coherence(
    object_field: np.ndarray,
    num_modes: int = 4,
    mode_decay_ratio: float = 0.5,
    detector_size: tuple[int, int] = (512, 512),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Simulate partial spatial coherence via coherent mode decomposition.

    I_total(q) = sum_m mu_m * |FFT[ phi_m(r) * O(r) ]|^2
    where phi_m are orthogonal Hermite-Gaussian spatial illumination modes.

    Parameters
    ----------
    object_field : np.ndarray
        Real or complex object transmission function.
    num_modes : int
        Number of mutually incoherent modes.
    mode_decay_ratio : float
        Eigenvalue decay ratio (fraction of power in subsequent modes).
    detector_size : tuple[int, int]
        Detector size (512, 512).
    rng : np.random.Generator or None
        Random generator for slight modal perturbations.

    Returns
    -------
    total_intensity : np.ndarray
        Incoherent sum of modal diffraction intensities.
    mode_weights : list of float
        Normalized modal weights mu_m.
    """
    if rng is None:
        rng = np.random.default_rng()

    ny_obj, nx_obj = object_field.shape[-2:]
    y = np.linspace(-1, 1, ny_obj)
    x = np.linspace(-1, 1, nx_obj)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Generate Hermite-Gaussian-like spatial modes
    weights = [mode_decay_ratio**m for m in range(num_modes)]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    total_intensity = np.zeros(detector_size, dtype=np.float32)

    for m, w in enumerate(norm_weights):
        # Mode spatial profile: Gaussian envelope * Hermite polynomials
        if m == 0:
            mode_profile = np.exp(-(xx**2 + yy**2) / 0.8)
        elif m == 1:
            mode_profile = 2 * xx * np.exp(-(xx**2 + yy**2) / 0.8)
        elif m == 2:
            mode_profile = 2 * yy * np.exp(-(xx**2 + yy**2) / 0.8)
        else:
            mode_profile = (4 * xx * yy) * np.exp(-(xx**2 + yy**2) / 0.8)

        # Modulate object by illumination mode
        modal_exit_wave = object_field * mode_profile
        _, mode_inten = simulate_fraunhofer_coherent(modal_exit_wave, detector_size=detector_size)
        total_intensity += (w * mode_inten).astype(np.float32)

    return total_intensity, norm_weights
