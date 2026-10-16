"""Benchmark B2: Diffraction Pattern Denoising Baseline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from opendiffraction.tasks.metrics import psnr, compute_r_factor


def denoise_diffraction_pattern(
    noisy_intensity: np.ndarray,
    method: str = "median_gaussian",
    sigma: float = 1.0,
    median_size: int = 3,
) -> np.ndarray:
    """Apply classical filtering to noisy diffraction pattern.

    Parameters
    ----------
    noisy_intensity : np.ndarray
        Raw noisy detector readout with Poisson + Gaussian noise.
    method : str
        Filtering pipeline ('median', 'gaussian', or 'median_gaussian').
    sigma : float
        Gaussian kernel standard deviation.
    median_size : int
        Median filter kernel dimension.

    Returns
    -------
    np.ndarray
        Denoised intensity estimate.
    """
    clipped = np.maximum(noisy_intensity, 0.0)

    if method == "median":
        return median_filter(clipped, size=median_size)
    elif method == "gaussian":
        return gaussian_filter(clipped, sigma=sigma)
    elif method == "median_gaussian":
        med = median_filter(clipped, size=median_size)
        return gaussian_filter(med, sigma=sigma)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_denoising(
    noisy_intensity: np.ndarray,
    expected_intensity: np.ndarray,
) -> dict[str, float]:
    """Evaluate baseline denoising against ground-truth expected intensity."""
    denoised = denoise_diffraction_pattern(noisy_intensity)
    p_val = psnr(expected_intensity, denoised)
    r_val = compute_r_factor(expected_intensity, denoised)

    return {
        "psnr_db": float(p_val),
        "r_factor": float(r_val),
    }


if __name__ == "__main__":
    from opendiffraction.objects.specimens import make_core_shell_nanoparticle
    from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction

    rng = np.random.default_rng(42)
    obj = make_core_shell_nanoparticle(64, rng=rng)
    cfg = SimulationConfig(photon_budget=1e6)
    sim = simulate_diffraction(obj, cfg, rng=rng)

    metrics = evaluate_denoising(sim.intensity_measured, sim.intensity_expected)
    print("Denoising Baseline Results:", metrics)
