"""Detector degradation models: Poisson shot noise, Gaussian readout noise, saturation, and hot pixels."""

from __future__ import annotations

import numpy as np


FWHM_ONE_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.4246609, giving FWHM = 1


def scale_to_photon_budget(
    intensity: np.ndarray,
    photon_budget: float,
) -> tuple[np.ndarray, float]:
    """Scale an intensity pattern so its sum equals the target photon budget.

    Parameters
    ----------
    intensity : np.ndarray
        Nonnegative intensity pattern.
    photon_budget : float
        Total expected photon count.

    Returns
    -------
    expected_photons : np.ndarray
        Pixel-wise expected photon arrival rate lambda_i.
    scale_factor : float
        Scalar multiplier s = photon_budget / sum(intensity).
    """
    total = float(np.sum(intensity))
    if total <= 0:
        return np.zeros_like(intensity, dtype=np.float64), 0.0

    scale_factor = float(photon_budget / total)
    expected_photons = (intensity * scale_factor).astype(np.float64)
    return expected_photons, scale_factor


def apply_poisson_noise(
    expected_photons: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample discrete photon counts from Poisson distribution.

    Parameters
    ----------
    expected_photons : np.ndarray
        Expected photon rate lambda_i >= 0.
    rng : np.random.Generator
        Deterministic NumPy random generator.

    Returns
    -------
    np.ndarray
        Integer/float Poisson sampled counts.
    """
    safe_rate = np.maximum(expected_photons, 0.0)
    return rng.poisson(safe_rate).astype(np.float64)


def apply_gaussian_readout_noise(
    detected_photons: np.ndarray,
    readout_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian detector readout noise to photon counts.

    Note: Readout noise may produce negative values for low photon counts.
    These signed values are preserved to prevent dark-subtraction bias.

    Parameters
    ----------
    detected_photons : np.ndarray
        Photon counts per pixel.
    readout_std : float
        Standard deviation of Gaussian readout noise. If <= 0, no noise is added.
    rng : np.random.Generator
        Deterministic NumPy random generator.

    Returns
    -------
    np.ndarray
        Readout values (can include negative values).
    """
    if readout_std <= 0:
        return detected_photons.copy()

    noise = rng.normal(0.0, readout_std, size=detected_photons.shape)
    return detected_photons + noise


def apply_saturation_and_defects(
    measured: np.ndarray,
    saturation_threshold: float | None = None,
    hot_pixel_coords: list[tuple[int, int]] | None = None,
    hot_pixel_value: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply detector full-well saturation and hot pixel defects.

    Parameters
    ----------
    measured : np.ndarray
        Detector readouts.
    saturation_threshold : float or None
        Maximum linear detector capacity. Values exceeding this are clipped.
    hot_pixel_coords : list of (y, x) or None
        Coordinates of defective saturated pixels.
    hot_pixel_value : float
        Value assigned to hot pixels.

    Returns
    -------
    out : np.ndarray
        Modified readout array.
    defect_mask : np.ndarray
        Boolean mask of defects (True for hot pixels or saturated pixels).
    """
    out = measured.copy()
    defect_mask = np.zeros(measured.shape[-2:], dtype=bool)

    if saturation_threshold is not None and saturation_threshold > 0:
        sat_indices = out >= saturation_threshold
        out[sat_indices] = saturation_threshold
        defect_mask |= sat_indices.squeeze()

    if hot_pixel_coords:
        for py, px in hot_pixel_coords:
            if 0 <= py < out.shape[-2] and 0 <= px < out.shape[-1]:
                out[..., py, px] = hot_pixel_value
                defect_mask[py, px] = True

    return out, defect_mask
