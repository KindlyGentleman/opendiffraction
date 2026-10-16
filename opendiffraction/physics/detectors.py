"""Detector degradation models: Gain nonuniformity, baseline drift, charge sharing/PSF blur, saturation, and ADC quantization."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_pixel_gain_nonuniformity(
    readout: np.ndarray,
    gain_std: float = 0.03,  # 3% gain variation
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pixel-to-pixel gain variations (flat-field response).

    Measured: Y_i = G_i * Signal_i
    where G_i ~ Normal(1.0, gain_std).

    Parameters
    ----------
    readout : np.ndarray
        Detector readouts.
    gain_std : float
        Fractional standard deviation of pixel gain.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    degraded_readout : np.ndarray
        Gain-modulated readout array.
    gain_matrix : np.ndarray
        True pixel-wise gain matrix G.
    """
    if rng is None:
        rng = np.random.default_rng()

    if gain_std <= 0:
        return readout.copy(), np.ones_like(readout)

    gain_matrix = rng.normal(1.0, gain_std, size=readout.shape).astype(np.float32)
    gain_matrix = np.clip(gain_matrix, 0.5, 1.5)
    return (readout * gain_matrix).astype(np.float32), gain_matrix


def apply_dark_current_and_baseline_drift(
    readout: np.ndarray,
    baseline_offset: float = 20.0,
    drift_std: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Add exposure-dependent baseline offset and drift.

    Parameters
    ----------
    readout : np.ndarray
        Detector readouts.
    baseline_offset : float
        Nominal electronic baseline ADU.
    drift_std : float
        Frame-to-frame drift fluctuation standard deviation.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    offset_readout : np.ndarray
        Readout with added baseline.
    actual_offset : float
        Applied baseline value.
    """
    if rng is None:
        rng = np.random.default_rng()

    actual_offset = baseline_offset + float(rng.normal(0.0, drift_std))
    return (readout + actual_offset).astype(np.float32), actual_offset


def apply_charge_spreading_blur(
    readout: np.ndarray,
    psf_sigma_pixels: float = 0.7,
) -> np.ndarray:
    """Simulate charge sharing between adjacent detector pixels via Gaussian PSF convolution.

    Parameters
    ----------
    readout : np.ndarray
        Input photon / electron counts.
    psf_sigma_pixels : float
        Point spread function standard deviation in pixels.

    Returns
    -------
    np.ndarray
        Blurred readout array conserving total signal.
    """
    if psf_sigma_pixels <= 0:
        return readout.copy()

    return gaussian_filter(readout, sigma=psf_sigma_pixels).astype(np.float32)


def apply_adc_quantization(
    readout: np.ndarray,
    adc_bits: int = 14,
    adu_per_photon: float = 1.0,
) -> np.ndarray:
    """Simulate finite Analog-to-Digital Converter (ADC) quantization.

    Parameters
    ----------
    readout : np.ndarray
        Continuous detector charge / signal.
    adc_bits : int
        ADC resolution (e.g. 12-bit, 14-bit, 16-bit).
    adu_per_photon : float
        Conversion gain (Digital ADU count per detected photon).

    Returns
    -------
    np.ndarray
        Discrete quantized ADU values clipped to [0, 2^bits - 1].
    """
    max_adu = (2**adc_bits) - 1
    scaled = readout * adu_per_photon
    quantized = np.round(scaled)
    return np.clip(quantized, 0, max_adu).astype(np.float32)
