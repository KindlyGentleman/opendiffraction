"""Spatial coherence modeling using the Gaussian Schell-model (GSM).

The implementation supports:
1. The corrected physical model (default): Gaussian damping centered at zero autocorrelation lag.
2. The legacy model (DPR v1.0-legacy): unshifted grid coordinates for regression verification.
"""

from __future__ import annotations

import numpy as np


def compute_lag_grid_sq(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generate squared Euclidean lag grid with DC centered.

    For shape (512, 512), coordinates span [-256, 255] along each axis.
    """
    ny, nx = shape
    y = np.linspace(-ny // 2, ny // 2 - 1, ny, dtype=np.float64)
    x = np.linspace(-nx // 2, nx // 2 - 1, nx, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return yy**2 + xx**2


def create_coherence_kernel(
    l_coh: float,
    shape: tuple[int, int] = (512, 512),
    legacy: bool = False,
) -> np.ndarray:
    """Create the Gaussian Schell-model coherence kernel.

    Parameters
    ----------
    l_coh : float
        Spatial coherence length in pixels. If inf or <= 0, returns all ones (fully coherent).
    shape : tuple[int, int], default (512, 512)
        Detector intensity grid dimensions.
    legacy : bool, default False
        If True, reproduces DPR v1.0-legacy unshifted grid convention.
        If False (default), correctly aligns the kernel peak at zero autocorrelation lag (0, 0).

    Returns
    -------
    np.ndarray
        Float64 kernel array matching shape.
    """
    if np.isinf(l_coh) or l_coh <= 0:
        return np.ones(shape, dtype=np.float64)

    r_sq = compute_lag_grid_sq(shape)
    if not legacy:
        # ifft2 output has zero lag at index [0, 0], so shift kernel center to [0, 0]
        r_sq = np.fft.ifftshift(r_sq)

    kernel = np.exp(-r_sq / (2.0 * (l_coh**2)))
    return kernel


def apply_partial_coherence(
    intensity: np.ndarray,
    l_coh: float,
    legacy: bool = False,
) -> np.ndarray:
    """Apply spatial partial coherence to centered coherent diffraction intensity.

    Parameters
    ----------
    intensity : np.ndarray
        2D or 3D/4D array of diffraction intensities with DC at center (fftshifted).
    l_coh : float
        Coherence length in pixels.
    legacy : bool, default False
        Use legacy unshifted coherence model if True.

    Returns
    -------
    np.ndarray
        Partially coherent intensity, nonnegative, matching input shape.
    """
    if np.isinf(l_coh) or l_coh <= 0:
        return np.maximum(intensity, 0.0)

    ny, nx = intensity.shape[-2:]
    kernel = create_coherence_kernel(l_coh, shape=(ny, nx), legacy=legacy)

    # Autocorrelation via IFFT of centered intensity
    # ifftshift shifts DC from center back to index [0, 0] for ifft2
    autocorr = np.fft.ifft2(np.fft.ifftshift(intensity, axes=(-2, -1)), axes=(-2, -1))
    blurred_autocorr = autocorr * kernel

    # Transform back to diffraction intensity domain
    out = np.fft.fftshift(np.fft.fft2(blurred_autocorr, axes=(-2, -1)), axes=(-2, -1))
    return np.maximum(np.real(out), 0.0)
