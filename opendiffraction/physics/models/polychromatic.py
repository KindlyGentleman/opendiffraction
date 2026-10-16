"""Model F5: Polychromatic Diffraction.

Models finite spectral bandwidth (SASE XFEL spikes or pink beam) by numerically integrating
far-field diffraction patterns over a wavelength spectrum with wavelength-dependent reciprocal coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from .fraunhofer import simulate_fraunhofer_coherent


def simulate_polychromatic_diffraction(
    object_field: np.ndarray,
    central_wavelength_m: float = 1e-10,  # 0.1 nm (12.4 keV)
    spectral_bandwidth_ratio: float = 0.05,  # Delta lambda / lambda (e.g. 5% pink beam or 0.2% SASE)
    num_wavelength_bins: int = 7,
    detector_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Simulate polychromatic diffraction by spectral integration.

    Radial coordinate scaling on detector scales linearly with wavelength: r_det(lambda) ~ lambda.
    Shorter wavelengths compress fringes towards center; longer wavelengths stretch fringes outwards.

    Parameters
    ----------
    object_field : np.ndarray
        2D object transmission function.
    central_wavelength_m : float
        Nominal beam wavelength lambda_0 in meters.
    spectral_bandwidth_ratio : float
        Fractional bandwidth Delta lambda / lambda.
    num_wavelength_bins : int
        Number of discrete wavelength integration steps.
    detector_size : tuple[int, int]
        Detector size (512, 512).

    Returns
    -------
    polychromatic_intensity : np.ndarray
        Integrated polychromatic diffraction intensity.
    spectral_components : list of (wavelength, weight)
        Sampled wavelengths and normalized weights.
    """
    ny, nx = detector_size
    cy, cx = ny // 2, nx // 2

    # Gaussian spectral weight distribution
    half_bw = spectral_bandwidth_ratio / 2.0
    lambdas = np.linspace(
        central_wavelength_m * (1.0 - half_bw),
        central_wavelength_m * (1.0 + half_bw),
        num_wavelength_bins,
    )
    sigma_lam = (central_wavelength_m * half_bw) / 2.0
    raw_weights = np.exp(-((lambdas - central_wavelength_m) ** 2) / (2.0 * sigma_lam**2))
    norm_weights = raw_weights / np.sum(raw_weights)

    # 1. Monochromatic base pattern at lambda_0
    _, base_intensity = simulate_fraunhofer_coherent(object_field, detector_size=detector_size)

    # 2. Integrate scaled patterns
    polychromatic_intensity = np.zeros(detector_size, dtype=np.float32)
    spectral_components = []

    for lam, w in zip(lambdas, norm_weights):
        spectral_components.append((float(lam), float(w)))
        scale = lam / central_wavelength_m  # scale > 1 stretches fringes, scale < 1 compresses

        if abs(scale - 1.0) < 1e-4:
            polychromatic_intensity += (w * base_intensity).astype(np.float32)
        else:
            # Rescale array around center (cy, cx) using bilinear interpolation
            # Coordinate mapping: r_new = r_old / scale
            y, x = np.mgrid[:ny, :nx]
            y_src = (y - cy) / scale + cy
            x_src = (x - cx) / scale + cx

            from scipy.ndimage import map_coordinates
            scaled_I = map_coordinates(base_intensity, [y_src, x_src], order=1, mode="constant", cval=0.0)
            polychromatic_intensity += (w * scaled_I).astype(np.float32)

    return polychromatic_intensity, spectral_components
