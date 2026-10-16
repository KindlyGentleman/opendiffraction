"""Model F4: Near-Field Fresnel Optical Propagation.

Uses the Band-Limited Angular Spectrum Method (ASM) with physical SI units:
wavelength (m), propagation distance (m), and pixel pitch (m).
"""

from __future__ import annotations

import numpy as np


def simulate_fresnel_diffraction(
    input_field: np.ndarray,
    wavelength_m: float = 1e-10,  # 0.1 nm (12.4 keV X-ray) or 632.8 nm (HeNe laser)
    propagation_distance_m: float = 0.5,  # 50 cm
    pixel_pitch_m: float = 75e-6,  # 75 um (e.g. Pilatus pixel)
    detector_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate an optical field over distance z using the band-limited Angular Spectrum Method.

    Parameters
    ----------
    input_field : np.ndarray
        2D field at source plane, padded into detector_size frame.
    wavelength_m : float
        X-ray or optical wavelength in meters.
    propagation_distance_m : float
        Propagation distance z in meters.
    pixel_pitch_m : float
        Pixel sampling interval Delta x in meters.
    detector_size : tuple[int, int]
        Grid dimensions (Ny, Nx).

    Returns
    -------
    propagated_field : np.ndarray
        Complex field at distance z.
    intensity : np.ndarray
        Diffraction intensity |propagated_field|^2.
    """
    ny, nx = detector_size
    if input_field.shape != detector_size:
        # Center input_field into detector_size
        padded = np.zeros(detector_size, dtype=np.complex128)
        sy0 = (ny - input_field.shape[0]) // 2
        sx0 = (nx - input_field.shape[1]) // 2
        padded[sy0 : sy0 + input_field.shape[0], sx0 : sx0 + input_field.shape[1]] = input_field
        input_field = padded

    # Spatial frequency grid in cycles/meter
    df_y = 1.0 / (ny * pixel_pitch_m)
    df_x = 1.0 / (nx * pixel_pitch_m)
    fy = np.fft.fftfreq(ny, d=pixel_pitch_m)
    fx = np.fft.fftfreq(nx, d=pixel_pitch_m)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    f_sq = fyy**2 + fxx**2

    # Bandwidth limit to avoid spatial aliasing in transfer function (Matsushima & Shimobaba 2009)
    # The phase derivative with respect to f must not exceed pi per frequency step
    lam = wavelength_m
    z = propagation_distance_m
    L_y = ny * pixel_pitch_m
    L_x = nx * pixel_pitch_m

    # Critical cutoffs
    f_limit_y = 1.0 / (lam * np.sqrt((2.0 * df_y * z)**2 + 1.0))
    f_limit_x = 1.0 / (lam * np.sqrt((2.0 * df_x * z)**2 + 1.0))
    valid_transfer = (np.abs(fyy) <= f_limit_y) & (np.abs(fxx) <= f_limit_x) & (f_sq < (1.0 / lam**2))

    # Transfer function: H(fx, fy) = exp(i * 2 * pi * z * sqrt(1/lam^2 - f^2))
    kz = np.zeros_like(f_sq)
    kz[valid_transfer] = np.sqrt((1.0 / lam**2) - f_sq[valid_transfer])
    H = np.zeros_like(f_sq, dtype=np.complex128)
    H[valid_transfer] = np.exp(1j * 2.0 * np.pi * z * kz[valid_transfer])

    # Propagate via Fourier transform
    F_in = np.fft.fft2(input_field)
    F_out = F_in * H
    propagated_field = np.fft.ifft2(F_out)

    # Shift DC to center for detector convention
    field_centered = np.fft.fftshift(propagated_field)
    intensity = np.maximum(np.real(field_centered * np.conj(field_centered)), 0.0).astype(np.float32)

    return field_centered, intensity


def simulate_fresnel_nearfield(
    input_field: np.ndarray,
    wavelength: float = 1e-10,
    distance: float = 0.5,
    pixel_size: float = 75e-6,
    detector_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray]:
    """Ergonomic alias for simulate_fresnel_diffraction."""
    return simulate_fresnel_diffraction(
        input_field,
        wavelength_m=wavelength,
        propagation_distance_m=distance,
        pixel_pitch_m=pixel_size,
        detector_size=detector_size,
    )

