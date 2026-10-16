"""Model F3: Thin-Object Coherent Transmission.

Models an incident optical probe P(r) interacting with a thin specimen transmission function T(r).
Exit wave: psi_exit(r) = P(r) * T(r).
"""

from __future__ import annotations

import numpy as np
from .fraunhofer import simulate_fraunhofer_coherent


def generate_probe(
    size: int = 64,
    fwhm: float = 30.0,
    defocus_rad: float = 0.0,
    astigmatism_rad: float = 0.0,
    astigmatism_angle: float = 0.0,
    coma_rad: float = 0.0,
    jitter_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate a 2D complex optical probe field with aberrations and pointing jitter.

    Parameters
    ----------
    size : int
        Grid dimension (default 64).
    fwhm : float
        Beam full-width at half-maximum in pixels.
    defocus_rad : float
        Quadratic wavefront phase curvature (radians at margin).
    astigmatism_rad : float
        Astigmatic phase amplitude.
    astigmatism_angle : float
        Astigmatism axis in radians.
    coma_rad : float
        Comatic wavefront aberration amplitude.
    jitter_offset : tuple[float, float]
        Pointing jitter displacement (dy, dx) in pixels.

    Returns
    -------
    probe : np.ndarray of complex128
        Incident complex probe field.
    """
    c = (size - 1.0) / 2.0
    y, x = np.mgrid[:size, :size] - c
    y_jit = y - jitter_offset[0]
    x_jit = x - jitter_offset[1]
    r_jit = np.sqrt(y_jit**2 + x_jit**2)
    theta_jit = np.arctan2(y_jit, x_jit)

    # Gaussian amplitude envelope
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    amp = np.exp(-(r_jit**2) / (2.0 * (sigma**2)))

    # Wavefront aberrations: W(r) = defocus * r^2 + astigmatism * r^2 * cos(2(theta - phi)) + coma * r^3 * cos(theta)
    r_norm = r_jit / (size / 2.0)
    phase = defocus_rad * (r_norm**2)
    if abs(astigmatism_rad) > 1e-5:
        phase += astigmatism_rad * (r_norm**2) * np.cos(2 * (theta_jit - astigmatism_angle))
    if abs(coma_rad) > 1e-5:
        phase += coma_rad * (r_norm**3) * np.cos(theta_jit)

    probe = (amp * np.exp(1j * phase)).astype(np.complex128)
    return probe


def simulate_thin_object_transmission(
    specimen_transmission: np.ndarray,
    probe_field: np.ndarray | None = None,
    detector_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate transmission through a thin specimen under probe illumination.

    Parameters
    ----------
    specimen_transmission : np.ndarray
        Complex or real specimen transmission T(r), shape (64, 64).
    probe_field : np.ndarray or None
        Incident probe P(r). If None, defaults to uniform plane wave.
    detector_size : tuple[int, int]
        Detector grid size (512, 512).

    Returns
    -------
    exit_wave : np.ndarray
        Exit surface field psi_exit = P * T.
    diffraction_field : np.ndarray
        Far-field Fraunhofer amplitude.
    intensity : np.ndarray
        Far-field Fraunhofer intensity |diffraction_field|^2.
    """
    if probe_field is None:
        exit_wave = specimen_transmission.astype(np.complex128)
    else:
        exit_wave = (probe_field * specimen_transmission).astype(np.complex128)

    field, intensity = simulate_fraunhofer_coherent(exit_wave, detector_size=detector_size)
    return exit_wave, field, intensity
