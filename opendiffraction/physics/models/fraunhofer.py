"""Model F1: Coherent Fraunhofer Diffraction.

Far-field intensity from real or complex 2D transmission functions under monochromatic plane-wave illumination.
"""

from __future__ import annotations

import numpy as np


def simulate_fraunhofer_coherent(
    object_field: np.ndarray,
    detector_size: tuple[int, int] = (512, 512),
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate coherent Fraunhofer far-field diffraction field and intensity.

    Parameters
    ----------
    object_field : np.ndarray
        Real or complex 2D object transmission function, shape (Ny_obj, Nx_obj).
    detector_size : tuple[int, int]
        Detector grid dimensions (Ny_det, Nx_det), default (512, 512).
    offset : tuple[float, float]
        Sub-pixel or integer displacement (dy, dx) of object relative to frame center.

    Returns
    -------
    field : np.ndarray
        Complex centered diffraction field, shape detector_size.
    intensity : np.ndarray
        Real nonnegative intensity |field|^2, shape detector_size.
    """
    ny_det, nx_det = detector_size
    ny_obj, nx_obj = object_field.shape[-2:]

    # Place object at center of detector frame
    y0 = (ny_det - ny_obj) // 2
    x0 = (nx_det - nx_obj) // 2

    dy_int, dx_int = int(round(offset[0])), int(round(offset[1]))
    y0_adj = y0 + dy_int
    x0_adj = x0 + dx_int

    frame = np.zeros(detector_size, dtype=object_field.dtype)
    sy0 = max(0, y0_adj)
    sy1 = min(ny_det, y0_adj + ny_obj)
    sx0 = max(0, x0_adj)
    sx1 = min(nx_det, x0_adj + nx_obj)

    oy0 = max(0, -y0_adj)
    oy1 = oy0 + (sy1 - sy0)
    ox0 = max(0, -x0_adj)
    ox1 = ox0 + (sx1 - sx0)

    if sy1 > sy0 and sx1 > sx0:
        frame[sy0:sy1, sx0:sx1] = object_field[oy0:oy1, ox0:ox1]

    # Sub-pixel shift via Fourier phase ramp if needed
    rem_y = offset[0] - dy_int
    rem_x = offset[1] - dx_int
    if abs(rem_y) > 1e-4 or abs(rem_x) > 1e-4:
        ky = np.fft.fftfreq(ny_det)
        kx = np.fft.fftfreq(nx_det)
        kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
        phase_ramp = np.exp(-2j * np.pi * (kyy * rem_y + kxx * rem_x))
        F_frame = np.fft.fft2(frame)
        frame = np.fft.ifft2(F_frame * phase_ramp)
        if np.isrealobj(object_field):
            frame = np.real(frame)

    # 2D Centered FFT
    field = np.fft.fftshift(np.fft.fft2(frame))
    intensity = np.maximum(np.real(field * np.conj(field)), 0.0).astype(np.float32)

    return field, intensity
