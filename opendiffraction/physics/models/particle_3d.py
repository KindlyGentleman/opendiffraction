"""Model F6: 3D Particle Scattering.

Simulates diffraction from an explicit 3D scattering density volume rho(x, y, z)
under 3D Euler rotations using the Fourier Slice Theorem and first Born projection approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform
from .fraunhofer import simulate_fraunhofer_coherent


def rotation_matrix_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Generate 3x3 rotation matrix for Z-Y-Z Euler angles in radians."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return Rz1 @ Ry @ Rz2


def rotate_volume_3d(
    volume: np.ndarray,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Rotate a 3D scalar volume around its spatial center (Nz//2, Ny//2, Nx//2).

    Parameters
    ----------
    volume : np.ndarray
        3D float array of shape (Nz, Ny, Nx).
    angles : tuple[float, float, float]
        Euler angles (alpha, beta, gamma) in radians.

    Returns
    -------
    np.ndarray
        Rotated 3D volume matching input shape.
    """
    R = rotation_matrix_euler(*angles)
    inv_R = R.T  # Inverse rotation for pull-back coordinate mapping

    center = np.array(volume.shape) / 2.0 - 0.5
    offset = center - inv_R @ center

    rotated = affine_transform(
        volume,
        matrix=inv_R,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return rotated.astype(np.float32)


def project_volume_to_2d(
    volume_3d: np.ndarray,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compute 2D transmission projection integral of rotated 3D volume along z-axis.

    P(y, x) = integral rho(x, y, z) dz (First Born projection approximation).
    """
    if any(abs(a) > 1e-4 for a in angles):
        vol_rot = rotate_volume_3d(volume_3d, angles=angles)
    else:
        vol_rot = volume_3d

    # Integrate along depth axis z (axis 0)
    projection = np.sum(vol_rot, axis=0)
    # Normalize projection peak to [0, 1]
    p_max = np.max(projection)
    if p_max > 0:
        projection = projection / p_max

    return projection.astype(np.float32)


def simulate_3d_scattering(
    volume_3d: np.ndarray,
    euler_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    detector_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate single-particle coherent diffraction from a true 3D scattering density.

    Parameters
    ----------
    volume_3d : np.ndarray
        3D scattering density volume of shape (64, 64, 64).
    euler_angles : tuple[float, float, float]
        Euler orientation angles (alpha, beta, gamma) in radians.
    detector_size : tuple[int, int]
        Detector size (512, 512).

    Returns
    -------
    projection_2d : np.ndarray
        Projected 2D ground truth transmission density, shape (64, 64).
    diffraction_field : np.ndarray
        Complex 2D diffraction field on detector.
    diffraction_intensity : np.ndarray
        2D diffraction intensity |field|^2 on detector.
    """
    projection_2d = project_volume_to_2d(volume_3d, angles=euler_angles)
    field, intensity = simulate_fraunhofer_coherent(projection_2d, detector_size=detector_size)
    return projection_2d, field, intensity
