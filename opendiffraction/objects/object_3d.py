"""Synthesis of 3D scattering density volumes for Model F6.

Procedural generation of 3D scalar volumes representing nanoscale specimens:
spheres, core-shell particles, ellipsoids, nanorods, and multi-particle clusters.
"""

from __future__ import annotations

import numpy as np


def make_coordinate_grid_3d(size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate centered 3D coordinate grids (z, y, x) and radial distance r."""
    coords = np.arange(size) - size / 2.0 + 0.5
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(z**2 + y**2 + x**2)
    return z, y, x, r


def make_3d_sphere(
    size: int = 64,
    radius: float = 16.0,
    density: float = 1.0,
) -> np.ndarray:
    """Generate a 3D homogeneous solid sphere."""
    _, _, _, r = make_coordinate_grid_3d(size)
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[r <= radius] = density
    return vol


def make_3d_core_shell(
    size: int = 64,
    core_radius: float = 10.0,
    shell_radius: float = 18.0,
    core_density: float = 1.0,
    shell_density: float = 0.4,
) -> np.ndarray:
    """Generate a concentric 3D core-shell sphere."""
    _, _, _, r = make_coordinate_grid_3d(size)
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[r <= shell_radius] = shell_density
    vol[r <= core_radius] = core_density
    return vol


def make_3d_ellipsoid(
    size: int = 64,
    radii: tuple[float, float, float] = (18.0, 12.0, 8.0),
    density: float = 1.0,
) -> np.ndarray:
    """Generate an oriented 3D triaxial ellipsoid."""
    z, y, x, _ = make_coordinate_grid_3d(size)
    rz, ry, rx = radii
    normalized_dist_sq = (z / rz) ** 2 + (y / ry) ** 2 + (x / rx) ** 2
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[normalized_dist_sq <= 1.0] = density
    return vol


def make_3d_nanorod(
    size: int = 64,
    length: float = 28.0,
    radius: float = 6.0,
    density: float = 1.0,
) -> np.ndarray:
    """Generate a 3D cylinder aligned along the z-axis with hemispherical caps."""
    z, y, x, _ = make_coordinate_grid_3d(size)
    r_xy = np.sqrt(x**2 + y**2)
    half_cyl = max(0.0, (length / 2.0) - radius)

    vol = np.zeros((size, size, size), dtype=np.float32)
    # Cylindrical body
    body_mask = (r_xy <= radius) & (np.abs(z) <= half_cyl)
    # End caps
    top_cap = (r_xy**2 + (z - half_cyl)**2) <= radius**2
    bot_cap = (r_xy**2 + (z + half_cyl)**2) <= radius**2

    vol[body_mask | top_cap | bot_cap] = density
    return vol


def make_3d_cluster(
    size: int = 64,
    n_particles: int = 5,
    min_radius: float = 5.0,
    max_radius: float = 9.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a cluster of bonded spherical particles."""
    if rng is None:
        rng = np.random.default_rng()

    z, y, x, _ = make_coordinate_grid_3d(size)
    vol = np.zeros((size, size, size), dtype=np.float32)

    centers = [[0.0, 0.0, 0.0]]
    for _ in range(n_particles):
        parent_idx = rng.integers(0, len(centers))
        px, py, pz = centers[parent_idx]
        r_part = rng.uniform(min_radius, max_radius)

        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        sep = rng.uniform(min_radius * 1.2, max_radius * 1.6)

        cx, cy, cz = px + direction[0] * sep, py + direction[1] * sep, pz + direction[2] * sep
        # Keep inside bounding volume
        if abs(cx) < size / 2.5 and abs(cy) < size / 2.5 and abs(cz) < size / 2.5:
            centers.append([cx, cy, cz])
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            density = rng.uniform(0.6, 1.0)
            vol[dist_sq <= r_part**2] = np.maximum(vol[dist_sq <= r_part**2], density)

    return vol


def generate_3d_volume(
    shape_type: str = "sphere",
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 3D volume according to specified shape type."""
    if rng is None:
        rng = np.random.default_rng()

    if shape_type == "sphere":
        return make_3d_sphere(size=size, radius=float(rng.uniform(12.0, 18.0)))
    elif shape_type == "core_shell":
        cr = float(rng.uniform(6.0, 10.0))
        sr = cr + float(rng.uniform(4.0, 8.0))
        return make_3d_core_shell(size=size, core_radius=cr, shell_radius=sr)
    elif shape_type == "ellipsoid":
        radii = (
            float(rng.uniform(14.0, 20.0)),
            float(rng.uniform(9.0, 14.0)),
            float(rng.uniform(6.0, 10.0)),
        )
        return make_3d_ellipsoid(size=size, radii=radii)
    elif shape_type == "nanorod":
        return make_3d_nanorod(
            size=size,
            length=float(rng.uniform(20.0, 32.0)),
            radius=float(rng.uniform(4.0, 7.0)),
        )
    elif shape_type == "cluster":
        return make_3d_cluster(size=size, n_particles=int(rng.integers(4, 8)), rng=rng)
    else:
        return make_3d_sphere(size=size, radius=15.0)
