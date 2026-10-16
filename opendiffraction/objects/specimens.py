"""Group 1: 12 Real-World Specimen Phantoms (V01 - V12).

Procedural synthesis of realistic physical proxies for biological CDI, nanomaterials,
and materials science benchmarking. Fully deterministic and parameter-controlled.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from opendiffraction.objects.procedural import make_coordinate_grid


def make_cell_phantom(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V01: Cell-like biological phantom with outer membrane, nucleus, and organelles."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, theta = make_coordinate_grid(size)

    # Deformable outer membrane boundary: r_cell(theta)
    r_cell = 20.0 + 3.0 * np.sin(2 * theta) + 2.0 * np.cos(3 * theta + rng.uniform(0, np.pi))
    cytoplasm = r <= r_cell

    # Nucleus (dense internal organelle, slightly displaced)
    ny_off, nx_off = rng.uniform(-4, 4), rng.uniform(-4, 4)
    r_nuc = np.sqrt((y - ny_off) ** 2 + (x - nx_off) ** 2)
    nucleus = r_nuc <= rng.uniform(6.0, 9.0)

    # Subcellular vesicles / mitochondria
    vesicles = np.zeros((size, size), dtype=bool)
    for _ in range(rng.integers(4, 9)):
        vy, vx = rng.uniform(-14, 14), rng.uniform(-14, 14)
        if (vy**2 + vx**2 < 16.0**2) and not ((vy - ny_off)**2 + (vx - nx_off)**2 < 8.0**2):
            vesicles |= (y - vy)**2 + (x - vx)**2 <= rng.uniform(1.5, 2.5)**2

    density = np.zeros((size, size), dtype=np.float32)
    density[cytoplasm] = 0.4
    density[nucleus] = 0.85
    density[vesicles] = 0.7

    # Smooth internal cytoplasmic texture
    texture = gaussian_filter(rng.standard_normal((size, size)), sigma=2.0)
    texture = 0.1 * (texture / (np.max(np.abs(texture)) + 1e-6))
    density = np.clip(density + texture * cytoplasm, 0.0, 1.0)

    return density.astype(np.float32)


def make_virus_particle(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V02: Virus-like particle with protein capsid shell and surface spike projections."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, theta = make_coordinate_grid(size)

    r_capsid = rng.uniform(12.0, 16.0)
    capsid = r <= r_capsid
    core = r <= (r_capsid - 3.5)

    # Capsid shell has higher electron density than genome core
    density = np.zeros((size, size), dtype=np.float32)
    density[capsid] = 0.85
    density[core] = 0.45

    # Surface spike glycoproteins (e.g. 12-18 radial protrusions)
    n_spikes = int(rng.integers(12, 19))
    spike_len = rng.uniform(3.0, 5.5)
    spike_ang_width = 0.12

    for i in range(n_spikes):
        ang = i * (2 * np.pi / n_spikes) + rng.uniform(-0.05, 0.05)
        d_theta = np.abs(((theta - ang + np.pi) % (2 * np.pi)) - np.pi)
        spike = (d_theta <= spike_ang_width) & (r >= r_capsid) & (r <= r_capsid + spike_len)
        density[spike] = 0.75

    return density.astype(np.float32)


def make_protein_aggregate(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V03: Heterogeneous multi-lobed protein complex proxy."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, _, _ = make_coordinate_grid(size)
    density = np.zeros((size, size), dtype=np.float32)

    # 3-6 interlocking globular subunits
    n_lobes = int(rng.integers(3, 7))
    for _ in range(n_lobes):
        cy = rng.uniform(-10, 10)
        cx = rng.uniform(-10, 10)
        rp = rng.uniform(5.0, 9.0)
        lobe_d = (y - cy)**2 + (x - cx)**2
        lobe_mask = lobe_d <= rp**2
        lobe_dens = float(rng.uniform(0.5, 0.95))
        density[lobe_mask] = np.maximum(density[lobe_mask], lobe_dens)

    # Smooth the junction boundaries
    density = gaussian_filter(density, sigma=1.2)
    density = density / (np.max(density) + 1e-6)
    return density.astype(np.float32)


def make_core_shell_nanoparticle(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V04: Core-shell nanoparticle (dense inorganic core + lighter shell)."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, _ = make_coordinate_grid(size)
    r_core = rng.uniform(6.0, 10.0)
    shell_thickness = rng.uniform(5.0, 10.0)
    r_shell = r_core + shell_thickness

    density = np.zeros((size, size), dtype=np.float32)
    density[r <= r_shell] = float(rng.uniform(0.3, 0.5))  # Lower-density shell
    density[r <= r_core] = float(rng.uniform(0.9, 1.0))   # High-density core

    return density.astype(np.float32)


def make_ellipsoidal_nanoparticle(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V05: Anisotropic ellipsoidal nanoparticle with variable aspect ratio (1.5 - 4.5)."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, _, _ = make_coordinate_grid(size)
    aspect_ratio = float(rng.uniform(1.8, 4.2))
    b = float(rng.uniform(5.0, 8.0))
    a = b * aspect_ratio
    theta = float(rng.uniform(0, np.pi))

    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)

    ellipse = (xp / a)**2 + (yp / b)**2 <= 1.0
    density = np.zeros((size, size), dtype=np.float32)
    density[ellipse] = float(rng.uniform(0.8, 1.0))

    return density.astype(np.float32)


def make_nanorod(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V06: High-aspect-ratio curved or straight nanorod / nanowire."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, _, _ = make_coordinate_grid(size)
    length = rng.uniform(28.0, 44.0)
    radius = rng.uniform(2.5, 4.5)
    rot = rng.uniform(0, np.pi)
    curvature = rng.uniform(-0.015, 0.015)

    xp = x * np.cos(rot) + y * np.sin(rot)
    yp = -x * np.sin(rot) + y * np.cos(rot)
    yp_curved = yp - curvature * (xp**2)

    rod = (np.abs(xp) <= length / 2.0) & (np.abs(yp_curved) <= radius)
    # Hemispherical caps
    c1 = (xp - length/2.0)**2 + yp_curved**2 <= radius**2
    c2 = (xp + length/2.0)**2 + yp_curved**2 <= radius**2
    support = rod | c1 | c2

    density = np.zeros((size, size), dtype=np.float32)
    density[support] = float(rng.uniform(0.85, 1.0))
    return density.astype(np.float32)


def make_nanoparticle_cluster(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V07: Oligomer nanoparticle cluster with controlled interparticle gap."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, _, _ = make_coordinate_grid(size)
    n_particles = int(rng.integers(2, 6))
    gap = float(rng.uniform(0.0, 6.0))  # Touching (0) to separated
    density = np.zeros((size, size), dtype=np.float32)

    positions = []
    base_r = float(rng.uniform(4.5, 7.5))
    for i in range(n_particles):
        ang = i * (2 * np.pi / n_particles) + rng.uniform(-0.1, 0.1)
        dist = base_r + gap + rng.uniform(-1, 1)
        cy = dist * np.sin(ang)
        cx = dist * np.cos(ang)
        rp = base_r * rng.uniform(0.85, 1.15)
        p_mask = (y - cy)**2 + (x - cx)**2 <= rp**2
        density[p_mask] = float(rng.uniform(0.8, 1.0))

    return density.astype(np.float32)


def make_porous_material(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V08: Porous solid material with stochastically interconnected voids."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, _ = make_coordinate_grid(size)
    r_outer = float(rng.uniform(18.0, 24.0))
    solid = r <= r_outer

    # Stochastic interconnected pore network via thresholded Gaussian field
    field = gaussian_filter(rng.standard_normal((size, size)), sigma=rng.uniform(2.5, 4.0))
    pore_fraction = float(rng.uniform(0.35, 0.55))
    pore_thresh = np.percentile(field[solid], 100.0 * (1.0 - pore_fraction))
    pores = (field >= pore_thresh) & solid

    density = np.zeros((size, size), dtype=np.float32)
    density[solid & (~pores)] = float(rng.uniform(0.8, 1.0))

    return density.astype(np.float32)


def make_colloidal_assembly(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V09: Colloidal assembly (close-packed hexagonal array of identical spheres)."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, _ = make_coordinate_grid(size)
    r_outer = float(rng.uniform(18.0, 24.0))
    pitch = float(rng.uniform(7.0, 10.0))
    particle_r = pitch * 0.42

    density = np.zeros((size, size), dtype=np.float32)

    # Hexagonal lattice points: (ix * pitch + 0.5*pitch*(iy%2), iy * sqrt(3)/2 * pitch)
    dx_hex = pitch
    dy_hex = pitch * np.sqrt(3) / 2.0

    for iy in range(-4, 5):
        for ix in range(-4, 5):
            cx = ix * dx_hex + (0.5 * dx_hex if (iy % 2) != 0 else 0.0)
            cy = iy * dy_hex
            if cx**2 + cy**2 <= (r_outer - particle_r)**2:
                # Stochastic missing defect (vacancy defect)
                if rng.random() > 0.08:
                    p_mask = (y - cy)**2 + (x - cx)**2 <= particle_r**2
                    density[p_mask] = 0.95

    return density.astype(np.float32)


def make_microfabricated_sample(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V10: Microfabricated lithographic test pattern (zone plate or Siemens star proxy)."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, theta = make_coordinate_grid(size)
    mode = rng.choice(["zone_plate", "siemens_star", "crosshair_aperture"])

    density = np.zeros((size, size), dtype=np.float32)

    if mode == "zone_plate":
        # Fresnel zone plate: binary rings at radii ~ sqrt(n)
        dr_sq = rng.uniform(25.0, 45.0)
        rings = (np.sin(np.pi * (r**2) / dr_sq) >= 0) & (r <= 24.0)
        density[rings] = 1.0

    elif mode == "siemens_star":
        n_spokes = int(rng.integers(16, 25))
        spokes = (np.sin(n_spokes * theta) >= 0) & (r <= 23.0) & (r >= 2.0)
        density[spokes] = 1.0

    else:
        # Crosshairs and alignment boxes
        box = (np.abs(x) <= 20) & (np.abs(y) <= 20)
        cross_h = (np.abs(y) <= 2.0) & (np.abs(x) <= 22)
        cross_v = (np.abs(x) <= 2.0) & (np.abs(y) <= 22)
        density[box | cross_h | cross_v] = 1.0

    return density.astype(np.float32)


def make_crack_void_phantom(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V11: Material defect phantom with internal cracks, fissures, and inclusion voids."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, _ = make_coordinate_grid(size)
    solid = (np.abs(x) <= 22) & (np.abs(y) <= 22)

    # Crack path: random walk line
    crack_mask = np.zeros((size, size), dtype=bool)
    n_points = 20
    curr_x = float(rng.uniform(-15, -10))
    curr_y = float(rng.uniform(-5, 5))

    for _ in range(n_points):
        curr_x += rng.uniform(1.0, 2.5)
        curr_y += rng.uniform(-1.5, 1.5)
        iy, ix = int(round(curr_y + 31.5)), int(round(curr_x + 31.5))
        if 0 <= iy < size and 0 <= ix < size:
            crack_mask[iy, ix] = True
            # Crack width
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= iy + dy < size and 0 <= ix + dx < size:
                    crack_mask[iy + dy, ix + dx] = True

    density = np.zeros((size, size), dtype=np.float32)
    density[solid] = float(rng.uniform(0.85, 0.95))
    density[crack_mask] = 0.0

    return density.astype(np.float32)


def make_mixed_density_specimen(size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """V12: Heterogeneous multi-phase object with continuous density gradients and textures."""
    if rng is None:
        rng = np.random.default_rng()

    y, x, r, theta = make_coordinate_grid(size)
    r_outer = float(rng.uniform(18.0, 24.0))
    support = r <= r_outer

    # Multi-component continuous gradient + layered phases
    phase_layer = np.where(x > 0, 0.8, 0.4)
    gradient = 0.3 * (1.0 - r / r_outer)
    texture = gaussian_filter(rng.standard_normal((size, size)), sigma=3.0)
    texture = 0.2 * (texture / (np.max(np.abs(texture)) + 1e-6))

    density = (phase_layer + gradient + texture) * support
    density = np.clip(density, 0.1, 1.0) * support.astype(np.float32)

    return density.astype(np.float32)


# Ergonomic and profile aliases
make_cluster_assembly = make_nanoparticle_cluster
make_porous_matrix = make_porous_material
make_colloidal_crystal = make_colloidal_assembly
make_lithographic_pattern = make_microfabricated_sample
make_crack_void = make_crack_void_phantom
make_mixed_density = make_mixed_density_specimen

