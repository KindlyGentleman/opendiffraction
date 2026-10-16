"""Procedural object generation across five shape families (A-E) and density types.

No external data downloads required. Fully deterministic given a NumPy random generator.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.ndimage import gaussian_filter


ShapeFamily = Literal[
    "basic_geometric",
    "particle_blob",
    "multiple_scatterer",
    "porous_heterogeneous",
    "structured_pattern",
]

DensityType = Literal[
    "uniform",
    "binary",
    "graded",
    "textured",
    "piecewise_constant",
]


def make_coordinate_grid(size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered Cartesian and polar coordinate grids for size x size array.

    Grid centers are at (size/2 - 0.5, size/2 - 0.5).
    """
    c = (size - 1.0) / 2.0
    y, x = np.mgrid[:size, :size] - c
    r = np.sqrt(y**2 + x**2)
    theta = np.arctan2(y, x)
    return y, x, r, theta


def generate_gaussian_texture(
    size: int = 64,
    sigma: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate smooth correlated random density texture normalized to [0.2, 1.0]."""
    if rng is None:
        rng = np.random.default_rng()

    white_noise = rng.standard_normal((size, size))
    filtered = gaussian_filter(white_noise, sigma=sigma)
    f_min, f_max = filtered.min(), filtered.max()
    if f_max > f_min:
        norm = (filtered - f_min) / (f_max - f_min)
    else:
        norm = np.ones((size, size))

    return 0.2 + 0.8 * norm


# -----------------------------------------------------------------------------
# Family A: Basic Geometric Objects
# -----------------------------------------------------------------------------
def make_basic_geometric(
    subtype: str | None = None,
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate basic geometric shapes: circle, ellipse, rectangle, triangle, polygon, annulus, slit."""
    if rng is None:
        rng = np.random.default_rng()

    subtypes = ["circle", "ellipse", "rectangle", "triangle", "polygon", "annulus", "slit"]
    st = subtype if subtype in subtypes else rng.choice(subtypes)

    y, x, r, theta = make_coordinate_grid(size)
    support = np.zeros((size, size), dtype=bool)

    if st == "circle":
        radius = rng.uniform(10.0, 22.0)
        support = r <= radius

    elif st == "ellipse":
        r_a = rng.uniform(12.0, 24.0)
        r_b = rng.uniform(6.0, 16.0)
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        yp = -x * np.sin(rot) + y * np.cos(rot)
        support = (xp / r_a) ** 2 + (yp / r_b) ** 2 <= 1.0

    elif st == "rectangle":
        hw = rng.uniform(8.0, 20.0)
        hh = rng.uniform(6.0, 18.0)
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        yp = -x * np.sin(rot) + y * np.cos(rot)
        support = (np.abs(xp) <= hw) & (np.abs(yp) <= hh)

    elif st == "triangle":
        side = rng.uniform(18.0, 36.0)
        rot = rng.uniform(0, 2 * np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        yp = -x * np.sin(rot) + y * np.cos(rot)
        h = side * np.sqrt(3) / 2
        support = (yp >= -h / 3) & (yp <= 2 * h / 3 - np.sqrt(3) * np.abs(xp))

    elif st == "polygon":
        n_sides = int(rng.integers(5, 9))
        rad = rng.uniform(12.0, 22.0)
        rot = rng.uniform(0, 2 * np.pi)
        # Regular polygon support using polar equation
        d_theta = np.pi / n_sides
        theta_mod = ((theta - rot) % (2 * d_theta)) - d_theta
        r_poly = rad * np.cos(d_theta) / np.cos(theta_mod)
        support = r <= r_poly

    elif st == "annulus":
        r_outer = rng.uniform(16.0, 24.0)
        r_inner = rng.uniform(6.0, r_outer - 5.0)
        support = (r <= r_outer) & (r >= r_inner)

    elif st == "slit":
        length = rng.uniform(20.0, 38.0)
        width = rng.uniform(3.0, 8.0)
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        yp = -x * np.sin(rot) + y * np.cos(rot)
        support = (np.abs(xp) <= length / 2) & (np.abs(yp) <= width / 2)

    return support.astype(np.float32)


# -----------------------------------------------------------------------------
# Family B: Particle-like Objects
# -----------------------------------------------------------------------------
def make_particle_blob(
    subtype: str | None = None,
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate irregular particles, smooth blobs, elongated, asymmetric, multi-lobed shapes."""
    if rng is None:
        rng = np.random.default_rng()

    subtypes = ["blob", "irregular", "elongated", "asymmetric", "multi_lobed"]
    st = subtype if subtype in subtypes else rng.choice(subtypes)

    y, x, r, theta = make_coordinate_grid(size)
    support = np.zeros((size, size), dtype=bool)

    if st == "blob":
        # Smooth random Fourier boundary: r(theta) = r0 + sum(a_k cos(k theta + phi_k))
        r0 = rng.uniform(12.0, 18.0)
        r_boundary = np.full_like(theta, r0)
        for k in [2, 3, 4, 5]:
            amp = rng.uniform(1.0, 4.0) / (k**0.7)
            phase = rng.uniform(0, 2 * np.pi)
            r_boundary += amp * np.cos(k * theta + phase)
        support = r <= np.maximum(r_boundary, 4.0)

    elif st == "irregular":
        # Thresholded filtered Gaussian noise envelope
        field = gaussian_filter(rng.standard_normal((size, size)), sigma=rng.uniform(3.0, 5.0))
        envelope = np.exp(-((r / 20.0) ** 4))
        support = (field * envelope) > (0.1 * np.max(np.abs(field)))

    elif st == "elongated":
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        yp = -x * np.sin(rot) + y * np.cos(rot)
        a = rng.uniform(18.0, 26.0)
        b = rng.uniform(5.0, 9.0)
        # Add slight curvature
        curvature = rng.uniform(-0.02, 0.02)
        yp_curved = yp - curvature * (xp**2)
        support = (xp / a) ** 2 + (yp_curved / b) ** 2 <= 1.0

    elif st == "asymmetric":
        r0 = rng.uniform(14.0, 20.0)
        r_boundary = r0 + 5.0 * np.cos(theta) + 3.0 * np.sin(2 * theta)
        support = r <= np.maximum(r_boundary, 3.0)

    elif st == "multi_lobed":
        n_lobes = int(rng.integers(3, 7))
        r0 = rng.uniform(12.0, 17.0)
        amp = rng.uniform(4.0, 7.0)
        rot = rng.uniform(0, np.pi)
        r_boundary = r0 + amp * np.cos(n_lobes * (theta - rot))
        support = r <= r_boundary

    return support.astype(np.float32)


# -----------------------------------------------------------------------------
# Family C: Multiple-Scatterer Objects
# -----------------------------------------------------------------------------
def make_multiple_scatterers(
    subtype: str | None = None,
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate particle pairs, clusters, sparse, and dense arrangements."""
    if rng is None:
        rng = np.random.default_rng()

    subtypes = ["pair_varying_sep", "cluster", "sparse", "dense"]
    st = subtype if subtype in subtypes else rng.choice(subtypes)

    y, x, _, _ = make_coordinate_grid(size)
    support = np.zeros((size, size), dtype=bool)

    if st == "pair_varying_sep":
        sep = rng.uniform(12.0, 32.0)
        angle = rng.uniform(0, np.pi)
        dx = (sep / 2) * np.cos(angle)
        dy = (sep / 2) * np.sin(angle)
        r1 = rng.uniform(4.0, 8.0)
        r2 = rng.uniform(4.0, 8.0)
        s1 = (x - dx) ** 2 + (y - dy) ** 2 <= r1**2
        s2 = (x + dx) ** 2 + (y + dy) ** 2 <= r2**2
        support = s1 | s2

    elif st == "cluster":
        n_particles = int(rng.integers(3, 7))
        for _ in range(n_particles):
            cx = rng.uniform(-14.0, 14.0)
            cy = rng.uniform(-14.0, 14.0)
            rp = rng.uniform(4.0, 7.5)
            support |= (x - cx) ** 2 + (y - cy) ** 2 <= rp**2

    elif st == "sparse":
        n_particles = int(rng.integers(2, 5))
        for _ in range(n_particles):
            cx = rng.uniform(-20.0, 20.0)
            cy = rng.uniform(-20.0, 20.0)
            rp = rng.uniform(3.0, 6.0)
            support |= (x - cx) ** 2 + (y - cy) ** 2 <= rp**2

    elif st == "dense":
        n_particles = int(rng.integers(8, 15))
        for _ in range(n_particles):
            cx = rng.uniform(-18.0, 18.0)
            cy = rng.uniform(-18.0, 18.0)
            rp = rng.uniform(2.5, 5.0)
            support |= (x - cx) ** 2 + (y - cy) ** 2 <= rp**2

    return support.astype(np.float32)


# -----------------------------------------------------------------------------
# Family D: Porous and Heterogeneous Structures
# -----------------------------------------------------------------------------
def make_porous_heterogeneous(
    subtype: str | None = None,
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate binary porous particles, internal voids, and core-shell structures."""
    if rng is None:
        rng = np.random.default_rng()

    subtypes = ["porous_matrix", "internal_voids", "core_shell"]
    st = subtype if subtype in subtypes else rng.choice(subtypes)

    y, x, r, _ = make_coordinate_grid(size)
    outer_r = rng.uniform(16.0, 23.0)
    outer_support = r <= outer_r

    if st == "porous_matrix":
        # Base matrix with random pores subtracted
        n_pores = int(rng.integers(5, 12))
        pore_mask = np.zeros((size, size), dtype=bool)
        for _ in range(n_pores):
            px = rng.uniform(-outer_r * 0.75, outer_r * 0.75)
            py = rng.uniform(-outer_r * 0.75, outer_r * 0.75)
            pr = rng.uniform(2.0, 5.0)
            pore_mask |= (x - px) ** 2 + (y - py) ** 2 <= pr**2
        support = outer_support & (~pore_mask)

    elif st == "internal_voids":
        # 1-3 distinct internal voids
        n_voids = int(rng.integers(1, 4))
        void_mask = np.zeros((size, size), dtype=bool)
        for _ in range(n_voids):
            vx = rng.uniform(-outer_r * 0.5, outer_r * 0.5)
            vy = rng.uniform(-outer_r * 0.5, outer_r * 0.5)
            vr = rng.uniform(4.0, 8.0)
            void_mask |= (x - vx) ** 2 + (y - vy) ** 2 <= vr**2
        support = outer_support & (~void_mask)

    elif st == "core_shell":
        support = outer_support

    return support.astype(np.float32)


# -----------------------------------------------------------------------------
# Family E: Structured Patterns
# -----------------------------------------------------------------------------
def make_structured_pattern(
    subtype: str | None = None,
    size: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate periodic arrays, sparse lattices, radial structures, repeated motifs."""
    if rng is None:
        rng = np.random.default_rng()

    subtypes = ["periodic_grid", "sparse_lattice", "radial_spokes", "line_grating"]
    st = subtype if subtype in subtypes else rng.choice(subtypes)

    y, x, r, theta = make_coordinate_grid(size)
    support = np.zeros((size, size), dtype=bool)

    if st == "periodic_grid":
        pitch = rng.uniform(6.0, 10.0)
        dot_r = rng.uniform(1.5, 2.5)
        grid_mask = ((np.abs(x) % pitch <= dot_r) | (pitch - (np.abs(x) % pitch) <= dot_r)) & (
            (np.abs(y) % pitch <= dot_r) | (pitch - (np.abs(y) % pitch) <= dot_r)
        )
        support = grid_mask & (r <= 22.0)

    elif st == "sparse_lattice":
        pitch = rng.uniform(9.0, 14.0)
        dot_r = rng.uniform(2.0, 3.5)
        grid_mask = ((np.abs(x) % pitch <= dot_r) | (pitch - (np.abs(x) % pitch) <= dot_r)) & (
            (np.abs(y) % pitch <= dot_r) | (pitch - (np.abs(y) % pitch) <= dot_r)
        )
        support = grid_mask & (r <= 24.0)

    elif st == "radial_spokes":
        n_spokes = int(rng.integers(6, 13))
        spoke_w = rng.uniform(0.1, 0.25)
        spoke_mask = (np.abs(np.sin(n_spokes * theta)) <= spoke_w) & (r >= 4.0) & (r <= 22.0)
        support = spoke_mask

    elif st == "line_grating":
        pitch = rng.uniform(5.0, 8.0)
        line_w = pitch * rng.uniform(0.3, 0.5)
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        lines = (np.abs(xp) % pitch) <= line_w
        support = lines & (r <= 22.0)

    return support.astype(np.float32)


# -----------------------------------------------------------------------------
# Master Object Synthesizer (Shape + Density)
# -----------------------------------------------------------------------------
def generate_procedural_object(
    family: ShapeFamily = "basic_geometric",
    density_type: DensityType = "uniform",
    size: int = 64,
    rng: np.random.Generator | None = None,
    shape_subtype: str | None = None,
    shape_family: ShapeFamily | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a complete object with defined shape family and density texture."""
    if shape_family is not None:
        family = shape_family

    if rng is None:
        rng = np.random.default_rng()

    # 1. Generate binary support
    if family == "basic_geometric":
        support_f = make_basic_geometric(shape_subtype, size=size, rng=rng)
    elif family == "particle_blob":
        support_f = make_particle_blob(shape_subtype, size=size, rng=rng)
    elif family == "multiple_scatterer":
        support_f = make_multiple_scatterers(shape_subtype, size=size, rng=rng)
    elif family == "porous_heterogeneous":
        support_f = make_porous_heterogeneous(shape_subtype, size=size, rng=rng)
    elif family == "structured_pattern":
        support_f = make_structured_pattern(shape_subtype, size=size, rng=rng)
    else:
        support_f = make_basic_geometric("circle", size=size, rng=rng)

    support = support_f > 0.5
    # If degenerate (empty), default to circle
    if not np.any(support):
        _, _, r, _ = make_coordinate_grid(size)
        support = r <= 15.0

    # 2. Modulate density
    y, x, r, _ = make_coordinate_grid(size)

    if density_type == "uniform":
        density = np.ones((size, size), dtype=np.float32)

    elif density_type == "binary":
        density = np.ones((size, size), dtype=np.float32)

    elif density_type == "graded":
        # Radial gradient from center outwards or linear ramp
        grad_mode = rng.choice(["radial", "linear"])
        if grad_mode == "radial":
            r_max = np.max(r[support]) if np.any(support) else 20.0
            density = np.clip(1.0 - 0.7 * (r / max(r_max, 1e-3)), 0.2, 1.0)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            proj = (x * np.cos(angle) + y * np.sin(angle)) / 25.0
            density = np.clip(0.6 + 0.4 * proj, 0.2, 1.0)

    elif density_type == "textured":
        density = generate_gaussian_texture(size=size, sigma=rng.uniform(2.0, 4.0), rng=rng)

    elif density_type == "piecewise_constant":
        # Concentric steps or two distinct halves
        step_mode = rng.choice(["concentric", "partition"])
        if step_mode == "concentric":
            r_med = np.median(r[support]) if np.any(support) else 10.0
            density = np.where(r <= r_med, 1.0, 0.45)
        else:
            rot = rng.uniform(0, np.pi)
            xp = x * np.cos(rot) + y * np.sin(rot)
            density = np.where(xp >= 0, 1.0, 0.5)

    else:
        density = np.ones((size, size), dtype=np.float32)

    obj = (support.astype(np.float32) * density.astype(np.float32)).clip(0.0, 1.0)
    return obj, support
