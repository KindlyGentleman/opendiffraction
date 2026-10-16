"""Complex-valued object synthesis with controlled phase distributions.

Physical form: O(r) = A(r) * exp(i * phi(r))

Phase distributions:
1. Constant phase: global phase offset
2. Smooth phase gradient: linear wedge / tilt
3. Piecewise constant: stepped phase plate / binary phase domains
4. Random smooth phase field: correlated Gaussian random phase
5. Optical vortex: topological charge m * theta

Note on Phase Retrieval Ambiguities:
A single intensity measurement |FFT[O]|^2 possesses fundamental non-uniqueness:
- Global phase shift: O(r) -> O(r) * exp(i * phi_0)
- Spatial translation: O(r) -> O(r - r_0) (manifested as Fourier phase tilt)
- Twin image / Conjugate inversion: O(r) -> O*(-r) produces identical intensity |FFT[O]|^2!
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.ndimage import gaussian_filter

from .procedural import make_coordinate_grid


PhaseType = Literal[
    "constant",
    "smooth_gradient",
    "piecewise_constant",
    "random_smooth",
    "optical_vortex",
]


def generate_phase_field(
    phase_type: str = "random_smooth",
    size: int = 64,
    max_phase_depth: float = np.pi,
    rng: np.random.Generator | None = None,
    max_phase: float | None = None,
) -> np.ndarray:
    """Generate a 2D phase field phi(x, y) in radians."""
    if max_phase is not None:
        max_phase_depth = max_phase

    if phase_type == "smooth":
        phase_type = "random_smooth"
    elif phase_type == "vortex":
        phase_type = "optical_vortex"

    if rng is None:
        rng = np.random.default_rng()

    y, x, r, theta = make_coordinate_grid(size)

    if phase_type == "constant":
        offset = rng.uniform(-np.pi, np.pi)
        phi = np.full((size, size), offset, dtype=np.float32)

    elif phase_type == "smooth_gradient":
        angle = rng.uniform(0, 2 * np.pi)
        proj = (x * np.cos(angle) + y * np.sin(angle)) / (size / 2.0)
        phi = (proj * max_phase_depth).astype(np.float32)

    elif phase_type == "piecewise_constant":
        rot = rng.uniform(0, np.pi)
        xp = x * np.cos(rot) + y * np.sin(rot)
        phi = np.where(xp >= 0, max_phase_depth / 2, -max_phase_depth / 2).astype(np.float32)

    elif phase_type == "random_smooth":
        raw = rng.standard_normal((size, size))
        smooth = gaussian_filter(raw, sigma=rng.uniform(3.0, 6.0))
        s_min, s_max = smooth.min(), smooth.max()
        if s_max > s_min:
            norm = (smooth - s_min) / (s_max - s_min) - 0.5
        else:
            norm = np.zeros_like(smooth)
        phi = (norm * 2.0 * max_phase_depth).astype(np.float32)

    elif phase_type == "optical_vortex":
        charge = int(rng.choice([-2, -1, 1, 2]))
        phi = (charge * theta).astype(np.float32)

    else:
        phi = np.zeros((size, size), dtype=np.float32)

    return phi
