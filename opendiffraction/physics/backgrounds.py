"""Experimental background models: Gas scattering, sample-holder film, solvent halo, parasitic flare, and cosmic hits."""

from __future__ import annotations

import numpy as np


def make_polar_grid(shape: tuple[int, int] = (512, 512)) -> tuple[np.ndarray, np.ndarray]:
    """Return radial distance and angle arrays centered at frame midpoint."""
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    return r, theta


def generate_gas_scattering_background(
    shape: tuple[int, int] = (512, 512),
    peak_intensity: float = 50.0,
    correlation_length_px: float = 40.0,
) -> np.ndarray:
    """Simulate diffuse isotropic gas/air scattering (Lorentzian profile)."""
    r, _ = make_polar_grid(shape)
    bg = peak_intensity / (1.0 + (r / correlation_length_px) ** 2)
    return bg.astype(np.float32)


def generate_support_film_background(
    shape: tuple[int, int] = (512, 512),
    peak_intensity: float = 30.0,
    ring_radius_px: float = 80.0,
    ring_width_px: float = 25.0,
) -> np.ndarray:
    """Simulate amorphous carbon support-film or silicon nitride membrane scattering halo."""
    r, _ = make_polar_grid(shape)
    halo = peak_intensity * np.exp(-((r - ring_radius_px) ** 2) / (2.0 * ring_width_px**2))
    return halo.astype(np.float32)


def generate_solvent_halo_background(
    shape: tuple[int, int] = (512, 512),
    peak_intensity: float = 40.0,
    water_ring_radius_px: float = 140.0,
    ring_width_px: float = 30.0,
) -> np.ndarray:
    """Simulate diffuse solvent/water scattering ring (e.g. liquid jet in XFEL or buffer solution)."""
    r, _ = make_polar_grid(shape)
    ring = peak_intensity * np.exp(-((r - water_ring_radius_px) ** 2) / (2.0 * ring_width_px**2))
    return ring.astype(np.float32)


def generate_parasitic_beamline_scattering(
    shape: tuple[int, int] = (512, 512),
    streak_intensity: float = 150.0,
    num_streaks: int = 4,
    streak_width_rad: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate parasitic scattering from beamline aperture slits and optical mirrors (cross-streaks and flare)."""
    if rng is None:
        rng = np.random.default_rng()

    r, theta = make_polar_grid(shape)
    ny, nx = shape
    cy, cx = ny // 2, nx // 2

    flare = np.zeros(shape, dtype=np.float32)
    angles = [i * (np.pi / (num_streaks / 2)) for i in range(num_streaks)]

    for ang in angles:
        ang_diff = np.abs(((theta - ang + np.pi) % (2 * np.pi)) - np.pi)
        streak = streak_intensity * np.exp(-(ang_diff**2) / (2.0 * streak_width_rad**2)) / (1.0 + r / 30.0)
        flare += streak.astype(np.float32)

    return flare


def generate_cosmic_ray_hits(
    shape: tuple[int, int] = (512, 512),
    num_hits: int = 3,
    hit_intensity_range: tuple[float, float] = (5000.0, 20000.0),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate isolated high-energy cosmic ray tracks or defective energy deposition events."""
    if rng is None:
        rng = np.random.default_rng()

    cosmic_frame = np.zeros(shape, dtype=np.float32)
    hit_coords = []

    ny, nx = shape
    for _ in range(num_hits):
        py = rng.integers(0, ny)
        px = rng.integers(0, nx)
        val = rng.uniform(*hit_intensity_range)
        cosmic_frame[py, px] = val
        hit_coords.append((int(py), int(px)))
        # Possible 2-pixel track
        if rng.random() > 0.5:
            dy = rng.choice([-1, 1])
            dx = rng.choice([-1, 1])
            if 0 <= py + dy < ny and 0 <= px + dx < nx:
                cosmic_frame[py + dy, px + dx] = val * 0.7
                hit_coords.append((int(py + dy), int(px + dx)))

    return cosmic_frame, hit_coords


def generate_cosmic_ray_hit(
    shape: tuple[int, int] = (512, 512),
    num_hits: int = 3,
    hit_intensity_range: tuple[float, float] = (5000.0, 20000.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate cosmic ray frame returning just the image array."""
    frame, _ = generate_cosmic_ray_hits(shape, num_hits, hit_intensity_range, rng)
    return frame

