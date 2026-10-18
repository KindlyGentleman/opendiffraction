"""Procedural detector masks: central beam stops, module gaps, dead pixels, stripes, and irregular patches.

All masks follow the convention:
- True: valid pixel
- False: missing / dead / beamstop pixel
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def create_centered_beam_stop(
    shape: tuple[int, int] = (512, 512),
    half_width: int = 16,
    half_height: int = 16,
    offset_y: int = 0,
    offset_x: int = 0,
    circular: bool = False,
) -> np.ndarray:
    """Create a central beam stop mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Detector dimensions (H, W).
    half_width : int
        Half-width of the beam stop.
    half_height : int
        Half-height of the beam stop.
    offset_y : int
        Displacement of beam stop center along y axis.
    offset_x : int
        Displacement of beam stop center along x axis.
    circular : bool
        If True, creates a circular beam stop with radius max(half_width, half_height).

    Returns
    -------
    mask : np.ndarray of bool
        True for valid pixels, False for beam-stop pixels.
    """
    ny, nx = shape
    cy = ny // 2 + offset_y
    cx = nx // 2 + offset_x
    mask = np.ones(shape, dtype=bool)

    if circular:
        r = max(half_width, half_height)
        y, x = np.ogrid[:ny, :nx]
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        mask[dist_sq <= r**2] = False
    else:
        y0 = max(0, cy - half_height)
        y1 = min(ny, cy + half_height)
        x0 = max(0, cx - half_width)
        x1 = min(nx, cx + half_width)
        mask[y0:y1, x0:x1] = False

    return mask


def create_beam_stop_holder_shadow(
    shape: tuple[int, int] = (512, 512),
    arm_angle_rad: float = 0.0,
    arm_width: int = 5,
    offset_y: int = 0,
    offset_x: int = 0,
) -> np.ndarray:
    """Create linear shadow mask corresponding to the mechanical beam-stop holder arm."""
    ny, nx = shape
    cy = ny // 2 + offset_y
    cx = nx // 2 + offset_x
    y, x = np.mgrid[:ny, :nx]

    # Coordinate system centered at beamstop
    yp = y - cy
    xp = x - cx

    # Distance perpendicular to arm axis
    arm_dist = np.abs(-np.sin(arm_angle_rad) * xp + np.cos(arm_angle_rad) * yp)
    # Projection along arm: arm extends outwards from beamstop
    arm_proj = np.cos(arm_angle_rad) * xp + np.sin(arm_angle_rad) * yp

    mask = np.ones(shape, dtype=bool)
    # Shadow extends from center outwards along the arm direction
    shadow = (arm_dist <= arm_width / 2.0) & (arm_proj >= 0)
    mask[shadow] = False
    return mask


def create_module_gaps(
    shape: tuple[int, int] = (512, 512),
    gap_width: int = 4,
    horizontal_gaps: list[int] | None = None,
    vertical_gaps: list[int] | None = None,
) -> np.ndarray:
    """Create detector tile/module gap mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Detector dimensions.
    gap_width : int
        Width of inactive tile borders in pixels.
    horizontal_gaps : list of int or None
        y-coordinates of horizontal module boundaries (default: [256]).
    vertical_gaps : list of int or None
        x-coordinates of vertical module boundaries (default: [256]).

    Returns
    -------
    mask : np.ndarray of bool
        True for valid pixels, False for gaps.
    """
    ny, nx = shape
    mask = np.ones(shape, dtype=bool)
    h_gaps = horizontal_gaps if horizontal_gaps is not None else [ny // 2]
    v_gaps = vertical_gaps if vertical_gaps is not None else [nx // 2]

    half_w = max(1, gap_width // 2)
    for gy in h_gaps:
        y0 = max(0, gy - half_w)
        y1 = min(ny, gy + half_w + (gap_width % 2))
        mask[y0:y1, :] = False

    for gx in v_gaps:
        x0 = max(0, gx - half_w)
        x1 = min(nx, gx + half_w + (gap_width % 2))
        mask[:, x0:x1] = False

    return mask


def create_dead_pixels_and_clusters(
    shape: tuple[int, int] = (512, 512),
    random_pixel_fraction: float = 0.005,
    num_clusters: int = 4,
    cluster_radius_range: tuple[int, int] = (2, 6),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create isolated dead pixels and localized dead clusters."""
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = shape
    mask = np.ones(shape, dtype=bool)

    if random_pixel_fraction > 0:
        dead_rand = rng.random(shape) < random_pixel_fraction
        mask[dead_rand] = False

    y, x = np.ogrid[:ny, :nx]
    for _ in range(num_clusters):
        cy = rng.integers(0, ny)
        cx = rng.integers(0, nx)
        r = rng.integers(cluster_radius_range[0], cluster_radius_range[1] + 1)
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        mask[dist_sq <= r**2] = False

    return mask


def create_stripes(
    shape: tuple[int, int] = (512, 512),
    num_horizontal_stripes: int = 2,
    num_vertical_stripes: int = 2,
    stripe_width: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create readout line dropouts (dead columns / rows)."""
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = shape
    mask = np.ones(shape, dtype=bool)

    for _ in range(num_horizontal_stripes):
        y0 = rng.integers(0, ny - stripe_width)
        mask[y0 : y0 + stripe_width, :] = False

    for _ in range(num_vertical_stripes):
        x0 = rng.integers(0, nx - stripe_width)
        mask[:, x0 : x0 + stripe_width] = False

    return mask


def create_irregular_patches(
    shape: tuple[int, int] = (512, 512),
    coverage_fraction: float = 0.15,
    smoothness: float = 18.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create smooth procedural irregular missing patches (like NVIDIA disocclusion masks)."""
    if rng is None:
        rng = np.random.default_rng()

    field = rng.standard_normal(shape)
    smooth_field = gaussian_filter(field, sigma=smoothness)
    threshold = np.percentile(smooth_field, 100.0 * coverage_fraction)
    mask = smooth_field >= threshold
    return mask.astype(bool)


def composite_detector_mask(
    shape: tuple[int, int] = (512, 512),
    include_beam_stop: bool = True,
    beam_stop_hw: int = 16,
    beam_stop_hh: int = 16,
    beam_stop_offset: tuple[int, int] = (0, 0),
    circular_beam_stop: bool = False,
    include_holder_arm: bool = False,
    arm_angle_rad: float = 0.0,
    arm_width: int = 5,
    include_module_gaps: bool = False,
    gap_width: int = 4,
    random_pixel_fraction: float = 0.0,
    num_clusters: int = 0,
    cluster_radius_range: tuple[int, int] = (2, 6),
    num_stripes_h: int = 0,
    num_stripes_v: int = 0,
    irregular_coverage: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a combined detector validity mask."""
    if rng is None:
        rng = np.random.default_rng()

    mask = np.ones(shape, dtype=bool)

    if include_beam_stop:
        bs_mask = create_centered_beam_stop(
            shape=shape,
            half_width=beam_stop_hw,
            half_height=beam_stop_hh,
            offset_y=beam_stop_offset[0],
            offset_x=beam_stop_offset[1],
            circular=circular_beam_stop,
        )
        mask &= bs_mask

        if include_holder_arm:
            arm_mask = create_beam_stop_holder_shadow(
                shape=shape,
                arm_angle_rad=arm_angle_rad,
                arm_width=arm_width,
                offset_y=beam_stop_offset[0],
                offset_x=beam_stop_offset[1],
            )
            mask &= arm_mask

    if include_module_gaps:
        gaps_mask = create_module_gaps(shape=shape, gap_width=gap_width)
        mask &= gaps_mask

    if random_pixel_fraction > 0 or num_clusters > 0:
        dead_mask = create_dead_pixels_and_clusters(
            shape=shape,
            random_pixel_fraction=random_pixel_fraction,
            num_clusters=num_clusters,
            cluster_radius_range=cluster_radius_range,
            rng=rng,
        )
        mask &= dead_mask

    if num_stripes_h > 0 or num_stripes_v > 0:
        stripes_mask = create_stripes(
            shape=shape,
            num_horizontal_stripes=num_stripes_h,
            num_vertical_stripes=num_stripes_v,
            stripe_width=1,
            rng=rng,
        )
        mask &= stripes_mask

    if irregular_coverage > 0:
        irreg_mask = create_irregular_patches(
            shape=shape,
            coverage_fraction=irregular_coverage,
            smoothness=18.0,
            rng=rng,
        )
        mask &= irreg_mask

    # Ensure mask never completely wipes out every single pixel on the detector
    if not np.any(mask):
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

    return mask
