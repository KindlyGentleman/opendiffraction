"""Benchmark B4: Hit Finding and Frame Classification Baseline."""

from __future__ import annotations

from typing import Any
import numpy as np


def classify_hit_binary(
    intensity: np.ndarray,
    energy_threshold: float = 1e4,
    peak_threshold: float = 50.0,
    noise_sigma: float = 0.5,
) -> dict[str, Any]:
    """Classify frame into binary hit vs non-hit (Task 4A).

    Gates readout noise floor (3 * noise_sigma) so zero-mean Gaussian readouts
    do not integrate to false signal.
    """
    threshold = 3.0 * noise_sigma
    signal_mask = intensity > threshold
    total_energy = float(np.sum(intensity[signal_mask])) if np.any(signal_mask) else 0.0
    peak_val = float(np.max(intensity)) if intensity.size > 0 else 0.0

    is_hit = (total_energy >= energy_threshold) and (peak_val >= peak_threshold)

    return {
        "prediction": "hit" if is_hit else "non_hit",
        "is_hit": bool(is_hit),
        "total_energy": total_energy,
        "peak_intensity": peak_val,
    }


def classify_hit_multiclass(
    intensity: np.ndarray,
    energy_threshold: float = 1e4,
    peak_threshold: float = 50.0,
    offset_threshold: float = 4.0,
    satellite_ratio_threshold: float = 0.38,
    noise_sigma: float = 0.5,
) -> dict[str, Any]:
    """Classify frame into four physical hit categories (Task 4B).

    Categories:
    - non_hit: Background and electronic readout noise only.
    - off_center_hit: Beam or particle translation resulting in asymmetric centroid offset.
    - multi_particle_hit: Multiple scatterers causing interference fringe modulation and
      satellite peaks in real-space Patterson autocorrelation.
    - single_hit: Centered single-particle coherent diffraction.
    """
    bin_res = classify_hit_binary(
        intensity,
        energy_threshold=energy_threshold,
        peak_threshold=peak_threshold,
        noise_sigma=noise_sigma,
    )
    if not bin_res["is_hit"]:
        return {
            "prediction": "non_hit",
            "is_hit": False,
            "total_energy": bin_res["total_energy"],
            "peak_intensity": bin_res["peak_intensity"],
            "com_dist": 0.0,
            "satellite_ratio": 0.0,
        }

    # Centroid displacement calculation
    threshold = 3.0 * noise_sigma
    sig = intensity > threshold
    ny, nx = intensity.shape
    cy, cx = ny // 2, nx // 2
    y_grid, x_grid = np.mgrid[:ny, :nx]
    com_y = float(np.sum(y_grid[sig] * intensity[sig])) / max(bin_res["total_energy"], 1e-12)
    com_x = float(np.sum(x_grid[sig] * intensity[sig])) / max(bin_res["total_energy"], 1e-12)
    com_dist = float(np.sqrt((com_y - cy) ** 2 + (com_x - cx) ** 2))

    if com_dist >= offset_threshold:
        return {
            "prediction": "off_center_hit",
            "is_hit": True,
            "total_energy": bin_res["total_energy"],
            "peak_intensity": bin_res["peak_intensity"],
            "com_dist": com_dist,
            "satellite_ratio": 0.0,
        }

    # Patterson autocorrelation: ifft2 of intensity yields real-space spatial autocorrelation
    patterson = np.fft.fftshift(np.abs(np.fft.ifft2(intensity)))
    crop_radius = min(32, ny // 4)
    crop_pat = patterson[cy - crop_radius : cy + crop_radius, cx - crop_radius : cx + crop_radius]
    crop_no_dc = crop_pat.copy()
    dc_mask_radius = 4
    crop_no_dc[
        crop_radius - dc_mask_radius : crop_radius + dc_mask_radius,
        crop_radius - dc_mask_radius : crop_radius + dc_mask_radius,
    ] = 0.0
    sat_peak = float(np.max(crop_no_dc))
    dc_peak = float(patterson[cy, cx])
    sat_ratio = sat_peak / max(dc_peak, 1e-12)

    pred = "multi_particle_hit" if sat_ratio >= satellite_ratio_threshold else "single_hit"

    return {
        "prediction": pred,
        "is_hit": True,
        "total_energy": bin_res["total_energy"],
        "peak_intensity": bin_res["peak_intensity"],
        "com_dist": com_dist,
        "satellite_ratio": sat_ratio,
    }


def classify_hit(
    intensity: np.ndarray,
    energy_threshold: float = 1e4,
    peak_threshold: float = 50.0,
    noise_sigma: float = 0.5,
) -> dict[str, Any]:
    """Backward-compatible hit classification entry point."""
    bin_res = classify_hit_binary(
        intensity,
        energy_threshold=energy_threshold,
        peak_threshold=peak_threshold,
        noise_sigma=noise_sigma,
    )
    multi_res = classify_hit_multiclass(
        intensity,
        energy_threshold=energy_threshold,
        peak_threshold=peak_threshold,
        noise_sigma=noise_sigma,
    )
    return {
        "prediction": multi_res["prediction"],
        "is_hit": bin_res["is_hit"],
        "total_energy": bin_res["total_energy"],
        "peak_intensity": bin_res["peak_intensity"],
        "com_dist": multi_res["com_dist"],
        "satellite_ratio": multi_res["satellite_ratio"],
    }
