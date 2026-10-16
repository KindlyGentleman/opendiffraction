"""Benchmark B3: Missing-Pixel and Beam-Stop Inpainting Baseline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from opendiffraction.tasks.metrics import aligned_mse, psnr


def inpaint_nearest_neighbor(
    intensity: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Inpaint missing detector pixels using Euclidean distance transform nearest valid pixel.

    Parameters
    ----------
    intensity : np.ndarray
        Measured diffraction pattern with zeros or artifacts in masked areas.
    valid_mask : np.ndarray
        Boolean or uint mask (True = valid pixel, False = defect/missing).

    Returns
    -------
    np.ndarray
        Inpainted pattern matching input shape.
    """
    valid = valid_mask.astype(bool)
    if np.all(valid):
        return intensity.copy()

    # Find nearest valid pixel coordinates
    _, indices = distance_transform_edt(~valid, return_indices=True)
    inpainted = intensity[tuple(indices)]
    return inpainted.astype(np.float32)


def evaluate_inpainting(
    measured_intensity: np.ndarray,
    valid_mask: np.ndarray,
    ground_truth_ideal: np.ndarray,
) -> dict[str, float]:
    """Evaluate inpainting error specifically within masked defect regions."""
    inpainted = inpaint_nearest_neighbor(measured_intensity, valid_mask)

    defect_region = ~valid_mask.astype(bool)
    if np.any(defect_region):
        defect_mse = float(np.mean((inpainted[defect_region] - ground_truth_ideal[defect_region])**2))
    else:
        defect_mse = 0.0

    overall_psnr = psnr(ground_truth_ideal, inpainted)

    return {
        "defect_region_mse": float(defect_mse),
        "overall_psnr_db": float(overall_psnr),
    }
