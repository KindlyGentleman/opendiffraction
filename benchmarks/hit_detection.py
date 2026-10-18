"""Benchmark B4: Hit Finding and Frame Classification Baseline."""

from __future__ import annotations

from typing import Any
import numpy as np


def classify_hit(
    intensity: np.ndarray,
    energy_threshold: float = 1e4,
    peak_threshold: float = 50.0,
    noise_sigma: float = 0.5,
) -> dict[str, Any]:
    """Classify frame into hit vs non-hit based on integrated photon energy and peak Bragg intensity.

    Gates readout noise floor (3 * noise_sigma) so zero-mean Gaussian readouts do not integrate to false signal.
    """
    threshold = 3.0 * noise_sigma
    signal_mask = intensity > threshold
    total_energy = float(np.sum(intensity[signal_mask])) if np.any(signal_mask) else 0.0
    peak_val = float(np.max(intensity)) if intensity.size > 0 else 0.0

    is_hit = (total_energy >= energy_threshold) and (peak_val >= peak_threshold)

    return {
        "prediction": "single_hit" if is_hit else "non_hit",
        "is_hit": bool(is_hit),
        "total_energy": total_energy,
        "peak_intensity": peak_val,
    }


def evaluate_hit_finding_batch(
    frames: list[np.ndarray],
    ground_truth_labels: list[str],
) -> dict[str, float]:
    """Compute precision, recall, and accuracy across batch of frames."""
    correct = 0
    tp, fp, fn, tn = 0, 0, 0, 0

    for f, gt in zip(frames, ground_truth_labels):
        res = classify_hit(f)
        pred = res["prediction"]
        if pred == gt:
            correct += 1

        is_hit_true = gt in ("single_hit", "multi_particle_hit", "off_center_hit")
        is_hit_pred = res["is_hit"]

        if is_hit_pred and is_hit_true:
            tp += 1
        elif is_hit_pred and not is_hit_true:
            fp += 1
        elif not is_hit_pred and is_hit_true:
            fn += 1
        else:
            tn += 1

    accuracy = correct / max(len(frames), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
