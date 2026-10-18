"""Manifest generation for the 12 Machine Learning Benchmark Tasks (T1 - T12)."""

from __future__ import annotations

from typing import Any
import pandas as pd


TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "T1": {
        "name": "Phase Retrieval",
        "type": "Inverse Problem",
        "input_modalities": ["intensity_measured", "mask"],
        "target_modalities": ["target_dpr", "object_amplitude"],
        "primary_metric": "aligned_mse",
        "secondary_metrics": ["aligned_ssim", "r_factor", "prtf"],
        "description": "Recover the missing Fourier phase and reconstruct the real-space object transmission density.",
    },
    "T2": {
        "name": "Diffraction Pattern Denoising",
        "type": "Image Restoration",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["intensity_expected"],
        "primary_metric": "psnr",
        "secondary_metrics": ["relative_l1", "ssim"],
        "description": "Remove Poisson shot noise and Gaussian electronic readout noise from detector readouts.",
    },
    "T3": {
        "name": "Detector Inpainting",
        "type": "Spatial Inpainting",
        "input_modalities": ["intensity_measured", "mask"],
        "target_modalities": ["intensity_expected"],
        "primary_metric": "masked_psnr",
        "secondary_metrics": ["mae_in_hole"],
        "description": "Predict missing diffraction intensities behind beam stops, holder arm shadows, and module gaps.",
    },
    "T4": {
        "name": "Hit Finding and Classification",
        "type": "Classification",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["hit_status"],
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": ["macro_f1"],
        "description": "Classify diffraction frames into single-particle hit, multi-particle collision, or blank non-hit.",
    },
    "T4A": {
        "name": "Binary Hit Finding",
        "type": "Binary Classification",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["is_hit"],
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": ["accuracy", "precision", "recall", "f1_score"],
        "description": "Binary screening of diffraction frames into valid scattering events (hits) vs blank detector readouts (non-hits).",
    },
    "T4B": {
        "name": "Four-Class Hit Characterization",
        "type": "Multiclass Classification",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["hit_status"],
        "primary_metric": "macro_f1",
        "secondary_metrics": ["four_class_accuracy", "per_class_recall", "confusion_matrix"],
        "description": "Fine-grained physical classification into single-particle hit, off-center hit, multi-particle collision, or blank non-hit.",
    },
    "T5": {
        "name": "Detector Defect Segmentation",
        "type": "Semantic Segmentation",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["defect_mask"],
        "primary_metric": "iou",
        "secondary_metrics": ["dice_score"],
        "description": "Segment anomalous dead pixels, hot clusters, and transient cosmic ray tracks.",
    },
    "T6": {
        "name": "Specimen Morphology Classification",
        "type": "Classification",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["morphology_class"],
        "primary_metric": "top1_accuracy",
        "secondary_metrics": ["macro_f1"],
        "description": "Directly classify particle architecture (cell, virus, core-shell, nanorod, cluster, etc.) from diffraction.",
    },
    "T7": {
        "name": "Physical Parameter Estimation",
        "type": "Regression",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["photon_budget", "coherence_length"],
        "primary_metric": "relative_mae",
        "secondary_metrics": ["r2_score"],
        "description": "Estimate physical beamline and specimen parameters directly from scattering patterns.",
    },
    "T8": {
        "name": "Out-of-Distribution Anomaly Detection",
        "type": "Anomaly Detection",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["is_ood"],
        "primary_metric": "auroc",
        "secondary_metrics": ["auprc"],
        "description": "Detect irregular specimens (cracks/voids, non-hits, severe artifacts) without labeled training data.",
    },
    "T9": {
        "name": "Domain Generalization",
        "type": "Domain Adaptation",
        "input_modalities": ["intensity_measured", "mask"],
        "target_modalities": ["target_dpr"],
        "primary_metric": "transfer_degradation_ratio",
        "secondary_metrics": ["aligned_mse"],
        "description": "Train on standard coherent synchrotron data and evaluate on partially coherent or XFEL regimes.",
    },
    "T10": {
        "name": "Self-Supervised Representation Learning",
        "type": "Self-Supervised Learning",
        "input_modalities": ["intensity_measured"],
        "target_modalities": ["representation_vector"],
        "primary_metric": "linear_probe_accuracy",
        "secondary_metrics": ["knn_accuracy"],
        "description": "Pretrain foundation representations on unlabeled diffraction patterns via masked autoencoding.",
    },
    "T11": {
        "name": "Uncertainty Quantification",
        "type": "Probabilistic Inverse Problem",
        "input_modalities": ["intensity_measured", "mask"],
        "target_modalities": ["target_dpr", "pixelwise_variance"],
        "primary_metric": "expected_calibration_error",
        "secondary_metrics": ["negative_log_likelihood"],
        "description": "Provide calibrated pixel-wise confidence intervals on inverse phase reconstructions.",
    },
    "T12": {
        "name": "Forward-Inverse Physics Consistency",
        "type": "Constraint Verification",
        "input_modalities": ["reconstructed_object", "intensity_measured", "mask"],
        "target_modalities": ["consistency_error"],
        "primary_metric": "r_factor",
        "secondary_metrics": ["friedel_symmetry_violation"],
        "description": "Verify physical self-consistency by repropagating reconstructions through forward model F1.",
    },
}


def build_task_manifest(
    df: pd.DataFrame,
    task_id: str,
    balance_classes: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Filter and construct sample manifest for a given task ID."""
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task: {task_id}. Choose T1 to T12.")

    task = TASK_DEFINITIONS[task_id]

    prof_col = "profile_id" if "profile_id" in df.columns else ("variation_family" if "variation_family" in df.columns else None)

    if task_id == "T1":
        # All samples with valid targets
        manifest = df[df["split"].isin(["train", "valid", "test", "test_ood"])].copy()
    elif task_id in ("T2", "T3"):
        # Denoising & inpainting
        manifest = df[df["split"].isin(["train", "valid", "test"])].copy()
    elif task_id in ("T4", "T4A", "T4B"):
        # Hit finding: prioritize Group 5 profiles (V37-V42) specifically designed for hit finding
        if prof_col is not None and df[prof_col].isin(["V37", "V38", "V39", "V40", "V41", "V42"]).any():
            manifest = df[df[prof_col].isin(["V37", "V38", "V39", "V40", "V41", "V42"])].copy()
        elif "hit_status" in df.columns and df["hit_status"].notna().any():
            manifest = df[df["hit_status"].notna()].copy()
        else:
            manifest = df.copy()

        if "hit_status" in manifest.columns:
            manifest["is_hit"] = manifest["hit_status"] != "non_hit"

        if balance_classes and "hit_status" in manifest.columns:
            counts = manifest["hit_status"].value_counts()
            min_count = int(counts.min())
            balanced_dfs = []
            for cls_name in sorted(counts.index):
                sub = manifest[manifest["hit_status"] == cls_name]
                balanced_dfs.append(sub.sample(n=min_count, random_state=seed))
            manifest = pd.concat(balanced_dfs, ignore_index=True)
    elif task_id == "T5":
        # Defect segmentation: profiles with module gaps, dead pixels, or cosmic rays
        if prof_col is not None:
            manifest = df[df[prof_col].isin(["V27", "V28", "V30", "V36"])].copy()
        else:
            manifest = df.copy()
    elif task_id == "T6":
        # Morphology classification: Group 1 profiles V01 - V12
        if prof_col is not None:
            manifest = df[df[prof_col].isin([f"V{i:02d}" for i in range(1, 13)])].copy()
        else:
            manifest = df.copy()
    elif task_id == "T7":
        # Parameter estimation
        manifest = df[df["split"].isin(["train", "valid", "test"])].copy()
    elif task_id == "T8":
        # Anomaly detection: balanced evaluation
        manifest = df.copy()
    elif task_id == "T9":
        # Domain generalization: train on clean/coherent, test on partial coherence / OOD
        manifest = df.copy()
    else:
        manifest = df.copy()

    manifest["task_id"] = task_id
    manifest["task_name"] = task["name"]
    return manifest
