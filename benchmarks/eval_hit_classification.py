"""Hit Finding and Frame Classification Benchmark Evaluation (Task 4).

Evaluates baseline thresholding against ground truth hit labels on the validation manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError:
    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0

    def confusion_matrix(y_true, y_pred, labels=None):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm

    def precision_recall_fscore_support(y_true, y_pred, labels=None, average=None, zero_division=0):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        precisions = []
        recalls = []
        f1s = []
        supports = []
        for l in labels:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == l and yp == l)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != l and yp == l)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == l and yp != l)
            supp = sum(1 for yt in y_true if yt == l)
            prec = tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
            rec = tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else float(zero_division)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            supports.append(supp)

        if average == "binary":
            return precisions[0], recalls[0], f1s[0], supports[0]
        elif average == "macro":
            return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s)), None
        return np.array(precisions), np.array(recalls), np.array(f1s), np.array(supports)

    def balanced_accuracy_score(y_true, y_pred):
        labels = sorted(list(set(y_true)))
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
        return float(np.mean(rec)) if len(rec) else 0.0

sys.path.insert(0, os.path.abspath("."))

from benchmarks.hit_detection import classify_hit
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset


def run_hit_classification_benchmark(
    dataset_dir: str = "output_v3",
    manifest_path: str = "manifests/task_4/val.csv",
    num_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate Hit Detection (T4A) and Multi-Class Hit Characterization (T4B) independently."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading Hit Classification evaluation data from {manifest_path}...")
    df_val = pd.read_csv(manifest_path)
    if "sample_id" not in df_val.columns:
        raise ValueError(f"Manifest at {manifest_path} is missing required 'sample_id' column.")
    if df_val["sample_id"].duplicated().any():
        dups = df_val[df_val["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Manifest contains duplicate sample IDs: {dups[:5]}")

    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(
        dataset_dir, split=None, task="hit_classification", return_dict=True, manifest_path=manifest_path
    )

    y_true_binary: list[int] = []
    y_pred_binary: list[int] = []
    y_true_multi: list[str] = []
    y_pred_multi: list[str] = []
    evaluated_sids: list[str] = []

    t0 = time.time()
    for idx, row in df_val.iterrows():
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            raise ValueError(f"Manifest row {idx} is missing required 'sample_id' field.")

        sample = ds.get_by_sample_id(sid)
        evaluated_sids.append(sid)

        inp = sample["input"][0].numpy()  # (512, 512)
        gt_status = str(sample.get("hit_status") or row.get("hit_status", "single_hit"))

        # Ground truth
        gt_is_hit = gt_status != "non_hit"

        # Predictions
        pred_res = classify_hit(inp, energy_threshold=1e4, peak_threshold=50.0)
        pred_is_hit = bool(pred_res["is_hit"])
        pred_status = str(pred_res["prediction"])

        y_true_binary.append(1 if gt_is_hit else 0)
        y_pred_binary.append(1 if pred_is_hit else 0)
        y_true_multi.append(gt_status)
        y_pred_multi.append(pred_status)

    ds.close()
    elapsed = time.time() - t0

    # T4A: Binary Hit Detection Metrics
    acc_bin = accuracy_score(y_true_binary, y_pred_binary)
    bal_acc_bin = balanced_accuracy_score(y_true_binary, y_pred_binary)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0
    )

    # T4B: 4-Class Hit Characterization Metrics
    classes_4 = ["single_hit", "off_center_hit", "multi_particle_hit", "non_hit"]
    acc_4 = accuracy_score(y_true_multi, y_pred_multi)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_multi, y_pred_multi, labels=classes_4, average="macro", zero_division=0
    )
    _, rec_per_cls, _, _ = precision_recall_fscore_support(
        y_true_multi, y_pred_multi, labels=classes_4, average=None, zero_division=0
    )
    cm_4 = confusion_matrix(y_true_multi, y_pred_multi, labels=classes_4).tolist()

    metrics: dict[str, Any] = {
        "task_4a_binary": {
            "accuracy": float(acc_bin),
            "balanced_accuracy": float(bal_acc_bin),
            "precision": float(prec_bin),
            "recall": float(rec_bin),
            "f1_score": float(f1_bin),
        },
        "task_4b_multiclass": {
            "four_class_accuracy": float(acc_4),
            "macro_f1": float(f1_macro),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "per_class_recall": {cls_name: float(rec) for cls_name, rec in zip(classes_4, rec_per_cls)},
            "confusion_matrix": cm_4,
            "classes": classes_4,
        },
        "accuracy": float(acc_bin),
        "balanced_accuracy": float(bal_acc_bin),
        "precision": float(prec_bin),
        "recall": float(rec_bin),
        "f1_score": float(f1_bin),
        "four_class_accuracy": float(acc_4),
        "macro_f1_multiclass": float(f1_macro),
        "evaluated_samples": len(y_true_binary),
        "evaluated_sample_ids": evaluated_sids,
        "wall_time_s": float(elapsed),
        "throughput_sps": float(len(y_true_binary) / max(elapsed, 1e-4)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hit Classification baseline.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifest", default="manifests/task_4/val.csv", help="Task 4 manifest")
    parser.add_argument("--samples", type=int, default=None, help="Number of validation samples")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metrics = run_hit_classification_benchmark(
        dataset_dir=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.samples,
    )
    print("\nHit Classification Baseline Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
