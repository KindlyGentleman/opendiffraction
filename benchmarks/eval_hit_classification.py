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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath("."))

from benchmarks.hit_detection import classify_hit
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset


def run_hit_classification_benchmark(
    dataset_dir: str = "output_v3",
    manifest_path: str = "manifests/task_4/val.csv",
    num_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate Hit Classification baseline on Task 4 validation samples."""
    print(f"Loading Hit Classification evaluation data from {manifest_path}...")
    df_val = pd.read_csv(manifest_path)
    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(
        dataset_dir, split=None, task="hit_classification", return_dict=True, manifest_path=manifest_path
    )

    y_true_binary = []
    y_pred_binary = []

    t0 = time.time()
    for idx, row in df_val.iterrows():
        sid = str(row.get("sample_id", ""))
        if sid and hasattr(ds, "get_by_sample_id"):
            try:
                sample = ds.get_by_sample_id(sid)
            except KeyError:
                continue
        else:
            if idx >= len(ds):
                break
            sample = ds[idx]
        inp = sample["input"][0].numpy()  # (512, 512)
        target_label = sample["target"]  # label string or binary indicator

        if hasattr(target_label, "item"):
            gt_is_hit = int(target_label.item()) != 2
        elif isinstance(target_label, str):
            gt_is_hit = target_label != "non_hit"
        else:
            gt_is_hit = bool(target_label)
        pred_res = classify_hit(inp, energy_threshold=1e4, peak_threshold=50.0)
        pred_is_hit = pred_res["is_hit"]

        y_true_binary.append(1 if gt_is_hit else 0)
        y_pred_binary.append(1 if pred_is_hit else 0)

    ds.close()
    elapsed = time.time() - t0

    acc = accuracy_score(y_true_binary, y_pred_binary)
    bal_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "evaluated_samples": len(y_true_binary),
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
