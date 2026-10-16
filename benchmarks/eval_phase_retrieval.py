"""Phase Retrieval Benchmark Evaluation (Task 1).

Evaluates classical HIO + Error Reduction iterative phase retrieval baseline
against ground truth specimen targets on the validation manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath("."))

from benchmarks.phase_retrieval import evaluate_phase_retrieval
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset


def run_phase_retrieval_benchmark(
    dataset_dir: str = "output_v3",
    manifest_path: str = "manifests/task_1/val.csv",
    num_samples: int = 20,
    num_hio_iters: int = 40,
    num_er_iters: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate HIO/ER baseline on Task 1 validation samples."""
    print(f"Loading Phase Retrieval evaluation data from {manifest_path}...")
    df_val = pd.read_csv(manifest_path)
    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(dataset_dir, split="valid", task="phase_retrieval", return_dict=True)

    ssim_list = []
    mse_list = []
    r_factor_list = []
    pearson_r_list = []

    t0 = time.time()
    for idx in range(min(len(df_val), len(ds))):
        sample = ds[idx]
        inp = sample["input"][0].numpy()  # (512, 512)
        target = sample["target"][0].numpy()  # (64, 64)
        mask = sample["mask"][0].numpy()  # (512, 512)

        res = evaluate_phase_retrieval(
            measured_intensity=inp,
            ground_truth_object=target,
            num_hio_iters=num_hio_iters,
            num_er_iters=num_er_iters,
            seed=seed + idx,
        )

        ssim_list.append(res["aligned_ssim"])
        mse_list.append(res["aligned_mse"])
        r_factor_list.append(res["final_r_factor"])

        # Compute Pearson correlation on flattened arrays
        r_val, _ = pearsonr(target.ravel(), np.clip(target.ravel(), 0, 1))
        pearson_r_list.append(float(r_val) if np.isfinite(r_val) else 0.0)

    ds.close()
    elapsed = time.time() - t0

    metrics = {
        "mean_aligned_ssim": float(np.mean(ssim_list)),
        "std_aligned_ssim": float(np.std(ssim_list)),
        "mean_aligned_mse": float(np.mean(mse_list)),
        "mean_final_r_factor": float(np.mean(r_factor_list)),
        "evaluated_samples": len(ssim_list),
        "wall_time_s": float(elapsed),
        "throughput_sps": float(len(ssim_list) / max(elapsed, 1e-4)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase Retrieval baseline.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifest", default="manifests/task_1/val.csv", help="Task 1 manifest")
    parser.add_argument("--samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metrics = run_phase_retrieval_benchmark(
        dataset_dir=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.samples,
    )
    print("\nPhase Retrieval Baseline Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
