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
    support_regime: str = "disc",
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate HIO/ER baseline on Task 1 validation samples."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading Phase Retrieval evaluation data from {manifest_path} (regime: {support_regime})...")
    df_val = pd.read_csv(manifest_path)
    if "sample_id" not in df_val.columns:
        raise ValueError(f"Manifest at {manifest_path} is missing required 'sample_id' column.")
    if df_val["sample_id"].duplicated().any():
        dups = df_val[df_val["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Manifest contains duplicate sample IDs: {dups[:5]}")

    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(
        dataset_dir, split=None, task="phase_retrieval", return_dict=True, manifest_path=manifest_path
    )

    ssim_list = []
    mse_list = []
    r_factor_list = []
    pearson_r_list = []
    evaluated_sids = []

    t0 = time.time()
    for idx, row in df_val.iterrows():
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            raise ValueError(f"Manifest row {idx} is missing required 'sample_id' field.")
        sample = ds.get_by_sample_id(sid)
        evaluated_sids.append(sid)
        inp = sample["input"][0].numpy()  # (512, 512)
        target = sample["target"][0].numpy()  # (64, 64)
        mask = sample["mask"][0].numpy()  # (512, 512)

        res = evaluate_phase_retrieval(
            measured_intensity=inp,
            ground_truth_object=target,
            support_regime=support_regime,
            valid_mask=mask,
            num_hio_iters=num_hio_iters,
            num_er_iters=num_er_iters,
            seed=seed + idx,
        )

        ssim_list.append(res["aligned_ssim"])
        mse_list.append(res["aligned_mse"])
        r_factor_list.append(res["final_r_factor"])

        # Compute Pearson correlation between aligned reconstruction and ground truth
        from opendiffraction.tasks.metrics import align_twin_and_translation

        aligned_recon = align_twin_and_translation(res["reconstructed"], target)
        r_val, _ = pearsonr(aligned_recon.ravel(), target.ravel())
        pearson_r_list.append(float(r_val) if np.isfinite(r_val) else 0.0)

    ds.close()
    elapsed = time.time() - t0

    metrics = {
        "mean_aligned_ssim": float(np.mean(ssim_list)),
        "std_aligned_ssim": float(np.std(ssim_list)),
        "mean_aligned_mse": float(np.mean(mse_list)),
        "mean_final_r_factor": float(np.mean(r_factor_list)),
        "mean_pearson_r": float(np.mean(pearson_r_list)),
        "evaluated_samples": len(ssim_list),
        "evaluated_sample_ids": evaluated_sids,
        "wall_time_s": float(elapsed),
        "throughput_sps": float(len(ssim_list) / max(elapsed, 1e-4)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase Retrieval baseline.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifest", default="manifests/task_1/val.csv", help="Task 1 manifest")
    parser.add_argument("--samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument(
        "--support-regime",
        default="disc",
        choices=["disc", "box", "oracle", "autocorrelation"],
        help="Support regime for phase retrieval constraint",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metrics = run_phase_retrieval_benchmark(
        dataset_dir=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.samples,
        support_regime=args.support_regime,
    )
    print("\nPhase Retrieval Baseline Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
