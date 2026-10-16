"""Diffraction Pattern Denoising Benchmark Evaluation (Task 2).

Evaluates classical spatial filtering (Median + Gaussian) baseline against
clean target patterns on the validation manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.abspath("."))

from benchmarks.denoising import denoise_diffraction_pattern
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
from opendiffraction.tasks.metrics import psnr, compute_r_factor


def run_denoising_benchmark(
    dataset_dir: str = "output_v3",
    manifest_path: str = "manifests/task_2/val.csv",
    num_samples: int = 50,
) -> dict[str, float]:
    """Evaluate Denoising baseline on Task 2 validation samples."""
    print(f"Loading Denoising evaluation data from {manifest_path}...")
    df_val = pd.read_csv(manifest_path)
    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(dataset_dir, split="valid", task="denoising", return_dict=True)

    psnr_list = []
    ssim_list = []
    r_factor_list = []

    t0 = time.time()
    for idx in range(min(len(df_val), len(ds))):
        sample = ds[idx]
        noisy_inp = sample["input"][0].numpy()  # (512, 512)
        target = sample["target"][0].numpy()  # (512, 512) or clean intensity

        # Baseline filtering
        denoised = denoise_diffraction_pattern(noisy_inp, method="median_gaussian", sigma=1.0, median_size=3)

        # Compute PSNR and R-factor
        p_val = psnr(target, denoised)
        r_val = compute_r_factor(target, denoised)

        # Normalized SSIM
        drange = max(float(np.max(target) - np.min(target)), 1e-6)
        s_val = ssim(target, denoised, data_range=drange)

        psnr_list.append(float(p_val))
        ssim_list.append(float(s_val))
        r_factor_list.append(float(r_val))

    ds.close()
    elapsed = time.time() - t0

    metrics = {
        "mean_psnr_db": float(np.mean(psnr_list)),
        "std_psnr_db": float(np.std(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_r_factor": float(np.mean(r_factor_list)),
        "evaluated_samples": len(psnr_list),
        "wall_time_s": float(elapsed),
        "throughput_sps": float(len(psnr_list) / max(elapsed, 1e-4)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Denoising baseline.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifest", default="manifests/task_2/val.csv", help="Task 2 manifest")
    parser.add_argument("--samples", type=int, default=50, help="Number of validation samples")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metrics = run_denoising_benchmark(
        dataset_dir=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.samples,
    )
    print("\nDenoising Baseline Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
