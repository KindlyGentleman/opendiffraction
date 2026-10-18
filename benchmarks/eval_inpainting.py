"""Detector Inpainting Benchmark Evaluation (Task 3).

Evaluates classical nearest-neighbor and boundary inpainting baselines against
unmasked ground truth diffraction patterns on the validation manifest.
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

from benchmarks.missing_pixel_recovery import inpaint_nearest_neighbor
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
from opendiffraction.tasks.metrics import psnr


def run_inpainting_benchmark(
    dataset_dir: str = "output_v3",
    manifest_path: str = "manifests/task_3/val.csv",
    num_samples: int = 50,
) -> dict[str, float]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading Inpainting evaluation data from {manifest_path}...")
    df_val = pd.read_csv(manifest_path)
    if "sample_id" not in df_val.columns:
        raise ValueError(f"Manifest at {manifest_path} is missing required 'sample_id' column.")
    if df_val["sample_id"].duplicated().any():
        dups = df_val[df_val["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Manifest contains duplicate sample IDs: {dups[:5]}")

    if num_samples is not None and len(df_val) > num_samples:
        df_val = df_val.head(num_samples)

    ds = OpenDiffractionTorchDataset(
        dataset_dir, split=None, task="inpainting", return_dict=True, manifest_path=manifest_path
    )

    mae_masked_list = []
    psnr_list = []
    ssim_list = []
    evaluated_sids = []

    t0 = time.time()
    for idx, row in df_val.iterrows():
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            raise ValueError(f"Manifest row {idx} is missing required 'sample_id' field.")
        sample = ds.get_by_sample_id(sid)
        evaluated_sids.append(sid)
        masked_inp = sample["input"][0].numpy()  # (512, 512)
        target = sample["target"][0].numpy()  # unmasked expected intensity
        mask = sample["mask"][0].numpy().astype(bool)  # (512, 512)

        inpainted = inpaint_nearest_neighbor(masked_inp, mask)

        # Compute error strictly on masked/missing pixels
        missing = ~mask
        if np.any(missing):
            mae_hole = float(np.mean(np.abs(inpainted[missing] - target[missing])))
        else:
            mae_hole = 0.0

        p_val = psnr(target, inpainted)
        drange = max(float(np.max(target) - np.min(target)), 1e-6)
        s_val = ssim(target, inpainted, data_range=drange)

        mae_masked_list.append(mae_hole)
        psnr_list.append(float(p_val))
        ssim_list.append(float(s_val))

    ds.close()
    elapsed = time.time() - t0

    metrics = {
        "mean_masked_region_mae": float(np.mean(mae_masked_list)),
        "mean_overall_psnr_db": float(np.mean(psnr_list)),
        "mean_overall_ssim": float(np.mean(ssim_list)),
        "evaluated_samples": len(psnr_list),
        "evaluated_sample_ids": evaluated_sids,
        "wall_time_s": float(elapsed),
        "throughput_sps": float(len(psnr_list) / max(elapsed, 1e-4)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Inpainting baseline.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifest", default="manifests/task_3/val.csv", help="Task 3 manifest")
    parser.add_argument("--samples", type=int, default=50, help="Number of validation samples")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metrics = run_inpainting_benchmark(
        dataset_dir=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.samples,
    )
    print("\nInpainting Baseline Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
