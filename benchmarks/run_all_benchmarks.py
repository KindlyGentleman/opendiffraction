"""Unified evaluation runner for all four OpenDiffraction 3.0 benchmark tasks.

Executes baseline evaluations for:
- Task 1: Phase Retrieval (HIO/ER)
- Task 2: Denoising (Median + Gaussian)
- Task 3: Missing Detector Recovery / Inpainting (Nearest Neighbor)
- Task 4: Hit and Miss Classification (Energy & Peak Thresholding)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from benchmarks.eval_phase_retrieval import run_phase_retrieval_benchmark
from benchmarks.eval_denoising import run_denoising_benchmark
from benchmarks.eval_inpainting import run_inpainting_benchmark
from benchmarks.eval_hit_classification import run_hit_classification_benchmark


def run_all_benchmarks(
    dataset_dir: str = "output_v3",
    manifests_dir: str = "manifests",
    pr_samples: int = 20,
    dn_samples: int = 50,
    in_samples: int = 50,
    hit_samples: int | None = None,
) -> dict[str, Any]:
    print("============================================================")
    print("Running OpenDiffraction 3.0 Benchmark Suite across Tasks 1-4")
    print("============================================================")

    suite_results = {}
    t_start = time.time()

    def _find_manifest(m_dir: str, task_key: str, fallback_sub: str) -> str:
        candidates = [
            os.path.join(m_dir, f"{task_key}_balanced_eval.csv"),
            os.path.join(m_dir, f"{task_key}_manifest.csv"),
            os.path.join(m_dir, fallback_sub, "val.csv"),
            os.path.join(m_dir, fallback_sub, "test.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return os.path.join(m_dir, fallback_sub, "val.csv")

    # 1. Phase Retrieval
    print("\n[1/4] Running Task 1: Phase Retrieval (HIO/ER)...")
    pr_manifest = _find_manifest(manifests_dir, "T1", "task_1")
    suite_results["T1_Phase_Retrieval"] = run_phase_retrieval_benchmark(
        dataset_dir=dataset_dir,
        manifest_path=pr_manifest,
        num_samples=pr_samples,
    )

    # 2. Denoising
    print("\n[2/4] Running Task 2: Diffraction Pattern Denoising...")
    dn_manifest = _find_manifest(manifests_dir, "T2", "task_2")
    suite_results["T2_Denoising"] = run_denoising_benchmark(
        dataset_dir=dataset_dir,
        manifest_path=dn_manifest,
        num_samples=dn_samples,
    )

    # 3. Inpainting
    print("\n[3/4] Running Task 3: Detector Inpainting...")
    in_manifest = _find_manifest(manifests_dir, "T3", "task_3")
    suite_results["T3_Inpainting"] = run_inpainting_benchmark(
        dataset_dir=dataset_dir,
        manifest_path=in_manifest,
        num_samples=in_samples,
    )

    # 4. Hit Classification
    print("\n[4/4] Running Task 4: Hit Finding and Classification...")
    hit_manifest = _find_manifest(manifests_dir, "T4", "task_4")
    suite_results["T4_Hit_Classification"] = run_hit_classification_benchmark(
        dataset_dir=dataset_dir,
        manifest_path=hit_manifest,
        num_samples=hit_samples,
    )

    total_time = time.time() - t_start
    suite_results["total_suite_time_s"] = float(total_time)
    return suite_results


def generate_baseline_validation_report(results: dict[str, Any], report_path: str = "reports/BASELINE_VALIDATION.md") -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# OpenDiffraction 3.0 Baseline Benchmark Validation\n\n")
        f.write("Results of classical baseline algorithms across the four primary benchmark tasks:\n\n")

        # Task 1
        t1 = results["T1_Phase_Retrieval"]
        f.write("## Task 1: Phase Retrieval (HIO/ER Iterative Baseline)\n\n")
        f.write("- **Algorithm**: Hybrid Input-Output (40 iters, beta=0.8) + Error Reduction (20 iters)\n")
        f.write(f"- **Evaluated Samples**: {t1['evaluated_samples']}\n")
        f.write(f"- **Mean Aligned SSIM**: {t1['mean_aligned_ssim']:.4f} $\\pm$ {t1['std_aligned_ssim']:.4f}\n")
        f.write(f"- **Mean Aligned MSE**: {t1['mean_aligned_mse']:.6f}\n")
        f.write(f"- **Mean Final R-Factor**: {t1['mean_final_r_factor']:.4f}\n")
        f.write(f"- **Throughput**: {t1['throughput_sps']:.2f} samples/s\n\n")

        # Task 2
        t2 = results["T2_Denoising"]
        f.write("## Task 2: Diffraction Pattern Denoising (Spatial Filter Baseline)\n\n")
        f.write(f"- **Algorithm**: Median Filter (size=3) + Gaussian Filter ($\\sigma=1.0$)\n")
        f.write(f"- **Evaluated Samples**: {t2['evaluated_samples']}\n")
        f.write(f"- **Mean PSNR**: {t2['mean_psnr_db']:.2f} dB $\\pm$ {t2['std_psnr_db']:.2f} dB\n")
        f.write(f"- **Mean SSIM**: {t2['mean_ssim']:.4f}\n")
        f.write(f"- **Mean R-Factor**: {t2['mean_r_factor']:.4f}\n")
        f.write(f"- **Throughput**: {t2['throughput_sps']:.2f} samples/s\n\n")

        # Task 3
        t3 = results["T3_Inpainting"]
        f.write("## Task 3: Detector Inpainting (Nearest-Neighbor Baseline)\n\n")
        f.write(f"- **Algorithm**: Euclidean Distance Transform Nearest Valid Pixel Inpainting\n")
        f.write(f"- **Evaluated Samples**: {t3['evaluated_samples']}\n")
        f.write(f"- **Masked Region MAE**: {t3['mean_masked_region_mae']:.4e}\n")
        f.write(f"- **Overall Pattern PSNR**: {t3['mean_overall_psnr_db']:.2f} dB\n")
        f.write(f"- **Overall SSIM**: {t3['mean_overall_ssim']:.4f}\n")
        f.write(f"- **Throughput**: {t3['throughput_sps']:.2f} samples/s\n\n")

        # Task 4
        t4 = results["T4_Hit_Classification"]
        f.write("## Task 4: Hit Finding and Classification (Energy & Peak Threshold Baseline)\n\n")
        f.write("- **Algorithm**: Integrated Photon Energy (E >= 1e4) + Peak Bragg Readout (I_max >= 50)\n")
        f.write(f"- **Evaluated Samples**: {t4['evaluated_samples']}\n")
        f.write(f"- **Accuracy**: {t4['accuracy']*100:.2f}%\n")
        f.write(f"- **Balanced Accuracy**: {t4['balanced_accuracy']*100:.2f}%\n")
        f.write(f"- **Precision**: {t4['precision']*100:.2f}%\n")
        f.write(f"- **Recall**: {t4['recall']*100:.2f}%\n")
        f.write(f"- **F1 Score**: {t4['f1_score']:.4f}\n")
        f.write(f"- **Throughput**: {t4['throughput_sps']:.2f} samples/s\n\n")

        f.write(f"Total benchmark execution time: {results['total_suite_time_s']:.2f}s.\n")
    print(f"Generated {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete OpenDiffraction 3.0 benchmark suite.")
    parser.add_argument("--dataset", default="output_v3", help="Dataset directory")
    parser.add_argument("--manifests", default="manifests", help="Manifests directory")
    parser.add_argument("--output", default="benchmarks/benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = run_all_benchmarks(dataset_dir=args.dataset, manifests_dir=args.manifests)
    print("\n================ BENCHMARK SUITE SUMMARY ================")
    print(json.dumps(results, indent=2))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    generate_baseline_validation_report(results, "reports/BASELINE_VALIDATION.md")


if __name__ == "__main__":
    main()
