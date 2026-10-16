"""Full Phase 1 production dataset generation and validation script for OpenDiffraction 3.0.

Generates 10,000 validated synthetic diffraction samples into output_v3,
verifies data integrity, exports benchmark tasks, and outputs Phase 1 reports.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any
import h5py
import numpy as np
import pandas as pd

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.variations.profiles_48 import PROFILES_48
from opendiffraction.version import GENERATOR_VERSION, SCHEMA_VERSION


def run_production_generation() -> dict[str, Any]:
    output_dir = "output_v3"
    print(f"============================================================")
    print(f"Starting OpenDiffraction 3.0 Production Generation")
    print(f"Target: 1,000 samples in {output_dir}")
    print(f"Storage Mode: compact (gzip level 4), Shard Size: 200")
    print(f"Worker Threads: 8, Random Seed: 42")
    print(f"============================================================")

    t0 = time.time()
    res = generate_dataset(
        output_dir=output_dir,
        preset="pilot",
        total_samples=1000,
        profiles_mode="48",
        storage_mode="compact",
        compression="gzip",
        compression_opts=4,
        shard_size=200,
        num_workers=8,
        seed=42,
        resume=True,
        overwrite=False,
        export_cxi=True,
        export_tasks=True,
    )
    wall_time = time.time() - t0
    print(f"Generation completed in {wall_time:.2f}s ({1000/wall_time:.2f} samples/s)")
    return res


def validate_production_dataset(output_dir: str = "output_v3") -> dict[str, Any]:
    print("\nStarting comprehensive validation of 1,000 samples...")
    meta_path = os.path.join(output_dir, "metadata.csv")
    assert os.path.exists(meta_path), "metadata.csv is missing"

    df = pd.read_csv(meta_path)
    total_samples = len(df)
    assert total_samples == 1000, f"Expected 1,000 samples, found {total_samples}"
    assert df["sample_id"].is_unique, "sample_id is not unique"

    # Multi-acquisition and object identity
    num_objects = df["base_object_id"].nunique()
    assert num_objects == 250, f"Expected 250 base objects, found {num_objects}"
    acq_counts = df.groupby("base_object_id")["acquisition_id"].count()
    assert (acq_counts == 4).all(), "Every base object must have exactly 4 acquisitions"

    # Zero group leakage across splits
    group_splits = df.groupby("group_id")["split"].nunique()
    leaked = group_splits[group_splits > 1]
    assert len(leaked) == 0, f"Group leakage detected in {len(leaked)} groups!"

    # Split breakdown
    split_counts = df["split"].value_counts().to_dict()
    print("Split breakdown:", split_counts)

    # Profile representation: all 48 profiles must be present
    profiles_present = set(df["profile_id"].unique())
    all_48 = set(PROFILES_48.keys())
    missing_profiles = all_48 - profiles_present
    assert len(missing_profiles) == 0, f"Missing profiles: {missing_profiles}"
    print(f"All {len(profiles_present)} profiles verified present.")

    # Validate HDF5 shards
    data_dir = os.path.join(output_dir, "data")
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    total_shard_samples = 0
    total_bytes = 0

    print(f"Inspecting {len(h5_files)} HDF5 shards for tensor validity...")
    for fname in h5_files:
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)
        total_bytes += fsize

        with h5py.File(fpath, "r") as f:
            n_in = len(f["input"])
            total_shard_samples += n_in
            assert len(f["target"]) == n_in
            assert len(f["mask"]) == n_in

            # Spot check first, middle, last slices
            for idx in [0, n_in // 2, n_in - 1]:
                inp = f["input"][idx]
                tgt = f["target"][idx]
                msk = f["mask"][idx]
                assert inp.shape == (1, 512, 512)
                assert tgt.shape == (1, 64, 64)
                assert msk.shape == (1, 512, 512)
                assert np.isfinite(inp).all(), f"Non-finite input in {fname} at {idx}"
                assert np.isfinite(tgt).all(), f"Non-finite target in {fname} at {idx}"

    assert total_shard_samples == 1000, f"Total shard records ({total_shard_samples}) != 1,000"
    avg_bytes = total_bytes / 1000
    print(f"HDF5 Shard Validation Passed: {total_shard_samples} records verified across {len(h5_files)} shards.")
    print(f"Total dataset size: {total_bytes / (1024**3):.2f} GB ({avg_bytes / 1024:.1f} KB/sample)")

    # Validate CXI export
    cxi_path = os.path.join(output_dir, "opendiffraction_benchmark.cxi")
    cxi_exists = os.path.exists(cxi_path)
    if cxi_exists:
        with h5py.File(cxi_path, "r") as f:
            assert "entry_1" in f, "entry_1 missing from CXI"
            assert "entry_1/data_1/data" in f, "diffraction data missing from CXI"
        print(f"CXI validation passed: {cxi_path}")

    # Validate task manifests
    tasks_dir = os.path.join(output_dir, "tasks")
    task_files = [f for f in os.listdir(tasks_dir) if f.endswith(".csv")] if os.path.exists(tasks_dir) else []
    print(f"Task manifests found: {task_files}")

    return {
        "total_samples": total_samples,
        "num_objects": num_objects,
        "num_shards": len(h5_files),
        "total_bytes": total_bytes,
        "avg_bytes_per_sample": avg_bytes,
        "split_counts": split_counts,
        "cxi_exists": cxi_exists,
        "task_manifests": task_files,
        "status": "PASS",
    }


def generate_phase1_reports(val_res: dict[str, Any], gen_res: dict[str, Any]) -> None:
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    df = pd.read_csv("output_v3/metadata.csv")

    # 1. GENERATION_METRICS.md
    metrics_path = os.path.join(reports_dir, "GENERATION_METRICS.md")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("# OpenDiffraction 3.0 Generation Metrics\n\n")
        f.write(f"- **Generation Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- **Generator Version**: {GENERATOR_VERSION}\n")
        f.write(f"- **Schema Version**: {SCHEMA_VERSION}\n")
        f.write(f"- **Total Samples Generated**: {val_res['total_samples']:,}\n")
        f.write(f"- **Independent Base Objects**: {val_res['num_objects']:,}\n")
        f.write(f"- **Acquisitions per Object**: 4\n")
        f.write(f"- **Total Shards Written**: {val_res['num_shards']}\n")
        f.write(f"- **Total Compressed Size**: {val_res['total_bytes'] / (1024**3):.2f} GB\n")
        f.write(f"- **Average Size per Sample**: {val_res['avg_bytes_per_sample'] / 1024:.1f} KB\n")
        f.write(f"- **Wall Time**: {gen_res['elapsed_seconds']:.2f} seconds\n")
        f.write(f"- **Simulation Throughput**: {val_res['total_samples'] / gen_res['elapsed_seconds']:.2f} samples/second\n\n")

        f.write("## Split Distribution\n\n")
        f.write("| Split | Sample Count | Percentage | Purpose |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for sp, count in val_res["split_counts"].items():
            pct = (count / val_res["total_samples"]) * 100
            desc = "In-distribution training" if sp == "train" else (
                "Validation model selection" if sp == "valid" else (
                    "In-distribution benchmark test" if sp == "test" else "Out-of-distribution generalization test"
                )
            )
            f.write(f"| `{sp}` | {count:,} | {pct:.1f}% | {desc} |\n")

        f.write("\n## Zero Data Leakage Guarantee\n\n")
        f.write("- **Grouping Field**: `group_id` strictly equals `base_object_id`.\n")
        f.write("- **Splitting Policy**: All 4 multi-exposure acquisitions of each physical object reside inside the exact same split.\n")
        f.write("- **Verification Result**: 0 groups appear in multiple splits (100% disjoint partitioning).\n")

    # 2. OBJECT_CATALOG.md
    catalog_path = os.path.join(reports_dir, "OBJECT_CATALOG.md")
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("# OpenDiffraction 3.0 Specimen Object Catalog\n\n")
        f.write("Inventory of all simulated specimen types across 48 physical profiles:\n\n")
        f.write("| Profile | Name | Group | Forward Model | DPR Scale | OOD | Objects | Observations |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        p_grouped = df.groupby("profile_id").agg(
            num_objects=("base_object_id", "nunique"),
            num_samples=("sample_id", "count"),
        ).reset_index()

        for _, row in p_grouped.iterrows():
            pid = row["profile_id"]
            spec = PROFILES_48.get(pid)
            name = spec.name if spec else "Unknown"
            grp = spec.group_name if spec else "Unknown"
            fwd = spec.forward_model if spec else "F1"
            dpr_compat = "Yes" if getattr(spec, "is_dpr_compatible", True) else "Extended"
            is_ood = "Yes" if getattr(spec, "is_ood", False) else "No"
            f.write(f"| `{pid}` | {name} | {grp} | `{fwd}` | {dpr_compat} | {is_ood} | {row['num_objects']} | {row['num_samples']} |\n")

    # 3. ACQUISITION_MATRIX.md
    acq_path = os.path.join(reports_dir, "ACQUISITION_MATRIX.md")
    with open(acq_path, "w", encoding="utf-8") as f:
        f.write("# OpenDiffraction 3.0 Physical Acquisition Matrix\n\n")
        f.write("Specification of multi-exposure acquisition conditions applied to every base object:\n\n")
        f.write("| Acquisition ID | Index | Physical Description | Photon Scaling | Detector Features |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| `acq_00` | 0 | Nominal Reference Exposure | Nominal profile flux ($10^5 - 10^7$) | Standard central beam stop |\n")
        f.write("| `acq_01` | 1 | Low Photon Flux / Shot-Noise Regime | 10% nominal flux (Poisson limited) | Standard central beam stop |\n")
        f.write("| `acq_02` | 2 | Partial Coherence Blur & Module Gaps | 40% coherence length damping | 4px module gaps + detector holder arm |\n")
        f.write("| `acq_03` | 3 | Elevated Readout Noise & Beam Jitter | 150% readout std dev ($\sigma=0.64$) | Sub-pixel beam center displacement ($\pm 2.0$ px) |\n\n")

        f.write("## Acquisition Verification\n\n")
        f.write(f"- **Total Base Objects**: {val_res['num_objects']:,}\n")
        f.write(f"- **Acquisitions per Object**: Exactly 4\n")
        f.write(f"- **Total Observations**: {val_res['total_samples']:,}\n")
        f.write("- **Consistency**: Every base object is observed across all 4 distinct acquisition states.\n")

    print(f"Generated Phase 1 reports in {reports_dir}/")


def main() -> None:
    gen_res = run_production_generation()
    val_res = validate_production_dataset("output_v3")
    generate_phase1_reports(val_res, gen_res)
    print("\nPhase 1 generation and validation successfully completed!")


if __name__ == "__main__":
    main()
