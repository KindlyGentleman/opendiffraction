"""Command-line interface for OpenDiffraction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
import yaml

from opendiffraction.generator.pipeline import generate_dataset
from opendiffraction.package.kaggle import package_kaggle_release
from opendiffraction.validation.auditor import DatasetAuditor
from opendiffraction.validation.reports import (
    generate_known_limitations,
    generate_quality_audit,
    generate_validation_report,
)
from opendiffraction.visualization.animation import (
    generate_photon_sweep_gif,
    generate_rotation_sweep_gif,
)
from opendiffraction.visualization.figures import generate_all_visualizations
from opendiffraction.visualization.figures_16 import generate_all_16_figures


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m opendiffraction",
        description="OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Dataset Generator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # generate command
    p_gen = subparsers.add_parser("generate", help="Generate synthetic diffraction dataset")
    p_gen.add_argument("--preset", choices=["quick", "starter", "full"], default="quick", help="Dataset size preset")
    p_gen.add_argument("--profiles-mode", choices=["48", "12"], default="48", help="48 profiles (v2.0) or 12 profiles (v1.0)")
    p_gen.add_argument("--samples", type=int, default=None, help="Explicit total sample count override")
    p_gen.add_argument("--output", default="./output", help="Target output directory")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--batch-size", type=int, default=48, help="Batch simulation size")
    p_gen.add_argument("--shard-size", type=int, default=480, help="Samples per HDF5 shard")
    p_gen.add_argument("--device", default="cpu", help="Compute device (cpu)")
    p_gen.add_argument("--resume", action="store_true", default=True, help="Resume previous generation run")
    p_gen.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume; overwrite/restart")
    p_gen.add_argument("--max-disk-gb", type=float, default=None, help="Maximum allowed disk space in GB")
    p_gen.add_argument("--dry-run", action="store_true", help="Perform preflight check without writing data")
    p_gen.add_argument("--export-cxi", action="store_true", help="Export validation subset to official CXI NeXus format")
    p_gen.add_argument("--export-tasks", action="store_true", help="Export independent task manifests (T1 - T12)")
    p_gen.add_argument("--config", default=None, help="Optional YAML config file path")

    # validate command
    p_val = subparsers.add_parser("validate", help="Validate physics, statistics, and HDF5 integrity")
    p_val.add_argument("--dataset", default="./output", help="Path to generated dataset directory")
    p_val.add_argument("--reports-dir", default="./reports", help="Directory to save markdown reports")

    # visualize command
    p_vis = subparsers.add_parser("visualize", help="Generate publication-quality figures and previews")
    p_vis.add_argument("--dataset", default="./output", help="Path to generated dataset directory")
    p_vis.add_argument("--output", default="./output/figures", help="Target figures directory")
    p_vis.add_argument("--suite", choices=["16", "8", "all"], default="all", help="Visualization suite (16 IEEE, 8 standard, or all)")

    # animate command
    p_anim = subparsers.add_parser("animate", help="Generate animated parameter sweep GIFs")
    p_anim.add_argument("--output", default="./output/figures/sweeps", help="Target directory for animations")
    p_anim.add_argument("--type", choices=["rotation", "photon", "both"], default="both", help="Animation type")

    # package command
    p_pkg = subparsers.add_parser("package", help="Package dataset for Kaggle release")
    p_pkg.add_argument("--dataset", default="./output", help="Source dataset directory")
    p_pkg.add_argument("--output", default="./kaggle_release", help="Kaggle package destination directory")
    p_pkg.add_argument("--username", default="YOUR_KAGGLE_USERNAME", help="Kaggle username placeholder")
    p_pkg.add_argument("--format", default="kaggle", choices=["kaggle"], help="Packaging format")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        # Check if YAML config provided
        preset = args.preset
        total_samples = args.samples
        output_dir = args.output
        seed = args.seed
        export_cxi = args.export_cxi
        export_tasks = args.export_tasks
        profiles_mode = args.profiles_mode

        if args.config and os.path.exists(args.config):
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
            preset = cfg_data.get("preset", cfg_data.get("name", preset))
            total_samples = cfg_data.get("num_samples", total_samples)
            output_dir = cfg_data.get("output_dir", output_dir)
            seed = cfg_data.get("random_seed", seed)
            export_cxi = cfg_data.get("export_cxi", export_cxi)
            export_tasks = cfg_data.get("export_tasks", export_tasks)
            print(f"[OpenDiffraction] Loaded configuration from {args.config}")

        print(f"[OpenDiffraction] Starting generation (preset='{preset}', mode='{profiles_mode}', samples={total_samples})...")
        res = generate_dataset(
            output_dir=output_dir,
            preset=preset,
            total_samples=total_samples,
            profiles_mode=profiles_mode,
            seed=seed,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            resume=args.resume,
            max_disk_gb=args.max_disk_gb,
            dry_run=args.dry_run,
            export_cxi=export_cxi,
            export_tasks=export_tasks,
        )
        print(f"[OpenDiffraction] Generation completed: {res}")
        return 0

    elif args.command == "validate":
        print(f"[OpenDiffraction] Running validation on {args.dataset}...")
        auditor = DatasetAuditor(args.dataset)
        audit_res = auditor.run_full_audit()

        os.makedirs(args.reports_dir, exist_ok=True)
        rep_val = os.path.join(args.reports_dir, "VALIDATION_REPORT.md")
        rep_audit = os.path.join(args.reports_dir, "QUALITY_AUDIT.md")
        rep_limit = os.path.join(args.reports_dir, "KNOWN_LIMITATIONS.md")

        generate_validation_report(audit_res, rep_val)
        generate_quality_audit(audit_res, rep_audit)
        generate_known_limitations(rep_limit)

        print(f"[OpenDiffraction] Audit status: {'PASSED' if audit_res['all_passed'] else 'FAILED'}")
        print(f"[OpenDiffraction] Reports written to {args.reports_dir}")
        return 0 if audit_res["all_passed"] else 1

    elif args.command == "visualize":
        print(f"[OpenDiffraction] Generating visualization figures to {args.output}...")
        os.makedirs(args.output, exist_ok=True)
        if args.suite in ("8", "all"):
            generate_all_visualizations(args.output)
        if args.suite in ("16", "all"):
            generate_all_16_figures(args.output)
        print(f"[OpenDiffraction] Figures successfully saved to {args.output}")
        return 0

    elif args.command == "animate":
        print(f"[OpenDiffraction] Generating parameter sweep animations in {args.output}...")
        os.makedirs(args.output, exist_ok=True)
        if args.type in ("rotation", "both"):
            p_rot = generate_rotation_sweep_gif(os.path.join(args.output, "3d_rotation_sweep.gif"))
            print(f"[OpenDiffraction] Generated {p_rot}")
        if args.type in ("photon", "both"):
            p_pho = generate_photon_sweep_gif(os.path.join(args.output, "photon_starvation_sweep.gif"))
            print(f"[OpenDiffraction] Generated {p_pho}")
        return 0

    elif args.command == "package":
        print(f"[OpenDiffraction] Packaging dataset from {args.dataset} to {args.output}...")
        pkg_res = package_kaggle_release(
            source_dir=args.dataset,
            release_dir=args.output,
            username=args.username,
        )
        print(f"[OpenDiffraction] Packaging complete: {pkg_res['shards_packaged']} shards packaged.")
        print(f"[OpenDiffraction] Kaggle initialization command: {pkg_res['cli_init_command']}")
        print(f"[OpenDiffraction] Kaggle publication command:    {pkg_res['cli_create_command']}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
