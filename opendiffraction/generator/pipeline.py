"""Dataset generation pipeline: preflight checks, batch simulation, sharding, and orchestration."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any
import numpy as np
from tqdm import tqdm

from opendiffraction.objects.procedural import generate_procedural_object
from opendiffraction.physics.forward import simulate_diffraction
from opendiffraction.variations.profiles import PROFILE_SPECS, sample_profile_config
from .splits import assign_group_to_split
from .storage import DatasetWriter


PRESET_CONFIGS = {
    "quick": 48,  # 4 samples x 12 profiles
    "starter": 2400,  # 200 samples x 12 profiles
    "full": 9600,  # 800 samples x 12 profiles
}

ESTIMATED_BYTES_PER_SAMPLE_COMPRESSED = 450_000  # ~450 KB per sample with gzip level 4


def check_disk_space(output_dir: str, total_samples: int, max_disk_gb: float | None = None) -> None:
    """Preflight check: ensure sufficient free disk space exists before simulation."""
    os.makedirs(output_dir, exist_ok=True)
    total, used, free = shutil.disk_usage(output_dir)

    estimated_bytes = total_samples * ESTIMATED_BYTES_PER_SAMPLE_COMPRESSED
    safety_margin_bytes = 1024 * 1024 * 1024  # 1 GB margin
    required_bytes = estimated_bytes + safety_margin_bytes

    free_gb = free / (1024**3)
    est_gb = estimated_bytes / (1024**3)

    if max_disk_gb is not None and est_gb > max_disk_gb:
        raise RuntimeError(
            f"Estimated disk usage ({est_gb:.2f} GB) exceeds user limit --max-disk-gb={max_disk_gb} GB."
        )

    if free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space on {output_dir}. "
            f"Available: {free_gb:.2f} GB, Estimated required: {est_gb:.2f} GB (+ 1 GB safety margin)."
        )


def generate_dataset(
    output_dir: str,
    preset: str = "quick",
    total_samples: int | None = None,
    seed: int = 42,
    batch_size: int = 48,
    shard_size: int = 480,
    resume: bool = True,
    max_disk_gb: float | None = None,
    dry_run: bool = False,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Execute complete dataset generation for all 12 profiles.

    Parameters
    ----------
    output_dir : str
        Target output directory for HDF5 shards and metadata.
    preset : str
        One of 'quick' (48), 'starter' (2400), 'full' (9600).
    total_samples : int or None
        Override sample count. Must be >= 12.
    seed : int
        Master random seed.
    batch_size : int
        Processing batch size.
    shard_size : int
        Maximum samples per HDF5 shard.
    resume : bool
        If True, skip already generated shards.
    max_disk_gb : float or None
        Maximum disk limit in GB.
    dry_run : bool
        If True, runs preflight and logs plan without writing data.
    progress_callback : callable or None
        Optional callback for tracking progress.

    Returns
    -------
    dict
        Summary statistics of generated dataset.
    """
    if total_samples is None:
        total_samples = PRESET_CONFIGS.get(preset, 48)

    check_disk_space(output_dir, total_samples, max_disk_gb)

    if dry_run:
        return {
            "status": "dry_run_success",
            "preset": preset,
            "total_samples": total_samples,
            "estimated_gb": (total_samples * ESTIMATED_BYTES_PER_SAMPLE_COMPRESSED) / (1024**3),
        }

    writer = DatasetWriter(output_dir, shard_size=shard_size)
    completed_ids = writer.get_completed_samples() if resume else set()

    profiles = list(PROFILE_SPECS.keys())  # V01 to V12
    samples_per_profile = total_samples // len(profiles)
    remainder = total_samples % len(profiles)

    # Master plan: list of sample tasks
    tasks = []
    global_idx = 0
    for p_idx, pid in enumerate(profiles):
        n_prof = samples_per_profile + (1 if p_idx < remainder else 0)
        spec = PROFILE_SPECS[pid]
        for s_idx in range(n_prof):
            sample_id = f"OD_{global_idx:06d}_{pid}"
            # Unique group ID per base object
            group_id = f"grp_{pid}_{s_idx:04d}"
            split = assign_group_to_split(group_id, seed=seed, is_ood=spec.is_ood)

            tasks.append(
                {
                    "global_idx": global_idx,
                    "sample_id": sample_id,
                    "profile_id": pid,
                    "index_in_profile": s_idx,
                    "total_in_profile": n_prof,
                    "group_id": group_id,
                    "split": split,
                    "is_ood": spec.is_ood,
                }
            )
            global_idx += 1

    # Filter completed if resuming
    pending_tasks = [t for t in tasks if t["sample_id"] not in completed_ids]

    if not pending_tasks:
        return {
            "status": "already_completed",
            "total_samples": total_samples,
            "completed_samples": len(completed_ids),
        }

    # Group pending tasks by split and shard
    split_buckets: dict[str, list[dict[str, Any]]] = {}
    for t in pending_tasks:
        split_buckets.setdefault(t["split"], []).append(t)

    start_time = time.time()
    total_written = 0

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for split_name, split_tasks in split_buckets.items():
        # Partition into shards of at most shard_size
        n_shards = int(np.ceil(len(split_tasks) / shard_size))
        for sh_idx in range(n_shards):
            shard_tasks = split_tasks[sh_idx * shard_size : (sh_idx + 1) * shard_size]
            shard_name = f"{split_name}_{sh_idx:03d}.h5"
            shard_path = os.path.join(data_dir, shard_name)

            shard_results = []
            manifest_rows = []

            for row_idx, task in enumerate(tqdm(shard_tasks, desc=f"Simulating {shard_name}")):
                task_rng = np.random.default_rng(seed + task["global_idx"])

                cfg, fam, dens, subtype = sample_profile_config(
                    profile_id=task["profile_id"],
                    index_within_profile=task["index_in_profile"],
                    total_in_profile=task["total_in_profile"],
                    rng=task_rng,
                )

                # Generate base object
                obj, _ = generate_procedural_object(
                    family=fam,
                    density_type=dens,
                    size=cfg.object_size[0],
                    rng=task_rng,
                    shape_subtype=subtype,
                )

                # Run physics forward simulation
                sim_res = simulate_diffraction(obj, cfg, rng=task_rng)

                # Compile sample dictionary
                sample_dict = {
                    "object_amplitude": sim_res.object_amplitude,
                    "object_phase": sim_res.object_phase,
                    "target_dpr": sim_res.target_dpr,
                    "target_physical": sim_res.target_physical,
                    "intensity_ideal": sim_res.intensity_ideal,
                    "intensity_expected": sim_res.intensity_expected,
                    "intensity_measured": sim_res.intensity_measured,
                    "valid_mask": sim_res.valid_mask,
                }
                shard_results.append(sample_dict)

                # Metadata record
                manifest_rows.append(
                    {
                        "sample_id": task["sample_id"],
                        "source_object_id": f"obj_{fam}_{subtype}_{task['group_id']}",
                        "group_id": task["group_id"],
                        "split": task["split"],
                        "variation_family": task["profile_id"],
                        "object_family": fam,
                        "density_type": dens,
                        "photon_budget": f"{cfg.photon_budget:.2e}",
                        "coherence_length": (
                            "inf" if np.isinf(cfg.coherence_length) else f"{cfg.coherence_length:.1f}"
                        ),
                        "readout_noise_std": f"{cfg.readout_std:.4f}",
                        "beam_stop_geometry": f"{cfg.beam_stop_hw}x{cfg.beam_stop_hh}",
                        "missing_pixel_fraction": f"{1.0 - float(np.mean(sim_res.valid_mask)):.4f}",
                        "detector_saturation_threshold": (
                            "None"
                            if cfg.saturation_threshold is None
                            else f"{cfg.saturation_threshold:.1e}"
                        ),
                        "beam_center_offset": f"({cfg.beam_center_offset[0]:.1f},{cfg.beam_center_offset[1]:.1f})",
                        "random_seed": seed + task["global_idx"],
                        "shard_name": shard_name,
                        "row_index": row_idx,
                        "simulation_version": "0.1.0",
                        "model_compatibility": "DPR_and_Standard_CDI",
                    }
                )

            # Atomic shard write
            writer.write_shard_atomic(shard_path, shard_results)
            writer.append_manifest_rows(manifest_rows)
            writer.update_progress([t["sample_id"] for t in shard_tasks])
            total_written += len(shard_tasks)

    elapsed = time.time() - start_time
    return {
        "status": "success",
        "preset": preset,
        "total_samples": total_samples,
        "written_samples": total_written,
        "elapsed_seconds": elapsed,
        "output_dir": output_dir,
    }
