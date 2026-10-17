"""Dataset generation pipeline: preflight checks, batch simulation, sharding, and crash-safe orchestration."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from opendiffraction.objects.complex_objects import generate_phase_field
from opendiffraction.objects.object_3d import generate_3d_volume
from opendiffraction.objects.procedural import (
    generate_procedural_object,
    make_multiple_scatterers,
)
from opendiffraction.physics.forward import (
    SimulationConfig,
    simulate_diffraction,
)
from opendiffraction.physics.masks import composite_detector_mask
from opendiffraction.scenarios import get_scenario_physics
from opendiffraction.variations.profiles import PROFILE_SPECS, sample_profile_config
from opendiffraction.variations.profiles_48 import PROFILES_48, sample_profile_48
from opendiffraction.version import (
    DEFAULT_TARGET_SAMPLES,
    GENERATOR_VERSION,
    MODEL_VERSION,
    SCHEMA_VERSION,
)
from .splits import assign_group_to_split
from .storage import DatasetWriter, compute_config_fingerprint


PRESET_CONFIGS = {
    "quick": 48,
    "starter": 480,
    "pilot": 1000,
    "full": 10000,
}

# Estimated bytes per sample compressed (gzip level 4)
BYTES_PER_SAMPLE_COMPACT = 930_000  # ~930 KB empirical
BYTES_PER_SAMPLE_EXTENDED = 3_800_000  # ~3.8 MB empirical


def check_disk_space(
    output_dir: str,
    total_samples: int,
    storage_mode: str = "extended",
    max_disk_gb: float | None = None,
) -> None:
    """Preflight check: ensure sufficient free disk space exists before simulation."""
    os.makedirs(output_dir, exist_ok=True)
    _, _, free = shutil.disk_usage(output_dir)

    bytes_per_sample = (
        BYTES_PER_SAMPLE_COMPACT if storage_mode == "compact" else BYTES_PER_SAMPLE_EXTENDED
    )
    estimated_bytes = total_samples * bytes_per_sample
    safety_margin_bytes = 1024 * 1024 * 1024  # 1 GB safety buffer
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
            f"Available: {free_gb:.2f} GB, Estimated required: {est_gb:.2f} GB (+ 1 GB safety buffer)."
        )


def build_master_task_plan(
    total_samples: int,
    seed: int = 42,
    profiles_mode: str = "48",
    acquisitions_per_object: int = 4,
    shard_size: int = 480,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Construct an immutable, deterministic master task plan.

    Every sample is assigned a fixed base_object_id, scene_id, group_id, split,
    shard_name, and row_index before generation begins.
    """
    if profiles_mode == "48":
        profile_ids = list(PROFILES_48.keys())
    else:
        profile_ids = list(PROFILE_SPECS.keys())

    num_objects = int(np.ceil(total_samples / acquisitions_per_object))
    master_tasks: list[dict[str, Any]] = []
    global_sample_idx = 0

    for obj_idx in range(num_objects):
        pid = profile_ids[obj_idx % len(profile_ids)]
        is_ood = PROFILES_48[pid].is_ood if profiles_mode == "48" else PROFILE_SPECS[pid].is_ood
        group_name = PROFILES_48[pid].group_name if profiles_mode == "48" else "Legacy"

        # Unique, stable object identity
        base_object_id = f"obj_{pid}_{obj_idx:05d}"
        scene_id = f"scene_{pid}_{obj_idx:05d}"
        group_id = base_object_id
        object_seed = seed + obj_idx * 10007

        # Group-safe split assignment: all acquisitions of base_object_id share this split
        split = assign_group_to_split(group_id, seed=seed, is_ood=is_ood)

        for acq_idx in range(acquisitions_per_object):
            if global_sample_idx >= total_samples:
                break

            sample_id = f"OD_{global_sample_idx:06d}_{pid}_a{acq_idx}"
            acquisition_id = f"acq_{acq_idx:02d}"
            acquisition_seed = object_seed + acq_idx * 101

            master_tasks.append(
                {
                    "global_sample_idx": global_sample_idx,
                    "sample_id": sample_id,
                    "base_object_id": base_object_id,
                    "scene_id": scene_id,
                    "acquisition_id": acquisition_id,
                    "group_id": group_id,
                    "split": split,
                    "profile_id": pid,
                    "object_index": obj_idx,
                    "acq_index": acq_idx,
                    "is_ood": is_ood,
                    "object_seed": object_seed,
                    "acquisition_seed": acquisition_seed,
                    "group_name": group_name,
                    "total_objects": num_objects,
                }
            )
            global_sample_idx += 1

    # Deterministically partition tasks within each split into immutable shards
    shards: dict[str, list[dict[str, Any]]] = {}
    split_tasks: dict[str, list[dict[str, Any]]] = {}
    for t in master_tasks:
        split_tasks.setdefault(t["split"], []).append(t)

    for split_name, t_list in split_tasks.items():
        n_split_shards = int(np.ceil(len(t_list) / shard_size))
        for sh_idx in range(n_split_shards):
            shard_name = f"{split_name}_{sh_idx:03d}.h5"
            batch = t_list[sh_idx * shard_size : (sh_idx + 1) * shard_size]
            for row_idx, task in enumerate(batch):
                task["shard_name"] = shard_name
                task["row_index"] = row_idx
            shards[shard_name] = batch

    return master_tasks, shards


def simulate_single_task(
    task: dict[str, Any],
    profiles_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate object specimen and execute forward diffraction for a single task."""
    obj_rng = np.random.default_rng(task["object_seed"])
    acq_rng = np.random.default_rng(task["acquisition_seed"])

    pid = task["profile_id"]
    acq_idx = task["acq_index"]

    if profiles_mode == "48":
        cfg, obj_amp, obj_phase, meta = sample_profile_48(
            profile_id=pid,
            index=task["object_index"],
            total=task.get("total_objects", 1000),
            rng=obj_rng,
        )
        fam = meta.get("group_name", "Specimen")
        fwd_model = meta.get("forward_model", "F1")
        subtype = pid
    else:
        cfg, fam, dens, subtype = sample_profile_config(
            profile_id=pid,
            index_within_profile=task["object_index"],
            total_in_profile=task.get("total_objects", 1000),
            rng=obj_rng,
        )
        obj_amp, _ = generate_procedural_object(
            family=fam,
            density_type=dens,
            size=cfg.object_size[0],
            rng=obj_rng,
            shape_subtype=subtype,
        )
        obj_phase = None
        fwd_model = "F1"
        meta = {}

    # Multi-acquisition physical variations
    # acq 0: Reference condition
    # acq 1: Low photon flux / high shot noise
    # acq 2: Coherence damping
    # acq 3: Higher readout noise & beam displacement
    if acq_idx == 1:
        cfg.photon_budget = max(1e4, cfg.photon_budget * 0.1)
    elif acq_idx == 2:
        if not np.isinf(cfg.coherence_length):
            cfg.coherence_length = max(30.0, cfg.coherence_length * 0.6)
    elif acq_idx == 3:
        cfg.readout_std = cfg.readout_std * 1.5
        cfg.beam_center_offset = (
            float(acq_rng.uniform(-2.0, 2.0)),
            float(acq_rng.uniform(-2.0, 2.0)),
        )

    # Forward physical simulation dispatched to profile forward_model
    sim_res = simulate_diffraction(
        obj_amp,
        cfg,
        rng=acq_rng,
        object_phase=obj_phase,
        forward_model=fwd_model,
        metadata=meta,
    )

    sample_dict = {
        "object_amplitude": sim_res.object_amplitude,
        "object_phase": sim_res.object_phase,
        "target_dpr": sim_res.target_dpr,
        "target_physical": sim_res.target_physical,
        "intensity_ideal": sim_res.intensity_ideal,
        "intensity_expected": sim_res.intensity_expected,
        "intensity_raw": sim_res.intensity_raw,
        "intensity_observed": sim_res.intensity_observed,
        "intensity_measured": sim_res.intensity_observed,
        "valid_mask": sim_res.valid_mask,
    }

    scen_phys = get_scenario_physics(task["profile_id"], meta)

    manifest_record = {
        "sample_id": task["sample_id"],
        "base_object_id": task["base_object_id"],
        "scene_id": task["scene_id"],
        "acquisition_id": task["acquisition_id"],
        "group_id": task["group_id"],
        "split": task["split"],
        "profile_id": task["profile_id"],
        "variation_family": task["profile_id"],
        "scenario_id": scen_phys["scenario_id"],
        "scenario_name": scen_phys["scenario_name"],
        "object_family": fam,
        "density_type": meta.get("density_type", "continuous"),
        "forward_model": fwd_model,
        "photon_budget": f"{sim_res.photon_budget:.2e}",
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
        "hit_status": meta.get("hit_status", "single_hit"),
        "random_seed": task["acquisition_seed"],
        "object_seed": task["object_seed"],
        "acquisition_seed": task["acquisition_seed"],
        "photon_energy_kev": f"{scen_phys['photon_energy_kev']:.3f}",
        "wavelength_m": float(scen_phys["wavelength_m"]),
        "pixel_pitch_m": float(scen_phys["pixel_pitch_m"]),
        "detector_distance_m": float(scen_phys["detector_distance_m"]),
        "ground_truth_available": True,
        "shard_name": task.get("shard_name", "shard_000.h5"),
        "row_index": task.get("row_index", 0),
        "simulation_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_compatibility": "DPR_and_Standard_CDI",
    }

    return sample_dict, manifest_record


def generate_dataset(
    output_dir: str,
    preset: str = "quick",
    total_samples: int | None = None,
    profiles_mode: str = "48",  # "48" or "12"
    seed: int = 42,
    batch_size: int = 48,
    shard_size: int = 480,
    resume: bool = True,
    overwrite: bool = False,
    storage_mode: str = "extended",  # "compact" or "extended"
    compression: str = "gzip",
    compression_opts: int = 4,
    max_disk_gb: float | None = None,
    dry_run: bool = False,
    export_cxi: bool = False,
    export_tasks: bool = False,
    progress_callback: Any = None,
    max_shards_to_write: int | None = None,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Execute complete crash-safe dataset generation with immutable shard planning."""
    if total_samples is None:
        total_samples = PRESET_CONFIGS.get(preset, DEFAULT_TARGET_SAMPLES)

    check_disk_space(output_dir, total_samples, storage_mode, max_disk_gb)

    if dry_run:
        bytes_per_sample = (
            BYTES_PER_SAMPLE_COMPACT if storage_mode == "compact" else BYTES_PER_SAMPLE_EXTENDED
        )
        return {
            "status": "dry_run_success",
            "preset": preset,
            "profiles_mode": profiles_mode,
            "total_samples": total_samples,
            "storage_mode": storage_mode,
            "estimated_gb": (total_samples * bytes_per_sample) / (1024**3),
        }

    writer = DatasetWriter(
        output_dir,
        shard_size=shard_size,
        compression=compression,
        compression_opts=compression_opts,
        storage_mode=storage_mode,
    )

    data_dir = os.path.join(output_dir, "data")
    existing_shards = (
        [f for f in os.listdir(data_dir) if f.endswith(".h5")] if os.path.exists(data_dir) else []
    )

    if overwrite:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(writer.metadata_csv):
            os.remove(writer.metadata_csv)
        if os.path.exists(writer.progress_file):
            os.remove(writer.progress_file)
    elif not resume:
        if existing_shards:
            raise RuntimeError(
                f"Output directory {output_dir} already contains {len(existing_shards)} data shards. "
                "Specify resume=True to continue, or overwrite=True to restart cleanly."
            )

    # 1. Build immutable master plan
    acq_per_obj = 1 if (preset == "quick" or total_samples <= 48) else 4
    master_tasks, shards = build_master_task_plan(
        total_samples=total_samples,
        seed=seed,
        profiles_mode=profiles_mode,
        acquisitions_per_object=acq_per_obj,
        shard_size=shard_size,
    )

    config_fingerprint = compute_config_fingerprint(
        {
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "storage_mode": storage_mode,
            "profiles_mode": profiles_mode,
            "seed": seed,
            "total_samples": total_samples,
            "shard_size": shard_size,
        }
    )

    start_time = time.time()
    total_written = 0
    first_shard_path: str | None = None
    skipped_shards = 0
    shards_written = 0

    for shard_name, shard_tasks in shards.items():
        shard_path = os.path.join(data_dir, shard_name)
        if first_shard_path is None:
            first_shard_path = shard_path

        # 2. Check if shard is already complete and verified with matching config
        if resume and writer.verify_shard(
            shard_path, len(shard_tasks), expected_config_hash=config_fingerprint
        ):
            skipped_shards += 1
            continue

        # 3. Stream simulation directly to incremental shard writer with bounded RAM
        inc_writer = writer.create_incremental_shard_writer(
            shard_path,
            total_samples=len(shard_tasks),
            config_fingerprint=config_fingerprint,
            has_phase=True,
        )
        manifest_rows: list[dict[str, Any]] = []

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                sim_outputs = executor.map(
                    lambda t: simulate_single_task(t, profiles_mode), shard_tasks
                )
                for row_idx, (sample_dict, manifest_rec) in enumerate(sim_outputs):
                    inc_writer.write_sample(row_idx, sample_dict)
                    manifest_rows.append(manifest_rec)
        else:
            for row_idx, task in enumerate(tqdm(shard_tasks, desc=f"Simulating {shard_name}")):
                sample_dict, manifest_rec = simulate_single_task(task, profiles_mode)
                inc_writer.write_sample(row_idx, sample_dict)
                manifest_rows.append(manifest_rec)

        # 4. Atomic commit and verified manifest update
        inc_writer.finalize()
        writer.append_manifest_rows(manifest_rows)
        writer.update_progress([t["sample_id"] for t in shard_tasks])
        total_written += len(shard_tasks)
        shards_written += 1

        if progress_callback:
            progress_callback(total_written, total_samples)

        if max_shards_to_write is not None and shards_written >= max_shards_to_write:
            break

    # Optional CXI export
    cxi_path_str: str | None = None
    if export_cxi and first_shard_path and os.path.exists(first_shard_path):
        from opendiffraction.io.cxi import export_to_cxi

        cxi_out = os.path.join(output_dir, "opendiffraction_benchmark.cxi")
        export_to_cxi(first_shard_path, cxi_out)
        cxi_path_str = cxi_out

    # Optional task manifests export
    if export_tasks:
        from opendiffraction.tasks.manifests import TASK_DEFINITIONS, build_task_manifest

        if os.path.exists(writer.metadata_csv):
            df_meta = pd.read_csv(writer.metadata_csv)
            tasks_dir = os.path.join(output_dir, "tasks")
            os.makedirs(tasks_dir, exist_ok=True)
            for t_id in TASK_DEFINITIONS:
                t_df = build_task_manifest(df_meta, t_id)
                t_df.to_csv(os.path.join(tasks_dir, f"{t_id}_manifest.csv"), index=False)

    elapsed = time.time() - start_time
    status_str = "already_completed" if (total_written == 0 and skipped_shards > 0) else "success"
    return {
        "status": status_str,
        "preset": preset,
        "profiles_mode": profiles_mode,
        "total_samples": total_samples,
        "written_samples": total_written,
        "skipped_shards": skipped_shards,
        "elapsed_seconds": elapsed,
        "output_dir": output_dir,
        "cxi_path": cxi_path_str,
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
