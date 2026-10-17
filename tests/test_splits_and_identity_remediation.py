"""Tests for object identity invariance, group-disjoint splits, and deterministic replay."""

import csv
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import pytest

from opendiffraction.generator.pipeline import (
    build_master_task_plan,
    generate_dataset,
    simulate_single_task,
)
from opendiffraction.generator.splits import assign_group_to_split


def test_object_identity_invariance_across_acquisitions():
    """Verify that unscaled object amplitude and phase remain invariant across shots."""
    profiles_to_test = ["V01", "V05", "V14", "V22", "V33", "V41", "V43", "V45", "V47"]
    
    for pid in profiles_to_test:
        base_task = {
            "global_sample_idx": 0,
            "sample_id": f"test_{pid}",
            "base_object_id": f"obj_{pid}_00001",
            "scene_id": f"scene_{pid}_00001",
            "group_id": f"obj_{pid}_00001",
            "split": "train",
            "profile_id": pid,
            "object_index": 0,
            "object_seed": 777000,
            "total_objects": len(profiles_to_test),
        }
        
        amps = []
        phases = []
        for acq_idx in range(4):
            task = dict(base_task)
            task["acquisition_id"] = f"acq_{acq_idx:02d}"
            task["acq_index"] = acq_idx
            task["acquisition_seed"] = base_task["object_seed"] + acq_idx * 101
            sample_dict, _ = simulate_single_task(task, profiles_mode="48")
            amps.append(sample_dict["object_amplitude"])
            phases.append(sample_dict["object_phase"])
        
        for i in range(1, 4):
            np.testing.assert_array_equal(
                amps[i],
                amps[0],
                err_msg=f"Object amplitude variance detected across acquisitions for {pid}",
            )
            if phases[0] is not None:
                np.testing.assert_array_equal(
                    phases[i],
                    phases[0],
                    err_msg=f"Object phase variance detected across acquisitions for {pid}",
                )


def test_group_disjoint_splitting_zero_leakage():
    """Verify train, valid, test, and test_ood splits contain zero overlapping objects."""
    master_tasks, _ = build_master_task_plan(
        total_samples=192,
        seed=42,
        profiles_mode="48",
        acquisitions_per_object=4,
    )
    
    split_objects: dict[str, set[str]] = defaultdict(set)
    for task in master_tasks:
        split_objects[task["split"]].add(task["base_object_id"])
    
    splits = list(split_objects.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_objects[s1].intersection(split_objects[s2])
            assert len(overlap) == 0, f"Object leakage detected between {s1} and {s2}: {overlap}"


def test_ood_profile_assignment_isolation():
    """Verify that OOD profiles are strictly partitioned into test_ood."""
    master_tasks, _ = build_master_task_plan(
        total_samples=192,
        seed=42,
        profiles_mode="48",
        acquisitions_per_object=4,
    )
    
    for task in master_tasks:
        if task["is_ood"]:
            assert task["split"] == "test_ood", (
                f"OOD profile {task['profile_id']} assigned to non-OOD split {task['split']}"
            )
        else:
            assert task["split"] in ("train", "valid", "test"), (
                f"In-distribution profile {task['profile_id']} assigned to {task['split']}"
            )


def test_deterministic_replay_from_manifest():
    """Verify exact bitwise replay of simulated samples from manifest seeds against HDF5."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        generate_dataset(
            output_dir=str(tmp_dir),
            total_samples=16,
            seed=999,
            profiles_mode="48",
            shard_size=8,
            storage_mode="extended",
            num_workers=1,
        )
        
        meta_path = tmp_dir / "metadata.csv"
        with open(meta_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 16
        
        # Test replay on a subset of rows
        for test_idx in [0, 3, 7, 11, 15]:
            row = rows[test_idx]
            shard_name = row["shard_name"]
            row_idx = int(row["row_index"])
            
            with h5py.File(tmp_dir / "data" / shard_name, "r") as h5f:
                stored_obs = h5f["intensity_observed"][row_idx, 0]
                stored_raw = h5f["intensity_raw"][row_idx, 0]
                stored_mask = h5f["valid_mask"][row_idx, 0]
                stored_obj = h5f["object_amplitude"][row_idx, 0]
            
            acq_idx = int(row["acquisition_id"].split("_")[-1])
            task_dict = {
                "global_sample_idx": test_idx,
                "sample_id": row["sample_id"],
                "base_object_id": row["base_object_id"],
                "scene_id": row["scene_id"],
                "acquisition_id": row["acquisition_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "profile_id": row["profile_id"],
                "object_index": test_idx // 4,
                "acq_index": acq_idx,
                "object_seed": int(row["object_seed"]),
                "acquisition_seed": int(row["acquisition_seed"]),
                "total_objects": 4,
            }
            
            replayed_dict, _ = simulate_single_task(task_dict, profiles_mode="48")
            
            np.testing.assert_array_equal(replayed_dict["intensity_observed"], stored_obs)
            np.testing.assert_array_equal(replayed_dict["intensity_raw"], stored_raw)
            np.testing.assert_array_equal(replayed_dict["valid_mask"], stored_mask)
            np.testing.assert_array_equal(replayed_dict["object_amplitude"], stored_obj)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
