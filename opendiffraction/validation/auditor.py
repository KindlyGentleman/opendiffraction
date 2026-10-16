"""Scientific validation and quality audit engine for OpenDiffraction datasets."""

from __future__ import annotations

import csv
import os
from typing import Any
import h5py
import numpy as np


class DatasetAuditor:
    """Executes mandatory physical, numerical, and split integrity audits."""

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.metadata_csv = os.path.join(self.dataset_dir, "metadata.csv")

    def run_full_audit(self) -> dict[str, Any]:
        """Run all verification tests and return structured audit report."""
        results: dict[str, Any] = {
            "dataset_dir": self.dataset_dir,
            "shards_found": 0,
            "total_samples": 0,
            "checks": {},
            "all_passed": True,
        }

        if not os.path.exists(self.data_dir):
            results["checks"]["data_directory_exists"] = {
                "passed": False,
                "error": f"Data directory {self.data_dir} does not exist.",
            }
            results["all_passed"] = False
            return results

        shards = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".h5")])
        results["shards_found"] = len(shards)
        if len(shards) == 0:
            results["checks"]["shards_present"] = {
                "passed": False,
                "error": "No HDF5 shards found.",
            }
            results["all_passed"] = False
            return results

        # 1. Inspect shards: dimensions, types, finiteness
        dim_passed = True
        finite_passed = True
        mask_passed = True
        total_samples = 0

        for sname in shards:
            spath = os.path.join(self.data_dir, sname)
            with h5py.File(spath, "r") as f:
                n = len(f["input"])
                total_samples += n

                # Shape check
                if f["input"].shape != (n, 1, 512, 512):
                    dim_passed = False
                if f["target"].shape != (n, 1, 64, 64):
                    dim_passed = False
                if f["mask"].shape != (n, 1, 512, 512):
                    dim_passed = False

                # Finiteness check
                inp_sample = f["input"][:]
                if not np.all(np.isfinite(inp_sample)):
                    finite_passed = False

                # Mask boolean check
                msk_sample = f["mask"][:]
                if msk_sample.dtype != bool:
                    mask_passed = False

        results["total_samples"] = total_samples
        results["checks"]["dimensions_and_shapes"] = {"passed": dim_passed}
        results["checks"]["finite_values"] = {"passed": finite_passed}
        results["checks"]["mask_boolean_semantics"] = {"passed": mask_passed}

        # 2. Check metadata manifest and split disjointness
        split_passed = True
        leakage_passed = True
        duplicates_detected = 0

        if os.path.exists(self.metadata_csv):
            rows = []
            with open(self.metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            train_groups = set()
            valid_groups = set()
            test_groups = set()

            for r in rows:
                gid = r.get("group_id")
                sp = r.get("split")
                if sp == "train":
                    train_groups.add(gid)
                elif sp == "valid":
                    valid_groups.add(gid)
                elif sp in ("test", "test_ood"):
                    test_groups.add(gid)

            # Assert empty intersections
            tv_inter = train_groups.intersection(valid_groups)
            tt_inter = train_groups.intersection(test_groups)
            vt_inter = valid_groups.intersection(test_groups)

            if len(tv_inter) > 0 or len(tt_inter) > 0 or len(vt_inter) > 0:
                leakage_passed = False

            results["checks"]["split_disjointness_no_leakage"] = {
                "passed": leakage_passed,
                "train_groups": len(train_groups),
                "valid_groups": len(valid_groups),
                "test_groups": len(test_groups),
                "leakage_count": len(tv_inter) + len(tt_inter) + len(vt_inter),
            }
        else:
            results["checks"]["metadata_manifest_present"] = {"passed": False}
            leakage_passed = False

        # 3. Physics checks: Nonnegativity of ideal intensity and Friedel symmetry
        nonneg_passed = True
        friedel_passed = True

        # Find an unshifted sample (beam_center_offset == "(0.0,0.0)") for Friedel symmetry
        target_shard = shards[0]
        target_row = 0
        if os.path.exists(self.metadata_csv):
            for r in rows:
                if r.get("beam_center_offset") == "(0.0,0.0)" and r.get("shard_name") in shards:
                    target_shard = r["shard_name"]
                    target_row = int(r["row_index"])
                    break

        shard_path = os.path.join(self.data_dir, target_shard)
        with h5py.File(shard_path, "r") as f:
            if "intensity_ideal" in f:
                i_ideal = f["intensity_ideal"][:]
                if np.any(i_ideal < -1e-6):
                    nonneg_passed = False

                sample = i_ideal[target_row, 0]
                # Center is at pixel (256, 256). An odd-length window centered at 256
                # inverts perfectly under np.flip (which maps index 256 to 256).
                K = 32
                win = sample[256 - K : 256 + K + 1, 256 - K : 256 + K + 1]
                win_flipped = np.flip(win)
                friedel_err = float(np.mean(np.abs(win - win_flipped)) / max(np.mean(win), 1e-6))
                if friedel_err > 1e-4:
                    friedel_passed = False

        results["checks"]["nonnegative_ideal_intensity"] = {"passed": nonneg_passed}
        results["checks"]["friedel_symmetry"] = {"passed": friedel_passed}

        # Overall status
        all_checks_passed = all(
            c.get("passed", False) for c in results["checks"].values()
        )
        results["all_passed"] = all_checks_passed

        return results
