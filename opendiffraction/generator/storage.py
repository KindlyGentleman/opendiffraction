"""HDF5 storage, chunking, atomic writes, manifest tracking, and resume management."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from typing import Any
import h5py
import numpy as np

from opendiffraction.version import GENERATOR_VERSION, SCHEMA_VERSION


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DatasetWriter:
    """Manages compressed HDF5 shards and metadata manifests with atomic write semantics."""

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 480,
        compression: str = "gzip",
        compression_opts: int = 4,
        storage_mode: str = "extended",  # "compact" or "extended"
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.shard_size = shard_size
        self.compression = compression
        self.compression_opts = compression_opts
        self.storage_mode = storage_mode

        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.progress_file = os.path.join(self.output_dir, "progress.json")
        self.metadata_csv = os.path.join(self.output_dir, "metadata.csv")
        self.manifest_dir = os.path.join(self.output_dir, "manifests")
        os.makedirs(self.manifest_dir, exist_ok=True)

    def get_completed_samples(self) -> set[str]:
        """Read list of already completed sample IDs from progress file."""
        if not os.path.exists(self.progress_file):
            return set()
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("completed_sample_ids", []))
        except Exception:
            return set()

    def update_progress(self, new_sample_ids: list[str]) -> None:
        """Atomically update progress manifest."""
        completed = self.get_completed_samples()
        completed.update(new_sample_ids)
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "completed_sample_ids": sorted(list(completed)),
                    "total_completed": len(completed),
                    "generator_version": GENERATOR_VERSION,
                    "schema_version": SCHEMA_VERSION,
                },
                f,
                indent=2,
            )
        os.replace(temp_file, self.progress_file)

    def verify_shard(self, shard_path: str, expected_count: int) -> bool:
        """Verify that an HDF5 shard exists, is uncorrupted, and has expected record count."""
        if not os.path.exists(shard_path):
            return False
        try:
            with h5py.File(shard_path, "r") as f:
                if "input" not in f or "target" not in f or "mask" not in f:
                    return False
                if len(f["input"]) != expected_count:
                    return False
                if self.storage_mode == "extended":
                    if "object_amplitude" not in f or "intensity_measured" not in f:
                        return False
                # Try reading first and last slice to verify dataset readability
                _ = f["input"][0]
                _ = f["input"][-1]
            return True
        except Exception:
            return False

    def write_shard_atomic(
        self,
        shard_path: str,
        samples: list[dict[str, Any]],
    ) -> str:
        """Write a batch of samples to an HDF5 shard atomically.

        Uses temporary file and atomic replace only after successful completion.
        Returns SHA-256 checksum of the written shard.
        """
        N = len(samples)
        if N == 0:
            return ""

        tmp_path = shard_path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # Select compression parameters
        comp = None if self.compression in (None, "none") else self.compression
        comp_opts = None if comp is None else self.compression_opts

        with h5py.File(tmp_path, "w") as f:
            # Standard training datasets (always present)
            ds_input = f.create_dataset(
                "input",
                shape=(N, 1, 512, 512),
                dtype=np.float32,
                chunks=(1, 1, 512, 512),
                compression=comp,
                compression_opts=comp_opts,
            )
            ds_target = f.create_dataset(
                "target",
                shape=(N, 1, 64, 64),
                dtype=np.float32,
                chunks=(1, 1, 64, 64),
                compression=comp,
                compression_opts=comp_opts,
            )
            ds_mask = f.create_dataset(
                "mask",
                shape=(N, 1, 512, 512),
                dtype=bool,
                chunks=(1, 1, 512, 512),
                compression=comp,
                compression_opts=comp_opts,
            )

            # Extended scientific datasets
            has_phase = any(s.get("object_phase") is not None for s in samples)
            if self.storage_mode == "extended":
                ds_obj_amp = f.create_dataset(
                    "object_amplitude",
                    shape=(N, 1, 64, 64),
                    dtype=np.float32,
                    chunks=(1, 1, 64, 64),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                ds_tgt_phys = f.create_dataset(
                    "target_physical",
                    shape=(N, 1, 64, 64),
                    dtype=np.float32,
                    chunks=(1, 1, 64, 64),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                ds_inten_ideal = f.create_dataset(
                    "intensity_ideal",
                    shape=(N, 1, 512, 512),
                    dtype=np.float32,
                    chunks=(1, 1, 512, 512),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                ds_inten_exp = f.create_dataset(
                    "intensity_expected",
                    shape=(N, 1, 512, 512),
                    dtype=np.float32,
                    chunks=(1, 1, 512, 512),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                ds_inten_meas = f.create_dataset(
                    "intensity_measured",
                    shape=(N, 1, 512, 512),
                    dtype=np.float32,
                    chunks=(1, 1, 512, 512),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                ds_valid_mask = f.create_dataset(
                    "valid_mask",
                    shape=(N, 1, 512, 512),
                    dtype=bool,
                    chunks=(1, 1, 512, 512),
                    compression=comp,
                    compression_opts=comp_opts,
                )
                if has_phase:
                    ds_phase = f.create_dataset(
                        "object_phase",
                        shape=(N, 1, 64, 64),
                        dtype=np.float32,
                        chunks=(1, 1, 64, 64),
                        compression=comp,
                        compression_opts=comp_opts,
                    )

            # Populate datasets
            for i, s in enumerate(samples):
                meas = s["intensity_measured"]
                target = s["target_dpr"]
                mask = s["valid_mask"]

                ds_input[i, 0] = meas
                ds_target[i, 0] = target
                ds_mask[i, 0] = mask

                if self.storage_mode == "extended":
                    ds_obj_amp[i, 0] = s["object_amplitude"]
                    ds_tgt_phys[i, 0] = s.get("target_physical", s["target_dpr"])
                    ds_inten_ideal[i, 0] = s["intensity_ideal"]
                    ds_inten_exp[i, 0] = s["intensity_expected"]
                    ds_inten_meas[i, 0] = meas
                    ds_valid_mask[i, 0] = mask

                    if has_phase and s.get("object_phase") is not None:
                        ds_phase[i, 0] = s["object_phase"]

        # Atomic commit
        os.replace(tmp_path, shard_path)
        return compute_file_sha256(shard_path)

    def append_manifest_rows(self, rows: list[dict[str, Any]]) -> None:
        """Append metadata records to metadata.csv without duplicating existing sample_id records."""
        if not rows:
            return

        existing_ids = set()
        file_exists = os.path.exists(self.metadata_csv)

        if file_exists:
            try:
                with open(self.metadata_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        sid = r.get("sample_id")
                        if sid:
                            existing_ids.add(sid)
            except Exception:
                pass

        # Filter out rows that are already recorded
        new_rows = [r for r in rows if r.get("sample_id") not in existing_ids]
        if not new_rows:
            return

        fieldnames = [
            "sample_id",
            "base_object_id",
            "scene_id",
            "acquisition_id",
            "group_id",
            "split",
            "profile_id",
            "variation_family",
            "object_family",
            "density_type",
            "forward_model",
            "photon_budget",
            "coherence_length",
            "readout_noise_std",
            "beam_stop_geometry",
            "missing_pixel_fraction",
            "detector_saturation_threshold",
            "beam_center_offset",
            "hit_status",
            "random_seed",
            "object_seed",
            "acquisition_seed",
            "wavelength_m",
            "pixel_pitch_m",
            "detector_distance_m",
            "ground_truth_available",
            "shard_name",
            "row_index",
            "simulation_version",
            "schema_version",
            "model_compatibility",
        ]

        # Write to temporary file first if creating or updating
        temp_csv = self.metadata_csv + ".tmp"
        if file_exists:
            import shutil
            shutil.copyfile(self.metadata_csv, temp_csv)
            mode = "a"
        else:
            mode = "w"

        with open(temp_csv, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for r in new_rows:
                writer.writerow(r)

        os.replace(temp_csv, self.metadata_csv)
