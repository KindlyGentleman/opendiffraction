"""HDF5 storage, chunking, atomic writes, manifest tracking, and resume management."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from typing import Any
import h5py
import numpy as np


class DatasetWriter:
    """Manages compressed HDF5 shards and metadata manifests with atomic write semantics."""

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 480,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.shard_size = shard_size
        self.compression = compression
        self.compression_opts = compression_opts

        os.makedirs(self.output_dir, exist_ok=True)
        self.progress_file = os.path.join(self.output_dir, "progress.json")
        self.metadata_csv = os.path.join(self.output_dir, "metadata.csv")

    def get_completed_samples(self) -> set[str]:
        """Read list of already completed sample IDs from progress file."""
        if not os.path.exists(self.progress_file):
            return set()
        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                return set(data.get("completed_sample_ids", []))
        except Exception:
            return set()

    def update_progress(self, new_sample_ids: list[str]) -> None:
        """Atomically update progress manifest."""
        completed = self.get_completed_samples()
        completed.update(new_sample_ids)
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump({"completed_sample_ids": sorted(list(completed))}, f, indent=2)
        shutil.move(temp_file, self.progress_file)

    def write_shard_atomic(
        self,
        shard_path: str,
        samples: list[dict[str, Any]],
    ) -> None:
        """Write a batch of samples to an HDF5 shard atomically.

        Uses temporary file and renames only after successful completion.
        """
        N = len(samples)
        if N == 0:
            return

        tmp_path = shard_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        with h5py.File(tmp_path, "w") as f:
            # Create DPR-compatible datasets
            ds_input = f.create_dataset(
                "input",
                shape=(N, 1, 512, 512),
                dtype=np.float32,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_target = f.create_dataset(
                "target",
                shape=(N, 1, 64, 64),
                dtype=np.float32,
                chunks=(1, 1, 64, 64),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_mask = f.create_dataset(
                "mask",
                shape=(N, 1, 512, 512),
                dtype=bool,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # Extended scientific datasets
            ds_obj_amp = f.create_dataset(
                "object_amplitude",
                shape=(N, 1, 64, 64),
                dtype=np.float32,
                chunks=(1, 1, 64, 64),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_inten_ideal = f.create_dataset(
                "intensity_ideal",
                shape=(N, 1, 512, 512),
                dtype=np.float32,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_inten_exp = f.create_dataset(
                "intensity_expected",
                shape=(N, 1, 512, 512),
                dtype=np.float32,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_inten_meas = f.create_dataset(
                "intensity_measured",
                shape=(N, 1, 512, 512),
                dtype=np.float32,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            ds_valid_mask = f.create_dataset(
                "valid_mask",
                shape=(N, 1, 512, 512),
                dtype=bool,
                chunks=(1, 1, 512, 512),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            has_phase = any(s.get("object_phase") is not None for s in samples)
            if has_phase:
                ds_phase = f.create_dataset(
                    "object_phase",
                    shape=(N, 1, 64, 64),
                    dtype=np.float32,
                    chunks=(1, 1, 64, 64),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            # Write records
            for i, s in enumerate(samples):
                # DPR fields
                # Note: 'input' in DPR is the unmasked noisy measurement; mask is applied by consumer
                meas = s["intensity_measured"]
                target = s["target_dpr"]
                mask = s["valid_mask"]

                ds_input[i, 0] = meas
                ds_target[i, 0] = target
                ds_mask[i, 0] = mask

                # Scientific fields
                ds_obj_amp[i, 0] = s["object_amplitude"]
                ds_inten_ideal[i, 0] = s["intensity_ideal"]
                ds_inten_exp[i, 0] = s["intensity_expected"]
                ds_inten_meas[i, 0] = meas
                ds_valid_mask[i, 0] = mask

                if has_phase and s.get("object_phase") is not None:
                    ds_phase[i, 0] = s["object_phase"]

        # Rename atomically
        shutil.move(tmp_path, shard_path)

    def append_manifest_rows(self, rows: list[dict[str, Any]]) -> None:
        """Append metadata records to metadata.csv."""
        if not rows:
            return

        file_exists = os.path.exists(self.metadata_csv)
        fieldnames = [
            "sample_id",
            "source_object_id",
            "group_id",
            "split",
            "variation_family",
            "object_family",
            "density_type",
            "photon_budget",
            "coherence_length",
            "readout_noise_std",
            "beam_stop_geometry",
            "missing_pixel_fraction",
            "detector_saturation_threshold",
            "beam_center_offset",
            "random_seed",
            "shard_name",
            "row_index",
            "simulation_version",
            "model_compatibility",
        ]

        with open(self.metadata_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in rows:
                filtered_row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(filtered_row)
