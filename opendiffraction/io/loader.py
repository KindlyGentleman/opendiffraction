"""Unified dataset loader for OpenDiffraction and DPR-compatible workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from typing import Any
import h5py
import numpy as np


@dataclass
class DiffractionSample:
    """A single loaded diffraction sample."""

    sample_id: str
    input: np.ndarray  # (1, 512, 512) float32 unmasked detector readout
    target: np.ndarray  # (1, 64, 64) float32 DPR scaled object
    mask: np.ndarray  # (1, 512, 512) bool valid pixel mask
    object_amplitude: np.ndarray  # (1, 64, 64) float32 unscaled true amplitude
    intensity_ideal: np.ndarray  # (1, 512, 512) float32 coherent intensity
    intensity_expected: np.ndarray  # (1, 512, 512) float32 expected photon rate
    intensity_measured: np.ndarray  # (1, 512, 512) float32 signed readout
    valid_mask: np.ndarray  # (1, 512, 512) bool valid pixel mask
    object_phase: np.ndarray | None  # (1, 64, 64) float32 or None
    metadata: dict[str, Any]


class OpenDiffractionDataset:
    """High-level reader for OpenDiffraction datasets and manifests."""

    def __init__(
        self,
        dataset_dir: str,
        split: str | None = None,
        variation: str | None = None,
    ) -> None:
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.metadata_csv = os.path.join(self.dataset_dir, "metadata.csv")

        self.records: list[dict[str, Any]] = []
        if os.path.exists(self.metadata_csv):
            with open(self.metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if split is not None and r.get("split") != split:
                        continue
                    if variation is not None and r.get("variation_family") != variation:
                        continue
                    self.records.append(r)
        else:
            # Fallback: scan data directory directly for .h5 files
            if os.path.exists(self.data_dir):
                h5_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".h5")])
                for fname in h5_files:
                    path = os.path.join(self.data_dir, fname)
                    with h5py.File(path, "r") as f:
                        n_items = len(f["input"])
                        sp = fname.split("_")[0]
                        for idx in range(n_items):
                            if split is not None and sp != split:
                                continue
                            self.records.append(
                                {
                                    "sample_id": f"{fname}_{idx}",
                                    "shard_name": fname,
                                    "row_index": idx,
                                    "split": sp,
                                }
                            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DiffractionSample:
        if idx < 0 or idx >= len(self.records):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.records)}")

        rec = self.records[idx]
        shard_path = os.path.join(self.data_dir, rec["shard_name"])
        row = int(rec["row_index"])

        with h5py.File(shard_path, "r") as f:
            inp = f["input"][row]
            tgt = f["target"][row]
            msk = f["mask"][row]
            obj_amp = f["object_amplitude"][row]
            i_ideal = f["intensity_ideal"][row]
            i_exp = f["intensity_expected"][row]
            i_meas = f["intensity_measured"][row]
            v_msk = f["valid_mask"][row]
            obj_phase = f["object_phase"][row] if "object_phase" in f else None

        return DiffractionSample(
            sample_id=rec.get("sample_id", f"sample_{idx}"),
            input=inp,
            target=tgt,
            mask=msk,
            object_amplitude=obj_amp,
            intensity_ideal=i_ideal,
            intensity_expected=i_exp,
            intensity_measured=i_meas,
            valid_mask=v_msk,
            object_phase=obj_phase,
            metadata=rec,
        )
