"""PyTorch Dataset adapter for OpenDiffraction with worker-safe lazy HDF5 loading."""

from __future__ import annotations

import csv
import os
from typing import Any, Callable
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from opendiffraction.version import GENERATOR_VERSION, SCHEMA_VERSION


class OpenDiffractionTorchDataset(Dataset):
    """PyTorch Dataset for OpenDiffraction benchmark tasks with multiprocess safety.

    Parameters
    ----------
    root : str
        Root directory of the OpenDiffraction dataset (containing data/ and metadata.csv).
    split : str or None
        One of 'train', 'valid', 'test', 'test_ood', or None for all splits.
    task : str
        Target ML task: 'phase_retrieval', 'denoising', 'inpainting', 'hit_classification', or 'raw'.
    transform : callable or None
        Optional transform applied to the input tensor.
    target_transform : callable or None
        Optional transform applied to the target tensor.
    family : str or None
        Optional object family filter (e.g. 'Specimen', 'Synthetic').
    variation : str or None
        Optional profile filter (e.g. 'V01', 'V05').
    return_dict : bool
        If True, returns a dictionary of named tensors instead of a standard tuple.
    """

    def __init__(
        self,
        root: str,
        split: str | None = "train",
        task: str = "phase_retrieval",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        family: str | None = None,
        variation: str | None = None,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        self.family = family
        self.variation = variation
        self.return_dict = return_dict

        self.data_dir = os.path.join(self.root, "data")
        self.metadata_csv = os.path.join(self.root, "metadata.csv")

        # Worker-safe lazy file handle cache: keyed by PID and shard path
        self._file_handles: dict[str, h5py.File] = {}
        self._pid: int | None = None

        self.records: list[dict[str, Any]] = []
        self._load_index()

    def _load_index(self) -> None:
        """Load and filter sample records from metadata.csv or data directory."""
        if os.path.exists(self.metadata_csv):
            with open(self.metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if self.split is not None and r.get("split") != self.split:
                        continue
                    if self.family is not None and r.get("object_family") != self.family:
                        continue
                    if self.variation is not None and r.get("profile_id") != self.variation:
                        continue
                    self.records.append(r)
        elif os.path.exists(self.data_dir):
            # Fallback: scan shards directly
            h5_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".h5")])
            for fname in h5_files:
                sp = fname.split("_")[0]
                if self.split is not None and sp != self.split:
                    continue
                path = os.path.join(self.data_dir, fname)
                with h5py.File(path, "r") as f:
                    n_samples = len(f["input"])
                for row_idx in range(n_samples):
                    self.records.append(
                        {
                            "sample_id": f"{fname}_{row_idx}",
                            "shard_name": fname,
                            "row_index": row_idx,
                            "split": sp,
                        }
                    )

    def _get_file_handle(self, shard_path: str) -> h5py.File:
        """Safely retrieve or open an HDF5 file handle for the current worker process."""
        current_pid = os.getpid()
        if self._pid != current_pid:
            # Forked or spawned worker process detected: clear old file handles
            self._pid = current_pid
            self._file_handles = {}

        if shard_path not in self._file_handles:
            self._file_handles[shard_path] = h5py.File(shard_path, "r")
        return self._file_handles[shard_path]

    def close(self) -> None:
        """Close all cached file handles in the current process."""
        if hasattr(self, "_file_handles"):
            for handle in list(self._file_handles.values()):
                try:
                    handle.close()
                except Exception:
                    pass
            self._file_handles.clear()

    def __enter__(self) -> OpenDiffractionTorchDataset:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self.records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.records)}")

        rec = self.records[idx]
        shard_path = os.path.join(self.data_dir, rec["shard_name"])
        row = int(rec["row_index"])

        h5_file = self._get_file_handle(shard_path)

        # Base modalities
        inp_arr = h5_file["input"][row]  # (1, 512, 512) float32
        tgt_arr = h5_file["target"][row]  # (1, 64, 64) float32
        msk_arr = h5_file["mask"][row]  # (1, 512, 512) bool

        inp_t = torch.from_numpy(inp_arr)
        tgt_t = torch.from_numpy(tgt_arr)
        msk_t = torch.from_numpy(msk_arr)

        if self.transform is not None:
            inp_t = self.transform(inp_t)
        if self.target_transform is not None:
            tgt_t = self.target_transform(tgt_t)

        # Task-specific target selection
        if self.task == "denoising":
            if "intensity_expected" in h5_file:
                clean_arr = h5_file["intensity_expected"][row]
            elif "intensity_ideal" in h5_file:
                clean_arr = h5_file["intensity_ideal"][row]
            else:
                clean_arr = inp_arr
            task_target_t = torch.from_numpy(clean_arr)
        elif self.task == "inpainting":
            if "intensity_expected" in h5_file:
                clean_arr = h5_file["intensity_expected"][row]
            else:
                clean_arr = inp_arr
            task_target_t = torch.from_numpy(clean_arr)
        elif self.task == "hit_classification":
            hit_str = rec.get("hit_status", "single_hit")
            hit_map = {"single_hit": 0, "multi_hit": 1, "multi_particle_hit": 1, "off_center_hit": 0, "non_hit": 2, "blank": 2}
            task_target_t = torch.tensor(hit_map.get(hit_str, 0), dtype=torch.long)
        else:
            task_target_t = tgt_t

        if self.return_dict:
            return {
                "sample_id": rec.get("sample_id", f"sample_{idx}"),
                "input": inp_t,
                "target": task_target_t,
                "mask": msk_t,
                "split": rec.get("split", ""),
                "profile_id": rec.get("profile_id", ""),
            }

        # Task-specific standard tuples
        if self.task == "phase_retrieval":
            return inp_t, task_target_t, msk_t
        elif self.task == "denoising":
            return inp_t, task_target_t
        elif self.task == "inpainting":
            return inp_t, task_target_t, msk_t
        elif self.task == "hit_classification":
            return inp_t, task_target_t

        # Default fallback
        return inp_t, task_target_t, msk_t
