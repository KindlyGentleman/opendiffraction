"""Example: PyTorch Dataset & DataLoader integration for OpenDiffraction."""

from __future__ import annotations

import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OpenDiffractionDataset(Dataset):
    """PyTorch Dataset for OpenDiffraction HDF5 shards.

    Compatible with Deep Phase Retrieval (DPR) network training.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        apply_log1p: bool = True,
        mask_input: bool = True,
    ):
        self.shard_paths = [str(p) for p in shard_paths]
        self.apply_log1p = apply_log1p
        self.mask_input = mask_input

        # Build index mapping (shard_idx, local_row)
        self.index_map = []
        for s_idx, path in enumerate(self.shard_paths):
            with h5py.File(path, "r") as f:
                n = f["input"].shape[0]
                for r in range(n):
                    self.index_map.append((s_idx, r))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s_idx, r_idx = self.index_map[idx]
        with h5py.File(self.shard_paths[s_idx], "r") as f:
            inp = f["input"][r_idx]    # (1, 512, 512) or (512, 512)
            tgt = f["target"][r_idx]   # (1, 64, 64) or (64, 64)
            msk = f["mask"][r_idx]     # (1, 512, 512) or (512, 512)

        if inp.ndim == 2:
            inp = inp[None, :, :]
        if tgt.ndim == 2:
            tgt = tgt[None, :, :]
        if msk.ndim == 2:
            msk = msk[None, :, :]

        msk_bool = msk.astype(bool)
        if self.mask_input:
            inp = np.where(msk_bool, inp, 0.0)

        if self.apply_log1p:
            inp = np.log1p(np.maximum(inp, 0.0))

        return {
            "input": torch.from_numpy(inp.astype(np.float32)),
            "target": torch.from_numpy(tgt.astype(np.float32)),
            "mask": torch.from_numpy(msk_bool),
        }


def create_dataloader(
    shard_paths: list[str | Path],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Create PyTorch DataLoader for OpenDiffraction."""
    ds = OpenDiffractionDataset(shard_paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    shard = "output_quick/data/train_000.h5"
    if os.path.exists(shard):
        loader = create_dataloader([shard], batch_size=4)
        batch = next(iter(loader))
        print("Batch loaded successfully:")
        print("  input:", batch["input"].shape)
        print("  target:", batch["target"].shape)
        print("  mask:", batch["mask"].shape)
