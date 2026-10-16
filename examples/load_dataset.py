"""Example: Load and inspect OpenDiffraction HDF5 and CXI datasets."""

import os
import h5py
import numpy as np


def load_hdf5_shard(file_path: str) -> dict[str, np.ndarray]:
    """Read standard OpenDiffraction HDF5 shard."""
    print(f"Loading HDF5: {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {
            "input": f["input"][:],
            "target": f["target"][:],
            "mask": f["mask"][:],
        }
    print(f"  Inputs shape: {data['input'].shape}, dtype: {data['input'].dtype}")
    print(f"  Targets shape: {data['target'].shape}, dtype: {data['target'].dtype}")
    print(f"  Masks shape: {data['mask'].shape}, dtype: {data['mask'].dtype}")
    return data


def load_cxi_file(file_path: str) -> None:
    """Inspect official CXI NeXus diffraction file."""
    print(f"\nLoading CXI NeXus: {file_path}")
    with h5py.File(file_path, "r") as f:
        print("  Root attributes:", dict(f.attrs))
        if "entry_1" in f:
            entry = f["entry_1"]
            print("  NXentry keys:", list(entry.keys()))
            if "data_1" in entry:
                d = entry["data_1/data"]
                m = entry["data_1/mask"]
                print(f"  Diffraction data: {d.shape}, units: {d.attrs.get('units', 'counts')}")
                print(f"  Detector mask: {m.shape}")


if __name__ == "__main__":
    sample_h5 = "output_quick/data/train_000.h5"
    sample_cxi = "output_quick/opendiffraction_benchmark.cxi"

    if os.path.exists(sample_h5):
        load_hdf5_shard(sample_h5)
    if os.path.exists(sample_cxi):
        load_cxi_file(sample_cxi)
