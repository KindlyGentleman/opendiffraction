"""Tests for official CXI NeXus format exporter and validator."""

import os
import tempfile
import h5py
import numpy as np
import pytest

from opendiffraction.io.cxi import export_to_cxi, validate_cxi


def test_cxi_export_and_validation():
    """Test exporting standard OpenDiffraction HDF5 file to official CXI format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_h5 = os.path.join(tmpdir, "test_source.h5")
        out_cxi = os.path.join(tmpdir, "test_output.cxi")

        # Create dummy source HDF5 conforming to OpenDiffraction schema
        n_samples = 4
        inputs = np.ones((n_samples, 512, 512), dtype=np.float32) * 50.0
        targets = np.ones((n_samples, 64, 64), dtype=np.float32) * 0.8
        masks = np.ones((n_samples, 512, 512), dtype=np.uint8)

        with h5py.File(src_h5, "w") as f:
            f.create_dataset("input", data=inputs)
            f.create_dataset("target", data=targets)
            f.create_dataset("mask", data=masks)

        # Export to CXI
        cxi_path = export_to_cxi(src_h5, out_cxi, scenario_metadata={"sample_name": "Gold Nanoparticle"})
        assert os.path.exists(cxi_path)

        # Validate with official compliance auditor
        is_valid, msgs = validate_cxi(cxi_path)
        assert is_valid, f"CXI validation failed: {msgs}"

        # Verify NeXus layout
        with h5py.File(cxi_path, "r") as f:
            assert "entry_1" in f
            entry = f["entry_1"]
            assert entry.attrs["NX_class"] == "NXentry"
            assert "data_1" in entry
            assert "data" in entry["data_1"]
            assert "mask" in entry["data_1"]
            assert entry["data_1"]["data"].shape == (n_samples, 512, 512)
            assert "instrument_1" in entry
            assert "image_1" in entry
            assert entry["image_1"]["data"].shape == (n_samples, 64, 64)
