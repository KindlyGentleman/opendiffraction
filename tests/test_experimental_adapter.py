"""Unit tests for the experimental CXI adapter."""

from __future__ import annotations

import os
import h5py
import numpy as np
import pytest

from opendiffraction.io.experimental import CXIExperimentalAdapter


def test_experimental_adapter_on_cxi_benchmark() -> None:
    cxi_path = "output_v3/opendiffraction_benchmark.cxi"
    if not os.path.exists(cxi_path):
        pytest.skip("Benchmark CXI file output_v3/opendiffraction_benchmark.cxi not found")

    adapter = CXIExperimentalAdapter(cxi_path)
    assert len(adapter) > 0, "Adapter reported 0 frames"

    frame = adapter.get_frame(0, target_size=(512, 512))
    assert frame.intensity.shape == (512, 512)
    assert frame.valid_mask.shape == (512, 512)
    assert np.isfinite(frame.intensity).all()
    assert frame.source_shape == (512, 512)
    assert frame.photon_energy_ev is not None
    assert frame.wavelength_m is not None


def test_experimental_adapter_geometry_crop_and_pad(tmp_path) -> None:
    # 1. Create a 1024x1024 mock experimental CXI file
    mock_cxi = tmp_path / "mock_1024.cxi"
    with h5py.File(mock_cxi, "w") as f:
        f.attrs["cxi_version"] = 140
        entry = f.create_group("entry_1")
        entry.attrs["NX_class"] = "NXentry"
        data_grp = entry.create_group("data_1")
        data_grp.attrs["NX_class"] = "NXdata"

        data = np.ones((2, 1024, 1024), dtype=np.float32) * 42.0
        mask = np.ones((2, 1024, 1024), dtype=np.int32)
        mask[:, 510:514, 510:514] = 0  # beam stop

        data_grp.create_dataset("data", data=data)
        data_grp.create_dataset("mask", data=mask)

        # Source energy
        inst = entry.create_group("instrument_1")
        src = inst.create_group("source_1")
        src.create_dataset("energy", data=9500.0)

    adapter = CXIExperimentalAdapter(str(mock_cxi))
    assert len(adapter) == 2
    frame = adapter.get_frame(1, target_size=(512, 512))

    # Assert cropped to 512x512
    assert frame.intensity.shape == (512, 512)
    assert frame.valid_mask.shape == (512, 512)
    assert frame.source_shape == (1024, 1024)
    assert np.allclose(frame.intensity, 42.0)
    assert frame.photon_energy_ev == 9500.0
    assert frame.wavelength_m is not None
    assert abs(frame.wavelength_m - 1.23984e-6 / 9500.0) < 1e-12
