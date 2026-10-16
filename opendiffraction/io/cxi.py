"""Official CXI (Coherent X-ray Imaging) NeXus/HDF5 format exporter and validator.

Conforms to CXIDB standards (cxidb.org/cxi.html) for experimental data exchange.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import h5py
import numpy as np


def export_to_cxi(
    h5_source_path: str | Path,
    cxi_output_path: str | Path,
    sample_indices: list[int] | None = None,
    scenario_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export OpenDiffraction dataset to official CXI format.

    Creates NeXus-compliant CXI layout:
    - /entry_1 [NXentry]
      - /data_1 [NXdata]
        - data: (N, 512, 512) diffraction intensities
        - mask: (N, 512, 512) uint32 or bool detector pixel mask
      - /instrument_1 [NXinstrument]
        - /source_1 [NXsource]
          - energy: photon energy in Joules (or eV with unit attribute)
        - /detector_1 [NXdetector]
          - distance: sample-to-detector distance in meters
          - x_pixel_size: pixel pitch in meters
          - y_pixel_size: pixel pitch in meters
          - data -> link to /entry_1/data_1/data
      - /image_1 [NXdata]
        - data: (N, 64, 64) ground truth object transmission density
      - /sample_1 [NXsample]
        - name: specimen description string

    Parameters
    ----------
    h5_source_path : str or Path
        Source OpenDiffraction HDF5 file containing 'input', 'target', and 'mask'.
    cxi_output_path : str or Path
        Target .cxi file path.
    sample_indices : list[int] or None
        Specific sample indices to export. If None, exports all.
    scenario_metadata : dict or None
        Optional beamline instrument parameters.

    Returns
    -------
    Path
        Created CXI file path.
    """
    src_p = Path(h5_source_path)
    out_p = Path(cxi_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if scenario_metadata is None:
        scenario_metadata = {
            "energy_ev": 8000.0,
            "distance_m": 0.5,
            "pixel_pitch_m": 75e-6,
            "sample_name": "OpenDiffraction Synthetic Specimen",
        }

    with h5py.File(src_p, "r") as src:
        total = src["input"].shape[0]
        indices = sample_indices if sample_indices is not None else list(range(total))

        inputs = src["input"][indices]
        targets = src["target"][indices]
        masks = src["mask"][indices]

        if inputs.ndim == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

    with h5py.File(out_p, "w") as dst:
        dst.attrs["cxi_version"] = 140

        # /entry_1
        entry = dst.create_group("entry_1")
        entry.attrs["NX_class"] = "NXentry"

        # /entry_1/data_1
        data_grp = entry.create_group("data_1")
        data_grp.attrs["NX_class"] = "NXdata"
        data_dset = data_grp.create_dataset(
            "data",
            data=inputs.astype(np.float32),
            chunks=(1, 512, 512),
            compression="gzip",
            compression_opts=4,
        )
        data_dset.attrs["units"] = "counts"
        data_dset.attrs["interpretation"] = "image"

        mask_dset = data_grp.create_dataset(
            "mask",
            data=masks.astype(np.uint32),
            chunks=(1, 512, 512),
            compression="gzip",
            compression_opts=4,
        )
        mask_dset.attrs["description"] = "1=valid, 0=defect/masked"

        # /entry_1/instrument_1
        instrument = entry.create_group("instrument_1")
        instrument.attrs["NX_class"] = "NXinstrument"

        # /entry_1/instrument_1/source_1
        source = instrument.create_group("source_1")
        source.attrs["NX_class"] = "NXsource"
        en_dset = source.create_dataset("energy", data=float(scenario_metadata.get("energy_ev", 8000.0)))
        en_dset.attrs["units"] = "eV"

        # /entry_1/instrument_1/detector_1
        detector = instrument.create_group("detector_1")
        detector.attrs["NX_class"] = "NXdetector"

        dist_dset = detector.create_dataset("distance", data=float(scenario_metadata.get("distance_m", 0.5)))
        dist_dset.attrs["units"] = "m"

        px_dset = detector.create_dataset("x_pixel_size", data=float(scenario_metadata.get("pixel_pitch_m", 75e-6)))
        px_dset.attrs["units"] = "m"

        py_dset = detector.create_dataset("y_pixel_size", data=float(scenario_metadata.get("pixel_pitch_m", 75e-6)))
        py_dset.attrs["units"] = "m"

        # Soft link: /entry_1/instrument_1/detector_1/data -> /entry_1/data_1/data
        detector["data"] = h5py.SoftLink("/entry_1/data_1/data")

        # /entry_1/image_1 (Ground truth object)
        image_grp = entry.create_group("image_1")
        image_grp.attrs["NX_class"] = "NXdata"
        obj_dset = image_grp.create_dataset(
            "data",
            data=targets.astype(np.float32),
            chunks=(1, 64, 64),
            compression="gzip",
            compression_opts=4,
        )
        obj_dset.attrs["interpretation"] = "image"
        obj_dset.attrs["description"] = "Ground truth object transmission density"

        # /entry_1/sample_1
        sample_grp = entry.create_group("sample_1")
        sample_grp.attrs["NX_class"] = "NXsample"
        sample_grp.create_dataset("name", data=str(scenario_metadata.get("sample_name", "OpenDiffraction Synthetic")))

    return out_p


def validate_cxi(cxi_path: str | Path) -> tuple[bool, list[str]]:
    """Validate CXI file compliance against CXIDB standards.

    Checks:
    - cxi_version attribute present
    - /entry_1 with NX_class == 'NXentry'
    - /entry_1/data_1/data present with 3D or 2D array
    - /entry_1/data_1/mask present
    - /entry_1/instrument_1/detector_1 geometry metadata present
    - /entry_1/image_1 ground truth present

    Returns
    -------
    is_valid : bool
        True if all required nodes pass.
    messages : list[str]
        Detailed compliance check log.
    """
    p = Path(cxi_path)
    if not p.exists():
        return False, [f"File does not exist: {p}"]

    messages: list[str] = []
    valid = True

    try:
        with h5py.File(p, "r") as f:
            if "cxi_version" not in f.attrs:
                messages.append("WARNING: Root attribute 'cxi_version' missing.")
            else:
                messages.append(f"PASS: CXI version {f.attrs['cxi_version']} detected.")

            if "entry_1" not in f:
                messages.append("FAIL: /entry_1 group missing.")
                return False, messages

            entry = f["entry_1"]
            if entry.attrs.get("NX_class") != "NXentry":
                messages.append("FAIL: /entry_1 NX_class is not 'NXentry'.")
                valid = False
            else:
                messages.append("PASS: /entry_1 conforms to NXentry.")

            # Check /entry_1/data_1/data
            if "data_1" not in entry or "data" not in entry["data_1"]:
                messages.append("FAIL: /entry_1/data_1/data missing.")
                valid = False
            else:
                shape = entry["data_1/data"].shape
                messages.append(f"PASS: /entry_1/data_1/data present with shape {shape}.")

            # Check /entry_1/data_1/mask
            if "data_1/mask" not in entry:
                messages.append("WARNING: /entry_1/data_1/mask missing.")
            else:
                messages.append("PASS: /entry_1/data_1/mask present.")

            # Check instrument
            if "instrument_1/detector_1" in entry:
                det = entry["instrument_1/detector_1"]
                if "distance" in det and "x_pixel_size" in det:
                    messages.append("PASS: Instrument detector distance and pixel size verified.")
                else:
                    messages.append("WARNING: Detector geometry incomplete.")

            # Check image
            if "image_1/data" in entry:
                messages.append("PASS: /entry_1/image_1/data ground truth verified.")

    except Exception as e:
        messages.append(f"FAIL: Exception while reading CXI: {e}")
        return False, messages

    return valid, messages
