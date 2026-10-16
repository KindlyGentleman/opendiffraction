# OpenDiffraction 3.0 Experimental CXI Adapter Validation Report

## 1. Overview

The `CXIExperimentalAdapter` module (`opendiffraction/io/experimental.py`) provides a robust, read-only interface for loading experimental Coherent X-ray Imaging (CXI) NeXus datasets and mapping them into the standardized tensor formats required by OpenDiffraction benchmark pipelines.

---

## 2. Supported NeXus Paths and Hierarchy

The adapter dynamically discovers diffraction and metadata nodes within standard NeXus CXI structures:

| Data Type | Primary Path | Fallback Path | Format |
| :--- | :--- | :--- | :--- |
| **Diffraction Intensity** | `/entry_1/data_1/data` | `/entry_1/instrument_1/detector_1/data` | 2D/3D float32 or int32 |
| **Detector Mask** | `/entry_1/data_1/mask` | `/entry_1/instrument_1/detector_1/mask` | 2D/3D uint32, int32, or bool |
| **Photon Energy** | `/entry_1/instrument_1/source_1/energy` | `/entry_1/source_1/energy` | Scalar float (eV) |
| **Detector Distance** | `/entry_1/instrument_1/detector_1/distance`| `/entry_1/detector_1/distance` | Scalar float (meters) |
| **Pixel Pitch** | `/entry_1/instrument_1/detector_1/x_pixel_size`| `/entry_1/detector_1/x_pixel_size` | Scalar float (meters) |

---

## 3. Spatial Geometry Standardization

Experimental X-ray detectors vary widely in physical dimensions (e.g., $1024 \times 1024$ for pnCCD, $1516 \times 1516$ for CSPAD, $2048 \times 2048$ for Rayonix). The adapter applies deterministic, centered bounding-box cropping or padding:
- **Large Detectors ($\ge 512 \times 512$)**: Centered region is cropped to exactly $512 \times 512$ pixels preserving the zero-frequency DC transmission center.
- **Small Detectors ($< 512 \times 512$)**: Centered into a zero-filled $512 \times 512$ canvas with the validity mask set to `False` outside the active sensor area.
- **Mask Alignment**: Sensor defect masks undergo the identical geometric transformation, guaranteeing pixel-level registration between intensity and mask arrays.

---

## 4. Verification Test Results

Two automated test suites were executed (`tests/test_experimental_adapter.py`):

1. **Benchmark CXI Verification (`test_experimental_adapter_on_cxi_benchmark`)**:
   - Source: `output_v3/opendiffraction_benchmark.cxi`
   - Frames: 200 diffraction patterns
   - Intensity Shape: $(512, 512)$ float32
   - Mask Shape: $(512, 512)$ bool
   - Photon Energy: 8,000 eV ($\lambda \approx 1.5498 \times 10^{-10}$ m)
   - Status: **PASS**

2. **Crop and Pad Resampling (`test_experimental_adapter_geometry_crop_and_pad`)**:
   - Source: Mock $1024 \times 1024$ experimental CXI
   - Intensity Cropping: $1024 \times 1024 \to 512 \times 512$
   - Mask Cropping: Verified beam stop preserved at center
   - Energy Extraction: 9,500 eV ($\lambda \approx 1.3051 \times 10^{-10}$ m)
   - Status: **PASS**

---

## 5. Summary and Reproducibility

The adapter operates completely in CPU memory, uses zero temporary disk writes, preserves native floating-point dynamic range, and provides clean Python dataclass records (`ExperimentalDiffractionFrame`) directly consumable by PyTorch and NumPy workflows.
