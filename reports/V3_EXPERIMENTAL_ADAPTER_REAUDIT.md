# OpenDiffraction 3.0 Experimental CXI Adapter Audit Report

## 1. Executive Summary

This report provides an independent audit of the CXI experimental adapter (`opendiffraction/io/experimental.py`) and its associated test suite (`tests/test_experimental_adapter.py`).

Critical Qualification:
Experimental-format adapter tested on synthetic fixtures; real beamline validation not yet performed.

The adapter provides a read-only bridge to parse Coherent X-ray Imaging (CXI) NeXus HDF5 files into standardized OpenDiffraction tensor structures. While the adapter successfully implements NeXus discovery, spatial cropping, padding, and physical wavelength conversion, its validation is currently restricted to synthetic NeXus files and test fixtures. Beamline validation on authentic raw experimental datasets (such as CXIDB entries) remains scheduled for subsequent evaluation phases.

| Subsystem / Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- |
| CXI NeXus Path Discovery | PASS | Resolves `/entry_1/data_1/data` and fallback detector paths |
| Spatial Standardization | PASS | Deterministic 512x512 cropping and zero-padding |
| Mask Registration | PASS | Mask undergoes identical spatial transform as intensity |
| Photon Energy to Wavelength | PASS | Computes lambda = h * c / E with physical constants |
| Synthetic CXI Benchmark Ingestion | PASS | Ingests all 200 frames of output_v3/opendiffraction_benchmark.cxi |
| Real Beamline Experimental Ingestion | NOT TESTED | Synthetic fixtures only; CXIDB downloads out of scope for pilot |

---

## 2. Technical Architecture Audit

The `CXIExperimentalAdapter` class encapsulates experimental ingestion without introducing external system dependencies beyond NumPy, h5py, and the standard library.

### 2.1 NeXus Hierarchy Discovery

The adapter inspects standard CXI NeXus hierarchy trees:

1. Diffraction Intensity:
   - Primary: `/entry_1/data_1/data`
   - Fallback: `/entry_1/instrument_1/detector_1/data`
2. Detector Mask:
   - Primary: `/entry_1/data_1/mask`
   - Fallback: `/entry_1/instrument_1/detector_1/mask`
3. Photon Energy:
   - Primary: `/entry_1/instrument_1/source_1/energy`
   - Fallback: `/entry_1/source_1/energy`
4. Detector Distance:
   - Primary: `/entry_1/instrument_1/detector_1/distance`
   - Fallback: `/entry_1/detector_1/distance`
5. Pixel Pitch:
   - Primary: `/entry_1/instrument_1/detector_1/x_pixel_size`
   - Fallback: `/entry_1/detector_1/x_pixel_size`

### 2.2 Geometry Normalization

The target output spatial dimension for OpenDiffraction benchmark pipelines is exactly 512x512 pixels:
- Oversized Detectors (height, width >= 512): Centered symmetric cropping extracts the 512x512 region around the optical axis (center pixel).
- Undersized Detectors (height, width < 512): Centered insertion into a 512x512 zero-filled canvas. Pixels outside the native detector boundary are marked invalid (mask = False).
- Mixed Aspect Ratios: Handled per-axis independently with center alignment.

### 2.3 Physical Constant Conversion

Photon energy E (measured in electron-volts, eV) is converted to radiation wavelength lambda (measured in meters) via:

lambda = (h * c) / (E * e)

where:
- Planck constant h = 6.62607015e-34 J s
- Speed of light c = 299792458.0 m / s
- Elementary charge e = 1.602176634e-19 C

Validation check: For E = 8,000 eV (standard Cu-Kalpha / hard X-ray benchmark energy), computed lambda = 1.549802e-10 m (1.5498 Angstroms). This matches the reference value within 1e-6 relative tolerance.

---

## 3. Test Suite and Execution Audit

The test suite `tests/test_experimental_adapter.py` executes two automated test functions:

1. `test_experimental_adapter_on_cxi_benchmark`:
   - Dataset: `output_v3/opendiffraction_benchmark.cxi`
   - Frame count: 200 diffraction patterns
   - Intensity shape: (512, 512) float32
   - Mask shape: (512, 512) bool
   - Photon energy: 8,000.0 eV
   - Status: PASS

2. `test_experimental_adapter_geometry_crop_and_pad`:
   - Mock source: 1024x1024 synthetic frame with central beamstop
   - Transformation: 1024x1024 to 512x512 center crop
   - Energy: 9,500.0 eV
   - Result: Intensity cropped correctly, beamstop centered, wavelength computed as 1.305096e-10 m
   - Status: PASS

---

## 4. Known Boundaries and Scientific Limitations

1. Synthetic vs Real Discrepancy:
   The tested CXI file (`opendiffraction_benchmark.cxi`) was produced synthetically by OpenDiffraction's pipeline following CXI format conventions. Real beamline data from LCLS, European XFEL, or SACLA often contain unexpected NeXus quirks:
   - Variable frame-by-frame beam center drifts.
   - Non-standard unit strings (e.g., 'keV', 'joules', or missing units).
   - Multi-panel tiled detector geometries (e.g., CSPAD 32-tile layouts requiring ASICS geometry files) before virtual assembly.
2. Read-Only Scope:
   The adapter is strictly read-only and does not write or modify external CXI files.
3. Multi-Panel Detector Assembly:
   The current adapter assumes assembled 2D detector images. Raw multi-panel formats requiring cheetah or crystfel geometry files (.geom) are not yet natively reconstructed within this adapter.

---

## 5. Certification and Recommendation

The `CXIExperimentalAdapter` module is certified as PASS for synthetic CXI NeXus ingestion and spatial alignment. It is ready for inclusion in the OpenDiffraction 3.0 release, with the clear scientific caveat that direct application to raw un-assembled beamline data requires prior geometry assembly.
