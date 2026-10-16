# Changelog

All notable changes to OpenDiffraction will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-10-16

### Added
- **Audited 1,000-Observation Training Pilot**:
  - Exactly 1,000 observations across 250 distinct base objects with 4 independent acquisitions per object.
  - Zero object-level leakage across train (644), valid (140), test (116), and test_ood (100) splits.
  - Full representation across all 48 variation profiles (V01-V48) and all 6 forward models (F1-F6).
  - High-performance HDF5 sharding (7 shards, gzip level 4 + shuffle filter, 895.35 MB storage).
- **Turnkey PyTorch Integration (`OpenDiffractionTorchDataset`)**:
  - Direct PyTorch Dataset and DataLoader support with worker-safe lazy handle caching.
  - Multiprocessing safe under Windows and Linux (65-132 samples/second throughput on CPU).
  - Harmonized task targets for phase retrieval, denoising, inpainting, and hit classification.
- **Independent Benchmark Baselines & Audits**:
  - Classical iterative phase retrieval (HIO/ER), spatial denoising, nearest-neighbor inpainting, and energy-based hit classification.
  - Documented physical failure modes: twin-image stagnation, central beamstop shadow underestimation, and synthetic hit-gap limitations.
- **Experimental CXI NeXus Adapter (`CXIExperimentalAdapter`)**:
  - Read-only ingestion of Coherent X-ray Imaging (CXI) files with dynamic NeXus path discovery.
  - Spatial standardization to 512x512 via center cropping or zero-padding with validity mask tracking.
  - Physical constant conversion from photon energy (eV) to radiation wavelength (m).

---

## [2.0.0] - 2026-10-16


### Added
- **6 Modular Forward Physics Models (F1 - F6)**:
  - F1: Coherent Fraunhofer far-field diffraction from 2D scalar fields.
  - F2: Gaussian Schell-model (GSM) partial coherence and orthogonal multimode decomposition.
  - F3: Thin-object transmission with probe aberrations (defocus, astigmatism, coma, jitter).
  - F4: Near-field Fresnel diffraction in SI units ($z$, $\lambda$, $\Delta x$).
  - F5: Polychromatic pink-beam diffraction with reciprocal-space coordinate dilation.
  - F6: 3D particle single-particle scattering with Euler rotations and Born projection.
- **48 Controlled Variation Profiles (V01 - V48)** organized into 6 physical groups.
- **6 Experimental Acquisition Scenarios (S1 - S6)**: XFEL, Synchrotron, Nanomaterials, Bio CDI, Optical Lab, Detector QA.
- **12 Machine Learning Benchmark Tasks (T1 - T12)** with standardized manifests and evaluation metrics.
- **Official CXI Format Exporter**: NeXus-compliant `.cxi` export (`/entry_1/data_1/...`) with automated validation.
- **Publication Visualization Suite**: 16 IEEE paper-quality figures and 3D/flux sweep animated GIFs.
- **Complete Test Suite**: 75 comprehensive tests covering physics, noise, profiles, CXI layout, and DPR compatibility.

---

## [0.1.0] - 2026-10-15

### Added
- Initial Fraunhofer coherent diffraction simulator.
- Corrected Gaussian Schell-model partial coherence implementation.
- Basic procedural object synthesis and DPR target normalization.
- Prototype HDF5 dataset writer and verification scripts.
