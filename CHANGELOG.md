# Changelog

All notable changes to OpenDiffraction will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
