# Changelog

All notable changes to OpenDiffraction will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] - 2026-10-17

### Fixed & Remediated
- **OD-001 (Forward Physics Dispatcher)**: Implemented `simulate_forward_model` in `opendiffraction/physics/forward.py`, routing simulation tasks dynamically to physical forward models F1 through F6 according to profile metadata.
- **OD-002 (Information Leakage Elimination)**: Separated detector readouts into `intensity_raw` (unmasked signed readouts) and `intensity_observed` (strictly masked observation where `observed[~valid_mask] == 0.0`). Guaranteed zero unmasked pixel leakage into model inputs.
- **OD-003 (Target Integrity & Silent Fallback Removal)**: Removed deceptive `clean_arr = inp_arr` fallback in `torch_dataset.py` and `loader.py`. Supervised tasks now require clean targets or explicit `self_supervised=True`.
- **OD-004 (Compact Schema Contract)**: Formalized compact vs extended storage contracts, ensuring clean ground-truth intensities are stored in extended shards.
- **OD-005 (Manifest Identity Enforcement)**: Added `manifest_path` support and `get_by_sample_id` indexing to `OpenDiffractionDataset` and `OpenDiffractionTorchDataset`, ensuring benchmark evaluators match exact sample identities.
- **OD-006 (Fresnel ASM Centering Bug)**: Eliminated spurious real-space `fftshift` in `fresnel.py`, restoring exact analytical $z=0$ identity ($\|U(z=0) - U_0\|_\infty < 10^{-14}$) and eliminating half-frame diagonal displacement.
- **OD-007 (Beam Displacement Semantics)**: Applied illumination wavevector tilt phase ramp in `pad_object_to_detector`, overcoming Fourier shift invariance and shifting far-field diffraction peaks on the detector frame.
- **OD-008 (Profile Aberrations Execution)**: Added spherical aberration ($W \propto r^4$) parameter `spherical_aberration_rad` to `generate_probe` and connected metadata in `forward.py` for profile V17.
- **OD-009 (Background Scattering Connection)**: Connected background halos, slit flare, and cosmic ray spikes to production forward pipeline.
- **OD-010 (F6 Euler Orientation)**: Cached 3D volume across acquisition shots and applied Euler orientation rotations before projection.
- **OD-011 (Physical Geometry Metadata)**: Synchronized wavelength, pixel pitch, photon energy, and detector distance dynamically from active `ScenarioSpec` S1-S6.
- **OD-012 (DPR Scaling Contract)**: Separated DPR flux-scaled target ($O \cdot s$) from physical ground truth amplitude ($O$).
- **OD-013 (HIO-to-ER Coordinate Transfer)**: Implemented direct real-space object state transfer into `error_reduction`, eliminating uncentered Fourier phase coordinate scrambling.
- **OD-014 (Masked Fourier Unconstrained)**: Left unmeasured Fourier pixels unconstrained during modulus updates, preventing artificial beamstop edge ringing.
- **OD-015 (Crystallographic R-Factor)**: Implemented standard crystallographic amplitude R-factor evaluated strictly over valid pixels.
- **OD-016 (Windowed SSIM Implementation)**: Implemented standard local windowed SSIM in `metrics.py` using 11x11 sliding window filter.
- **OD-017 (Diagnostic Pearson Correlation)**: Replaced trivial target vs clipped target correlation with true Pearson correlation between aligned reconstruction and ground truth in `eval_phase_retrieval.py`.
- **OD-018 (HIO Support Regimes)**: Implemented explicit support constraint regimes (disc, bounding box, oracle ground truth dilation, autocorrelation estimation) in `benchmarks/phase_retrieval.py`.
- **OD-019 (Severe Mask Policy)**: Removed silent mask reset (`mask[:] = True`) on severe defect patterns, preserving extreme detector masks.
- **OD-020 (Quick Preset Profile Coverage)**: Configured `acquisitions_per_object = 1` for quick preset, ensuring 100% coverage of profiles V01-V48 in smoke tests.
- **OD-021 (Bounded-Memory Incremental Streaming)**: Implemented `IncrementalShardWriter` in `storage.py`, pre-allocating HDF5 datasets and streaming samples to disk individually. Peak generation process RAM reduced to $<120$ MB.
- **OD-022 (Resume Safety & Fingerprinting)**: Added deterministic SHA-256 configuration fingerprinting stored in HDF5 root attributes; resume logic validates configuration compatibility before appending.
- **OD-023 (Hit Finding Class Balance)**: Added balanced stratified manifest generation for Task 4 (Hit Finding) with $3\sigma$ readout noise floor gating.
- **OD-024 (Localized Region Metrics)**: Inpainting benchmark evaluators compute Masked Region MAE and RMSE strictly over missing pixels.
- **OD-025 (Preset Consistency)**: Harmonized sample counts and versions across CLI, YAML configs, and documentation.
- **OD-026 (Release Version Consistency)**: Centralized v3.1.0 in `version.py`.
- **OD-027 (CI Test Suite Coverage)**: Updated GitHub Actions workflow `.github/workflows/tests.yml` to run full test suite across `tests/`.
- **OD-028 (Physics Oracles Verification)**: Validated 5 independent mathematical oracles (Airy disk, Parseval energy conservation, Friedel symmetry, Fresnel identity).
- **OD-029 (Truthful Task Scope)**: Clarified T1-T4 benchmarked tasks vs T5-T12 roadmap manifests.
- **OD-030 (CXI Scope Disclosure)**: Disclosed synthetic format compatibility testing versus facility beamtime data.

### Added
- **Multi-Acquisition Object Invariance**: Preserved unscaled base object identity across all 4 acquisition shots across all 48 profiles, including V47 3D particle Euler projections.
- **Sub-Pixel Twin Image Alignment**: Added sub-pixel cross-correlation registration in `align_twin_and_translation` using matrix-multiplication DFT (`skimage.registration.phase_cross_correlation`).
- **Unified 117-Test Regression Suite**: 100% automated test pass rate across physical forward models, detector masking, storage contracts, and deterministic replay.

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
