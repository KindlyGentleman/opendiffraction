# OpenDiffraction Scientific Remediation Progress Log

- **Last Updated**: 2026-10-17
- **Base Commit**: `345095b`
- **Active Working Branch**: `remediation/v3.1-scientific-audit`
- **Current Test Suite**: 117 tests passing (100% pass rate in 158.45s)
- **CI Test Suite**: Full `tests/` coverage verified in `.github/workflows/tests.yml`

---

## 1. Remediation Phase Progress Overview

| Phase | Description | Scope | Status | Evidence Document |
|:---:|:---|:---|:---:|:---|
| **Phase A** | Fail-First Reproduction of Defects | OD-001 through OD-030 | **COMPLETED (PASS)** | `tests/test_second_pass_reproduction.py` |
| **Phase B** | Reconstruct Physical Generation Pipeline | F1-F6 active routing, Fresnel ASM spatial centering | **COMPLETED (PASS)** | `reports/FORWARD_MODEL_REMEDIATION.md` |
| **Phase C** | Verify All 48 Profiles and Six Scenarios | Aberration routing (V17 spherical), beam shift (V19), S1-S6 parameters | **COMPLETED (PASS)** | `reports/PROFILE_ACTIVATION_AND_DIVERSITY_AUDIT.md` |
| **Phase D** | Detector Observability, Masks and Storage | Zero leakage guarantee, remove silent mask wipeout (OD-019) | **COMPLETED (PASS)** | `reports/DETECTOR_MASK_AND_LEAKAGE_AUDIT.md` |
| **Phase E** | Scaling, Identity, Splits and Reproducibility | DPR vs physical scaling, group-disjoint splits, deterministic replay | **COMPLETED (PASS)** | `reports/GROUP_SPLIT_AND_REPRODUCIBILITY_AUDIT.md` |
| **Phase F** | Repair Phase Retrieval and Evaluation Metrics | HIO-ER state transfer (OD-013), windowed SSIM (OD-016), correlation diagnostic (OD-017) | **COMPLETED (PASS)** | `reports/PHASE_RETRIEVAL_RECONSTRUCTION_AUDIT.md` |
| **Phase G** | Rebuild Benchmark Evaluation | Manifest sample_id enforcement (OD-005), balanced Hit Finding (OD-023) | **COMPLETED (PASS)** | `reports/BENCHMARK_PROTOCOL_AND_FAIRNESS_AUDIT.md` |
| **Phase H** | Resource Safety, Resume and Performance | IncrementalShardWriter streaming (<120 MiB RAM), SHA-256 config resume | **COMPLETED (PASS)** | `reports/RESOURCE_AND_RESUME_SAFETY_AUDIT.md` |
| **Phase I** | Scientific Figures and Visual Verification | Figures A through G regenerated at 300 DPI + vector PDF | **COMPLETED (PASS)** | `reports/FIGURE_AND_VISUALIZATION_AUDIT.md` |
| **Phase J** | Independent Adversarial Re-Audit | 5 physical oracles verified, boundary stress testing | **COMPLETED (PASS)** | `reports/INDEPENDENT_SCIENTIFIC_REAUDIT.md` |
| **Phase K** | Release Consistency and Versioning | Release metadata, DATASET_CARD.md, CHANGELOG.md, CI workflows | **COMPLETED (PASS)** | `reports/PUBLICATION_READINESS.md` |

---

## 2. Issue Resolution Log

- **OD-001 (F1-F6 Dispatch)**: VERIFIED. `simulate_forward_model` routes F1-F6 in `forward.py`.
- **OD-002 (Mask Leakage)**: VERIFIED. `intensity_observed[~mask] == 0.0` strictly enforced.
- **OD-003 (Clean Target Fallback)**: VERIFIED. Explicit `KeyError` raised in `torch_dataset.py`.
- **OD-004 (Compact Schema Contract)**: VERIFIED. Compact and extended schemas formalized.
- **OD-005 (Manifest Identity Enforcement)**: VERIFIED. Loaders and evaluators enforce strict `sample_id` mapping.
- **OD-006 (Fresnel ASM Centering)**: VERIFIED. Post-ifft2 spatial `fftshift` removed; zero-distance identity verified.
- **OD-007 (Beam Displacement Semantics)**: VERIFIED. Real-space object translation replaced with illumination wavevector tilt phase ramp.
- **OD-008 (Profile Aberrations Execution)**: VERIFIED. Spherical aberration ($W \propto r^4$) added to `generate_probe` and connected for profile V17.
- **OD-009 (Background Scattering Connection)**: VERIFIED. Halos, slit flare, and cosmic hits connected.
- **OD-010 (F6 Euler Orientation)**: VERIFIED. 3D volume cached and rotated per Euler angles.
- **OD-011 (Physical Geometry Metadata)**: VERIFIED. Connected S1-S6 scenario presets to pipeline metadata dynamically.
- **OD-012 (DPR Scaling Contract)**: VERIFIED. Target scaling separated from physical amplitude.
- **OD-013 (HIO-to-ER Coordinate Transfer)**: VERIFIED. Direct real-space object state transfer into ER eliminates phase coordinate scrambling.
- **OD-014 (Masked Fourier Unconstrained)**: VERIFIED. Masked Fourier pixels updated from iterate.
- **OD-015 (Crystallographic R-Factor)**: VERIFIED. Standard crystallographic formula implemented.
- **OD-016 (Windowed SSIM Implementation)**: VERIFIED. Local sliding window SSIM implemented in `metrics.py`.
- **OD-017 (Diagnostic Pearson Correlation)**: VERIFIED. Replaced target-with-clip(target) with aligned reconstruction correlation.
- **OD-018 (HIO Support Regimes)**: VERIFIED. Supported disc, box, oracle, and autocorrelation support masks in benchmarks.
- **OD-019 (Severe Mask Policy)**: VERIFIED. Removed silent `mask[:] = True` reset in `masks.py`.
- **OD-020 (Quick Preset Profile Coverage)**: VERIFIED. Configured `acquisitions_per_object = 1` for quick preset, covering all 48 profiles V01-V48.
- **OD-021 (Bounded Memory Generation)**: VERIFIED. `IncrementalShardWriter` keeps RAM < 120 MiB.
- **OD-022 (Cryptographic Resume Validation)**: VERIFIED. SHA-256 config hash checked during resume.
- **OD-023 (Hit Finding Class Balance)**: VERIFIED. Balanced 4-class evaluation manifest created.
- **OD-024 (Localized Region Metrics)**: VERIFIED. Masked Region MAE/RMSE evaluated.
- **OD-025 (Preset Consistency)**: VERIFIED. Harmonized CLI, YAML, and README sample counts and version 3.1.0.
- **OD-026 (Release Version Consistency)**: VERIFIED. Centralized v3.1.0 in `version.py`.
- **OD-027 (CI Test Suite Coverage)**: VERIFIED. Expanded `.github/workflows/tests.yml` to run full suite `pytest tests/ -v`.
- **OD-028 (Physics Oracles Verification)**: VERIFIED. Airy disk, Parseval, Friedel verified.
- **OD-029 (Truthful Task Scope)**: VERIFIED. Clarified T1-T4 benchmarked vs T5-T12 roadmap.
- **OD-030 (CXI Scope Disclosure)**: VERIFIED. Disclosed synthetic format test vs real beamtime.
