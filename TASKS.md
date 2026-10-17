# OpenDiffraction Remediation Tasks

Branch: `remediation/v3.1-scientific-audit`
Target: OpenDiffraction 3.1.0 Scientific Remediation

## Prioritized Issue Register

| ID | Sev | Phase | Category | Description | Status |
|---|---|---|---|---|---|
| OD-001 | P0 | Phase I | physics_dispatch | Production pipeline ignores forward_model and always calls F1 | VERIFIED |
| OD-002 | P0 | Phase III | masking_leakage | intensity_measured unmasked data leaks into model input | VERIFIED |
| OD-003 | P0 | Phase I | fresnel_centering | Spurious real-space fftshift in Fresnel ASM breaks z=0 identity | VERIFIED |
| OD-004 | P0 | Phase IV | hdf5_targets | Silent fallback uses noisy input as clean target in compact mode | VERIFIED |
| OD-005 | P0 | Phase VI | phase_retrieval | HIO/ER forces masked pixels to zero; metric is squared error | VERIFIED |
| OD-006 | P1 | Phase II | pipeline_masks | Spurious holder arm forced on acq_idx == 2 | VERIFIED |
| OD-007 | P1 | Phase II | manifest_physics | Wavelength, pitch, and distance hardcoded regardless of scenario | VERIFIED |
| OD-008 | P1 | Phase VIII | memory_safety | Shard buffered in RAM before write; exceeds tight RAM limits | VERIFIED |
| OD-009 | P1 | Phase VIII | resume_safety | verify_shard checks only count; no config hash verification | VERIFIED |
| OD-010 | P1 | Phase II | sample_planning | Hardcoded total=1000 in sample_profile_48 | VERIFIED |

## Remediation Roadmap and Acceptance Gates

### Gate 0: Baseline Freeze
- [x] Record repository baseline state, versions, and test results (83 passed).
- [x] Create dedicated branch `remediation/v3.1-scientific-audit`.
- [x] Create `reports/issue_register.json` and `reports/REMEDIATION_BASELINE_AUDIT.md`.
- Status: **PASS**

### Gate I: Forward-Physics Dispatcher (OD-001, OD-003)
- [x] Implement `simulate_forward_model` in `opendiffraction/physics/forward.py`.
- [x] Connect pipeline `simulate_single_task` to `simulate_forward_model`.
- [x] Fix Fresnel spatial centering in `opendiffraction/physics/models/fresnel.py`.
- [x] Implement regression tests for F1-F6 models and dispatch in `tests/test_forward_dispatcher_remediation.py`.
- [x] Deliver `reports/FORWARD_MODEL_REMEDIATION.md`.
- Status: **PASS**

### Gate III: Detector Masking and Information Leakage (OD-002, OD-006)
- [x] Standardize modalities: `intensity_raw` (unmasked readout) and `intensity_observed` (masked observation).
- [x] Enforce `observed[~valid_mask] == 0` strictly across generator and storage.
- [x] Update loaders and models to consume `intensity_observed`.
- [x] Fix holder arm generation so it is strictly profile-driven (OD-006).
- [x] Deliver `reports/DETECTOR_MASK_AND_LEAKAGE_AUDIT.md`.
- Status: **PASS**

### Gate IV: HDF5 Storage, Targets, and DPR Scaling (OD-004)
- [x] Define explicit target requirements per task (T1-T12).
- [x] Remove silent fallback to input in `torch_dataset.py` and `loader.py`; raise clear exception.
- [x] Validate DPR target scaling ($s \cdot O$ vs $\sqrt{s} \cdot O$) with upstream reference.
- [x] Deliver `reports/HDF5_SCHEMA_AND_TARGET_AUDIT.md`.
- Status: **PASS**

### Gate II: Profile Activation (48 Variations & Scenarios S1-S6) (OD-007, OD-010)
- [x] Audit all 48 profiles to ensure declared effects modulate measurement.
- [x] Verify S1-S6 scenarios populate correct physical metadata in manifest (OD-007).
- [x] Fix hardcoded sample size scaling in `pipeline.py` (OD-010).
- [x] Deliver `reports/PROFILE_ACTIVATION_AND_DIVERSITY_AUDIT.md`.
- Status: **PASS**

### Gate V: Object Identity, Group Splits, and Replay
- [x] Verify unscaled base object identity invariant across acquisitions.
- [x] Verify group-disjoint splits with zero cross-set leakage.
- [x] Test deterministic replay from manifest seeds.
- [x] Deliver `reports/GROUP_SPLIT_AND_REPRODUCIBILITY_AUDIT.md`.
- Status: **PASS**

### Gate VI: Classical Phase Retrieval Baseline (OD-005)
- [x] Repair HIO and ER mask handling: keep unmeasured pixels unconstrained.
- [x] Add reconstruction registration (translation, twin image, phase).
- [x] Standardize crystallographic R-factor metric.
- [x] Deliver `reports/PHASE_RETRIEVAL_RECONSTRUCTION_AUDIT.md`.
- Status: **PASS**

### Gate VIII: Engineering Reliability and Memory Safety (OD-008, OD-009)
- [x] Implement bounded-memory incremental shard writes.
- [x] Add config hash and integrity verification to resume logic.
- [x] Execute fault-injection tests (corrupted shard, config mismatch, interruption).
- [x] Deliver `reports/RESOURCE_AND_RESUME_SAFETY_AUDIT.md`.
- Status: **PASS**

### Gate VII: Benchmark Revalidation (T1-T12)
- [x] Re-run baselines for T1 (Phase Retrieval), T2 (Denoising), T3 (Inpainting), T4 (Hit Finding).
- [x] Generate balanced evaluation set for T4.
- [x] Complete audit matrix for T5-T12.
- [x] Deliver `reports/BENCHMARK_PROTOCOL_AND_FAIRNESS_AUDIT.md`.
- Status: **PASS**

### Gate IX: Regression Test Suite and Packaging
- [x] Add comprehensive regression tests for OD-001 through OD-010.
- [x] Verify optional dependencies in `pyproject.toml` and `requirements.txt`.
- [x] Run complete test suite and verify 100% pass rate.
- Status: **PASS**

### Gate X: Figure and Visualization Provenance
- [x] Re-generate scientific figures A-G with verified provenance.
- [x] Deliver `reports/FIGURE_AND_VISUALIZATION_AUDIT.md`.
- Status: **PASS**

### Gate XI: Independent Scientific Re-Audit
- [x] Independent calculation routes and oracles.
- [x] Deliver `reports/INDEPENDENT_SCIENTIFIC_REAUDIT.md`.
- Status: **PASS**

### Gate XII: Release Consistency and Documentation
- [x] Update README, DATASET_CARD, and CHANGELOG to reflect version 3.1.0.
- [x] Deliver `reports/FINAL_REMEDIATION_REPORT.md` and `reports/PUBLICATION_READINESS.md`.
- Status: **PASS**
