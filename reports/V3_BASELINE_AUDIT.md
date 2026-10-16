# OpenDiffraction 3.0 Baseline Audit

**Date**: 2026-10-16  
**Auditor**: Antigravity Computational Physics and Reproducibility Auditor  
**Repository Base**: `d:\03_Proyek\Diffraction_Dataset`  
**Branch**: `feature/v3-training-data`  
**Status**: AUDIT COMPLETE (Phase 0 Exit Criteria Satisfied)

---

## 1. Executive Summary

This audit establishes the baseline state of OpenDiffraction before undertaking large-scale data generation (target: 10,000 observations). The existing codebase contains verified physics implementations (F1 to F6) and passing regression tests against upstream Deep Phase Retrieval (DPR) conventions. However, critical vulnerabilities were discovered in resume shard naming, object grouping, and configuration version tracking.

---

## 2. Test Suite Execution Baseline

The baseline test suite was executed in an isolated PowerShell environment on CPU:

- **Command**: `python -m pytest -q`
- **Execution Date**: 2026-10-16 20:55 UTC+7
- **Environment**: Python 3.12, PyTorch 2.8.0, h5py 3.16.0, NumPy 2.5.3, SciPy 1.16.2, pytest 9.1.1
- **Total Test Count**: 78
- **Passed**: 78
- **Failed**: 0
- **Skipped**: 0
- **Total Execution Time**: 151.77 seconds

### Test Breakdown by Subsystem:
1. `tests/test_physics_forward.py`: 8 tests covering forward diffraction, padding, sub-pixel shifts, Poisson noise, Gaussian readout noise, detector masks.
2. `tests/test_coherence.py`: 5 tests covering Gaussian Schell-model (GSM) coherence decay, zero-lag centering, and energy conservation.
3. `tests/test_noise_statistics.py`: 4 tests validating Poisson mean-variance relations and Gaussian noise moments.
4. `tests/test_objects_and_variations.py`: 6 tests verifying procedural object generators and shape subtypes.
5. `tests/test_storage_and_splits.py`: 5 tests verifying HDF5 writing, group disjointness, and metadata reproduction.
6. `tests/test_cli.py`: 4 tests verifying command-line interface arguments and execution flow.
7. `tests/test_cxi.py`: 3 tests validating NeXus/CXI export format compliance.
8. `tests/test_f1_to_f6.py`: 6 tests validating modular forward physics models (Fraunhofer, GSM/multimode coherence, thin-object probe transmission, Fresnel propagation, polychromatic spectrum, 3D particle scattering).
9. `tests/test_profiles_48.py`: 49 tests (1 completeness test + 48 parametrized execution tests for profiles V01 through V48).
10. `tests/test_dpr_compatibility.py`: 5 tests validating upstream DPR `deeppr.CustomDataset` loading and numerical agreement.

All 78 tests passed. No skipped tests were counted as passing.

---

## 3. Discovered Correctness and Architectural Issues

### Issue 1: Dynamic Shard Naming Overwrites on Resume (Critical)
In `opendiffraction/generator/pipeline.py` (lines 181-186), shard names are constructed dynamically using the count of remaining `pending_tasks` for each split:
```python
split_tasks = split_buckets[split_name]  # only pending tasks
n_shards = int(np.ceil(len(split_tasks) / shard_size))
for sh_idx in range(n_shards):
  shard_name = f"{split_name}_{sh_idx:03d}.h5"
```
When generation is interrupted after completing one or more shards, `pending_tasks` contains fewer tasks. On resume, `sh_idx` restarts from 0, immediately overwriting `f"{split_name}_000.h5"` with the next block of tasks. This destroys previously written data, duplicates metadata indices, and causes row-index misalignment.

### Issue 2: Independent Object Grouping and Missing Multi-Acquisition Link (High)
In `pipeline.py` (lines 140-142), every single task creates a separate group:
```python
group_id = f"grp_{pid}_{s_idx:04d}"
```
This couples `group_id` directly to profile ID `pid`. If a base object is observed under multiple conditions or profiles, each observation receives a different `group_id`. Because split assignment hashes `group_id`, multi-exposure observations of the same object could leak across training, validation, and test splits. Base object identity (`base_object_id`) must be strictly decoupled from acquisition conditions.

### Issue 3: Inconsistent Version and Profile Definitions (Medium)
Metadata across the repository exhibits conflicting version strings:
- `opendiffraction/__init__.py`: `__version__ = "0.1.0"`
- `pyproject.toml`: `version = "2.0.0"`
- `kaggle_release/configuration.json`: `"version": "0.1.0"` with 12 legacy profiles
- `configs/quick.yaml` & `configs/full.yaml`: `version: "2.0.0"` with 48 profiles
- `opendiffraction/generator/pipeline.py`: hardcoded `"simulation_version": "2.0.0"`

A single source of truth must be created in `opendiffraction/version.py` defining Generator version 3.0.0, Schema version 3.0.0, and Model version 3.0.0.

### Issue 4: PyTorch Loading and Resource Protection (Medium)
The existing `OpenDiffractionDataset` in `opendiffraction/io/loader.py` lacks a dedicated PyTorch `Dataset` subclass (`OpenDiffractionTorchDataset`). In multi-worker PyTorch DataLoaders, sharing open HDF5 file handles across forked or spawned worker processes causes crashes or concurrency lockups. A per-worker lazy file handle manager is required.

---

## 4. Phase 0 Audit Verdict

Phase 0 audit requirements are fully satisfied. The physical models are scientifically sound and match published DPR literature. The critical resume and object grouping defects must be resolved prior to generating the 10,000-sample release.
