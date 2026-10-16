# OpenDiffraction 3.0 Experiment and Audit Log

## Entry 001: Baseline Test Suite Run
- Timestamp: 2026-10-16 20:55 UTC+7
- Environment: Python 3.12 / PyTorch 2.8.0 / h5py 3.16.0 / pytest 9.1.1
- Command: `python -m pytest -q`
- Results:
  - Total tests: 78
  - Passed: 78
  - Failed: 0
  - Skipped: 0
  - Duration: 151.77s
- Observations:
  - Forward models F1 to F6 execute reliably without errors.
  - DPR numerical agreement test passed against local `deeppr` package.

## Entry 002: Phase 0 Pipeline Architecture and Resume Safety Fixes
- Timestamp: 2026-10-16 21:05 UTC+7
- Changes:
  - Created `opendiffraction/version.py` establishing version 3.0.0 as single source of truth.
  - Resolved dynamic shard naming bug in `pipeline.py` with immutable pre-planned shard assignment.
  - Implemented base object identity decoupling (`base_object_id`) to prevent multi-exposure split leakage.
  - Implemented atomic `os.replace` shard writes and de-duplicated metadata manifest appending.
  - Created `scripts/verify_resume.py` and executed fault injection tests:
    - Test 1 (interrupted run + orphaned `.tmp` + resume): PASSED
    - Test 2 (safe overwrite policy refusal): PASSED
    - Test 3 (deterministic bit-for-bit replay): PASSED
  - Implemented `opendiffraction/io/torch_dataset.py` with multi-worker lazy loading.
  - All unit tests in `tests/test_storage_and_splits.py`, `tests/test_cli.py`, and `tests/test_torch_dataset.py` PASSED.

## Entry 003: Phase 1 Staged Generation Execution
- Timestamp: 2026-10-16 21:26 UTC+7
- Target adjustment: User approved reducing primary target to 1,000 validated samples.
- Staged Verification Results (`scripts/verify_phase1_stages.py`):
  - Stage A (48 samples): PASSED in 4.36s (11.0 samples/s, 952.7 KB/sample). Tensor shapes and DPR values verified.
  - Stage B (480 samples): PASSED in 44.84s (10.7 samples/s, 908.3 KB/sample). Zero leakage across 120 groups.
  - Stage C (1,000 samples): PASSED in 89.96s (11.1 samples/s, 916.8 KB/sample). Mid-flight interruption and resume verified.
- Production Generation (`scripts/generate_v3_production.py`):
  - Generated: 1,000 samples in `output_v3/` across 7 HDF5 shards (0.87 GB total, 916.8 KB/sample).
  - Wall time: 90.51s (11.05 samples/s).
  - Validation: 0 NaN, 0 Inf, 250 base objects x 4 acquisitions, all 48 profiles present.

## Entry 004: Phase 3 Task Manifests and Scientific Audits
- Timestamp: 2026-10-16 21:39 UTC+7
- Command: `python scripts/build_phase3_manifests_and_audit.py`
- Manifests Exported:
  - `manifests/task_1/`: Phase Retrieval (644 train, 140 val, 116 test, 100 test_ood)
  - `manifests/task_2/`: Denoising (644 train, 140 val, 116 test)
  - `manifests/task_3/`: Inpainting (644 train, 140 val, 116 test)
  - `manifests/task_4/`: Hit Classification (644 train, 140 val, 116 test, 100 test_ood)
- Reports Created: `V3_FULL_DATASET_VALIDATION.md`, `V3_PHYSICS_VALIDATION.md`, `V3_DISTRIBUTION_AUDIT.md`, `V3_DATASET_CARD.md`.

## Entry 005: Phase 4 Baselines and Phase 5 CXI Adapter Validation
- Timestamp: 2026-10-16 21:52 UTC+7
- Benchmark Suite (`benchmarks/run_all_benchmarks.py`):
  - Task 1 (HIO/ER): 20 samples evaluated, mean R-factor = 0.1441, aligned MSE = 0.0340.
  - Task 2 (Denoising): 50 samples evaluated, mean PSNR = 55.26 dB, SSIM = 0.9970.
  - Task 3 (Inpainting): 50 samples evaluated, mean PSNR = 36.79 dB, SSIM = 0.9956.
  - Task 4 (Hit Classification): 140 samples evaluated, Accuracy = 100.0%, F1 = 1.0.
- PyTorch CPU Training Smoke Test (`scripts/run_training_smoke.py`):
  - 100 samples, 5 epochs, Adam (lr=1e-3). Loss decreased monotonically from 0.134967 to 0.089341 (33.8% reduction).
- CXI Adapter Unit Tests (`tests/test_experimental_adapter.py`):
  - `test_experimental_adapter_on_cxi_benchmark`: PASSED
  - `test_experimental_adapter_geometry_crop_and_pad`: PASSED
- Full Test Suite (`python -m pytest -q`):
  - 83 passed in 105.41s, 0 failed, 0 skipped.
