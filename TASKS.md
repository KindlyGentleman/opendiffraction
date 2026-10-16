# OpenDiffraction 3.0 Task Management

## Phase 0: Audit Existing Code and Data
- [x] Run existing test suite and record baseline results (78 passed, 0 failed, 0 skipped).
- [x] P0.1: Audit resume implementation and identify interruption vulnerabilities.
- [x] P0.2: Audit object identity and group data leakage across splits.
- [x] P0.3: Audit forward physics models, FFT centering, coherence, scaling, and DPR compatibility.
- [x] P0.4: Audit metadata schemas, configuration fields, and version consistency.
- [x] Fix critical resume, split, and metadata issues identified in Phase 0.
- [x] Deliver Phase 0 audit reports:
  - `reports/V3_BASELINE_AUDIT.md`
  - `reports/SCIENTIFIC_MODEL_AUDIT.md`
  - `reports/RESUME_SAFETY_AUDIT.md`
  - `reports/STORAGE_BASELINE.md`
- [x] Execute fault-injection resume tests (`scripts/verify_resume.py`: All 3 tests passed).

## Phase 1: Large-Scale Synthetic Dataset Generation (1,000 Target)
- [x] P1.1: Design dataset identity: 250 base objects x 4 diverse acquisitions = 1,000 observations.
- [x] P1.2: Verify physical model compatibility for supported profiles (F1 to F6).
- [x] P1.3: Procedural specimen generation across 15 specimen families with recorded seeds.
- [x] P1.4: Diffraction acquisition condition matrix (flux, partial coherence, masks, noise, background).
- [x] P1.5: Group-disjoint deterministic splitting (64.4% train, 14.0% val, 11.6% test, 10.0% test_ood).
- [x] P1.6: Metadata design with typed fields and documented units.
- [x] P1.7: Staged generation execution:
  - [x] Stage A: 48 samples (integration smoke test)
  - [x] Stage B: 480 samples (statistical and compression benchmarks)
  - [x] Stage C: 1,000 samples (engineering pilot)
  - [x] Stage D: 1,000 samples (production dataset in `output_v3/`)
- [x] Deliverables:
  - `output_v3/` dataset (7 HDF5 shards, 1,000 records, 0.87 GB)
  - `reports/GENERATION_METRICS.md`
  - `reports/OBJECT_CATALOG.md`
  - `reports/ACQUISITION_MATRIX.md`

## Phase 2: Optimize Storage, Resume, and PyTorch Loading
- [x] P2.1: Measure storage baseline on representative subsets.
- [x] P2.2: Benchmark compression filters (gzip-1, gzip-4, lzf, uncompressed).
- [x] P2.3: Implement compact training and extended scientific storage modes.
- [x] P2.4: Optimize HDF5 chunk shape, shard size, atomic writes, and manifest verification.
- [x] P2.5: Preflight resource protection (memory, disk, workers).
- [x] P2.6: Refactor dataset loader with lazy loading and stable indexing.
- [x] P2.7: Implement `OpenDiffractionTorchDataset` with safe per-worker lazy file opening and context management.
- [x] P2.8: Document and implement scientific normalization transforms.
- [x] Deliverables:
  - `opendiffraction/io/torch_dataset.py`
  - `opendiffraction/generator/storage.py`
  - `scripts/benchmark_storage.py`
  - `scripts/verify_resume.py`
  - `reports/STORAGE_OPTIMIZATION.md`

## Phase 3: Scientific Data Quality and Training-Ready Tasks
- [x] P3.1: Full structural validation across all 1,000 samples (0 NaN, 0 Inf, zero duplicates).
- [x] P3.2: Stratified physics validation (Fourier, Parseval, Poisson, coherence, mask integrity).
- [x] P3.3: Statistical distribution audit.
- [x] P3.4: Four ML benchmark task manifests (T1 Phase Retrieval, T2 Denoising, T3 Inpainting, T4 Hit Finding).
- [x] P3.5: Comprehensive dataset card (`reports/V3_DATASET_CARD.md`).
- [x] Deliverables:
  - `reports/V3_FULL_DATASET_VALIDATION.md`
  - `reports/V3_PHYSICS_VALIDATION.md`
  - `reports/V3_DISTRIBUTION_AUDIT.md`
  - `reports/V3_DATASET_CARD.md`
  - `manifests/task_1/` to `manifests/task_4/` (train.csv, val.csv, test.csv, test_ood.csv)

## Phase 4: Minimal Reproducible Benchmark Infrastructure
- [x] P4.1: Benchmark evaluation API and metric runners.
- [x] P4.2: Execute baseline algorithms for T1, T2, T3, T4 on CPU.
- [x] P4.3: PyTorch CPU training smoke test (lightweight ConvNet on CPU, 33.8% loss reduction in 5 epochs).
- [x] P4.4: Benchmark protocol and reproducibility guidelines.
- [x] Deliverables:
  - `benchmarks/eval_phase_retrieval.py`
  - `benchmarks/eval_denoising.py`
  - `benchmarks/eval_inpainting.py`
  - `benchmarks/eval_hit_classification.py`
  - `benchmarks/run_all_benchmarks.py`
  - `reports/BASELINE_VALIDATION.md`
  - `reports/TRAINING_SMOKE_TEST.md`
  - `reports/BENCHMARK_PROTOCOL.md`

## Phase 5: Limited Experimental Data Compatibility
- [x] P5.1: Experimental source selection from CXIDB (id-62, id-81, id-91, id-164).
- [x] P5.2: Read-only experimental CXI/HDF5 adapter (`opendiffraction/io/experimental.py`).
- [x] P5.3: Synthetic-to-real comparison and quantitative domain gap analysis.
- [x] P5.4: Adapter verification and automated unit testing (`tests/test_experimental_adapter.py`).
- [x] Deliverables:
  - `opendiffraction/io/experimental.py`
  - `reports/EXPERIMENTAL_DATA_CANDIDATES.md`
  - `reports/SYNTHETIC_REAL_DOMAIN_GAP.md`
  - `reports/EXPERIMENTAL_ADAPTER_VALIDATION.md`
