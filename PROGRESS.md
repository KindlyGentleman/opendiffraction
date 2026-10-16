# OpenDiffraction 3.0 Progress Tracking

## Milestone Status Overview
- **Phase 0 (Audit Existing Code and Data)**: COMPLETED
  - Discovered and resolved dynamic shard overwriting on resume.
  - Eliminated group data leakage by coupling specimen identity across all multi-exposure acquisitions.
  - Resolved conflicting version strings (`3.0.0` canonical single source of truth in `opendiffraction/version.py`).
  - Implemented transactional atomic writes with SHA-256 verification.
  - Passed fault-injection resume tests (`scripts/verify_resume.py`).

- **Phase 1 (Generate 1,000 Scientific Samples)**: COMPLETED
  - Executed staged verification (Stage A: 48, Stage B: 480, Stage C: 1,000).
  - Generated 1,000 validated synthetic diffraction samples in `output_v3/` (250 base objects x 4 physical acquisitions).
  - All 48 physical profiles (V01 - V48) and 6 forward models (F1 - F6) thoroughly represented.
  - Sharded across 7 verified HDF5 shards (0.87 GB total, ~916.8 KB/sample).
  - Zero group leakage (100% disjoint groups across train, val, test, test_ood).
  - Exported NeXus/CXI benchmark file (`output_v3/opendiffraction_benchmark.cxi`).

- **Phase 2 (Storage, Resume, and PyTorch Loading)**: COMPLETED
  - Implemented `OpenDiffractionTorchDataset` with multiprocess-safe per-worker lazy file handles and context management.
  - Tested compact storage mode with HDF5 chunking and gzip level 4 compression.
  - Supported task-specific tensor collation (`phase_retrieval`, `denoising`, `inpainting`, `hit_classification`).

- **Phase 3 (Scientific Data Quality and Tasks)**: COMPLETED
  - Validated structural and numerical integrity (0 NaN, 0 Inf, zero duplicates).
  - Exported four dedicated task benchmark manifests into `manifests/`:
    - `manifests/task_1/` (Phase Retrieval: train, val, test, test_ood)
    - `manifests/task_2/` (Denoising: train, val, test)
    - `manifests/task_3/` (Inpainting: train, val, test)
    - `manifests/task_4/` (Hit Classification: train, val, test, test_ood)
  - Produced scientific audit reports and official Dataset Card.

- **Phase 4 (Minimal Benchmark Infrastructure)**: COMPLETED
  - Implemented evaluation modules: `benchmarks/eval_phase_retrieval.py`, `benchmarks/eval_denoising.py`, `benchmarks/eval_inpainting.py`, `benchmarks/eval_hit_classification.py`.
  - Built master benchmark runner: `benchmarks/run_all_benchmarks.py`.
  - Executed classical baselines on CPU across all four tasks.
  - Ran PyTorch CPU training smoke test with `TinyDiffractionReconNet` (33.8% loss reduction over 5 epochs).
  - Generated `reports/BASELINE_VALIDATION.md`, `reports/TRAINING_SMOKE_TEST.md`, and `reports/BENCHMARK_PROTOCOL.md`.

- **Phase 5 (Experimental Compatibility Pathway)**: COMPLETED
  - Surveyed candidate experimental datasets on CXIDB (`reports/EXPERIMENTAL_DATA_CANDIDATES.md`).
  - Implemented read-only CXI experimental data adapter in `opendiffraction/io/experimental.py`.
  - Passed unit tests on CXI NeXus structures and spatial resizing (`tests/test_experimental_adapter.py`).
  - Documented quantitative domain-gap analysis in `reports/SYNTHETIC_REAL_DOMAIN_GAP.md` and adapter validation in `reports/EXPERIMENTAL_ADAPTER_VALIDATION.md`.
