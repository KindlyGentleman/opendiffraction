# OpenDiffraction Asset Cleanup and Optimization Proposal

**Audit Date**: October 16, 2026  
**Policy**: Lossless deduplication preserving 100% scientific data and legible figures  

---

## 1. Proposed Visual Asset Cleanup in `kaggle_release/examples/`

Retain only the 5 essential publication figures that are actively embedded in the Kaggle documentation:
1. `figure_1_workflow.png` (549 KB) - Full forward simulation pipeline
2. `figure_7_models_f1_to_f6.png` (433 KB) - Modular forward models F1 to F6
3. `figure_2_specimens.png` (127 KB) - Synthetic specimen morphologies
4. `figure_10_phase_retrieval.png` (57 KB) - Baseline phase retrieval reconstruction
5. `figure_16_tasks_summary.png` (506 KB) - 12 machine learning benchmark tasks specification

Remove the 4 unreferenced, uncompressed raw image files:
- Remove `kaggle_release/examples/kaggle_cover.png` (-8.28 MB)
- Remove `kaggle_release/examples/figure_4_photon_sweep.png` (-6.18 MB)
- Remove `kaggle_release/examples/figure_5_coherence_sweep.png` (-4.58 MB)
- Remove `kaggle_release/examples/figure_3_geometry_diffraction.png` (-3.18 MB)

**Impact**: Shrinks `examples/` from **23.9 MB down to 1.67 MB (93% reduction)** with ZERO loss of documentation figures.

---

## 2. Scientific Data Preservation Guarantee

- **Zero Data Loss**: All 4 HDF5 shards (`train_000.h5`, `valid_000.h5`, `test_000.h5`, `test_ood_000.h5`) will remain completely untouched and bitwise identical.
- **Zero Metadata Loss**: `metadata.csv` and all 12 task manifests in `tasks/` remain completely intact.
- **CXI Interoperability**: `opendiffraction_benchmark.cxi` will be preserved intact.

---

## 3. Checksum Regeneration Strategy

After deduplication, generate an exhaustive `checksums.sha256` that verifies:
- Root documentation (`README.md`, `DATASET_CARD.md`, `DATASHEET.md`, `LICENSE`, `LICENSE-DATA.md`, `CITATION.cff`)
- Metadata (`metadata.csv`, `configuration.json`)
- Standard containers (`opendiffraction_benchmark.cxi`)
- All HDF5 data shards (`data/*.h5`)
- All task manifests (`tasks/*.csv`)
- All embedded example figures (`examples/*.png`)
