# OpenDiffraction Redundancy and Waste Audit Report

**Audit Date**: October 16, 2026  
**Auditor**: Independent Research Data & Software Auditor  

---

## 1. Summary of Identified Redundancies

Our systematic cryptographic hash comparison and reference tracing across repository files identified three main categories of waste:

### A. Bloated & Unreferenced Visual Assets in `kaggle_release/examples/`
The `kaggle_release/examples/` directory was uploaded to Kaggle as a 23 MB ZIP archive (`examples.zip`). Analysis revealed that over **22 MB (95%)** of this payload consists of unreferenced or uncompressed files:

| File | Size | Referencing Document | Analysis |
| :--- | :--- | :--- | :--- |
| `kaggle_cover.png` | 8.28 MB | None (README references header) | Raw RGBA canvas (2979x1562). An optimized 1600x839 RGB banner (1.25 MB) already exists in `assets/kaggle_cover.png`. |
| `figure_4_photon_sweep.png` | 6.18 MB | None in README | Unreferenced debug plot. |
| `figure_5_coherence_sweep.png` | 4.58 MB | None in README | Unreferenced debug plot. |
| `figure_3_geometry_diffraction.png` | 3.18 MB | None in README | Unreferenced debug plot. |
| **Total Redundant Asset Weight** | **22.22 MB** | - | **Safe for cleanup & deduplication.** |

### B. Duplicate Markdown Documentation Files
Identical documentation files were duplicated across multiple directory trees:
- `DATASET_CARD.md`: Duplicated 3 times (`docs/DATASET_CARD.md`, root `DATASET_CARD.md`, `kaggle_release/DATASET_CARD.md`).
- `DATASHEET.md`: Duplicated 2 times (`DATASHEET.md`, `kaggle_release/DATASHEET.md`).
- `CITATION.cff`: Duplicated 2 times (`CITATION.cff`, `kaggle_release/CITATION.cff`).
- `LICENSE`: Duplicated 2 times (`LICENSE`, `kaggle_release/LICENSE`).

### C. CXI vs. HDF5 Shards Redundancy Evaluation
- The HDF5 shards (`data/*.h5`) contain the complete dataset (48 samples, 171 MB).
- `opendiffraction_benchmark.cxi` (4.81 MB) contains 5 representative samples formatted in standard NeXus CXIDB layout.
- **Recommendation**: Retain `opendiffraction_benchmark.cxi` because it provides vital domain interoperability with native crystallography software (CXIView, CrystFEL, NeXus tools) at a negligible storage cost (4.8 MB, <3% of payload).
