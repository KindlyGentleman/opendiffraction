# Publication State Snapshot

**Audit Date**: October 16, 2026  
**Auditor Role**: Independent Scientific Reproducibility Auditor & Research Software Maintainer  
**Repository**: `https://github.com/KindlyGentleman/opendiffraction`  
**Kaggle Dataset**: `https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi`  
**Creator & Maintainer**: Muhammad Kamal (GitHub: `KindlyGentleman`, Kaggle: `hamadmuhammadkamal`)

---

## 1. Git Repository State

| Property | Value | Audit Verification Method |
| :--- | :--- | :--- |
| **Commit SHA (Base)** | `8a3dbcc1663f272a2f073d3ea747517c5a0ce516` | `git log -1 --format="%H"` |
| **Audit Branch** | `audit/licensing-citation-release-integrity` | `git branch --show-current` |
| **Default Branch** | `main` (Protected with CI status checks) | `gh repo view` |
| **Git Tags** | None present | `git tag --list` |
| **Working Tree Status**| Clean | `git status --porcelain` |
| **Software Version** | `2.0.0` | `pyproject.toml` |
| **Dataset Version** | `2.0.0` (Quick-Release Benchmark) | `configuration.json`, `metadata.csv` |

---

## 2. Kaggle Dataset State

| Property | Value | Verification Method |
| :--- | :--- | :--- |
| **Dataset Slug** | `hamadmuhammadkamal/opendiffraction-synthetic-cdi` | Kaggle API / CLI |
| **Status** | `ready` | `python -m kaggle datasets status` |
| **Current Version** | Version 3 | Kaggle API / CLI |
| **Actual Published Samples** | **48 samples** (1 per profile V01–V48) | Direct remote audit via `./audit_download` |
| **Sample Breakdown** | Train: 35, Valid: 5, Test: 3, Test OOD: 5 | Direct HDF5 shape inspection |
| **CXI File Present** | `opendiffraction_benchmark.cxi` (5 patterns, 4.81 MB) | Direct HDF5 shape inspection |
| **Public Starter Notebook** | `hamadmuhammadkamal/opendiffraction-quickstart-eda-phase-retrieval` | `python -m kaggle kernels status` |
| **Notebook Status** | `KernelWorkerStatus.COMPLETE` | Executed and verified |

---

## 3. Discovered Discrepancies Requiring Immediate Correction

1. **Authorship & Creator Metadata**:
   - `CITATION.cff` listed `"OpenDiffraction Contributors"` / `"Computational Optics Team"` instead of **Muhammad Kamal**.
   - `pyproject.toml` listed `authors = [{ name = "OpenDiffraction Contributors" }]`.
   - Obsolete URL `https://github.com/muham/opendiffraction` was present in `CITATION.cff`.
2. **Foundational DPR Paper Attribution**:
   - `CITATION.cff` contained fabricated co-authors for the Lee et al. (2025) publication (`Dongil Kim`, `Minhyuk Kim`, `Chan Kim`) instead of the verified 7 authors: Sung Yun Lee, Do Hyung Cho, Chulho Jung, Daeho Sung, Daewoong Nam, Sangsoo Kim, and Changyong Song.
   - Journal name was frequently cited as `"Nature Communications / npj Computational Materials"` instead of the accurate journal title: ***npj Computational Materials***.
3. **Licensing Separation**:
   - Single root `LICENSE` file erroneously mixed BSD 2-Clause and CC BY 4.0 terms under an ambiguous `"OpenDiffraction Contributors"` copyright header.
   - Needs strict separation: BSD 2-Clause for software (`LICENSE`), CC BY 4.0 for data (`LICENSE-DATA.md`), and an explanatory `LICENSES.md`.
4. **Sample Count Reconciliation**:
   - The public dataset contains **48 samples** (1 sample per profile V01–V48).
   - Some documentation files ambiguously referred to a 2,400-sample starter release.
   - The release must be explicitly qualified as a **48-sample quick-release benchmark**, highlighting that 1 sample per profile is intended for validation, benchmarking, and inference testing rather than large-scale supervised neural network training.
5. **Hardcoded Local Paths**:
   - `tests/test_dpr_compatibility.py` and `reports/VALIDATION_REPORT.md` contained hardcoded Windows paths (`d:\03_Proyek\...`).
   - Must be sanitized for general cross-platform reproducibility.
6. **Figure Redundancy**:
   - `assets/` and `kaggle_release/examples/` contained duplicate uncompressed PNGs, including an 8.28 MB cover image, causing bloated release packages (~23 MB for examples alone).
