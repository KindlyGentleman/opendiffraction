# Release Consistency and Reconciliation Report

**Audit Date**: October 16, 2026  
**Status**: Audited & Reconciled  

---

## 1. Sample Count Reconciliation

| Release Manifest | Claimed Samples | Actual Remote Samples | Reconciliation Action |
| :--- | :--- | :--- | :--- |
| `kaggle_release/metadata.csv` | 48 | 48 | Confirmed exact match. |
| `output_quick/` (Local benchmark) | 48 | 48 | Confirmed exact match. |
| Legacy Documentation references | 2,400 | 48 | **Corrected**: All public documentation now explicitly identifies this release as the **48-sample Quick-Release Benchmark**. |

---

## 2. Deleted Shards Audit

- In earlier local development runs, test shards `train_001.h5`, `train_002.h5`, `train_003.h5` were generated during full parameter sweeps.
- **Audit Check**: Does `metadata.csv` reference any of these deleted shards?
- **Result**: **NO**. Every single row in `metadata.csv` references ONLY one of the four active shards: `train_000.h5`, `valid_000.h5`, `test_000.h5`, `test_ood_000.h5`.
- **Result**: Zero orphaned rows or dangling shard references exist in the dataset.

---

## 3. Checksum Manifest Reconciliation

- The downloaded `checksums.sha256` contained 10 hashes.
- `dataset-metadata.json` is consumed by the Kaggle API into dataset metadata and is not served as a discrete payload file by the Kaggle downloader.
- `tasks/` manifests and `examples/` figures were bundled inside Kaggle zip archives and were missing from the root checksum file.
- **Correction Action**: Generate a complete, authoritative `checksums.sha256` covering ALL release payload files.
