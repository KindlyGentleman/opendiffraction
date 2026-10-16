# OpenDiffraction 3.0 Kaggle Publication Report

## 1. Executive Summary

This report documents the audited publication and remote verification of OpenDiffraction 3.0 on Kaggle:
- **Dataset Title**: OpenDiffraction: Synthetic CDI Patterns
- **Kaggle URL**: [https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi](https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi)
- **Dataset ID**: 12630148
- **Owner**: `hamadmuhammadkamal` (Muhammad Kamal)
- **Release Version**: 3.0.0 (Validated 1,000-Observation Training Pilot)
- **Status**: PASS (Published, verified `ready`, remote file index confirmed, downloadable bundle verified).

---

## 2. Remote Dataset Inventory on Kaggle

The remote file inventory indexed on Kaggle confirms the presence of all 1,000 synthetic observations across 7 HDF5 shards:

| Remote Path | Size (Bytes) | Creation Date (UTC) | Content Description |
| :--- | :--- | :--- | :--- |
| `metadata.csv` | 282,305 | 2026-10-16 15:27:05 | Master catalog of 1,000 observations |
| `data/train_000.h5` | 191,107,013 | 2026-10-16 15:27:11 | Training split shard 0 (200 samples) |
| `data/train_001.h5` | 187,674,552 | 2026-10-16 15:27:11 | Training split shard 1 (200 samples) |
| `data/train_002.h5` | 187,802,707 | 2026-10-16 15:27:12 | Training split shard 2 (200 samples) |
| `data/train_003.h5` | 42,858,322 | 2026-10-16 15:27:09 | Training split shard 3 (44 samples) |
| `data/valid_000.h5` | 122,428,343 | 2026-10-16 15:27:12 | Validation split shard 0 (140 samples) |
| `data/test_000.h5` | 109,599,389 | 2026-10-16 15:27:08 | In-distribution test shard (116 samples) |
| `data/test_ood_000.h5` | 97,376,869 | 2026-10-16 15:27:08 | Out-of-distribution test shard (100 samples) |
| `opendiffraction_benchmark.cxi` | 109,717,345 | 2026-10-16 15:27:06 | Benchmark CXI NeXus container |
| `configuration.json` | 1,037 | 2026-10-16 15:26:54 | Dataset hyperparameters and settings |
| `checksums.sha256` | 3,021 | 2026-10-16 15:26:54 | SHA-256 manifest of all release files |
| `DATASET_CARD.md` | 4,413 | 2026-10-16 15:26:54 | Comprehensive dataset documentation |
| `README.md` | 4,413 | 2026-10-16 15:26:54 | Usage instructions and PyTorch examples |
| `tasks.zip` | 309,248 | 2026-10-16 15:26:54 | Standardized manifests for tasks T1-T12 |
| `examples/` | ~1.66 MB | 2026-10-16 15:27:05 | 5 visual workflow & forward model diagrams |

Total uncompressed payload: ~1.05 GB (0.98 GiB compressed archive).

---

## 3. Remote Verification Check

1. Metadata Verification:
   - Fetched `dataset-metadata.json` via Kaggle API: verified `title`, `description`, `keywords`, and `licenses`.
   - Verified that Version 3.0 description was rendered accurately.
2. File Download Verification:
   - Successfully downloaded `metadata.csv` directly from Kaggle.
   - Row count verified: exactly 1,000 samples.
   - SHA-256 checksum verified against local reference.
   - Full 0.98 GiB archive download initiated and verified through Kaggle API.

---

## 4. Publication Certification

The OpenDiffraction 3.0 dataset release is certified as successfully published, live, and publicly accessible on Kaggle.
