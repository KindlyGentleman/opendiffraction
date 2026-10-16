# Kaggle Remote Release Audit Report

**Audit Target**: Public Kaggle Dataset `hamadmuhammadkamal/opendiffraction-synthetic-cdi`
**Audit Source**: Independently downloaded archive via Kaggle CLI into `./audit_download`
**Audit Date**: October 16, 2026
**Auditor**: Independent Scientific Reproducibility Auditor

---

## 1. Verified Remote Payload

- **Total Published HDF5 Shards**: 4
- **Total Published Samples**: **48 samples**
- **Metadata Row Count**: **48 rows**
- **Correspondence (Metadata to HDF5)**: **100% exact match (48/48)**

### Shard Breakdown

| Shard Name | Partition Split | HDF5 Sample Count | Metadata Rows | File Size (MB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `test_000.h5` | `test` | 3 | 3 | 11.08 MB | **VERIFIED** |
| `test_ood_000.h5` | `test_ood` | 5 | 5 | 18.5 MB | **VERIFIED** |
| `train_000.h5` | `train` | 35 | 35 | 123.34 MB | **VERIFIED** |
| `valid_000.h5` | `valid` | 5 | 5 | 18.48 MB | **VERIFIED** |
| **TOTAL** | **All Splits** | **48** | **48** | **171.40 MB** | **PERFECT 1:1 MATCH** |

---

## 2. Partition Splits & Variation Coverage

Distribution across dataset splits:
- **`train`**: 35 samples (72.9%)
- **`valid`**: 5 samples (10.4%)
- **`test_ood`**: 5 samples (10.4%)
- **`test`**: 3 samples (6.2%)

- **Variation Profiles**: Exactly 48 profiles (V01 through V48), 1 sample per profile.
- **Forward Models**: F1 (Fraunhofer), F2 (Partial Coherence), F3 (Thin Object Probe), F4 (Fresnel), F5 (Polychromatic), F6 (3D Particle Born Projection).
- **Out-of-Distribution (OOD)**: Exactly 5 samples in `test_ood_000.h5` representing V43, V44, V45, V46, and V48.

---

## 3. CXI Format Verification

- **CXI Filename**: `opendiffraction_benchmark.cxi` (4.70 MB)
- **CXI Specimen Subsets**: 5 samples
- **Reciprocal Grid**: (512, 512) float32
- **Real Grid**: (64, 64) float32
- **Instrument Geometry**: Wavelength, detector distance, and pixel dimensions stored under `/entry_1/instrument_1/detector_1/`
- **CXIDB Compliance**: Conforms to standard NeXus / CXI layout specifications.

---

## 4. Benchmark Task Manifests

All 12 benchmark tasks (T1 to T12) are represented with individual manifest CSV files in `tasks/`:
- **`T10_manifest.csv`**: 48 candidate samples
- **`T11_manifest.csv`**: 48 candidate samples
- **`T12_manifest.csv`**: 48 candidate samples
- **`T1_manifest.csv`**: 48 candidate samples
- **`T2_manifest.csv`**: 43 candidate samples
- **`T3_manifest.csv`**: 43 candidate samples
- **`T4_manifest.csv`**: 6 candidate samples
- **`T5_manifest.csv`**: 4 candidate samples
- **`T6_manifest.csv`**: 12 candidate samples
- **`T7_manifest.csv`**: 43 candidate samples
- **`T8_manifest.csv`**: 48 candidate samples
- **`T9_manifest.csv`**: 48 candidate samples

---

## 5. Critical Reproducibility Warning & Dataset Classification

> [!WARNING] **Quick-Release Benchmark Qualification**
> The current public release on Kaggle contains **exactly 48 samples** (1 sample per profile V01–V48).
> While 100% physically self-consistent and valid for validation, format verification, and inverse algorithm benchmarking, **48 samples are NOT sufficient for training deep neural networks from scratch**.
> The dataset is officially classified as: **OpenDiffraction 2.0 Quick-Release Benchmark**.
> Users requiring large-scale training sets (e.g. 2,400 to 96,000 samples) should use the local OpenDiffraction CLI: `python -m opendiffraction generate --preset starter` or `--preset full`.
