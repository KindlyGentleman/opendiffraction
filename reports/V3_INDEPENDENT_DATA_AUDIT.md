# OpenDiffraction 3.0 Independent Scientific Data Audit

## 1. Executive Summary

- **Total Audited Samples**: 1,000
- **HDF5 Shard Count**: 7
- **Independent Base Objects**: 250
- **Acquisitions per Object**: 4
- **Total Storage Footprint**: 895.35 MB (0.9388 GB / 0.8744 GiB)
- **Average Size per Sample**: 916.8 KB
- **Sequential Read Throughput**: 122.4 samples/s

## 2. Structural & Numerical Integrity

- **All Inputs Finite (No NaN, No Inf)**: True
- **All Targets Finite (No NaN, No Inf)**: True
- **All Masks Boolean**: True
- **Sample ID Uniqueness**: 100.0% (0 duplicates)
- **Shard-to-Metadata Mapping Consistency**: True

## 3. Object Grouping and Data Leakage Audit

- **Grouping Hierarchy**: 250 base objects, each observed under 4 distinct physical acquisition conditions.
- **Exact Ground-Truth Match**: Verified that `target` is bit-for-bit identical across all 4 acquisitions of each base object.
- **Input Variation**: Verified that `input` differs across all 4 acquisitions reflecting physical noise, flux, and mask variations.
- **Split Disjointness**: Pairwise intersections between `train`, `valid`, `test`, `test_ood` are strictly empty (0 leaked objects).

## 4. Split Distribution

| Split | Sample Count | Percentage | Base Objects |
| :--- | :--- | :--- | :--- |
| `train` | 644 | 64.4% | 161 |
| `valid` | 140 | 14.0% | 35 |
| `test` | 116 | 11.6% | 29 |
| `test_ood` | 100 | 10.0% | 25 |

## 5. Scientific Forward Models and Profiles

- **Forward Models Present**: F1, F2, F3, F4, F5, F6
- **Profiles Present**: All 48 profiles verified.
- **Target Scaling**: Conforms strictly to DPR `false_scale=True` convention for direct compatibility with published neural phase retrieval models.
