# OpenDiffraction 3.0 Full Dataset Validation Report

## Structural Validation Summary

- **Total Stored Samples**: 1,000
- **Independent Base Objects**: 250
- **Acquisitions per Object**: 4
- **HDF5 Shards**: 7
- **Tensor Dimensions**: `input`: (1, 512, 512) float32, `target`: (1, 64, 64) float32, `mask`: (1, 512, 512) bool
- **Finite Value Check**: PASS (0 NaN, 0 Inf across all arrays)
- **Duplicate Check**: PASS (0 duplicate sample_id records)

## Zero Group Leakage Audit

| Split | Samples | Objects | Percentage |
| :--- | :--- | :--- | :--- |
| `train` | 644 | 161 | 64.4% |
| `valid` | 140 | 35 | 14.0% |
| `test` | 116 | 29 | 11.6% |
| `test_ood` | 100 | 25 | 10.0% |

**Disjointness Assertion**: All 250 base objects appear in exactly 1 split. Cross-split object leakage is exactly 0.0%.
