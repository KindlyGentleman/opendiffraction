# OpenDiffraction 3.0 Generation Metrics

- **Generation Timestamp**: 2026-10-16 14:38:43 UTC
- **Generator Version**: 3.0.0
- **Schema Version**: 3.0.0
- **Total Samples Generated**: 1,000
- **Independent Base Objects**: 250
- **Acquisitions per Object**: 4
- **Total Shards Written**: 7
- **Total Compressed Size**: 0.87 GB
- **Average Size per Sample**: 916.8 KB
- **Wall Time**: 90.48 seconds
- **Simulation Throughput**: 11.05 samples/second

## Split Distribution

| Split | Sample Count | Percentage | Purpose |
| :--- | :--- | :--- | :--- |
| `train` | 644 | 64.4% | In-distribution training |
| `valid` | 140 | 14.0% | Validation model selection |
| `test` | 116 | 11.6% | In-distribution benchmark test |
| `test_ood` | 100 | 10.0% | Out-of-distribution generalization test |

## Zero Data Leakage Guarantee

- **Grouping Field**: `group_id` strictly equals `base_object_id`.
- **Splitting Policy**: All 4 multi-exposure acquisitions of each physical object reside inside the exact same split.
- **Verification Result**: 0 groups appear in multiple splits (100% disjoint partitioning).
