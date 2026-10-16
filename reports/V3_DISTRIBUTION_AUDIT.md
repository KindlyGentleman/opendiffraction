# OpenDiffraction 3.0 Statistical Distribution Audit

## Global Intensity and Mask Statistics

- **Minimum Measured Intensity**: -17.348
- **Maximum Measured Intensity**: 2.500e+04
- **Mean Frame Intensity**: 6.994
- **Missing Pixel Fraction (Mean)**: 0.94%
- **Missing Pixel Fraction (Range)**: [0.00%, 3.02%]

## Hit Finding Classification Distribution

| Hit Status | Count | Percentage |
| :--- | :--- | :--- |
| `single_hit` | 940 | 94.0% |
| `multi_particle_hit` | 20 | 2.0% |
| `off_center_hit` | 20 | 2.0% |
| `non_hit` | 20 | 2.0% |

## Four Benchmark Task Manifest Breakdown

| Task | Folder | Train | Val | Test | Test OOD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TASK_1 | `manifests/task_1/` | 644 | 140 | 116 | 100 |
| TASK_2 | `manifests/task_2/` | 644 | 140 | 116 | 0 |
| TASK_3 | `manifests/task_3/` | 644 | 140 | 116 | 0 |
| TASK_4 | `manifests/task_4/` | 644 | 140 | 116 | 100 |
