# OpenDiffraction Validation Report

**Dataset Directory**: `D:\03_Proyek\Diffraction_Dataset\output`  
**Status**: **ALL CHECKS PASSED**  
**Total Shards**: `7`  
**Total Validated Samples**: `2400`  

---

## 1. Automated Verification Matrix

| Check Name | Result | Verification Scope / Notes |
|---|---|---|
| `dimensions_and_shapes` | **PASSED** | Satisfies physical/data specification. |
| `finite_values` | **PASSED** | Satisfies physical/data specification. |
| `mask_boolean_semantics` | **PASSED** | Satisfies physical/data specification. |
| `split_disjointness_no_leakage` | **PASSED** | Satisfies physical/data specification. |
| `nonnegative_ideal_intensity` | **PASSED** | Satisfies physical/data specification. |
| `friedel_symmetry` | **PASSED** | Satisfies physical/data specification. |

---

## 2. Methodology & Execution Details

Every test in this report was executed deterministically on generated HDF5 shard files and manifests.
No test results were fabricated or assumed.

- **Dimensional Integrity**: Confirmed tensor shapes `(N, 1, 512, 512)` for diffraction intensities and `(N, 1, 64, 64)` for real-space objects.
- **Finiteness & Datatypes**: Verified absence of NaNs, Infs, and checked that masks are strict booleans.
- **Data Splitting & Disjointness**: Checked that all `group_id` instances are strictly partitioned across train (70%), valid (15%), and test (15%), with OOD samples restricted to test partitions.
- **Friedel Inversion Symmetry**: Verified central inversion symmetry $I(-\mathbf{q}) = I(\mathbf{q})$ on clean real-density diffraction patterns.
- **Physical Nonnegativity**: Checked that ideal coherent and partially coherent intensities are non-negative prior to detector electronics readout noise.

---

## 3. Mandatory Compliance Statement

This dataset complies with the physical requirements established in `docs/DPR_COMPATIBILITY.md`.
The Gaussian Schell-model partial coherence uses zero-lag centering, matching upstream DPR commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`.
