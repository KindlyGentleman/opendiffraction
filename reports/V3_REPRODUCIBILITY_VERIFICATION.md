# OpenDiffraction 3.0 Reproducibility Verification Report

## 1. Objective and Scope

Verify that synthetic diffraction samples can be re-simulated from their recorded random seeds and physical parameters with bit-for-bit mathematical identity.

## 2. Test Execution

- **Evaluated Samples**: 10 representative observations across splits, forward models, and acquisition variations.
- **Bit-for-Bit Identity Rate**: 10 / 10 (100.0%)
- **Array Comparisons**: `target_dpr`, `intensity_measured`, `valid_mask` all strictly bit-identical (`np.array_equal == True`).
- **Status**: `PASS`

## 3. Crash and Resume Fault-Injection

- Verified via `scripts/verify_resume.py`:
  - Orphaned `.tmp` file recovery: PASS
  - Refusal to silently overwrite existing data without `--overwrite`: PASS
  - Interrupted generation resume with zero duplicate records: PASS
