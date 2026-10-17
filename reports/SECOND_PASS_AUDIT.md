# OpenDiffraction Second-Pass Scientific Remediation and Re-Audit

- **Document Version**: 2.0.0
- **Audit Date**: 2026-10-17
- **Base Commit**: `345095b` (`release: OpenDiffraction 3.1.0 scientific remediation and audit`)
- **Working Branch**: `remediation/v3.1-scientific-audit`
- **Audit Perspective**: Computational Optics Physicist, Scientific Python Engineer, ML Benchmark Auditor, Adversarial Research Reviewer, Reproducibility Engineer
- **Environment**: Python 3.13.7, Windows x86_64, CPU execution, h5py 3.13.0, NumPy 2.2.3, SciPy 1.15.2

---

## 1. Executive Summary & Baseline State

An independent second-pass audit was conducted on OpenDiffraction against the expanded 30-issue register (OD-001 through OD-030). The initial execution of the automated test suite confirmed that all 107 existing tests pass with 100% pass rate in 307.56 seconds.

However, deep adversarial source code inspection across physical optics, detector masking, benchmark loaders, metric definitions, and preset configurations reveals that while the primary P0 pipeline defects (OD-001 forward model routing, OD-002 detector mask leakage, OD-003 target fallback, OD-004 compact target modalities, OD-006 Fresnel spatial centering) are successfully resolved, several critical physical and evaluation discrepancies remain:

1. **Beam Displacement Optics Fallacy (OD-007)**: In `pad_object_to_detector`, `beam_center_offset` shifts the object in real space inside a zero-padded array. By the Fourier shift theorem, translating a compact object under plane-wave illumination does not shift the center of its far-field diffraction modulus on the detector.
2. **Missing Wavefront Aberrations (OD-008)**: `generate_probe` in `opendiffraction/physics/models/transmission.py` implements defocus, astigmatism, and coma, but lacks spherical aberration ($W \propto r^4$), despite `V17` setting `metadata["spherical_aberration_waves"]`.
3. **Severe Mask Silent Erasure (OD-019)**: In `opendiffraction/physics/masks.py:288-289`, `if np.mean(mask) < 0.2: mask[:] = True` silently erases severe masks and converts them into 100% valid, defect-free detectors.
4. **HIO-to-ER Fourier Coordinate Scrambling (OD-013)**: In `benchmarks/phase_retrieval.py:66`, `initial_phase = np.angle(np.fft.fft2(np.fft.ifftshift(hio_recon)))` produces uncentered Fourier coordinates, but `error_reduction` applies `ifftshift` to it, inverting the frequency coordinates by $(N/2, N/2)$ upon entry into Error Reduction.
5. **Trivial Correlation Diagnostic (OD-017)**: In `benchmarks/eval_phase_retrieval.py:68`, the diagnostic metric computes `pearsonr(target.ravel(), np.clip(target.ravel(), 0, 1))`, correlating the ground truth with clipped ground truth instead of evaluating the reconstruction.
6. **Global vs Windowed SSIM Divergence (OD-016)**: `ssim_2d` in `opendiffraction/tasks/metrics.py` evaluates single global scalar variances and covariances across the entire 2D image rather than local sliding Gaussian windows (Wang et al., 2004).
7. **Preset & Profile Smoke Mismatch (OD-020, OD-025)**: The `quick` preset generates 48 samples with `acquisitions_per_object = 4`, creating only 12 objects (V01 through V12) and failing to exercise profiles V13 through V48 during smoke testing.
8. **Manifest Identity Bypassing (OD-005)**: Benchmark scripts loop sequentially over dataset indices rather than mapping exact `sample_id` entries from task manifests.
9. **Incomplete CI Verification (OD-027)**: `.github/workflows/tests.yml` executes only 6 test files, ignoring loaders, storage, benchmarks, and regression suites.

---

## 2. 30-Issue Master Status Matrix

| ID | Sev | Category | Description | Status in Current Working Tree |
|:---:|:---:|:---|:---|:---:|
| **OD-001** | P0 | physics_dispatch | F1-F6 model labels dispatched through production physics | **VERIFIED** (commit 345095b) |
| **OD-002** | P0 | masking_leakage | Detector input strictly zeroed out behind occlusions | **VERIFIED** (commit 345095b) |
| **OD-003** | P0 | target_fallback | Explicit KeyError when clean supervised targets are missing | **VERIFIED** (commit 345095b) |
| **OD-004** | P0 | compact_modalities | Explicit storage schemas for compact vs extended tasks | **VERIFIED** (commit 345095b) |
| **OD-005** | P0 | manifest_enforcement | Benchmark loaders enforce exact manifest sample_id mapping | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-006** | P1 | fresnel_propagation | Fresnel propagation spatial centering and zero-distance identity | **VERIFIED** (commit 345095b) |
| **OD-007** | P1 | beam_displacement | Beam displacement shifts detector center rather than real object | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-008** | P1 | profile_effects | Profile parameters (spherical aberration, coma) modulate probe | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-009** | P1 | background_scattering | Background halos, flare, and cosmic hits connected to production | **VERIFIED** (commit 345095b) |
| **OD-010** | P1 | f6_euler_orientation | 3D Euler volume rotation invariant across shots | **VERIFIED** (commit 345095b) |
| **OD-011** | P1 | manifest_physics | Geometry metadata synchronized with active scenarios S1-S6 | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-012** | P1 | dpr_scaling_contract | DPR target scaling distinguished from physical amplitude | **VERIFIED** (commit 345095b) |
| **OD-013** | P1 | hio_er_conventions | Fourier coordinate and phase transfer between HIO and ER | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-014** | P1 | missing_data_constraints | Masked Fourier pixels unconstrained during modulus updates | **VERIFIED** (commit 345095b) |
| **OD-015** | P1 | r_factor_definition | Standard crystallographic Fourier R-factor implementation | **VERIFIED** (commit 345095b) |
| **OD-016** | P1 | ssim_windowing | Standard local windowed SSIM vs global scalar SSIM | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-017** | P1 | invalid_correlation | Replace target vs clip(target) with reconstruction correlation | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-018** | P1 | hio_support_assumptions | Separate support information regimes (oracle, loose, estimated) | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-019** | P1 | silent_mask_replacement | Remove silent mask reset when valid coverage < 20% | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-020** | P1 | quick_profile_coverage | Quick preset exercises all 48 profiles (1 per profile) | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-021** | P1 | memory_safety | IncrementalShardWriter streaming (<120 MiB RAM peak) | **VERIFIED** (commit 345095b) |
| **OD-022** | P1 | resume_validation | SHA-256 config hash and sample checksum verification | **VERIFIED** (commit 345095b) |
| **OD-023** | P1 | hit_finding_balance | Balanced 4-class Hit Finding with 3-sigma noise floor gating | **VERIFIED** (commit 345095b) |
| **OD-024** | P1 | localized_metrics | Inpainting evaluated with Masked Region MAE and RMSE | **VERIFIED** (commit 345095b) |
| **OD-025** | P2 | preset_consistency | Harmonize sample counts across CLI, YAML, and README | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-026** | P2 | release_consistency | Version 3.1.0 synchronized across metadata and documentation | **VERIFIED** (commit 345095b) |
| **OD-027** | P2 | ci_coverage | CI executes complete unit and regression test suite | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-028** | P2 | physics_unit_compatibility | Mathematical oracles (Airy, Parseval, Friedel, Fresnel) | **VERIFIED** (commit 345095b) |
| **OD-029** | P2 | task_scope_truthfulness | Truthful scope: T1-T4 benchmarked, T5-T12 manifest roadmaps | **SOURCE-CONFIRMED (NEEDS FIX)** |
| **OD-030** | P2 | cxi_experimental_scope | Explicit disclosure of synthetic format vs real beamtime | **SOURCE-CONFIRMED (NEEDS FIX)** |

---

## 3. Detailed Root Cause Analysis for Issues Requiring Remediation

### OD-007: Beam-Center Displacement via Object Translation
- **File**: `opendiffraction/physics/forward.py:81-125` (`pad_object_to_detector`)
- **Defect**: The function shifts the real-space object position:
  ```python
  y0_adj = y0 + dy_int
  x0_adj = x0 + dx_int
  ```
  In coherent far-field diffraction, the Fraunhofer intensity is:
  $$I(\mathbf{q}) = |\mathcal{F}\{f(\mathbf{r} - \mathbf{r}_0)\}|^2 = |F(\mathbf{q}) e^{-i 2\pi \mathbf{q} \cdot \mathbf{r}_0}|^2 = |F(\mathbf{q})|^2$$
  The intensity distribution on the detector is invariant under real-space translations of the specimen. Consequently, profiles V19 (beam pointing jitter) and V40 (off-center hit) do not exhibit the intended diffraction pattern shift on the detector array.
- **Remedy**:
  1. For detector-center miscalibration or beam pointing shifts, apply a coordinate shift to the diffraction intensity on the detector:
     $$I_{\text{observed}}(\mathbf{q}) = I_{\text{ideal}}(\mathbf{q} - \Delta\mathbf{q})$$
  2. For incident beam pointing jitter, tilt the illumination wavevector by adding a phase ramp in real space:
     $$\psi_{\text{inc}}(\mathbf{r}) = \exp(i 2\pi \mathbf{k}_{\perp} \cdot \mathbf{r})$$

### OD-008: Missing Spherical Aberration in Optical Probe
- **File**: `opendiffraction/physics/models/transmission.py:13-66` (`generate_probe`)
- **Defect**: `generate_probe` computes phase as:
  ```python
  phase = defocus_rad * (r_norm**2)
  if abs(astigmatism_rad) > 1e-5:
      phase += astigmatism_rad * (r_norm**2) * np.cos(2 * (theta_jit - astigmatism_angle))
  if abs(coma_rad) > 1e-5:
      phase += coma_rad * (r_norm**3) * np.cos(theta_jit)
  ```
  It has no term for primary spherical aberration ($W \propto r^4$). In `profiles_48.py:679`, profile `V17` sets `metadata["spherical_aberration_waves"] = float(rng.uniform(0.5, 1.5))`, but this value is never passed to `generate_probe` or evaluated.
- **Remedy**: Add `spherical_aberration_rad` parameter to `generate_probe`, evaluate $W_{\text{sph}} = \text{spherical\_aberration\_rad} \cdot r_{\text{norm}}^4$, and pass it from `forward.py` when profile `V17` is simulated.

### OD-019: Silent Mask Reset on Severe Degradations
- **File**: `opendiffraction/physics/masks.py:288-290`
- **Defect**:
  ```python
  # Ensure mask never completely wipes out the detector
  if np.mean(mask) < 0.2:
      mask[:] = True
  ```
  When severe compound defects (e.g., broad module gaps, large beamstops, and high irregular dropouts) reduce valid detector coverage below 20%, the code silently erases all masks and returns a 100% valid detector!
- **Remedy**: Remove the silent `mask[:] = True` overwrite. Provide an explicit mask policy: allow severe masks (clamped to a small non-zero minimum valid pixel threshold such as at least 100 valid pixels) or resample with warning, but never silently turn a defective mask into a pristine detector.

### OD-013: Fourier Coordinate Scrambling Between HIO and ER
- **File**: `benchmarks/phase_retrieval.py:66-74`
- **Defect**:
  ```python
  initial_phase = np.angle(np.fft.fft2(np.fft.ifftshift(hio_recon)))
  er_recon, er_errors = error_reduction(..., initial_phase=initial_phase)
  ```
  `np.fft.fft2` returns standard uncentered Fourier frequencies. However, `error_reduction` line 60 executes:
  ```python
  curr_phase = np.fft.ifftshift(initial_phase)
  ```
  Applying `ifftshift` to an already uncentered array shifts the zero frequency by $(N/2, N/2)$, inverting the phase spatial frequencies and degrading the starting point of Error Reduction.
- **Remedy**: Pass centered phase `np.fft.fftshift(np.angle(np.fft.fft2(np.fft.ifftshift(hio_recon))))`, or support passing the reconstructed real-space object `hio_recon` directly into ER to eliminate redundant phase conversions.

### OD-016: Global Scalar SSIM vs Standard Windowed SSIM
- **File**: `opendiffraction/tasks/metrics.py:50-64`
- **Defect**: `ssim_2d` calculates global mean, global variance, and global covariance across the entire image. Standard SSIM (Wang et al., 2004) computes these statistics locally inside an $11 \times 11$ sliding Gaussian window and averages the local SSIM map.
- **Remedy**: Integrate `skimage.metrics.structural_similarity` when `scikit-image` is installed, and implement a standard windowed fallback with uniform/Gaussian filtering for standalone CPU environments.

### OD-017: Diagnostic Pearson Correlation Bug
- **File**: `benchmarks/eval_phase_retrieval.py:68`
- **Defect**:
  ```python
  r_val, _ = pearsonr(target.ravel(), np.clip(target.ravel(), 0, 1))
  ```
  The code correlates `target` with `np.clip(target, 0, 1)`. The reconstructed object is completely ignored!
- **Remedy**: Compute the correlation between the aligned reconstructed object and the ground truth:
  ```python
  aligned_recon = align_twin_and_translation(res["reconstructed"], target)
  r_val, _ = pearsonr(aligned_recon.ravel(), target.ravel())
  ```

### OD-020 & OD-025: Preset & Smoke Test Profile Inconsistency
- **Files**: `opendiffraction/generator/pipeline.py:81-100`, `configs/quick.yaml`, `README.md`
- **Defect**: `build_master_task_plan` computes `num_objects = ceil(total_samples / acquisitions_per_object)`. With default `acquisitions_per_object = 4`, `--preset quick` (48 samples) only creates 12 base objects, sampling exclusively `V01` through `V12`. Profiles `V13` through `V48` are never exercised.
- **Remedy**: In `pipeline.py`, when preset is `quick`, set `acquisitions_per_object = 1` so that all 48 unique profiles are generated (1 sample per profile). Harmonize `quick.yaml`, `starter.yaml`, and `README.md`.

### OD-005: Benchmark Manifest Sample Identity Enforcement
- **Files**: `opendiffraction/io/loader.py`, `opendiffraction/io/torch_dataset.py`, `benchmarks/eval_*.py`
- **Defect**: `OpenDiffractionDataset` has no parameter for `manifest_path`. In `benchmarks/eval_*.py`, evaluations loop over `range(num_samples)` without verifying that the loaded samples match the manifest `sample_id` ordering.
- **Remedy**: Provide a unified `sample_id` index mapping in both `OpenDiffractionDataset` and `OpenDiffractionTorchDataset`, and ensure benchmark evaluators look up samples strictly by `sample_id`.

### OD-027: CI Test Suite Incompleteness
- **File**: `.github/workflows/tests.yml`
- **Defect**: Line 34 runs only 6 test files, omitting storage, dataset loaders, benchmark scripts, and all new remediation test suites.
- **Remedy**: Update `.github/workflows/tests.yml` to execute `python -m pytest tests/ -v`, covering all 107+ tests.

---

## 4. Remediation Execution Plan

The remediation will be executed sequentially following fail-first test principles:

1. **Step 1 (Fail-First Tests)**: Write failing regression tests for OD-005, OD-007, OD-008, OD-013, OD-016, OD-017, OD-019, OD-020, OD-025.
2. **Step 2 (Optics & Physics Corrections)**:
   - Implement beam displacement detector shift / tilt in `forward.py` (OD-007).
   - Add spherical aberration $r^4$ to `generate_probe` in `transmission.py` and connect V17 (OD-008).
   - Remove silent mask erasure in `masks.py` (OD-019).
3. **Step 3 (Algorithms & Metrics Corrections)**:
   - Fix HIO-to-ER state transfer and Fourier coordinate conventions (OD-013).
   - Implement windowed SSIM in `metrics.py` (OD-016).
   - Fix reconstruction diagnostic correlation in `eval_phase_retrieval.py` (OD-017).
   - Support oracle and loose support regimes in `phase_retrieval.py` (OD-018).
4. **Step 4 (Data Loaders & Presets Corrections)**:
   - Add manifest `sample_id` enforcement to `loader.py` and `torch_dataset.py` (OD-005).
   - Correct `--preset quick` profile coverage to all 48 profiles (OD-020, OD-025).
   - Synchronize scenario geometry metadata S1-S6 (OD-011).
5. **Step 5 (CI, Packaging & Documentation)**:
   - Expand `.github/workflows/tests.yml` to run full test suite (OD-027).
   - Update `TASKS.md`, `REMEDIATION_PROGRESS.md`, and issue register.
   - Run complete test suite and verify 100% pass rate.
