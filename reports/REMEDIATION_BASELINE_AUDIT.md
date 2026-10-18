# OpenDiffraction Baseline Remediation Audit

Date: 2026-10-17
Git commit: `f2fcab8`
Branch: `remediation/v3.1-scientific-audit`
Environment: Python 3.13.7 (Windows), 15.7 GiB total RAM (1.97 GiB free at baseline), 15.2 GiB free disk on D:.

## 1. Executive Summary

This audit assesses the state of OpenDiffraction 3.0 prior to remediation. While the existing test suite passes (83 tests passed in 156 seconds), inspection of the source code reveals five critical (P0) scientific and data integrity defects, alongside five major (P1) engineering issues.

Crucially:
1. All profiles in production dataset generation execute simple Fraunhofer diffraction. The forward model labels F3, F4, F5, and F6 exist only in CSV metadata and are not actually executed.
2. Inaccessible detector regions (beamstop, module gaps, dead pixels) are not properly masked in the stored model input. The unmasked detector array is stored and fed to models, introducing critical information leakage.
3. The Fresnel propagator contains an erroneous `fftshift` on the spatial output field, displacing the object by (N/2, N/2) even at zero propagation distance.
4. The PyTorch loader silently substitutes noisy measurements as their own clean targets when clean arrays are missing from compact shards.
5. Classical phase retrieval algorithms (HIO and ER) constrain masked pixels to zero amplitude, penalizing reconstructors with non-physical constraints and skewing benchmark conclusions.

## 2. Claim-to-Evidence Matrix

| Claim in Documentation | Source Code Location | Existing Test Status | Actual Physical Evidence | Finding |
|---|---|---|---|---|
| Six physical forward models (F1-F6) executed in dataset generation | `opendiffraction/generator/pipeline.py:222` | Passing (`test_pipeline.py`) | F3-F6 models are defined in `models/` but never dispatched; `simulate_diffraction` is called unconditionally | **FAIL (OD-001)** |
| Proper detector masking without ground truth leakage | `opendiffraction/physics/forward.py:240-249` | Passing (`test_forward.py`) | `observed_masked` is computed and discarded; unmasked `intensity_measured` is returned and stored | **FAIL (OD-002)** |
| Near-field Fresnel propagation via angular spectrum method | `opendiffraction/physics/models/fresnel.py:82` | Passing (`test_models.py`) | Spurious `fftshift` applied to real-space field; violates z=0 identity | **FAIL (OD-003)** |
| Supervised ground truth available for restoration benchmarks | `opendiffraction/io/torch_dataset.py:176-188` | Passing (`test_torch_dataset.py`) | Silent fallback sets clean target to noisy input when `intensity_expected` is missing | **FAIL (OD-004)** |
| Classical HIO/ER CDI baselines properly benchmarked | `opendiffraction/physics/phase_retrieval.py:58-62` | Passing (`test_phase_retrieval.py`) | Masked pixels forced to zero modulus; residual formula is normalized squared error, not R-factor | **FAIL (OD-005)** |
| Deterministic acquisition variation across 4 shots | `opendiffraction/generator/pipeline.py:225` | Passing | Random holder arm injected on shot 2 regardless of profile configuration | **FAIL (OD-006)** |
| Physical wavelength, pitch, and distance metadata | `opendiffraction/generator/pipeline.py:276-278` | Passing | Fixed constants hardcoded in manifest regardless of scenario S1-S6 | **FAIL (OD-007)** |
| Bounded memory dataset generation | `opendiffraction/generator/pipeline.py:383-401` | Passing on small tests | Shard accumulated in memory before atomic write; high peak RSS | **FAIL (OD-008)** |
| Resumable generation with integrity verification | `opendiffraction/generator/storage.py:81` | Passing (`test_storage.py`) | Verification checks row count only; ignores config hash and seeds | **FAIL (OD-009)** |
| Profile sampling scales with target sample count | `opendiffraction/generator/pipeline.py:178` | Passing | `total=1000` hardcoded into `sample_profile_48` | **FAIL (OD-010)** |

## 3. Confirmed P0 Defect Details

### OD-001: Forward Model Dispatcher Bypass
In `opendiffraction/generator/pipeline.py`:
```python
# line 182:
fwd_model = meta.get("forward_model", "F1")
...
# line 222:
sim_res = simulate_diffraction(obj_amp, cfg, rng=acq_rng, object_phase=obj_phase)
```
The variable `fwd_model` is stored in the manifest dictionary (line 258), but never passed to any simulation function. Consequently, profiles V45 (Fresnel), V16-V19 (aberrated probe), V20/V46 (polychromatic), and V47 (3D particle) all generate identical scalar Fraunhofer far-field patterns.

### OD-002: Missing Pixel Detector Leakage
In `opendiffraction/physics/forward.py`:
```python
# line 240:
observed_masked = intensity_measured * mask.astype(np.float32)
...
# lines 248-249:
intensity_expected=intensity_expected.astype(np.float32),
intensity_measured=intensity_measured.astype(np.float32),
```
`observed_masked` is calculated but never returned in `SimulationResult`. In `pipeline.py:243`, `intensity_measured` (which has non-zero signal behind beamstops and gaps) is written to HDF5 as the observation. Any model trained on this field receives unmasked signal behind detector occlusions.

### OD-003: Fresnel Real-Space Centering Inversion
In `opendiffraction/physics/models/fresnel.py`:
```python
# line 79-82:
propagated_field = np.fft.ifft2(F_out)
# Shift DC to center for detector convention
field_centered = np.fft.fftshift(propagated_field)
```
The angular spectrum method operates via Fourier transforms: `ifft2(fft2(field) * H)` returns the spatial field in standard coordinate ordering matching the input. Applying `fftshift` to `propagated_field` shifts the spatial origin by `(Ny//2, Nx//2)`. When tested with propagation distance `z = 0`, the field is shifted instead of remaining identical to the input.

### OD-004: Clean Target Fallback to Input
In `opendiffraction/io/torch_dataset.py`:
```python
# lines 176-181:
if "intensity_expected" in h5_file:
    clean_arr = h5_file["intensity_expected"][row]
elif "intensity_ideal" in h5_file:
    clean_arr = h5_file["intensity_ideal"][row]
else:
    clean_arr = inp_arr  # Silent substitution of noisy input!
```
In compact storage mode (`storage_mode="compact"`), `intensity_expected` and `intensity_ideal` are omitted to save disk space. The dataset loader quietly sets the clean restoration target equal to the noisy input, breaking all supervised denoising and inpainting tasks.

### OD-005: Phase Retrieval Mask and Metric Artifacts
In `opendiffraction/physics/phase_retrieval.py`:
In `hybrid_input_output` and `error_reduction`, the Fourier modulus replacement:
```python
G_mod = modulus * np.exp(1j * np.angle(G))
```
forces the Fourier amplitude to 0 at masked detector pixels (where `modulus` is 0). In classical CDI, masked pixels are unmeasured degrees of freedom and must not be forced to zero. Doing so forces an artificial high-frequency beamstop shape into the Fourier plane, preventing convergence. Additionally, the reported error metric is:
```python
error = float(np.sum(np.abs(np.abs(G) - modulus) ** 2) / max(np.sum(modulus**2), 1e-8))
```
which is a normalized squared residual, not the standard crystallographic Fourier R-factor `sum(| |G| - |F| |) / sum(|F|)`.

## 4. Remediation Baseline Gate (Gate 0)

Status: **PASS (Baseline Recorded)**
- Existing code and test suite recorded.
- Confirmed issues registered with unique identifiers (OD-001 through OD-010).
- Dedicated remediation branch `remediation/v3.1-scientific-audit` created.
- Proceeding to Phase I: Forward-Physics Dispatcher repair.
