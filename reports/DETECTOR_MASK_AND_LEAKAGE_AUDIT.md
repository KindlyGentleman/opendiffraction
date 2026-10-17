# Detector Masking, Information Leakage, and Geometry Audit Report (Gate III)

**Branch**: `remediation/v3.1-scientific-audit`  
**Date**: 2026-10-17  
**Status**: PASS  
**Addressed Issues**: OD-002 (Masking information leakage), OD-006 (Spurious holder arm injection)

---

## 1. Executive Summary

In OpenDiffraction 3.0, physical simulations computed a valid detector mask `valid_mask` and a masked array `observed_masked`, but the primary measurement field exposed to data loaders, PyTorch datasets, and HDF5 storage was `intensity_measured`, which contained unmasked intensity and noise everywhere (including behind opaque beamstops and inactive module gaps). Any model trained on `intensity_measured` received true diffraction signals in unmeasured physical regions, constituting catastrophic information leakage.

Additionally, in [`pipeline.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/pipeline.py), line 223 unconditionally forced a mechanical holder arm shadow onto acquisition shot index 2 for all profiles regardless of profile definition, confounding acquisition-index comparisons with undocumented detector geometry variations.

In this remediation:
1. Re-architected detector measurement outputs into two unambiguous, physically sound fields:
   - `intensity_raw`: Complete detector readout before hardware masking, preserving noise and physics for detector calibration or experimental modeling.
   - `intensity_observed`: Strictly masked observation where all pixels outside `valid_mask` are guaranteed to equal zero (`intensity_observed[~valid_mask] == 0.0`).
2. Maintained `intensity_measured` as a zero-leakage backward-compatibility property returning `intensity_observed`.
3. Updated HDF5 storage and dataset loaders ([storage.py](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/storage.py), [loader.py](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/io/loader.py), [torch_dataset.py](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/io/torch_dataset.py)) to store and serve `intensity_observed` for all machine learning inputs.
4. Decoupled detector geometry from acquisition indices in [`pipeline.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/pipeline.py), ensuring holder arms and module gaps are strictly profile-driven.
5. Implemented comprehensive leakage regression tests in [`test_mask_leakage_remediation.py`](file:///d:/03_Proyek/Diffraction_Dataset/tests/test_mask_leakage_remediation.py). All tests pass with zero leakage.

---

## 2. Information Leakage Analysis and Resolution

### 2.1 The Nature of the Defect (OD-002)
A physical detector consists of active pixels $\Omega_{\text{valid}}$ and inactive or occluded regions $\Omega_{\text{masked}} = \Omega \setminus \Omega_{\text{valid}}$ (such as central direct-beam stops, tungsten holder arms, inter-module ceramic gaps, and dead pixels).

In the upstream code:
```python
# Old implementation in forward.py
sim_res.intensity_measured = simulated_readout  # Contained valid signal in Omega_masked!
sim_res.observed_masked = simulated_readout * mask
```
When `storage.py` and `torch_dataset.py` consumed `intensity_measured`, neural networks and iterative solvers had direct access to high-intensity low-$q$ speckles behind the beamstop. Supervised inpainting models trained on this field were not inferring missing information from surrounding speckles, but simply reproducing the unmasked data.

### 2.2 The Zero-Leakage Architecture
The forward simulation output was restructured into:
1. `intensity_raw`: $I_{\text{raw}}(\mathbf{k}) = \text{Poisson}(I_{\text{conv}}(\mathbf{k})) + \mathcal{N}(0, \sigma_{\text{read}}^2)$ everywhere on $\Omega$.
2. `intensity_observed`: $I_{\text{obs}}(\mathbf{k}) = I_{\text{raw}}(\mathbf{k}) \cdot M(\mathbf{k})$, where $M(\mathbf{k}) \in \{0, 1\}$.
3. Backward compatibility: `intensity_measured` defined as a Python property alias returning `intensity_observed`.

Verification guarantee:
$$\forall \mathbf{k} \in \Omega_{\text{masked}}, \quad I_{\text{obs}}(\mathbf{k}) \equiv 0.0$$

In [`test_mask_leakage_remediation.py`](file:///d:/03_Proyek/Diffraction_Dataset/tests/test_mask_leakage_remediation.py), this condition is tested across all 48 profiles:
```python
assert np.all(sample_dict["intensity_observed"][~valid_mask] == 0.0)
assert np.all(sample_dict["intensity_measured"][~valid_mask] == 0.0)
```
Across all 48 variation profiles, zero non-zero pixels were found in masked regions.

---

## 3. Profile-Driven Geometry Resolution (OD-006)

### 3.1 Unintended Detector Variation
In OpenDiffraction 3.0, line 223 of [`pipeline.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/pipeline.py) executed:
```python
# Old pipeline.py
if meta.get("holder_arm", False) or acq_idx == 2:
    arm_mask = composite_detector_mask(..., include_holder_arm=True, ...)
```
This forced a randomized tungsten holder arm shadow on shot index 2 for all profiles, including profiles V01 to V20 (clean standard specimens) and V21 (clean beamstop-free detector). Consequently:
- Shot 0, 1, 3 had no holder arm.
- Shot 2 had a holder arm.

This violated physical experimental semantics: an experimental beamline setup does not insert or remove mechanical detector mounting hardware between consecutive shots of the same sample.

### 3.2 Remediation
1. Removed `or acq_idx == 2` from holder arm and module gap generation in [`pipeline.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/pipeline.py).
2. Integrated `include_holder_arm` directly into [`simulate_forward_model`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/physics/forward.py) governed strictly by `metadata.get("holder_arm", False)`.
3. Profiles explicitly requiring holder arms (such as V26 and V30) retain holder arm shadows deterministically across all acquisition shots. Profiles without holder arms (such as V01) remain strictly free of spurious shadows.
4. Tested in [`test_mask_leakage_remediation.py`](file:///d:/03_Proyek/Diffraction_Dataset/tests/test_mask_leakage_remediation.py), confirming V01 preserves consistent detector geometry across all four acquisition shots (`acq_00` to `acq_03`).

---

## 4. Acceptance Gate III Sign-Off

All Gate III criteria are fulfilled:
- Zero information leakage verified across all 48 profiles (`intensity_observed[~mask] == 0.0`).
- Detector fields cleanly partitioned into `intensity_raw` and `intensity_observed`.
- Backward-compatibility alias `intensity_measured` is leakage-free.
- Holder arm and module gap geometry are strictly profile-driven.
- 100% test pass rate across all masking tests.
