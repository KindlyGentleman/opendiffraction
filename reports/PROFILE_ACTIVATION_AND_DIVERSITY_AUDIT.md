# Variation Profile Activation and Diversity Audit Report (Gate II)

**Branch**: `remediation/v3.1-scientific-audit`  
**Date**: 2026-10-17  
**Status**: PASS  
**Addressed Issues**: OD-007 (Manifest physical parameters), OD-010 (Sample planning scaling), Profile execution across 48 variations and 6 experimental scenarios

---

## 1. Executive Summary

This audit verifies that all 48 variation profiles (V01 to V48) and all 6 experimental acquisition scenarios (S1 to S6) are fully active, physically distinct, and truthfully reflected in simulation outputs and manifest records.

In OpenDiffraction 3.0:
- The forward simulation unconditionally invoked standard Fraunhofer diffraction (F1), leaving declared profile models (F2 to F6) inactive.
- Manifest parameters for wavelength ($\lambda = 1.24 \times 10^{-10}$ m), detector pixel pitch ($75\ \mu\text{m}$), and sample-detector distance ($0.5$ m) were hardcoded constants regardless of whether the profile simulated soft X-rays, hard X-rays, or visible light lasers.
- Hardcoded `total=1000` was passed to profile sampling functions regardless of dataset sample size.

In this remediation:
1. Verified that each of the 48 variation profiles maps to its intended forward physical model and generates valid diffraction observations.
2. Verified that all 6 physical groups (Specimens, Coherence, Detector Realism, Background, Acquisition, and OOD/Advanced Physics) exhibit expected physical modulations.
3. Connected manifest fields to resolved physical parameters in [`pipeline.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/generator/pipeline.py), ensuring optical and soft X-ray scenarios record truthful physical constants.
4. Validated 100% test pass rate across [`tests/test_profiles_48.py`](file:///d:/03_Proyek/Diffraction_Dataset/tests/test_profiles_48.py).

---

## 2. Profile Registry and Physical Group Coverage

The 48 variation profiles are structured into six physical groups:

### 2.1 Group 1: Real-World Specimens (V01 to V12, Model F1)
- V01: Cell Phantoms (organelles, nucleus, membrane)
- V02: Virus Particles (icosahedral capsids, internal core)
- V03: Protein Aggregates (fractal globular assemblies)
- V04: Core-Shell Nanoparticles (differential core/shell scattering contrast)
- V05: Nanorods and Anisotropic Particles (high aspect ratio)
- V06: Colloidal Crystals (periodic Bragg interference lattice)
- V07: Porous Matrices (aerogels, spinodal decomposition)
- V08: Defect and Crack Structures (void networks, fracture lines)
- V09: Lithographic Nanostructures (test targets, Siemens stars)
- V10: Phase and Amplitude Objects (complex transmission)
- V11: Multiple Scatterers (multi-particle cluster fields)
- V12: Dense Clustered Assemblies (high-packing-fraction aggregates)

### 2.2 Group 2: Illumination and Coherence (V13 to V20, Models F1, F2, F3, F4, F5)
- V13: Fully Coherent Baseline (F1, $l_{\text{coh}} = \infty$)
- V14: Moderate Spatial Coherence (F2, $l_{\text{coh}} = 200\ \mu\text{m}$)
- V15: Low Spatial Coherence (F2, $l_{\text{coh}} = 50\ \mu\text{m}$)
- V16: Anisotropic Spatial Coherence (F2, elliptical Schell-model beam)
- V17: Multimode Beam (F2, incoherent mode mixture)
- V18: Aberrated Wavefront Transmission (F3, defocus, astigmatism, coma)
- V19: Near-Field Fresnel Propagation (F4, $z = 0.05$ m, ASM propagation)
- V20: Polychromatic Spectral Bandwidth (F5, $\Delta \lambda / \lambda = 1.5\%$)

### 2.3 Group 3: Detector Realism (V21 to V30, Model F1)
- V21: Ideal Photon Counter (zero readout noise, unmasked)
- V22: Pure Poisson Shot Noise (quantum-limited)
- V23: Mixed Poisson-Gaussian Noise ($\sigma_{\text{read}} = \text{FWHM}/\sqrt{8\ln 2}$)
- V24: Severe Electronic Readout Noise ($\sigma_{\text{read}} = 2.5$)
- V25: Circular Central Beam Stop (missing low-$q$ frequencies)
- V26: Central Beam Stop with Rigid Shadow Arm (asymmetric shadow)
- V27: Multi-Panel Module Gaps (CSPAD/Pilatus tiled cross-gap geometry)
- V28: Dead Pixels and Cluster Defects (random pixel dropouts)
- V29: Full-Well Detector Saturation (dynamic-range clipping)
- V30: Combined Realistic Detector (simultaneous noise, gaps, beamstop, arm)

### 2.4 Group 4: Background and Scattering (V31 to V36, Model F1)
- V31: Background-Free Vacuum (high-vacuum reference)
- V32: Diffuse Gas Scattering (low-$q$ Lorentzian background)
- V33: Support Membrane Halo (amorphous carbon / SiN ring)
- V34: Solvent / Water Ring (diffuse liquid jet scattering)
- V35: Aperture Slit Streaks (orthogonal parasitic knife-edge flare)
- V36: Cosmic Ray Strike Artifacts (localized streak deposition)

### 2.5 Group 5: Acquisition and Hit Finding (V37 to V42, Model F1)
- V37: High-Flux Hit ($10^7$ incident photons)
- V38: Low-Dose Hit ($5 \times 10^4$ incident photons)
- V39: Beam Center Drift / Jitter (sub-pixel pointing wobble)
- V40: Off-Center Sample Intersection (grazing interaction)
- V41: Multi-Particle Hit (split scatterers in focus)
- V42: Non-Hit Blank Frame (pure background and detector noise)

### 2.6 Group 6: Advanced Physics and Out-of-Distribution (V43 to V48, Models F1, F3, F4, F6)
- V43: Dynamic Phase Objects (F1, complex transmission phase)
- V44: Multi-Wavelength Exposure (F5, dual-energy pink beam)
- V45: Phase Gradient and Wedge Transmission (F3, spatial phase ramps)
- V46: Defocused Wavefront CDI (F4, near-field defocus regime)
- V47: 3D Particle Volume Scattering (F6, orientation-dependent projection)
- V48: Adversarial Compound Stress Test (overlapping high-severity defects)

---

## 3. Experimental Scenario Coverage (S1 to S6)

Each scenario configures beamline physics:

| Scenario ID | Name | Facility Type | Nominal Energy | Pixel Pitch | Distance |
|---|---|---|---|---|---|
| **S1** | XFEL Single-Particle Imaging | XFEL | 8.0 keV | $110\ \mu\text{m}$ | 0.50 m |
| **S2** | Synchrotron Pink-Beam CDI | Storage Ring | 12.0 keV | $75\ \mu\text{m}$ | 2.00 m |
| **S3** | Nanomaterials Inspection | Synchrotron / Lab | 9.5 keV | $55\ \mu\text{m}$ | 1.20 m |
| **S4** | Biological CDI & Soft Matter | Synchrotron / XFEL | 2.5 keV | $30\ \mu\text{m}$ | 0.80 m |
| **S5** | Educational Optical Bench | Laser Lab (He-Ne) | 0.00196 keV | $6.45\ \mu\text{m}$ | 0.15 m |
| **S6** | Detector Calibration QA | Metrology Beamline | 10.0 keV | $75\ \mu\text{m}$ | 1.00 m |

---

## 4. Acceptance Gate II Sign-Off

All Gate II criteria are fulfilled:
- All 48 profiles verified and producing valid, physically modulated diffraction patterns.
- Forward physical models F1 to F6 bound deterministically to corresponding profile definitions.
- Manifest parameters dynamically reflect active scenario geometries.
- 100% test pass rate across [`tests/test_profiles_48.py`](file:///d:/03_Proyek/Diffraction_Dataset/tests/test_profiles_48.py).
