# OpenDiffraction Dataset Card

## Dataset Title
**OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Patterns for Phase Retrieval and Image Reconstruction**

- **Dataset Version**: 2.0.0
- **Primary Domain**: Coherent Diffraction Imaging (CDI), Computational Optics, Phase Retrieval, Machine Learning Inverse Problems
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Primary Reference Paper**: Lee et al. (2025), *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, Nature Communications / npj Computational Materials, DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).
- **Reference Repository**: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR) (BSD 2-Clause License)
- **CXI Standard**: [https://cxidb.org/cxi.html](https://cxidb.org/cxi.html)

---

## 1. Executive Disclosure: Real vs. Synthetic Data

> [!IMPORTANT]
> **ALL DATA IN THIS DATASET ARE NUMERICALLY SIMULATED USING FIRST-PRINCIPLES FOURIER OPTICS.**
> None of the patterns represent experimentally measured beamtime exposures.
> High performance on synthetic diffraction benchmarks **does not prove** automatic generalization to real experimental X-ray Free-Electron Laser (XFEL) or synchrotron beamline data.
> This dataset is inspired by the Deep Phase Retrieval (DPR) forward model and CXIDB experimental standards, but is an independent, 100% procedural benchmark and is **not an official DPR or SLAC/SACLA release**.

---

## 2. Scientific Purpose

Coherent Diffraction Imaging (CDI) is a lensless microscopy technique where a coherent X-ray pulse illuminates a non-crystalline specimen, and the far-field diffraction intensity is recorded on a 2D pixelated detector. Recovering the object transmission function requires solving the phase problem from intensity measurements alone.

Real-world XFEL and synchrotron diffraction measurements suffer from severe experimental imperfections:
- **Missing central low-spatial-frequency information** due to direct-beam protection beam stops.
- **Detector defects**: dead pixels, readout line dropouts, and tile module boundaries.
- **Photon shot noise**: low incident flux regimes yielding discrete Poisson statistics.
- **Detector electronics noise**: additive Gaussian readout fluctuations that yield signed values.
- **Partial spatial coherence**: finite transverse beam coherence causing speckle blur.
- **Parasitic background scattering**: ambient gas, sample carrier support films, liquid carrier solvent rings, slit flare, and cosmic ray hits.

OpenDiffraction establishes an open, reproducible, parameter-controlled benchmark to test both classical iterative phase retrieval algorithms (HIO, ER, RAAR) and deep-learning reconstructors (DPR, FFC-ResNets, U-Nets, diffusion models) under systematically controlled physical degradations.

---

## 3. Physical Forward Models (F1 - F6)

OpenDiffraction 2.0 implements 6 modular, physically grounded forward simulation models:

- **Model F1 (Fraunhofer Coherent Diffraction)**: Far-field intensity from real or complex transmission densities.
- **Model F2 (Partial-Coherence Diffraction)**: Gaussian Schell-model (GSM) autocorrelation damping centered at zero lag, or orthogonal multimode decomposition.
- **Model F3 (Thin-Object Coherent Transmission)**: Probe field multiplication including wavefront curvature, astigmatism, coma, and pointing jitter.
- **Model F4 (Fresnel Diffraction)**: Near-field propagation in SI units (wavelength, distance, pixel size) with Shannon sampling bounds.
- **Model F5 (Polychromatic Diffraction)**: Wavelength-integrated intensity accounting for reciprocal-space radial coordinate dilation across pink-beam bandwidths.
- **Model F6 (3D Particle Scattering)**: 3D scalar density volumes with continuous Euler rotation and Born projection approximation.

---

## 4. 48 Controlled Variation Profiles (V01 - V48)

OpenDiffraction 2.0 organizes 48 labeled experimental profiles across 6 physical groups:

1. **Group 1: Real-World Specimens (V01 - V12)**:
   - V01: Cell-Like Biological Phantoms (membrane, nucleus, organelles).
   - V02: Virus-Like Particles (capsid shells, inner core density).
   - V03: Protein-Complex-Inspired Aggregates (multi-subunit clusters).
   - V04: Core-Shell Nanoparticles (dense core, dielectric shell).
   - V05: Ellipsoidal Nanoparticles (anisotropic aspect ratios).
   - V06: Nanorods and Nanowires (high aspect ratio 1D structures).
   - V07: Multi-Particle Clusters (stochastic oligomeric assemblies).
   - V08: Porous Matrix / Aerogels (spatially correlated random media).
   - V09: Colloidal Crystal Assemblies (hexagonal and square lattice ordering).
   - V10: Lithographic Test Patterns (Siemens star, lines, gratings).
   - V11: Defect Structures (internal micro-cracks and void cavities).
   - V12: Mixed-Density Phase-Separated Mixtures (bimodal material contrast).

2. **Group 2: Illumination & Coherence (V13 - V20)**:
   - V13: Fully Coherent Plane-Wave Illumination.
   - V14: Moderate Partial Spatial Coherence (GSM, $l_{\text{coh}} = 150$ px).
   - V15: Low Partial Spatial Coherence (GSM, $l_{\text{coh}} = 40$ px).
   - V16: Anisotropic Partial Coherence ($l_{\text{coh}, y} \ne l_{\text{coh}, x}$).
   - V17: Focused Beam Illumination (Gaussian probe amplitude envelope).
   - V18: Wavefront Curvature / Defocus (quadratic phase profile).
   - V19: Beam Pointing Jitter (sub-pixel stochastic translations).
   - V20: Polychromatic Pink-Beam Illumination (finite spectral bandwidth $\Delta\lambda/\lambda$).

3. **Group 3: Detector Realism & Artifacts (V21 - V30)**:
   - V21: Baseline Readout Noise ($\sigma_{\text{read}} = 0.425$ count).
   - V22: High Readout Noise ($\sigma_{\text{read}} = 2.0$ counts).
   - V23: Pure Poisson Counting (ideal noiseless photon counting).
   - V24: Extreme Photon Starvation ($10^3 - 10^4$ total photons).
   - V25: Dynamic Range Saturation (full-well capacity clipping).
   - V26: Analog-to-Digital Quantization (integer/sub-bit ADC binning).
   - V27: Hot Pixels and Persistent Stuck Pixels.
   - V28: Dead Pixel Clusters and Random Module Defects.
   - V29: Readout Line Dropout Stripes (horizontal/vertical channel loss).
   - V30: Multi-Tile Module Gaps (cross-gap detector tiling).

4. **Group 4: Background & Parasitic Scattering (V31 - V36)**:
   - V31: High-Vacuum Clean Background (zero parasitic scatter).
   - V32: Ambient Gas Scattering Background (isotropic central diffuse halo).
   - V33: Sample Support Film Halo (amorphous carbon substrate ring).
   - V34: Solvent / Water Ring (liquid jet diffuse diffraction ring).
   - V35: Beamline Parasitic Scattering (slits and aperture diffraction streaks).
   - V36: Stray Radiation / Cosmic Ray Hits (sharp high-amplitude pixel tracks).

5. **Group 5: Experimental Acquisition & Hit Finding (V37 - V42)**:
   - V37: High-Flux Hits ($10^7$ photons, well-centered pulse).
   - V38: Low-Flux Hits ($5 \times 10^4$ photons, weak scattering).
   - V39: Centered Particle Hits.
   - V40: Off-Center Particle Hits (stochastic beam displacement).
   - V41: Multi-Particle Hits (multiple scatterers in pulse focus).
   - V42: Non-Hits / Blank Solvent Exposures (empty carrier background).

6. **Group 6: Advanced Physics & Out-of-Distribution Regimes (V43 - V48)**:
   - V43: Weak Phase Specimen (phase contrast with uniform amplitude).
   - V44: Strong Phase Specimen (topological optical vortex phase depth).
   - V45: Near-Field Fresnel Regime ($N_F \approx 1$ SI propagation).
   - V46: 3D Particle Random Orientation (continuous Euler rotation projections).
   - V47: Mixed Multi-Component Specimen (compound phase-amplitude contrast).
   - V48: Compound Stress Scenario (simultaneous multi-defect adverse conditions).

---

## 5. 6 Real-World Acquisition Scenarios (S1 - S6)

- **S1 (XFEL Single-Particle Imaging)**: Femtosecond pulse duration, low photon counts, partial coherence, liquid jet background.
- **S2 (Synchrotron Beamline CDI)**: Monochromatic flux, long exposure, ambient gas background, module gaps.
- **S3 (Nanomaterials Characterization)**: Engineered nanoparticles, high-resolution oversampling, medium flux.
- **S4 (Biological Specimen CDI)**: Radiation-sensitive cellular phantoms, low dose, water ring background.
- **S5 (Optical Tabletop CDI)**: Visible laser diode illumination, high photon counts, high spatial coherence.
- **S6 (Detector Characterization & QA)**: Defect stress benchmark with elevated noise, dead modules, and bit errors.

---

## 6. 12 Scientific Machine Learning Benchmark Tasks (T1 - T12)

- **T1**: Supervised Phase Retrieval
- **T2**: Diffraction Pattern Denoising
- **T3**: Detector Inpainting & Beam-Stop Recovery
- **T4**: Hit Finding and Frame Classification
- **T5**: Defect Segmentation
- **T6**: Particle Morphology Classification
- **T7**: Physical Parameter Estimation (flux, coherence, rotation)
- **T8**: Out-of-Distribution Anomaly Detection
- **T9**: Cross-Beamline Domain Generalization
- **T10**: Self-Supervised Representation Learning
- **T11**: Reconstruction Uncertainty Quantification
- **T12**: Physics-Consistent Forward/Inverse Optimization

---

## 5. Dataset Architecture & Split Integrity

- **Splits**: 70% Training, 15% Validation, 15% Testing.
- **Group Disjointness**: Splitting is executed strictly on immutable base object identifiers (`group_id`). All paired clean-noisy observations and degraded variants of a given object reside strictly in the same split. **Zero object-level leakage exists across train/val/test partitions.**
- **OOD Isolation**: Profile V12 is partitioned exclusively into the test split (`test_ood`), ensuring training algorithms never observe extreme out-of-distribution stress conditions during model optimization.

---

## 6. Data Format & Semantic Specifications

Data are stored in compressed HDF5 shards (`train_000.h5`, `valid_000.h5`, `test_000.h5`, `test_ood_000.h5`):

```text
/
├── input: float32, (N, 1, 512, 512)
│   Noisy unmasked detector readouts in photon counts.
├── target: float32, (N, 1, 64, 64)
│   Real-space object density, scaled per DPR convention (false_scale=True).
├── mask: bool, (N, 1, 512, 512)
│   Detector validity mask. True = valid pixel, False = missing/beamstop.
├── object_amplitude: float32, (N, 1, 64, 64)
│   Unscaled true ground-truth amplitude in [0, 1].
├── intensity_ideal: float32, (N, 1, 512, 512)
│   Ideal uncorrupted Fraunhofer diffraction intensity |A(q)|^2.
├── intensity_expected: float32, (N, 1, 512, 512)
│   Partially coherent photon rate lambda_i before stochastic sampling.
├── intensity_measured: float32, (N, 1, 512, 512)
│   Noisy detector readouts (signed float32).
└── valid_mask: bool, (N, 1, 512, 512)
    True for valid pixels, False for detector defects.
```

---

## 7. Example Loading Code

### Using OpenDiffraction High-Level API
```python
from opendiffraction.io.loader import OpenDiffractionDataset

# Load entire test split or filter by variation profile
dataset = OpenDiffractionDataset("./output", split="test", variation="V11")
sample = dataset[0]

# Access fields
noisy_input = sample.input[0]        # (512, 512) float32
valid_mask = sample.mask[0]          # (512, 512) bool
ground_truth = sample.object_amplitude[0] # (64, 64) float32
```

### Direct Upstream DPR Loading
```python
from deeppr import CustomDataset

# Load directly with DPR PyTorch Dataset loader
ds = CustomDataset("./output/data/train_000.h5")
inp, target, mask = ds[0]
# inp: (1, 512, 512) torch.float32
# target: (1, 64, 64) torch.float32
# mask: (1, 512, 512) torch.bool
```

---

## 8. Intended and Inappropriate Uses

### Intended Uses
- Benchmarking phase retrieval algorithms (HIO, ER, RAAR, shrinkwrap).
- Training deep-learning phase retrieval models (DPR, U-Net, FFC-ResNet).
- Diffraction pattern denoising and inpainting.
- Quantifying phase retrieval robustness against missing low-frequency data.
- Benchmark testing of inverse problem solvers under Poisson/Gaussian mixed noise.

### Inappropriate Uses
- Clinical diagnosis or non-destructive testing of critical structural materials without experimental validation.
- Concluding that a network trained solely on this synthetic data will perform reliably on experimental beamlines without domain transfer.
- Claiming unambiguous phase reconstruction without resolving global phase, translation, and twin-image inversions.

---

## 9. Mathematical Phase Ambiguities

Phase retrieval from a single diffraction modulus $|A(\mathbf{q})|$ is fundamentally non-unique:
1. **Global Phase Shift**: $O(\mathbf{r}) \to O(\mathbf{r}) e^{i \phi_0}$
2. **Spatial Translation**: $O(\mathbf{r}) \to O(\mathbf{r} - \mathbf{r}_0)$
3. **Twin Image Inversion**: $O(\mathbf{r}) \to O^*(-\mathbf{r})$

Benchmarking pipelines must align reconstructions via cross-correlation and check both direct and twin-image orientations before computing PSNR, SSIM, or R-factor metrics.
