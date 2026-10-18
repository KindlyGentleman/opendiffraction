# OpenDiffraction Dataset Card

## Dataset Title
**OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Patterns for Phase Retrieval and Image Reconstruction**

- **Dataset Version**: 3.1.0 (Remediated Scientific Release)
- **Creator & Principal Maintainer**: OpenDiffraction computational optics and scientific machine learning contributors (Principal Maintainer: [Muhammad Kamal](https://github.com/KindlyGentleman), GitHub: `@KindlyGentleman`, Kaggle: `@hamadmuhammadkamal`)
- **Dataset DOI**: [10.34740/kaggle/dsv/20811988](https://doi.org/10.34740/kaggle/dsv/20811988)
- **Primary Domain**: Coherent Diffraction Imaging (CDI), Computational Optics, Phase Retrieval, Machine Learning Inverse Problems
- **Code License**: BSD 2-Clause (Copyright (c) 2026 Muhammad Kamal)
- **Data & Figures License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Primary Reference Paper**: Lee et al. (2025), *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, npj Computational Materials 11, 68, DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).
- **Reference Repository**: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR) (BSD 2-Clause License)
- **CXI Standard**: [https://cxidb.org/cxi.html](https://cxidb.org/cxi.html)

---

## 1. Executive Disclosure: Quick-Release Benchmark Scope & Synthetic Data

> [!WARNING]
> **48-SAMPLE QUICK-RELEASE BENCHMARK SCOPE:**  
> The published release on Kaggle contains **exactly 48 curated samples** (1 per physical profile `V01` through `V48`: 35 train, 5 validation, 3 test, 5 test_ood).  
> **48 samples are strictly insufficient for training supervised deep neural networks from scratch.**  
> The 48-sample release serves as:
> 1. An evaluation benchmark and testbed for pre-trained phase retrieval networks.
> 2. A verification suite for classical iterators (ER, HIO, Shrinkwrap, RAAR).
> 3. An out-of-distribution (OOD) stress test across partial coherence, beam stops, and phase objects.
> 
> To generate large-scale training volumes ($10^4$ to $10^5$ samples), execute:  
> `python -m opendiffraction.cli generate --preset full --samples 10000 --output ./my_full_dataset`

> [!IMPORTANT]
> **ALL DATA IN THIS DATASET ARE NUMERICALLY SIMULATED USING FIRST-PRINCIPLES FOURIER OPTICS.**  
> None of the patterns represent experimentally measured beamtime exposures.  
> High performance on synthetic diffraction benchmarks **does not prove** automatic generalization to real experimental X-ray Free-Electron Laser (XFEL) or synchrotron beamline data.  
> OpenDiffraction was independently developed by Muhammad Kamal. It is **not an official release of POSTECH, the FXDI research group, or the DPR authors**, and has not been reviewed, approved, funded, or endorsed by them.

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

OpenDiffraction 3.1 implements 6 modular, physically grounded forward simulation models with active physics dispatching and formal mathematical oracle validation:

- **Model F1 (Fraunhofer Coherent Diffraction)**: Far-field intensity from 2D complex transmission functions via discrete Fourier transform; verified against analytical circular aperture Airy disk and Parseval energy conservation.
- **Model F2 (Partial-Coherence Diffraction)**: Gaussian Schell-model (GSM) mutual coherence damping centered at zero spatial frequency lag, or orthogonal multimode decomposition.
- **Model F3 (Thin-Object Coherent Transmission)**: Probe field multiplication including wavefront curvature, astigmatism, coma, and stochastic sub-pixel pointing jitter.
- **Model F4 (Fresnel Diffraction)**: Near-field propagation via Angular Spectrum Method (ASM) in SI units (wavelength, propagation distance, pixel pitch) with centered coordinate grids; verified to identity at z = 0.
- **Model F5 (Polychromatic Diffraction)**: Pink-beam finite bandwidth spectral integration accounting for reciprocal-space radial coordinate dilation across multiple wavelength channels.
- **Model F6 (3D Particle Scattering)**: 3D scalar density volumes with continuous Euler rotation and transmission projection; guarantees exact bitwise object invariance across multiple acquisition shots.

---

## 4. 48 Controlled Variation Profiles (V01 - V48)

OpenDiffraction 3.1 organizes 48 labeled experimental profiles across 6 physical groups:

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

The released package contains 48 curated samples (1 sample per variation profile `V01` through `V48`):

- **Train Split (`train_000.h5`)**: 35 samples (72.9%) spanning diverse specimens, nominal illumination, and standard detector noise.
- **Validation Split (`valid_000.h5`)**: 5 samples (10.4%) for checkpoint validation (`V01`, `V06`, `V09`, `V20`, `V39`).
- **In-Distribution Test Split (`test_000.h5`)**: 3 samples (6.3%) for in-distribution testing (`V18`, `V19`, `V32`).
- **Out-of-Distribution Test Split (`test_ood_000.h5`)**: 5 samples (10.4%) quarantined exclusively for out-of-distribution stress evaluation (`V43`, `V44`, `V45`, `V46`, `V48`).
- **Group Disjointness**: Splitting is executed strictly on immutable base object identifiers (`group_id`). All paired clean-noisy observations and degraded variants of a given object reside strictly in the same split. **Zero object-level leakage exists across train/val/test partitions.**
- **OOD Isolation**: Profiles `V43` (weak phase), `V44` (strong phase vortex), `V45` (Fresnel near-field), `V46` (polychromatic radial stretch), and `V48` (adversarial compound defects) are strictly isolated in `test_ood`, guaranteeing that training algorithms never encounter extreme physics violations during optimization.

---

## 6. Data Format & Semantic Specifications

OpenDiffraction 3.1 establishes a strict zero-leakage contract and supports two standardized HDF5 storage profiles: Compact Storage (default) and Extended Storage (`--full-physics`).

### 6.1 The Zero Information Leakage Guarantee
In both formats, the observable input fields `input` and `intensity_observed` have detector defects, module gaps, and center beamstops strictly zeroed out:
$$\text{intensity\_observed}[\sim\text{mask}] = 0.0$$
Unmasked detector readouts (photons plus noise before beamstop interception) are preserved exclusively under `intensity_raw` in Extended Storage for specialized inpainting evaluations (e.g., Task T3). For downstream pipelines, `intensity_measured` provides a zero-leakage property alias pointing directly to `intensity_observed`.

### 6.2 Compact Storage Schema (Default)
Optimized for storage efficiency and fast supervised training. All clean target fields can be deterministically re-generated on demand using the recorded PRNG seeds in `metadata.csv`.

```text
/
├── input: float32, (N, 1, 512, 512)
│   Masked detector observation (masked pixels strictly 0.0; zero leakage).
├── target: float32, (N, 1, 64, 64)
│   DPR-compatible real-space object density scaled by scale_factor.
└── mask: bool, (N, 1, 512, 512)
    Detector validity mask. True = valid pixel, False = missing/beamstop.
```

### 6.3 Extended Storage Schema (--full-physics)
Comprehensive scientific schema containing all intermediate optical states, noise fields, and unscaled physical ground-truth quantities.

```text
/
├── input: float32, (N, 1, 512, 512)
│   Masked detector observation (alias to intensity_observed; zero leakage).
├── intensity_observed: float32, (N, 1, 512, 512)
│   Masked detector readout with intensity_observed[~mask] == 0.0.
├── intensity_raw: float32, (N, 1, 512, 512)
│   Unmasked detector readout (photons + noise) prior to masking.
├── intensity_measured: float32, (N, 1, 512, 512)
│   Zero-leakage alias to intensity_observed.
├── intensity_ideal: float32, (N, 1, 512, 512)
│   Ideal uncorrupted Fraunhofer diffraction intensity |A(q)|^2.
├── intensity_expected: float32, (N, 1, 512, 512)
│   Partially coherent photon rate lambda_i before stochastic Poisson sampling.
├── mask: bool, (N, 1, 512, 512)
│   Detector validity mask. True = valid pixel, False = missing/beamstop.
├── valid_mask: bool, (N, 1, 512, 512)
│   Compatibility alias for mask.
├── target: float32, (N, 1, 64, 64)
│   DPR-compatible real-space target density.
├── object_amplitude: float32, (N, 1, 64, 64)
│   Unscaled physical ground-truth object amplitude in [0, 1].
└── object_phase: float32, (N, 1, 64, 64)
    True object phase in radians.
```

### 6.4 DPR Scaling Convention
Upstream DPR trains on targets scaled by `scale_factor` ($O \cdot s$), which normalizes the maximum intensity or flux across batches. OpenDiffraction explicitly distinguishes between this training scaling convention (`target`) and the physical unscaled amplitude ground truth (`object_amplitude`). Supervised models evaluating against unscaled physical quantities should reference `object_amplitude`.

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

---

## 10. Citation & Acknowledgments

If you use OpenDiffraction in your research or benchmarks, please cite:
```bibtex
@misc{muhammad_kamal_2026,
  author    = {Muhammad Kamal},
  title     = {OpenDiffraction: Synthetic CDI Patterns},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/dsv/20811988},
  doi       = {10.34740/kaggle/dsv/20811988}
}
```

### Acknowledgments & Research Lineage
OpenDiffraction was developed by Muhammad Kamal and the OpenDiffraction computational optics and scientific machine learning contributors as part of **internship research acknowledging the Femtosecond X-ray Diffraction and Imaging (FXDI) Laboratory** at Pohang University of Science and Technology (POSTECH).

### Non-Endorsement & Independence Disclosure
OpenDiffraction is an independently developed research data and benchmark software project. It is not an official commercial product or institutional publication of POSTECH or the authors of DPR. None of these entities have officially endorsed or approved OpenDiffraction; references to published academic literature, institutions, and upstream repositories are provided for scholarly provenance and scientific acknowledgment.
