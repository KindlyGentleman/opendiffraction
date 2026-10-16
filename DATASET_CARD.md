# OpenDiffraction Dataset Card

## Dataset Title
**OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Patterns for Phase Retrieval and Image Reconstruction**

- **Dataset Version**: 0.1.0
- **Primary Domain**: Coherent Diffraction Imaging (CDI), Computational Optics, Phase Retrieval, Machine Learning Inverse Problems
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Primary Reference Paper**: Lee et al. (2025), *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, Nature Communications / npj Computational Materials, DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).
- **Reference Repository**: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR) (BSD 2-Clause License)

---

## 1. Executive Disclosure: Real vs. Synthetic Data

> [!IMPORTANT]
> **ALL DATA IN THIS DATASET ARE NUMERICALLY SIMULATED USING FIRST-PRINCIPLES FOURIER OPTICS.**
> None of the patterns represent experimentally measured beamtime exposures.
> High performance on synthetic diffraction benchmarks **does not prove** automatic generalization to real experimental X-ray Free-Electron Laser (XFEL) or synchrotron beamline data.
> This dataset is inspired by the Deep Phase Retrieval (DPR) forward model but is an independent, 100% procedural benchmark and is **not an official DPR or SLAC/SACLA release**.

---

## 2. Scientific Purpose

Coherent Diffraction Imaging (CDI) is a lensless microscopy technique where a coherent X-ray pulse illuminates a non-crystalline specimen, and the far-field diffraction intensity is recorded on a 2D pixelated detector. Recovering the object transmission function requires solving the phase problem from intensity measurements alone.

Real-world XFEL and synchrotron diffraction measurements suffer from severe experimental imperfections:
- **Missing central low-spatial-frequency information** due to direct-beam protection beam stops.
- **Detector defects**: dead pixels, readout line dropouts, and tile module boundaries.
- **Photon shot noise**: low incident flux regimes yielding discrete Poisson statistics.
- **Detector electronics noise**: additive Gaussian readout fluctuations that yield signed values.
- **Partial spatial coherence**: finite transverse beam coherence causing speckle blur.

OpenDiffraction establishes an open, reproducible, parameter-controlled benchmark to test both classical iterative phase retrieval algorithms (HIO, ER, RAAR) and deep-learning reconstructors (DPR, FFC-ResNets, U-Nets, diffusion models) under systematically controlled physical degradations.

---

## 3. Physical Forward Model

### 3.1 Fraunhofer Diffraction
For an object transmission density field $O(\mathbf{r})$ placed within a $64 \times 64$ window centered in a $512 \times 512$ detector frame:
$$A(\mathbf{q}) = \mathcal{F}_{2\text{D}}[O(\mathbf{r})]$$
$$I_{\text{ideal}}(\mathbf{q}) = |A(\mathbf{q})|^2$$
where $\mathcal{F}_{2\text{D}}$ is the centered 2D Fourier transform (zero-frequency DC situated at pixel `(256, 256)`). The oversampling ratio is $\sigma = 512 / 64 = 8$ along each Cartesian axis, satisfying the Shannon/Nyquist oversampling condition ($\sigma \ge 2$) required for phase retrieval.

### 3.2 Gaussian Schell-Model Partial Spatial Coherence
Under the Gaussian Schell-model (GSM), finite transverse spatial coherence degrades diffraction fringe visibility. The autocorrelation of the object is damped by a Gaussian envelope centered at **zero spatial lag**:
$$\Gamma(\Delta \mathbf{r}) = \mathcal{F}^{-1}[I_{\text{ideal}}](\Delta \mathbf{r}) \cdot \exp\left(-\frac{\|\Delta \mathbf{r}\|^2}{2 \ell_{\text{coh}}^2}\right)$$
$$I_{\text{expected}}(\mathbf{q}) = \left| \mathcal{F}[\Gamma(\Delta \mathbf{r})] \right|$$
OpenDiffraction strictly follows the upstream corrected standard (DPR commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`), ensuring monotonic fringe damping.

### 3.3 Photon Flux Normalization
$$s = \frac{N_{\text{photons}}}{\sum_{\mathbf{q}} I_{\text{expected}}(\mathbf{q})}, \quad \lambda_i = s \cdot I_{\text{expected}}(\mathbf{q}_i)$$
where $\lambda_i$ is the expected incident photon count on detector pixel $i$.

### 3.4 Poisson Shot Noise & Gaussian Readout Noise
$$K_i \sim \text{Poisson}(\lambda_i)$$
$$Y_i = K_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$$
where $\sigma_{\text{read}} = 1 / (2 \sqrt{2 \ln 2}) \approx 0.42466$ gives a full-width at half-maximum (FWHM) of 1 photon count. Signed detector readouts ($Y_i < 0$) are preserved in the scientific files to eliminate dark-current threshold bias.

---

## 4. Controlled Experimental Profiles (V01 - V12)

| Profile ID | Profile Name | Primary Swept Variable | Controlled / Fixed Variables | Purpose |
|---|---|---|---|---|
| **V01** | Clean Coherent Diffraction | Object Geometry | $\ell_{\text{coh}}=\infty$, No noise, No beam stop | Theoretical upper reconstruction limit |
| **V02** | Geometry Diversity | 5 Shape Families (A-E) | $\ell_{\text{coh}}=200$, $N_{\text{ph}}=3\times 10^6$, $\sigma_{\text{read}}=\text{std}$ | Shape and morphology generalization |
| **V03** | Density Heterogeneity | Uniform, binary, graded, textured | Standard beam stop and acquisition | Internal density and contrast recovery |
| **V04** | Photon Starvation | $N_{\text{photons}} \in [10^3, 10^8]$ | Fixed object morphology | Low-flux phase retrieval breakdown |
| **V05** | Partial Coherence | $\ell_{\text{coh}} \in [\infty, 25]$ px | Zero readout noise | Speckle fringe visibility decay |
| **V06** | Readout Noise | $\sigma_{\text{read}} \in [0, 3.0]$ counts | High flux ($10^7$ photons) | Disentangling additive electronics noise |
| **V07** | Beam-Stop Size | $4\times 4$ to $48\times 48$ px | Centered and shifted positions | Low-frequency missing data tolerance |
| **V08** | Irregular Defects | Dead pixels, clusters, stripes, gaps | Procedural defect generators | Robustness to incomplete detector data |
| **V09** | Saturation & Hot Pixels | Full-well threshold & hot pixels | High flux ($10^7$ photons) | Sensor nonlinearities and clipping |
| **V10** | Alignment Errors | Sub-pixel & integer shifts $(\delta y, \delta x)$ | Standard noise and beam stop | Beam-center miscalibration sensitivity |
| **V11** | Combined Realistic | Simultaneous composite degradations | Real-world synchrotron/XFEL parameters | Primary real-world benchmark |
| **V12** | OOD Stress Tests | Extreme multivariate adverse stress | Held-out fractals, extreme starvation | Out-of-distribution failure mode analysis |

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
