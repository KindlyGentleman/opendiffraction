# DPR Compatibility and Physical Specification

**Reference Project**: Deep Phase Retrieval (DPR)  
**Reference Paper**: Lee et al. (2025), *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, npj Computational Materials 11, 68, DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).  
**Upstream Repository**: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR)  
**Upstream Commit SHA**: `1952ee743a4913f1ae5fd3ca51a006c53709e8ae` (Date: Wed Sep 30 14:03:19 2026 +0900)  
**Upstream License**: BSD 2-Clause License (Copyright (c) 2024, sungyun98)  
**Dataset Creator & Maintainer**: Muhammad Kamal (GitHub: `@KindlyGentleman`, Kaggle: `@hamadmuhammadkamal`)

> [!NOTE]
> **Independence and Non-Endorsement Notice**:  
> OpenDiffraction was independently developed by Muhammad Kamal as a synthetic diffraction benchmark inspired by the published literature. It is not an official project of Pohang University of Science and Technology (POSTECH), the FXDI research group, or the DPR authors, and has not been reviewed, approved, funded, or endorsed by them.

---

## 1. Executive Summary

This document establishes the mathematical and implementation alignment between **OpenDiffraction** and the **DPR** codebase. OpenDiffraction is an independent, reproducible, physics-based synthetic diffraction generator. It can export datasets directly consumable by DPR models (`deeppr.CustomDataset`) while also providing an extended scientific data format that preserves unscaled physical ground truth, incident expected intensities, signed detector readouts, and comprehensive physical metadata.

---

## 2. Upstream Repository Inspection & History

### 2.1 Git History and the Coherence Bug Fix

Between the published legacy version tagged `v1.0-legacy` and commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`, a critical physical bug fix was made to the partial coherence simulation in `deeppr/dataset.py`:

```
commit 1952ee743a4913f1ae5fd3ca51a006c53709e8ae
Author: sungyun98 <sungyun98@g.postech.edu>
Date:   Wed Sep 30 14:03:19 2026 +0900

    fix(dataset): apply the coherence kernel at the zero lag
    
    GenerateDiffraction multiplies the autocorrelation from _ifft2, which has
    its zero lag at (0, 0), by a Gaussian Schell-model kernel centred on the
    512 x 512 grid. The weight therefore grew with the lag (0.44 at zero lag
    and 0.63 at (63, 63) for l_coh = 200), which gave negative intensities
    and a higher speckle contrast instead of a partially coherent blur.
    ifftshift the kernel so that its maximum lies at the zero lag.
```

### 2.2 Legacy vs. Corrected Formulation

1. **The Autocorrelation Domain**:
   By the Wiener–Khinchin theorem, the inverse Fourier transform of the coherent intensity $I(\mathbf{q}) = |A(\mathbf{q})|^2$ yields the spatial autocorrelation $\Gamma(\Delta \mathbf{r})$ of the object transmission function. Under the Gaussian Schell-model (GSM), spatial partial coherence acts as a Gaussian damping window $g(\Delta \mathbf{r}) = \exp(-\|\Delta \mathbf{r}\|^2 / (2 \ell_{\text{coh}}^2))$ multiplying the autocorrelation, which is then transformed back to intensity via forward FFT.

2. **The Lag Convention**:
   The standard inverse 2D FFT (`_ifft2` in DPR) evaluates coordinates such that the zero spatial lag $(\Delta x = 0, \Delta y = 0)$ is situated at array index `[0, 0]`.
   - **`v1.0-legacy` Defect**: The Gaussian kernel was generated on a grid centered at pixel index `[256, 256]`. When point-wise multiplied with the autocorrelation centered at `[0, 0]`, the zero-lag component was attenuated by $\exp(-(256^2 + 256^2)/(2 \ell_{\text{coh}}^2))$, while large lags (near index 256) received maximum weighting. This inverted the physical coherence dampening, artificially enhancing high-lag speckle contrast.
   - **Corrected Standard (`HEAD`)**: Applying `ifftshift` to the squared distance matrix shifts index `[256, 256]` to `[0, 0]`, ensuring that $g(0, 0) = 1.0$ and that autocorrelation damping decays monotonically with lag distance $\|\Delta \mathbf{r}\|$.

> [!IMPORTANT]
> **OpenDiffraction strictly defaults to the corrected physical Gaussian Schell-model**. A legacy mode (`coherence_mode="legacy"`) is provided strictly for regression verification against pre-fix DPR checkpoints.

---

## 3. Mathematical Forward Model Comparison

| Component | Upstream DPR (`GenerateDiffraction`) | OpenDiffraction |
|---|---|---|
| **Object Dimension** | $64 \times 64$ pixels | $64 \times 64$ pixels (procedural + extensible) |
| **Padded Frame** | $512 \times 512$ (`F.pad(..., 224)`) | $512 \times 512$ centered padding (oversampling $\sigma = 8$) |
| **Fourier Transform** | Centered 2D FFT (`fftshift(fft2(...))`) | Centered 2D FFT (identical convention) |
| **Coherent Intensity** | $\|A(q_x, q_y)\|^2$ | $\|A(q_x, q_y)\|^2$ (Parseval verified) |
| **Partial Coherence** | Gaussian Schell-model with zero-lag centering | Corrected GSM (default) + legacy toggle |
| **Photon Budget** | Flux $F \in [1, 10) \times 10^{\text{ph\_ord}}$ | Explicit photon budget $N_{\text{photons}} \in [10^3, 10^8]$ |
| **Photon Scaling** | $I \leftarrow I \times (F / \sum I)$ | $\lambda_i = F \cdot I_i / \sum I_j$ (Poisson rate) |
| **Target Object Scaling** | `false_scale=True` sets `target = obj * scale` | Provides both physical $O \cdot \sqrt{s}$ and DPR `false_scale` $O \cdot s$ |
| **Poisson Noise** | `torch.poisson(torch.clamp(I, min=0))` | True Poisson realization $N_i \sim \text{Poisson}(\lambda_i)$ |
| **Readout Noise** | `torch.normal(I_poiss, sig)` with $\text{FWHM}=1$ ($\sigma = 1 / 2.35482$) | Configurable Gaussian readout noise $\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$ |
| **Signed Readouts** | Stored directly (signed float) | Signed values preserved; masks stored separately |
| **Detector Masks** | External NVIDIA Irregular Masks + random central rectangle | Pure procedural masks (no external dependencies) + central beam stops |

---

## 4. Normalization and Scaling Conventions

### 4.1 Target Scaling (`false_scale`)
In linear physical optics, the measured diffraction intensity scales quadratically with object transmission density amplitude:
$$I(\mathbf{q}) \propto |O(\mathbf{r})|^2$$
When normalizing a diffraction pattern to total photon flux $F$, the scale factor is:
$$s = \frac{F}{\sum_{\mathbf{q}} I_{\text{raw}}(\mathbf{q})}$$
The physically consistent object amplitude under flux $F$ should therefore scale as:
$$O_{\text{phys}}(\mathbf{r}) = O_{\text{base}}(\mathbf{r}) \cdot \sqrt{s}$$
However, the upstream DPR repository uses `false_scale=True` during dataset generation (`generate_dataset.py`, line 182), which computes:
$$O_{\text{target}}(\mathbf{r}) = O_{\text{base}}(\mathbf{r}) \cdot s$$
This means that DPR neural network training targets are scaled linearly with intensity scale $s$, rather than with amplitude $\sqrt{s}$.

**OpenDiffraction Convention**:
- In DPR-compatible HDF5 outputs (`data/*.h5`), `target` uses the DPR `false_scale=True` convention to guarantee exact neural-network compatibility.
- In scientific HDF5 outputs and manifests, the true unscaled ground truth $O_{\text{base}}$ is stored under `object_amplitude`, alongside the physical scale factor $\sqrt{s}$ and the flux $F$.

---

## 5. Storage Formats and Field Semantics

### 5.1 DPR HDF5 Layout (`deeppr.CustomDataset`)
```text
/
├── input: float32, shape (N, 1, 512, 512)
│   Description: Noisy detector readout (unmasked), centered DC.
├── target: float32, shape (N, 1, 64, 64)
│   Description: Real-space object, scaled according to false_scale=True.
└── mask: bool, shape (N, 1, 512, 512)
    Description: Boolean array; True = valid pixel, False = missing/beamstop.
```

### 5.2 OpenDiffraction Extended Scientific Layout
```text
/
├── input: float32, shape (N, 1, 512, 512)           # DPR compatibility
├── target: float32, shape (N, 1, 64, 64)           # DPR compatibility
├── mask: bool, shape (N, 1, 512, 512)              # DPR compatibility
├── object_amplitude: float32, shape (N, 1, 64, 64) # Unscaled ground truth [0, 1]
├── intensity_ideal: float32, shape (N, 1, 512, 512)# Coherent Fraunhofer intensity
├── intensity_expected: float32, shape (N, 1, 512, 512) # Post-coherence photon rate λ_i
├── intensity_measured: float32, shape (N, 1, 512, 512) # Noisy readout (signed)
├── valid_mask: bool, shape (N, 1, 512, 512)        # Sensor validity mask
└── object_phase: float32, shape (N, 1, 64, 64)     # Phase (0 for real, [-π, π] for complex)
```

---

## 6. Procedural Independence from Proprietary / External Datasets

Upstream DPR relies on three external datasets:
1. **EMNIST** (for binary object shapes)
2. **CIFAR-100** (for grayscale internal densities)
3. **NVIDIA Irregular Mask Dataset** (disocclusion masks)

**OpenDiffraction Eliminates All External Dependencies**:
1. Shape geometries are generated procedurally across 5 families (geometric, blobs, multi-scatterers, porous, lattices).
2. Densities are synthesized via Perlin/Gaussian random fields, radial gradients, and procedural textures.
3. Detector defects (dead clusters, readout lines, module gaps, central beam stops) are generated via parametric mathematical models.

This guarantees:
- 100% deterministic reproducibility from random seeds.
- Zero licensing ambiguity (no unverified redistribution of third-party images).
- Fully controlled physical parameters (exact void size, exact cluster separation, exact gap dimensions).
