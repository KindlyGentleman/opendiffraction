# Source Provenance and Technical Derivation Guide

This document traces the technical and algorithmic provenance of OpenDiffraction, distinguishing between:
1. **Independent software architecture**
2. **Established mathematical and optical physics forward models**
3. **Upstream code and conventions adapted from Deep Phase Retrieval (DPR)**

---

## 1. Technical Delineation Matrix

| Feature / Component | Category | Technical Source / Attribution | Provenance & Implementation Notes |
| :--- | :--- | :--- | :--- |
| **Project Architecture & Pipeline** | Independent Implementation | Muhammad Kamal | Standalone Python package architecture with CLI, procedural generators, HDF5 chunked writers, and automated validation suite. |
| **Model F1: Fraunhofer Diffraction** | Standard Physics | Classical Fourier Optics (Goodman, *Introduction to Fourier Optics*) | 2D centered FFT: $I(\mathbf{q}) = \|\mathcal{F}\{O(\mathbf{r})\}\|^2$. Real-space zero-padding to 512x512 with DC at $(256, 256)$. |
| **Model F2: Spatial Partial Coherence** | Adapted Algorithm | Lee et al. (2025) / Upstream DPR commit `1952ee7` | Gaussian Schell-model (GSM) with zero-lag centered autocorrelation damping: $I_{\text{coh}} = \|\mathcal{F}\{\mathcal{F}^{-1}\{I\} \cdot \exp(-r^2 / 2\sigma_\mu^2)\}\|$. |
| **Model F3: Thin Object Probe**| Standard Physics | Ptychography / Wavefront Optics (Rodenburg & Faulkner 2004) | Specimen transmission multiplied by aberrated probe wavefield with defocus, astigmatism, and coma phase terms. |
| **Model F4: Fresnel Propagation** | Standard Physics | Wave Propagation (Goodman) | Angular spectrum method in SI units: $U_z = \mathcal{F}^{-1}\{\mathcal{F}\{U_0\} \cdot H(f_x, f_y; z, \lambda)\}$. |
| **Model F5: Polychromatic Integration** | Standard Physics | Spectrally-broadband diffraction | Discrete integration over wavelength bandwidth with reciprocal radial dilation $\mathbf{q} \propto 1/\lambda$. |
| **Model F6: 3D Born Projection** | Standard Physics | Fourier Slice Theorem / 3D Scattering | 3D ellipsoidal scattering volume rotated via continuous Euler angles $(\alpha, \beta, \gamma)$ and projected into 2D reciprocal space. |
| **Poisson & Gaussian Noise** | Standard Physics | Detector Physics / DPR Convention | Poisson photon shot noise ($k \sim \text{Poisson}(\lambda)$) followed by electronic readout noise with $\sigma = \text{FWHM}/(2\sqrt{2\ln 2}) \approx 0.4247$. |
| **Target Normalization (`false_scale`)** | Adapted Convention | Upstream DPR (`deeppr.dataset`) | DPR training target convention: $O_{\text{dpr}} = O \cdot s$, where $s = \Phi_{\text{target}} / \sum I$. (Physical amplitude is $O \cdot \sqrt{s}$). |
| **Detector Masks** | Independent & Adapted | DPR & NVIDIA Irregular Masks concepts | Composite procedural mask generator supporting central beamstops (square, circular, cross), detector gaps, dead pixels, and cluster defects. |
| **CXI Standard Export** | Open Scientific Standard | CXIDB / NeXus Specification | NeXus-compliant HDF5 hierarchy conforming to `cxidb.org/cxi.html` for single-particle and CDI diffraction data. |

---

## 2. Delineation of Upstream DPR Code vs. OpenDiffraction Code

### What OpenDiffraction Does NOT Redistribute:
- **No Pretrained Neural Network Weights**: OpenDiffraction does not bundle or redistribute `param_dpr1.pt` or VGG19 weights.
- **No Experimental Target Datasets**: OpenDiffraction generates 100% synthetic diffraction patterns procedurally from mathematical and noise models.
- **No Bundled Third-Party CNN Layers**: OpenDiffraction does not include partial convolution, FFC, or ASAM neural network implementations.
- **Zero Proprietary / External Dataset Dependencies**: Upstream DPR generation scripts rely on EMNIST, CIFAR-100, and NVIDIA irregular masks. OpenDiffraction replaces all external dependencies with 100% procedural generators for shapes, densities, and detector masks, guaranteeing clean-room mathematical reproducibility and zero intellectual property entanglement.

### What OpenDiffraction Adapts from DPR:
- **Mathematical GSM Coherence Kernel**: Damping of the zero-lag centered autocorrelation function.
- **HDF5 Tensor Layout**: Compatibility with `deeppr.CustomDataset` via `/input` (512x512), `/target` (64x64), and `/mask` (512x512).
- **Evaluation Protocols**: Benchmark Task T1 (Supervised Phase Retrieval) directly evaluates DPR-compatible reconstruction metrics (`ALIGNED_MSE`).

---

## 3. Independent Software Authorship

OpenDiffraction is created, engineered, and maintained by **Muhammad Kamal** (`KindlyGentleman` on GitHub, `hamadmuhammadkamal` on Kaggle) as an independent contribution to open scientific machine learning and reproducible computational diffraction imaging.
