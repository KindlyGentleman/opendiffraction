# OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Patterns for Phase Retrieval and Image Reconstruction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: BSD 2-Clause](https://img.shields.io/badge/Code_License-BSD_2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Physics: Fraunhofer CDI](https://img.shields.io/badge/physics-Fraunhofer_CDI-green.svg)](https://doi.org/10.1038/s41524-025-01569-7)

**OpenDiffraction** is an open-source, reproducible, physics-based synthetic diffraction imaging generator and benchmark suite designed for public publication on Kaggle. It models Fraunhofer coherent X-ray diffraction, Gaussian Schell-model partial spatial coherence, mixed Poisson shot noise, Gaussian detector electronics noise, central beam stops, and irregular detector gaps.

OpenDiffraction produces datasets that are directly compatible with the Deep Phase Retrieval ([DPR](https://github.com/sungyun98/DPR)) neural network while offering an extended scientific format containing unscaled physical ground-truth objects, uncorrupted ideal intensities, expected photon rates, and complete experimental manifests.

---

## Key Features

1. **First-Principles Fourier Optics**:
   - Fraunhofer far-field propagation via 2D FFT with centered zero frequency.
   - Padded real-space objects ($64 \times 64 \to 512 \times 512$, oversampling $\sigma = 8$).
   - Rigorous Parseval energy conservation and Friedel inversion symmetry.
2. **Corrected Gaussian Schell-Model Partial Coherence**:
   - Implements the corrected zero-lag autocorrelation damping matching upstream DPR commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`.
3. **Realistic Detector Degradations**:
   - Mixed Poisson shot noise and additive Gaussian readout noise ($\text{FWHM} = 1$).
   - Signed detector readouts preserved in scientific fields to eliminate dark-current threshold distortion.
   - Central beam stops, detector module gaps, dead pixel clusters, and readout stripes.
4. **100% Procedural Object Generation (Zero Proprietary Dependencies)**:
   - 5 shape families: geometric, smooth blobs, multi-scatterers, porous structures, and structured lattices.
   - 5 density textures: uniform, binary, graded, textured, and piecewise constant.
   - Zero dependency on external third-party datasets (no EMNIST, CIFAR-100, or NVIDIA masks required).
5. **12 Controlled Experimental Variation Profiles (V01 - V12)**:
   - From clean coherent baselines to photon starvation ($10^3 - 10^8$ photons) and out-of-distribution stress tests.
6. **Object-Disjoint Splitting**:
   - 70% train / 15% valid / 15% test split strictly on base object identity (`group_id`). Zero data leakage.
7. **Comprehensive Testing & Validation**:
   - Automated 19-point pytest test suite verifying physics, statistical distributions, determinism, and DPR loading.

---

## Quickstart

### Installation
```bash
git clone https://github.com/muham/opendiffraction.git
cd opendiffraction
pip install -e .
```

### Command-Line Interface

Generate quick smoke-test dataset (48 samples):
```bash
python -m opendiffraction generate --preset quick --output ./output --seed 42
```

Generate full starter benchmark (2,400 samples):
```bash
python -m opendiffraction generate --preset starter --output ./output --seed 42
```

Validate physics, statistics, and HDF5 integrity:
```bash
python -m opendiffraction validate --dataset ./output --reports-dir ./reports
```

Generate publication figures (Figures 1-8 and Kaggle cover):
```bash
python -m opendiffraction visualize --dataset ./output --output ./output/figures
```

Package dataset for Kaggle release:
```bash
python -m opendiffraction package --dataset ./output --output ./kaggle_release --username YOUR_KAGGLE_USERNAME
```

---

## Variation Profiles Overview

- **V01 — Clean Coherent Diffraction**: Ideal upper reconstruction limits.
- **V02 — Geometry Diversity**: Generalization across 5 procedural shape families.
- **V03 — Density Heterogeneity**: Continuous density recovery (graded, textured, piecewise).
- **V04 — Photon Starvation**: Flux sweep across 6 orders of magnitude ($10^3 - 10^8$ photons).
- **V05 — Partial Coherence**: Spatial coherence length sweep ($\ell_{\text{coh}} = \infty \to 25$ px).
- **V06 — Readout Noise**: Additive electronic noise sweep ($\sigma_{\text{read}} = 0 \to 3.0$ counts).
- **V07 — Beam-Stop Geometry**: Low-frequency loss across varying sizes and offsets.
- **V08 — Irregular Defects**: Inpainting benchmark with dead pixels, stripes, and module gaps.
- **V09 — Saturation & Hot Pixels**: Full-well capacity clipping and sensor hot spots.
- **V10 — Alignment Errors**: Beam-center miscalibration sensitivity.
- **V11 — Combined Realistic Degradation**: Composite synchrotron/XFEL degradation.
- **V12 — Out-of-Distribution Stress**: Extreme adverse conditions held out for model testing.

---

## Notebooks

- [`notebooks/01_explore_opendiffraction.ipynb`](notebooks/01_explore_opendiffraction.ipynb): Dataset inspection, shard loading, and visualization.
- [`notebooks/02_phase_retrieval_baseline.ipynb`](notebooks/02_phase_retrieval_baseline.ipynb): Iterative phase retrieval baseline using HIO and Error Reduction.

---

## References

1. Lee et al. (2025), *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, Nature Communications / npj Computational Materials, DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).
2. Upstream DPR Repository: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR) (BSD 2-Clause License).
3. Detailed compatibility and upstream history: [`docs/DPR_COMPATIBILITY.md`](docs/DPR_COMPATIBILITY.md).

---

## License

- Scientific Data: [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)
- Source Code: [BSD 2-Clause License](LICENSE)
