# OpenDiffraction 3.0 Benchmark Re-Audit Report

## 1. Executive Summary

Independent re-audit and evaluation of classical baseline methods on the verified 1,000-sample OpenDiffraction 3.0 dataset.

## 2. Task 1: Phase Retrieval (HIO/ER Iterative Baseline)

- **Evaluated Samples**: 20 validation frames
- **Mean Aligned SSIM**: 0.0128 $\pm$ 0.0170 (Median: 0.0058)
- **Mean Aligned MSE**: 0.025500
- **Mean Final R-Factor**: 0.1483
- **Scientific Analysis**: While HIO/ER achieves low Fourier space residuals (R-factor ~0.14), real-space reconstruction SSIM remains low (~0.02 - 0.05) on complex non-convex specimens. Classical iterative phase retrieval frequently stagnates in local minima or twin-image pseudo-solutions without tight dynamic shrinkwrap constraints, underscoring the necessity of deep learning approaches.

## 3. Task 2: Diffraction Pattern Denoising (Spatial Filtering Baseline)

- **Evaluated Samples**: 50 validation frames
- **Mean PSNR**: 55.26 dB $\pm$ 8.03 dB
- **Mean SSIM**: 0.9970
- **Mean R-Factor**: 0.3217
- **Dynamic Range Caveat**: The very high PSNR (>50 dB) reflects the large dynamic range of central Bragg diffraction peaks against dark outer speckle backgrounds. Outer speckles with photon counts <5 are partially blurred by classical Gaussian filters, making SSIM and Fourier ring correlation essential complementary metrics.

## 4. Task 3: Detector Inpainting (Nearest-Neighbor Baseline)

- **Evaluated Samples**: 50 validation frames
- **Masked Region MAE**: 754.50 photons (Median: 225.23)
- **Overall Pattern PSNR**: 36.79 dB
- **Overall SSIM**: 0.9956
- **Spatial Error Localization**: Global PSNR is ~36.8 dB because valid detector pixels (>98% of detector) are unmodified. However, within the central beamstop shadow where forward intensity peaks sharply, nearest-neighbor extrapolation severely underestimates photon counts (MAE ~750 photons), proving that radial symmetry-aware models or learned priors are required.

## 5. Task 4: Hit Finding & Frame Classification Baseline

- **Evaluated Samples**: 140 validation frames (136 hits, 4 non-hits)
- **Accuracy**: 100.00%
- **Balanced Accuracy**: 100.00%
- **Precision / Recall / F1**: 1.0000 / 1.0000 / 1.0000
- **Confusion Matrix [TN, FP, FN, TP]**: [[4, 0], [0, 136]]

### Benchmark Weakness and Honest Qualification

- **Synthetic Gap**: In synthetic generation, non-hit frames (Profile V42) contain zero specimen scatter, resulting in total integrated intensity equal to electronic readout noise (~0 ADU). Single-particle hits contain $10^4 - 10^7$ photons. Consequently, simple energy thresholding achieves 100% accuracy on synthetic data.
- **Real-World Limitation**: On physical XFEL beamlines, gas background scatter, liquid jet droplets, and dark current fluctuations create significant energy overlap between weak hits and non-hits. High synthetic hit classification accuracy must not be claimed as equivalent to robust experimental hit finding.
