# OpenDiffraction 3.0 Baseline Benchmark Validation

Results of classical baseline algorithms across the four primary benchmark tasks:

## Task 1: Phase Retrieval (HIO/ER Iterative Baseline)

- **Algorithm**: Hybrid Input-Output (40 iters, beta=0.8) + Error Reduction (20 iters)
- **Evaluated Samples**: 20
- **Mean Aligned SSIM**: 0.0184 $\pm$ 0.0250
- **Mean Aligned MSE**: 0.033978
- **Mean Final R-Factor**: 0.1441
- **Throughput**: 0.22 samples/s

## Task 2: Diffraction Pattern Denoising (Spatial Filter Baseline)

- **Algorithm**: Median Filter (size=3) + Gaussian Filter ($\sigma=1.0$)
- **Evaluated Samples**: 50
- **Mean PSNR**: 55.26 dB $\pm$ 8.03 dB
- **Mean SSIM**: 0.9970
- **Mean R-Factor**: 0.3217
- **Throughput**: 13.11 samples/s

## Task 3: Detector Inpainting (Nearest-Neighbor Baseline)

- **Algorithm**: Euclidean Distance Transform Nearest Valid Pixel Inpainting
- **Evaluated Samples**: 50
- **Masked Region MAE**: 7.5450e+02
- **Overall Pattern PSNR**: 36.79 dB
- **Overall SSIM**: 0.9956
- **Throughput**: 18.98 samples/s

## Task 4: Hit Finding and Classification (Energy & Peak Threshold Baseline)

- **Algorithm**: Integrated Photon Energy (E >= 1e4) + Peak Bragg Readout (I_max >= 50)
- **Evaluated Samples**: 140
- **Accuracy**: 100.00%
- **Balanced Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1 Score**: 1.0000
- **Throughput**: 124.04 samples/s

Total benchmark execution time: 97.83s.
