# OpenDiffraction 3.0 Baseline Benchmark Validation

Results of classical baseline algorithms across the four primary benchmark tasks:

## Task 1: Phase Retrieval (HIO/ER Iterative Baseline)

- **Algorithm**: Hybrid Input-Output (40 iters, beta=0.8) + Error Reduction (20 iters)
- **Evaluated Samples**: 20
- **Mean Aligned SSIM**: 0.3914 $\pm$ 0.1312
- **Mean Aligned MSE**: 0.079748
- **Mean Final R-Factor**: 0.6080
- **Throughput**: 0.21 samples/s

## Task 2: Diffraction Pattern Denoising (Spatial Filter Baseline)

- **Algorithm**: Median Filter (size=3) + Gaussian Filter ($\sigma=1.0$)
- **Evaluated Samples**: 50
- **Mean PSNR**: 34.05 dB $\pm$ 8.43 dB
- **Mean SSIM**: 0.9958
- **Mean R-Factor**: 0.6909
- **Throughput**: 5.11 samples/s

## Task 3: Detector Inpainting (Nearest-Neighbor Baseline)

- **Algorithm**: Euclidean Distance Transform Nearest Valid Pixel Inpainting
- **Evaluated Samples**: 50
- **Masked Region MAE**: 1.0748e+03
- **Overall Pattern PSNR**: 36.02 dB
- **Overall SSIM**: 0.9960
- **Throughput**: 7.40 samples/s

## Task 4: Hit Finding and Classification (Energy & Peak Threshold Baseline)

- **Algorithm**: Integrated Photon Energy (E >= 1e4) + Peak Bragg Readout (I_max >= 50)
- **Evaluated Samples**: 16
- **Accuracy**: 100.00%
- **Balanced Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1 Score**: 1.0000
- **Throughput**: 59.80 samples/s

Total benchmark execution time: 112.23s.
