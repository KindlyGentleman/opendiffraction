# OpenDiffraction Machine Learning Benchmark Tasks (T1 - T12)

OpenDiffraction 2.0 establishes 12 standardized, leakage-resistant benchmark tasks covering the full lifecycle of coherent diffraction data analysis, from low-level detector QA and hit finding to phase retrieval and uncertainty quantification.

---

## Benchmark Overview Table

| Task ID | Task Name | Input Tensor | Target Output | Primary Evaluation Metrics | Baseline Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | Supervised Phase Retrieval | `/input` (512x512) | `/target` (64x64) | Ambiguity-corrected SSIM, PSNR, R-factor | HIO+ER (SSIM 0.741) / DPR FFC-ResNet |
| **T2** | Diffraction Denoising | `/input` (512x512) | `/intensity_ideal` (512x512)| PSNR, Poisson deviance, SSIM | Median Filter / BM3D / U-Net (65.8 dB) |
| **T3** | Detector Inpainting | `/input` masked | `/intensity_ideal` on missing | Inpainted Region PSNR, Radial continuity | Biharmonic Inpainting (PSNR 31.2 dB) |
| **T4** | Hit Finding & Screening | `/input` (512x512) | `is_hit` $\in \{0, 1\}$ | ROC-AUC, F1-score, Precision @ 95% Recall | Radial Energy Threshold (AUC 1.000) |
| **T5** | Defect Segmentation | `/input` (512x512) | `/mask` (512x512 bool) | Mean IoU, Dice Coefficient | Otsu Thresholding / DeepLabV3+ |
| **T6** | Morphology Classification| `/input` (512x512) | Morphology class ID (1-12) | Macro F1, Top-1 and Top-3 Accuracy | ResNet-18 / Swin Transformer |
| **T7** | Parameter Estimation | `/input` (512x512) | `total_photons`, $l_{\text{coh}}$ | Mean Absolute Percentage Error (MAPE), $R^2$ | Feature Ridge Regression / MLP |
| **T8** | Anomaly Detection | `/input` (512x512) | Anomaly score on OOD (V12) | AUROC, False Alarm Rate @ 95% TPR | Deep SVDD / Normalized Reconstruction Error |
| **T9** | Domain Generalization | Scenario S1 inputs | Reconstructions on S2-S6 | Domain Generalization Gap ($\Delta\text{SSIM}$) | Empirical Risk Minimization (ERM) vs DANN |
| **T10**| Self-Supervised Learning| `/input` (unlabeled) | Self-supervised embeddings | Linear probing accuracy on T6 & T7 | SimCLR / Masked Autoencoder (MAE) |
| **T11**| Uncertainty Quantification| `/input` (512x512) | Object mean & variance maps | Expected Calibration Error (ECE), NLL | Monte Carlo Dropout / Deep Ensembles |
| **T12**| Physics-Loss Optimization| `/input` (512x512) | Physics-consistent object | Data fidelity R-factor, Support violation | Physics-Informed Neural Network (PINN) |

---

## Detailed Task Specifications

### Task T1: Supervised Phase Retrieval
- **Objective**: Recover real-space object transmission density $O(\mathbf{r})$ directly from a single intensity pattern $I(\mathbf{q})$.
- **Handling Phase Ambiguities**: Phase retrieval solutions possess trivial mathematical degeneracies:
  1. Global phase shift: $O(\mathbf{r}) \to O(\mathbf{r}) e^{i \phi_0}$.
  2. Spatial translation: $O(\mathbf{r}) \to O(\mathbf{r} - \mathbf{r}_0)$.
  3. Twin-image reflection: $O(\mathbf{r}) \to O^*(-\mathbf{r})$.
- **Standardized Metric Protocol**: All candidate reconstructions $\hat{O}$ must be cross-correlated against both $O_{\text{true}}$ and $O_{\text{true}}^*(-\mathbf{r})$ to find the optimal sub-pixel translation $(\Delta x, \Delta y)$ and orientation before evaluating SSIM and PSNR.
- **Crystallographic R-factor**:
  $$\mathcal{R}_{\text{factor}} = \frac{\sum_{\mathbf{q} \in M} \left| \sqrt{I_{\text{meas}}(\mathbf{q})} - \alpha |\mathcal{F}\{\hat{O}\}(\mathbf{q})| \right|}{\sum_{\mathbf{q} \in M} \sqrt{I_{\text{meas}}(\mathbf{q})}}$$
  where $\alpha = \frac{\sum \sqrt{I_{\text{meas}}} |\mathcal{F}\{\hat{O}\}|}{\sum |\mathcal{F}\{\hat{O}\}|^2}$ and $M$ is the valid detector mask.

### Task T2: Diffraction Pattern Denoising
- **Objective**: Estimate uncorrupted intensity $|A(\mathbf{q})|^2$ from mixed Poisson-Gaussian degraded measurements.
- **Physics Challenge**: Signal dynamic range spans up to 7 orders of magnitude between central low-$q$ speckles and high-$q$ detector boundaries. Low-$q$ pixels are photon shot-noise dominated, while high-$q$ pixels are electronic readout-noise dominated.
- **Metrics**: Peak Signal-to-Noise Ratio (PSNR in dB) evaluated in both linear and $\log_{10}(1 + I)$ scale, and Poisson deviance:
  $$D(y, \hat{y}) = 2 \sum_i \left[ y_i \ln\left(\frac{y_i}{\hat{y}_i}\right) - (y_i - \hat{y}_i) \right]$$

### Task T3: Detector Inpainting & Missing Pixel Recovery
- **Objective**: Restore missing reciprocal-space intensities obscured by the central beam stop (radius 10-45 px), detector inter-module gaps, and dead pixel clusters.
- **Physical Significance**: The central beam stop covers zero-frequency and low-spatial-frequency speckles that dictate the overall object shape and total scattering cross-section.
- **Metrics**: PSNR evaluated specifically over missing pixels ($\mathcal{M}_{\text{missing}} = 1 - \text{mask}$), and radial profile smoothness across the beam-stop transition boundary.

### Task T4: Hit Finding and Frame Classification
- **Objective**: Classify single-shot detector frames as either an informative particle hit (`is_hit = 1`) or a blank solvent/carrier non-hit (`is_hit = 0`).
- **Context**: At modern high-repetition-rate XFEL facilities (e.g., European XFEL at 27 kHz, LCLS-II at 1 MHz), streaming hit finding must reject $>95\%$ of non-hit frames in real-time to manage petabyte-scale data pipelines.
- **Metrics**: Area Under ROC Curve (AUROC), F1-score, and False Positive Rate at $95\%$ hit recall.

### Task T8: Out-of-Distribution Anomaly Detection
- **Objective**: Detect novel specimen morphologies or extreme experimental degradation regimes that deviate from the standard training distribution.
- **Protocol**: Models are trained on standard specimens (V01-V11) and tested against known test data vs. the held-out spinodal decomposition profile (V12) and compound stress regime (V48).
- **Metrics**: AUROC, False Alarm Rate at 95% True Positive Rate, and Out-of-Distribution Detection Error.

### Task T9: Cross-Beamline Domain Generalization
- **Objective**: Assess transferability of phase retrieval models trained on standard XFEL pulses (Scenario S1) when deployed on synchrotron data (Scenario S2), nanomaterials beamlines (S3), or tabletop optical setups (S5).
- **Metric**: Domain transfer degradation gap:
  $$\Delta \text{SSIM}(S_k) = \text{SSIM}_{\text{in-domain}}(S_1) - \text{SSIM}_{\text{transferred}}(S_k)$$

---

## Official Leaderboard Benchmark Protocol

To prevent test-set overfitting and promote reproducible scientific comparison:
1. **Zero Data Leakage**: Always train exclusively on `train` partition files. Do not evaluate training loss or metrics across `test` or `test_ood`.
2. **Object Disjointness**: Never split dataset rows randomly; always respect the `group_id` column to prevent sharing identical base objects between splits.
3. **Ambiguity Alignment**: Report phase retrieval scores with optimal twin-image and translation registration applied.
