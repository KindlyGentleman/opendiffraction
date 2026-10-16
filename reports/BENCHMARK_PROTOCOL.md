# OpenDiffraction 3.0 Benchmark Protocol and Evaluation Standards

## 1. Overview and Scientific Motivation

OpenDiffraction 3.0 provides four core benchmark tasks designed to evaluate machine learning algorithms on inverse diffraction problems, detector image restoration, and frame quality filtering in coherent X-ray imaging:

1. **Task 1: Phase Retrieval (T1)** — Non-convex phase reconstruction from measured intensity modulus.
2. **Task 2: Diffraction Pattern Denoising (T2)** — Shot noise and electronic readout noise removal.
3. **Task 3: Detector Inpainting (T3)** — Reconstruction of masked beam stop shadows and module gaps.
4. **Task 4: Hit Finding and Frame Classification (T4)** — Pre-reconstruction filtering of single-particle hits, multi-particle collisions, and blank exposures.

---

## 2. Standard Evaluation Splits

All benchmark evaluations must strictly observe the predefined split boundaries:
- **`train.csv`**: In-distribution training set.
- **`val.csv`**: Validation set for hyperparameter tuning and model selection.
- **`test.csv`**: Standard test set for reporting in-distribution generalization.
- **`test_ood.csv`**: Out-of-distribution stress test (Group 6 profiles: Fresnel near-field, polychromatic spectrum, strong phase objects, adversarial stress).

> [!IMPORTANT]
> Zero object leakage is mathematically guaranteed across splits: every base object specimen resides entirely in one split. Evaluating test samples derived from training objects is strictly prohibited.

---

## 3. Evaluation Metrics and Standards

### Task 1: Phase Retrieval
- **Aligned SSIM**: Structural Similarity Index computed after registering global translational shifts and resolving complex conjugate reflection ambiguities:
  $$\text{SSIM}_{\text{aligned}} = \max_{\Delta \mathbf{r}, \pm} \text{SSIM}(O_{\text{pred}}(\mathbf{r} \pm \Delta \mathbf{r}), O_{\text{true}}(\mathbf{r}))$$
- **Aligned MSE**: Normalized Mean Squared Error between aligned reconstruction and ground truth target.
- **R-Factor (Crystallographic Residual)**: Agreement between forward-simulated modulus and measured detector modulus:
  $$R = \frac{\sum_{\mathbf{q}} ||\mathcal{F}\{O_{\text{pred}}\}(\mathbf{q})| - \sqrt{I_{\text{meas}}(\mathbf{q})}|}{\sum_{\mathbf{q}} \sqrt{I_{\text{meas}}(\mathbf{q})}}$$

### Task 2: Diffraction Pattern Denoising
- **PSNR (Peak Signal-to-Noise Ratio)**: Measured in decibels (dB) against clean expected arrival rates:
  $$\text{PSNR} = 10 \log_{10} \left( \frac{\max(I_{\text{true}})^2}{\text{MSE}(I_{\text{denoised}}, I_{\text{true}})} \right)$$
- **SSIM**: Evaluated across full 512x512 diffraction intensity frame.

### Task 3: Detector Inpainting
- **Masked Region MAE**: Mean absolute error evaluated strictly over the union of missing pixels ($\mathbf{M} = 0$):
  $$\text{MAE}_{\text{hole}} = \frac{1}{\sum (1 - M_i)} \sum_{i: M_i = 0} |I_{\text{inpaint}, i} - I_{\text{true}, i}|$$
- **Overall PSNR**: Computed over the complete 512x512 array.

### Task 4: Hit Finding and Classification
- **Balanced Accuracy**: Arithmetic mean of sensitivity (recall) and specificity, avoiding bias from class imbalance.
- **Macro F1 Score**: Harmonic mean of precision and recall averaged across hit vs. non-hit classes.
- **Precision / Recall**: Diagnostic hit purity vs. retrieval rate.

---

## 4. Submission and Reproducibility Protocol

1. Models must be trained without test split data.
2. Predictions must be evaluated using the official scripts in `benchmarks/`.
3. Execution times and hardware configurations (CPU/GPU) must be reported alongside numerical metrics.
