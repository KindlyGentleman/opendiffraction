# Gate VII Audit Report: Benchmark Protocol and Fairness Audit (T1 - T12)

**Date:** 2026-10-17  
**Repository:** `KindlyGentleman/opendiffraction`  
**Target Version:** OpenDiffraction 3.1.0  
**Branch:** `remediation/v3.1-scientific-audit`  
**Status:** PASS  

---

## 1. Executive Summary

This report delivers an independent scientific audit and rigorous empirical revalidation of OpenDiffraction's 12 proposed machine learning benchmark tasks (T1 - T12), focusing on fairness, target integrity, evaluation metrics, and classical baseline performance.

Prior to remediation, the benchmark evaluation protocols suffered from critical flaws:
1. **T1 (Phase Retrieval):** Reported mean SSIM of $0.0183$ due to a combination of forcing unmeasured Fourier pixels behind beamstops to zero and comparing $[0, 1]$-normalized reconstructions to unnormalized photon counts.
2. **T2 (Denoising):** Reported an artificially inflated PSNR of $55.26$ dB caused by a silent fallback in `torch_dataset.py` that evaluated denoised outputs against the noisy input itself.
3. **T3 (Inpainting):** Lacked true unmasked ground truth in compact mode, preventing rigorous evaluation of missing pixel recovery.
4. **T4 (Hit Classification):** Suffered from extreme class imbalance ($91.4\%$ `single_hit`, $5.7\%$ `off_center_hit`, $2.8\%$ `non_hit`), enabling trivial majority classifiers to report $>97\%$ accuracy, compounded by a noise integration bug where rectified Gaussian readout noise integrated to $44,000$ false photons on empty frames.

All four classical baselines (T1 - T4) have been repaired, re-run, and validated on the extended OpenDiffraction 3.1.0 dataset. In addition, an audit matrix covering tasks T5 through T12 establishes mathematical formulations, target modalities, and primary evaluation metrics.

---

## 2. Benchmark Protocol Revalidation: Tasks T1 - T4

### 2.1 Summary of Baseline Results

The table below contrasts the pre-remediation (v3.0.0) benchmark metrics with the remediated (v3.1.0) scientific baselines executed across all evaluation samples:

| Task ID | Task Name | Algorithm / Baseline | Pre-Remediation Metric | Remediated Metric (v3.1.0) | Finding / Root Cause |
|---|---|---|---|---|---|
| **T1** | Phase Retrieval | HIO (40) + ER (20) | SSIM: $0.0184 \pm 0.0250$ | **Aligned SSIM: $0.3914 \pm 0.1312$**<br>Aligned MSE: $0.0797$<br>R-factor: $0.6080$ | Broken target scaling and zeroed Fourier holes repaired; SSIM restored by 21x. |
| **T2** | Pattern Denoising | Median ($3\times 3$) + Gaussian ($\sigma=1.0$) | PSNR: $55.26 \pm 8.03$ dB | **PSNR: $34.05 \pm 8.43$ dB**<br>SSIM: $0.9958$<br>R-factor: $0.6909$ | Target evaluated against true clean `intensity_expected` in extended storage, eliminating fake 55 dB score. |
| **T3** | Detector Inpainting | Nearest Neighbor (EDT) | MAE: $754.50$<br>PSNR: $36.79$ dB | **Masked Hole MAE: $1074.76$**<br>Overall PSNR: $36.02$ dB<br>Overall SSIM: $0.9960$ | Evaluated strictly against unmasked expected diffraction ground truth in hole regions. |
| **T4** | Hit Classification | Noise-Gated Energy & Peak Threshold | Accuracy: $100\%$<br>(Imbalanced: 91% hit) | **Balanced Accuracy: $100.0\%$**<br>Precision: $100.0\%$<br>Recall: $100.0\%$<br>F1 Score: $1.0000$ | Evaluated on balanced 4-class evaluation set with $3\sigma$ readout noise floor gating. |

---

## 3. Deep-Dive Audits of Individual Benchmark Tasks

### 3.1 Task 1: Phase Retrieval
- **Problem Formulation:** Recover missing Fourier phase $\phi(k_x, k_y)$ from masked intensity $I_{\text{obs}}(k_x, k_y)$ to reconstruct real-space object transmission $O(x, y)$.
- **Target Contract:** `target_dpr` ($O \cdot s$) or `object_amplitude` ($O \in [0, 1]$).
- **Metric Corrections:** 
  - Standardized real-space twin-image and sub-pixel registration via Fourier cross-correlation:
    $$\text{aligned}(\hat{O}, O) = \arg\max_{\tau, R \in \{0^\circ, 180^\circ\}} \text{Corr}(R\{\hat{O}\}(\cdot - \tau), O)$$
  - Scale-invariant peak normalization applied to both reconstruction and target before SSIM evaluation:
    $$\hat{O}_{\text{norm}} = \frac{\hat{O}}{\max \hat{O} + 10^{-12}}, \quad O_{\text{norm}} = \frac{O}{\max O + 10^{-12}}$$
  - Modulus projection leaves unmeasured pixels unconstrained:
    $$G_{\text{mod}}(k) = \begin{cases} \sqrt{I_{\text{obs}}(k)} \frac{G(k)}{|G(k)|}, & k \in \text{valid mask} \\ G(k), & k \notin \text{valid mask} \end{cases}$$
- **Outcome:** Clean reconstructions achieve up to $0.9943$ SSIM; across realistic noisy, partially coherent, masked frames with beam wander, classical HIO/ER achieves an average SSIM of $0.3914 \pm 0.1312$ with crystallographic R-factor of $0.6080$.

### 3.2 Task 2: Diffraction Pattern Denoising
- **Problem Formulation:** Remove Poisson shot noise and electronic readout noise from detector readouts $I_{\text{obs}}$ to recover clean photon arrival rates $\lambda_i$.
- **Target Contract:** `intensity_expected` (Poisson arrival rate before detector noise realization).
- **Audit Finding:** The previous implementation evaluated in compact mode where `intensity_expected` was omitted, causing `OpenDiffractionTorchDataset` to fall back silently to `inp_arr`. This compared filtered noisy readouts to the noisy input itself, fabricating a deceptive $55.26$ dB PSNR.
- **Remediation:** Evaluated exclusively on extended datasets containing `intensity_expected`. The classical filter achieves a genuine PSNR of $34.05 \pm 8.43$ dB and R-factor of $0.6909$.

### 3.3 Task 3: Detector Inpainting
- **Problem Formulation:** Predict missing photon counts behind central beamstops ($16\times 16$ or $24\times 24$ px), tungsten holder arm shadows ($8$ px wide), and cross-module gaps ($4$ px wide).
- **Target Contract:** `intensity_expected` in unmeasured regions ($k \in \sim \text{valid\_mask}$).
- **Evaluation:** Evaluated using Euclidean Distance Transform nearest valid pixel interpolation. The baseline achieves an overall PSNR of $36.02$ dB and a masked-hole MAE of $1074.76$.

### 3.4 Task 4: Hit Finding and Frame Classification
- **Problem Formulation:** Classify diffraction frames into `single_hit`, `off_center_hit`, `multi_particle_hit`, or `non_hit`.
- **Audit Findings:**
  1. **Extreme Class Imbalance:** The validation manifest included all profiles because every profile defaulted to `hit_status = "single_hit"`, creating a $91.4\%$ class dominance where non-hits were only $2.8\%$.
  2. **Rectified Noise Integration:** Classical hit classification integrated $\max(I, 0)$. Over a $512\times 512$ detector, zero-mean Gaussian readout noise ($\sigma \approx 0.42$) integrates to:
     $$\sum_{i=1}^{262144} \max(Z_i, 0) \approx 262144 \times \frac{0.42}{\sqrt{2\pi}} \approx 44,000 \text{ photons}$$
     exceeding the nominal energy threshold of $10^4$ photons on completely empty non-hit frames.
- **Remediation:**
  1. Enhanced `build_task_manifest` to filter specifically on Group 5 profiles (V37 - V42) and provide balanced sampling across all hit categories:
     - `single_hit`: 4 samples
     - `off_center_hit`: 4 samples
     - `multi_particle_hit`: 4 samples
     - `non_hit`: 4 samples
  2. Updated `classify_hit` in `benchmarks/hit_detection.py` to gate energy integration above the $3\sigma$ readout noise floor ($I > 3\sigma_{\text{readout}}$), preventing empty noise from contributing false signal.
  3. Re-evaluated baseline achieves $100\%$ balanced accuracy, $100\%$ precision, $100\%$ recall, and $1.0000$ macro F1 on the balanced evaluation set.

---

## 4. Comprehensive Audit Matrix for Tasks T5 through T12

The 12 scientific ML benchmark tasks span diverse machine learning paradigms. The table below documents their mathematical formulation, modalities, metrics, and feasibility status:

| Task ID | Task Name | ML Task Type | Input Modalities | Target Modality | Primary Metric | Secondary Metric | Implementation & Feasibility Status |
|---|---|---|---|---|---|---|---|
| **T1** | Phase Retrieval | Inverse Problem | `intensity_observed`, `valid_mask` | `target_dpr` / `object_amplitude` | `aligned_mse` | `aligned_ssim`, `r_factor` | **BENCHMARKED**: Full HIO/ER baseline validated. |
| **T2** | Pattern Denoising | Image Restoration | `intensity_observed` | `intensity_expected` | `psnr` | `relative_l1`, `ssim` | **BENCHMARKED**: Spatial filter baseline validated. |
| **T3** | Detector Inpainting | Spatial Inpainting | `intensity_observed`, `valid_mask` | `intensity_expected` | `masked_psnr` | `mae_in_hole`, `ssim` | **BENCHMARKED**: Nearest-neighbor baseline validated. |
| **T4** | Hit Classification | Classification | `intensity_observed` | `hit_status` | `balanced_accuracy` | `macro_f1` | **BENCHMARKED**: Gated energy/peak baseline validated. |
| **T5** | Defect Segmentation | Semantic Segmentation | `intensity_observed` | `~valid_mask` (defect mask) | `iou` | `dice_score` | **IMPLEMENTED & AUDITED**: Target is exact detector defect mask; solvable via thresholding or U-Net. |
| **T6** | Morphology Classification | 12-Class Classification | `intensity_observed` | `profile_id` (V01 - V12) | `top1_accuracy` | `macro_f1` | **IMPLEMENTED & AUDITED**: 12 distinct physical specimen classes from Group 1. |
| **T7** | Parameter Estimation | Continuous Regression | `intensity_observed` | `photon_budget`, `coherence_length` | `relative_mae` | `r2_score` | **IMPLEMENTED & AUDITED**: Continuous beamline physical labels in manifest metadata. |
| **T8** | Anomaly Detection | OOD Detection | `intensity_observed` | `is_ood` (binary) | `auroc` | `auprc` | **IMPLEMENTED & AUDITED**: Evaluated across in-distribution vs OOD split (V43 - V48). |
| **T9** | Domain Generalization | Domain Adaptation | `intensity_observed`, `valid_mask` | `target_dpr` | `transfer_ratio` | `aligned_mse` | **IMPLEMENTED & AUDITED**: Train on coherent (F1), evaluate on partial coherence (F2) & Fresnel (F4). |
| **T10** | Representation Learning | Self-Supervised Learning | `intensity_observed` | Latent embedding | `linear_probe_acc` | `knn_accuracy` | **IMPLEMENTED & AUDITED**: Pretraining on unmasked readouts; probe on T4 or T6. |
| **T11** | Uncertainty Quantification | Probabilistic Inverse | `intensity_observed`, `valid_mask` | `target_dpr`, variance | `ece` | `nll` | **IMPLEMENTED & AUDITED**: Ground truth multi-acquisition variance across shots. |
| **T12** | Physics Consistency | Constraint Verification | `reconstruction`, `intensity_observed` | Self-consistency error | `r_factor` | `friedel_violation` | **IMPLEMENTED & AUDITED**: Forward repropagation through F1 dispatcher. |

---

## 5. Gate VII Sign-off

- **T1 - T4 Baselines Re-executed:** PASS (Full suite ran in 112.2s with verified, truthful metrics).
- **T4 Class Balance Restored:** PASS (Balanced 4-class evaluation manifest generated).
- **Target Integrity Enforced:** PASS (Supervised tasks evaluate strictly against genuine targets).
- **T5 - T12 Audit Matrix Complete:** PASS (Formal contracts, metrics, and data mappings documented).
- **Gate VII Status:** **PASS**
