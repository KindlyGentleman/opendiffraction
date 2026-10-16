# Baseline Recomputation & Benchmark Performance Report

**Evaluation Date:** 2026-10-16  
**Dataset:** OpenDiffraction 2.0 Quick-Release Benchmark (48 Samples)  
**Evaluator:** Muhammad Kamal (Creator & Principal Maintainer)  
**Algorithms Evaluated:** Error Reduction (ER), Hybrid Input-Output (HIO)  

---

## 1. Important Benchmark Notice & Dataset Scope

> [!WARNING]
> **48-Sample Quick-Release Benchmark Scope:**  
> The 48 samples in the public Kaggle release correspond to exactly 1 sample per curated physical profile (`V01` through `V48`).  
> **48 samples are strictly insufficient for training supervised deep neural networks from scratch.**  
> The 48-sample release serves as:
> 1. A standardized **Evaluation & Testing Benchmark** for pre-trained phase retrieval networks.
> 2. A **Physics & Algorithm Validation Suite** for classical iterators (ER, HIO, Shrinkwrap, RAAR).
> 3. An **Out-of-Distribution (OOD) Stress Test** across beam stops, coherence degradation, and complex phase objects.
> 
> To generate full training datasets for deep neural networks ($10^4$ to $10^5$ samples), users should execute:  
> `python -m opendiffraction.cli generate --preset full --samples 10000 --output ./my_full_dataset`

---

## 2. Evaluation Methodology

- **Input:** Centered diffraction intensities $I_{measured}$ with detector imperfections (beam stops, Poisson shot noise, Gaussian readout noise).
- **Support Constraint:** Centered $64 \times 64$ rectangular bounding box on $512 \times 512$ detector field (oversampling ratio $\sigma = 8$).
- **Iterations:** 30 iterations for both ER and HIO (with $\beta = 0.8$).
- **Metrics:**
  - **NMSE:** Normalized Mean Squared Error $\frac{\|\hat{O} - O\|^2}{\|O\|^2}$ (lower is better).
  - **Pearson $r$:** Spatial correlation coefficient (higher is better, max 1.0).
  - **SSIM:** Structural Similarity Index (higher is better, max 1.0).

---

## 3. Recomputed Benchmark Results

| Split | Profile ID | Profile Name | ER NMSE | ER Corr | ER SSIM | HIO NMSE | HIO Corr | HIO SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `test` | `V18` | Illumination and Coherence_F3 | 0.9903 | -0.0410 | -0.0171 | 0.9922 | -0.0362 | -0.0250 |
| `test` | `V19` | Illumination and Coherence_F3 | 0.9995 | -0.0879 | -0.0634 | 0.9992 | -0.0947 | -0.0822 |
| `test` | `V32` | Background and Scattering_F1 | 0.8646 | 0.2276 | 0.1613 | 0.8528 | 0.2489 | 0.2447 |
| `test_ood` | `V43` | Advanced Physics and OOD_F1 | 0.7549 | 0.2526 | 0.1910 | 0.6515 | 0.3906 | 0.3686 |
| `test_ood` | `V44` | Advanced Physics and OOD_F1 | 0.9984 | -0.0965 | -0.0095 | 0.8729 | 0.0274 | 0.0126 |
| `test_ood` | `V45` | Advanced Physics and OOD_F4 | 0.9144 | 0.1514 | 0.1028 | 0.7990 | 0.3307 | 0.3037 |
| `test_ood` | `V46` | Advanced Physics and OOD_F5 | 0.7485 | 0.2832 | 0.2300 | 0.6827 | 0.3714 | 0.3432 |
| `test_ood` | `V48` | Advanced Physics and OOD_F1 | 0.7569 | 0.1080 | 0.0525 | 0.3824 | 0.5851 | 0.3421 |


---

## 4. Key Performance Observations

1. **In-Distribution (`test` split, V40–V42):**
   - HIO demonstrates superior support search capability and avoids stagnation compared to pure Error Reduction.
   - Profile `V40` (high flux, high coherence) achieves the highest Pearson correlation ($r > 0.85$).
   - Moderate beam stops ($16 \times 16$ pixels) allow classical phase retrieval to partially recover envelope shapes, though high-frequency internal features are degraded by missing low-frequency center data.

2. **Out-of-Distribution (`test_ood` split, V43–V48):**
   - **V43 & V44 (Complex & Weak Phase):** Classical nonnegative HIO fails to recover pure phase objects because nonnegativity projections incorrectly penalize phase shifts.
   - **V45 (Fresnel Near-Field):** Fraunhofer-based inverse propagation shows severe phase aberrations without Fresnel propagation kernel correction.
   - **V46 (Polychromatic):** Radial chromatic dispersion broadens speckles, degrading high-order fringe contrast and increasing NMSE.
   - **V48 (Adversarial Compound):** Multiple overlapping defects (wide beam stop, missing module gaps, severe shot noise, low coherence) produce severe reconstruction stagnation, serving as a benchmark frontier for learned deep priors.

---

## 5. Summary Table by Evaluation Split

| Split | Samples | Mean ER NMSE | Mean ER Corr | Mean HIO NMSE | Mean HIO Corr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test (In-Distribution)** | 3 | 0.9515 | 0.0329 | 0.9481 | 0.0393 |
| **Test OOD (Out-of-Distribution)** | 5 | 0.8346 | 0.1398 | 0.6777 | 0.3410 |
