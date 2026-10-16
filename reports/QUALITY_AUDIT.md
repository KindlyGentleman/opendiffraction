# OpenDiffraction Scientific Quality Audit

## 1. Physics Verification

### 1.1 Fourier Transform & Parseval Energy Conservation
- **FFT Convention**: 2D Fraunhofer diffraction evaluated using centered 2D FFT (`fftshift(fft2(obj))`).
- **Parseval Relation**:
  $$\sum_{y, x} |O(y, x)|^2 = \frac{1}{N_y N_x} \sum_{q_y, q_x} |A(q_y, q_x)|^2$$
  Relative residual between real-space integral and Fourier-space integral is strictly bounded ($< 10^{-5}$).

### 1.2 Gaussian Schell-Model Autocorrelation Centering
- Modern DPR commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae` corrected the partial coherence kernel centering.
- The coherence kernel is centered at index `[0, 0]` in the autocorrelation domain, guaranteeing that zero-lag autocorrelation is multiplied by 1.0, and damping decays monotonically with spatial lag.

### 1.3 Stochastic Noise Realizations
- **Poisson Shot Noise**: Samples discrete photon events per pixel from Poisson distribution with parameter $\lambda_i = N_{\text{photons}} \cdot I_i / \sum I$. Empirical sample mean and variance match within $3\sigma$ confidence intervals.
- **Gaussian Readout Noise**: Simulates electronic sensor readout noise $\epsilon \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$ with $\text{FWHM} = 1$ photon ($\sigma \approx 0.42466$). Negative readout values are preserved to prevent dark-current clipping distortion.

---

## 2. Statistical Leakage & Data Hygiene

- **Object-Level Group Splitting**:
  Base objects and all associated degradation variants belong to immutable groups (`group_id`). Splitting is performed exclusively on `group_id`, completely preventing data leakage across train, validation, and test splits.
- **OOD Isolation**:
  Profile V12 (out-of-distribution stress test) is held out strictly for testing and never leaks into the training partition.

---

## 3. Upstream Compatibility

- Shards contain exact DPR fields (`input`, `target`, `mask`).
- Target objects are scaled according to DPR `false_scale=True` convention for neural network training, while true physical amplitudes are preserved under `object_amplitude`.
