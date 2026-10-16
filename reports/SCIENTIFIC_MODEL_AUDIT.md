# Scientific Model Audit

**Date**: 2026-10-16  
**Auditor**: Computational Physics and Scientific Data Auditor  
**Project**: OpenDiffraction 3.0  
**Scope**: Forward Physics Models (F1 to F6), Coherence, Normalization, Scaling Conventions, and DPR Compatibility

---

## 1. Physical Assumptions and Mathematical Models

OpenDiffraction implements six modular physical forward models describing coherent X-ray scattering and optical propagation:

### F1: Coherent Fraunhofer Diffraction
- **Governing Physics**: Far-field diffraction where Fresnel number $N_F = a^2 / (\lambda z) \ll 1$.
- **Implementation**: Centered 2D Fast Fourier Transform (FFT) with `fftshift`:
  $$A(q_x, q_y) = \mathcal{F}\{O(x, y)\}, \quad I_{\text{ideal}}(q_x, q_y) = |A(q_x, q_y)|^2$$
- **Centering Convention**: Zero spatial frequency (DC component) is centered at pixel index `(256, 256)` on a $512 \times 512$ grid.
- **Padding and Oversampling**: The $64 \times 64$ specimen is zero-padded into a $512 \times 512$ detector array, providing an linear oversampling ratio $\sigma = 512 / 64 = 8.0$. This exceeds the Nyquist-Shannon requirement ($\sigma \ge 2.0$) necessary for iterative phase retrieval.
- **Parseval Relation**: Verified in `test_physics_forward.py`: $\sum |O(x, y)|^2 = \frac{1}{N^2} \sum |A(q_x, q_y)|^2$ within $10^{-5}$ relative tolerance.

### F2: Partial Coherence (Gaussian Schell-Model)
- **Governing Physics**: Partial spatial coherence modelled via the Gaussian Schell-model (GSM). The cross-spectral density decays as a Gaussian function of spatial separation (lag) $\Delta \mathbf{r}$:
  $$g(\Delta \mathbf{r}) = \exp\left(-\frac{\|\Delta \mathbf{r}\|^2}{2 \ell_{\text{coh}}^2}\right)$$
- **Upstream DPR Coherence Defect Fix**: In legacy DPR releases (`v1.0-legacy`), the GSM kernel was placed at grid center `(256, 256)` instead of zero lag `(0, 0)` in `_ifft2` coordinates. This attenuated the zero-lag component and artificially amplified high-frequency speckle contrast. OpenDiffraction applies `ifftshift` to the coordinate distance matrix, ensuring $g(0, 0) = 1.0$ and strictly monotonic decay.
- **Energy Conservation**: By Parseval theorem, multiplying autocorrelation by a damping window with $g(0, 0) = 1.0$ preserves total integrated photon energy. Verified in `test_19_gsm_coherence_decay_and_energy_conservation()`: $\sum I_{\text{ideal}} = \sum I_{\text{part}}$ within $10^{-5}$.

### F3: Thin-Object Probe Transmission
- **Governing Physics**: Illumination by a structured, aberrated X-ray beam probe $P(\mathbf{r})$ incident on a thin specimen $O(\mathbf{r})$:
  $$\psi(\mathbf{r}) = P(\mathbf{r}) \cdot O(\mathbf{r})$$
- **Probe Aberrations**: Defocus, astigmatism, and spherical aberration parameterized in the exit-wavefront pupil phase. Beam pointing jitter modelled as random sub-pixel positional shifts.

### F4: Fresnel Near-Field Diffraction
- **Governing Physics**: Near-field free-space propagation over distance $z$ where curvature of wavefronts cannot be neglected.
- **Implementation**: Angular spectrum method using the free-space transfer function:
  $$H(f_x, f_y) = \exp\left(j \frac{2\pi}{\lambda} z \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}\right)$$
- **Model-Specific Boundary**: Because near-field propagation violates Fraunhofer far-field assumptions, Fresnel observations must not be evaluated with Fraunhofer phase retrieval algorithms without propagation kernel inversion.

### F5: Polychromatic Diffraction
- **Governing Physics**: Pink-beam illumination with finite spectral bandwidth $\Delta \lambda / \lambda \sim 10^{-2}$ to $10^{-1}$.
- **Implementation**: Incoherent summation over spectral intensity components:
  $$I_{\text{poly}}(\mathbf{q}) = \sum_k w_k I(\mathbf{q}; \lambda_k)$$
  where wavelengths $\lambda_k$ radially scale the diffraction pattern.

### F6: Approximate 3D Particle Scattering
- **Governing Physics**: Scattering from 3D particle density distributions $V(x, y, z)$ under the first Born approximation.
- **Implementation**: Specimen rotated in 3D by random Euler angles $(\alpha, \beta, \gamma)$, followed by forward projection (projection-slice theorem) to obtain the 2D projected transmission density.

---

## 2. DPR Compatibility and Scaling Conventions

### 2.1 The `false_scale=True` Convention
In linear physical optics, measured diffraction intensity scales quadratically with object density amplitude:
$$I(\mathbf{q}) \propto |O(\mathbf{r})|^2$$
When normalizing a diffraction pattern to total photon budget $F$:
$$s = \frac{F}{\sum_{\mathbf{q}} I_{\text{raw}}(\mathbf{q})}$$
The true physical object amplitude scales as:
$$O_{\text{phys}}(\mathbf{r}) = O(\mathbf{r}) \cdot \sqrt{s}$$
However, upstream DPR (`sungyun98/DPR`) trains neural networks using `false_scale=True`, which defines:
$$O_{\text{target}}(\mathbf{r}) = O(\mathbf{r}) \cdot s$$
This means DPR neural network targets scale linearly with flux factor $s$, not with $\sqrt{s}$.

### 2.2 OpenDiffraction Resolution
OpenDiffraction explicitly preserves both representations:
1. `target` / `target_dpr`: scaled by $s$ to provide exact drop-in compatibility with upstream DPR models.
2. `target_physical`: scaled by $\sqrt{s}$ to preserve physical optics amplitude.
3. `object_amplitude`: unscaled normalized density $O(\mathbf{r}) \in [0, 1]$.
4. `scale_factor`: stores the exact float scalar $s$.

These representations are cleanly separated in metadata and dataset fields, avoiding ambiguity.

---

## 3. Stochastic Noise Models

1. **Poisson Counting Statistics**:
   Incident photon rate at pixel $i$ is $\lambda_i = F \cdot I_{\text{coh}, i} / \sum I_{\text{coh}, j}$. The observed photon count is $N_i \sim \text{Poisson}(\lambda_i)$. For $\lambda_i \gg 1$, the variance equals the mean ($\text{Var}(N_i) = \mathbb{E}[N_i] = \lambda_i$).
2. **Gaussian Electronic Readout Noise**:
   Readout noise from detector electronics is modelled as additive zero-mean Gaussian noise:
   $$M_i = N_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$$
   where default $\sigma_{\text{read}} = \text{FWHM} / 2.35482 \approx 0.42466$.
3. **Signed Readout Values**:
   Because Gaussian noise can produce negative values in dark regions ($N_i = 0, \epsilon_i < 0$), readouts are stored as signed float32 numbers. They are never artificially clipped or zeroed before masking.

---

## 4. Verification Matrix

| Physical Verification Item | Status | Test / Proof |
|---|---|---|
| Fraunhofer FFT Centering | PASSED | `test_model_f1_fraunhofer_coherent` (peak at DC) |
| GSM Coherence Zero-Lag | PASSED | `test_19_gsm_coherence_decay_and_energy_conservation` ($g(0,0)=1.0$) |
| Parseval Energy Conservation | PASSED | `test_physics_forward.py` |
| DPR Forward Agreement | PASSED | `test_18_numerical_agreement_with_dpr_forward_model` |
| Poisson Mean-Variance | PASSED | `test_noise_statistics.py` |
| DPR vs Physical Scaling | PASSED | `test_20_dpr_target_scaling_and_physical_scaling` |
| Signed Detector Readouts | PASSED | `test_physics_forward.py` |
| Mask Geometry & Beam Stop | PASSED | `test_21_detector_masks_and_masked_observations` |

The forward physics engine satisfies all scientific validation criteria for OpenDiffraction 3.0.
