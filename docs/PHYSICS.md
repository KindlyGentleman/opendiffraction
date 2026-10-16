# OpenDiffraction: Physical Forward Simulation Models

This document specifies the optical physics, numerical approximations, and forward simulation equations implemented in OpenDiffraction 2.0.

---

## 1. Fraunhofer Coherent Diffraction (Model F1)

In lensless coherent diffraction imaging (CDI), the detector is placed in the far-field zone, where the Fresnel number $N_F = a^2 / (\lambda z) \ll 1$.

For a 2D complex specimen transmission function $O(\mathbf{r}) = |O(\mathbf{r})| \exp(i \phi(\mathbf{r}))$, the exit wave propagates to the reciprocal-space scattering vector $\mathbf{q} = (q_x, q_y)$:

$$A(\mathbf{q}) = \mathcal{F}_{2\text{D}}[O(\mathbf{r})] = \iint O(\mathbf{r}) \exp(-i 2\pi \mathbf{q} \cdot \mathbf{r}) \, d^2\mathbf{r}$$

The coherent intensity recorded on the detector is the squared modulus of the complex amplitude:

$$I_{\text{ideal}}(\mathbf{q}) = |A(\mathbf{q})|^2$$

### Numerical Discretization and Centering
- Real-space object dimension: $64 \times 64$ pixels.
- Detector array dimension: $512 \times 512$ pixels.
- Oversampling ratio along each Cartesian axis: $\sigma = 512 / 64 = 8.0$.
- Zero-frequency DC component is situated at pixel coordinate `(256, 256)`.
- Parseval energy theorem is verified numerically: $\sum |O(\mathbf{r})|^2 = \frac{1}{N^2} \sum |A(\mathbf{q})|^2$.

---

## 2. Partial Spatial Coherence: Gaussian Schell-Model (Model F2)

Real synchrotron and X-ray Free-Electron Laser (XFEL) beams possess finite transverse spatial coherence.

### The Autocorrelation Damping Formulation
By the Wiener-Khinchin theorem, the inverse Fourier transform of the coherent intensity yields the specimen autocorrelation:

$$\Gamma(\Delta \mathbf{r}) = \mathcal{F}_{2\text{D}}^{-1}[I_{\text{ideal}}(\mathbf{q})](\Delta \mathbf{r})$$

Under the Gaussian Schell-model (GSM), spatial coherence loss is modeled by multiplying the autocorrelation with a Gaussian window centered strictly at **zero spatial lag** $\Delta \mathbf{r} = (0, 0)$:

$$g(\Delta \mathbf{r}) = \exp\left(-\frac{\|\Delta \mathbf{r}\|^2}{2 \ell_{\text{coh}}^2}\right)$$

$$\Gamma_{\text{damped}}(\Delta \mathbf{r}) = \Gamma(\Delta \mathbf{r}) \cdot g(\Delta \mathbf{r})$$

The partially coherent intensity is obtained by forward Fourier transform:

$$I_{\text{expected}}(\mathbf{q}) = \left| \mathcal{F}_{2\text{D}}[\Gamma_{\text{damped}}(\Delta \mathbf{r})] \right|$$

### Upstream DPR Bug Fix
In the uncorrected legacy implementation (`v1.0-legacy`), the Gaussian envelope was centered at `(256, 256)` instead of `(0, 0)`, which artificially suppressed low spatial lags and amplified high lags. OpenDiffraction strictly defaults to the corrected zero-lag formulation (upstream DPR commit `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`).

---

## 3. Thin-Object Coherent Transmission (Model F3)

When illumination is not a plane wave, an incident optical probe field $P(\mathbf{r})$ illuminates the specimen transmission $T(\mathbf{r})$:

$$\psi_{\text{exit}}(\mathbf{r}) = P(\mathbf{r}) \cdot T(\mathbf{r})$$

The incident probe includes:
1. **Gaussian amplitude envelope**: $A(r) = \exp(-r^2 / (2\sigma_p^2))$.
2. **Defocus wavefront curvature**: $\phi_{\text{defocus}}(r) = c_{\text{defocus}} \cdot (r / r_{\max})^2$.
3. **Primary astigmatism**: $\phi_{\text{astig}}(r, \theta) = c_{\text{astig}} \cdot (r / r_{\max})^2 \cos(2(\theta - \theta_0))$.
4. **Coma**: $\phi_{\text{coma}}(r, \theta) = c_{\text{coma}} \cdot (r / r_{\max})^3 \cos(\theta)$.
5. **Pointing jitter**: Sub-pixel translations $(y - \delta y, x - \delta x)$.

---

## 4. Near-Field Fresnel Diffraction (Model F4)

When the detector is at a finite distance $z$ such that $N_F \approx 1$, propagation is governed by the Fresnel transfer function:

$$H(q_x, q_y) = \exp\left(i \frac{2\pi z}{\lambda}\right) \exp\left(-i \pi \lambda z (q_x^2 + q_y^2)\right)$$

$$\psi_z(x, y) = \mathcal{F}^{-1}\left[ \mathcal{F}[\psi_0(x, y)] \cdot H(q_x, q_y) \right]$$

$$I_{\text{Fresnel}}(x, y) = |\psi_z(x, y)|^2$$

All parameters are parameterized in SI units: wavelength $\lambda$ (m), distance $z$ (m), and pixel pitch $\Delta x$ (m), observing the Nyquist limit $\Delta q = 1 / (N \Delta x)$.

---

## 5. Polychromatic Pink-Beam Diffraction (Model F5)

XFEL self-amplified spontaneous emission (SASE) pulses and synchrotron pink beams possess a finite spectral bandwidth $\Delta \lambda / \lambda \in [0.1\%, 5\%]$.

Because the physical detector coordinate $r_{\text{det}}$ corresponds to spatial frequency $q = r_{\text{det}} / (\lambda z)$, wavelength dilation scales reciprocal coordinates linearly with $\lambda$:

$$I_{\text{poly}}(\mathbf{q}_0) = \int S(\lambda) I\left(\mathbf{q}_0 \frac{\lambda}{\lambda_0}\right) \, d\lambda$$

OpenDiffraction implements spectral integration by radially interpolating monochromatic intensity slices across discrete wavelength bins weighted by a normalized Gaussian spectral profile $S(\lambda)$.

---

## 6. 3D Particle Scattering (Model F6)

For a 3D scalar scattering density $\rho(x, y, z)$, the specimen is rotated by 3D Euler angles $(\alpha, \beta, \gamma)$:

$$\rho_{\text{rot}}(\mathbf{r}) = \mathcal{R}_{\alpha, \beta, \gamma}[\rho(\mathbf{r})]$$

Under the first Born projection approximation along the beam propagation axis $z$:

$$P_{\text{proj}}(x, y) = \int \rho_{\text{rot}}(x, y, z) \, dz$$

The resulting 2D projection represents the effective transmission density entering the Fraunhofer forward model.

---

## 7. Noise Statistics and Detector Realism

### 7.1 Photon Counting (Poisson Shot Noise)
Given target photon budget $F$:

$$s = \frac{F}{\sum_{\mathbf{q}} I_{\text{expected}}(\mathbf{q})}, \quad \lambda_i = s \cdot I_{\text{expected}}(\mathbf{q}_i)$$

$$K_i \sim \text{Poisson}(\lambda_i)$$

### 7.2 Gaussian Electronics Readout Noise
$$Y_i = K_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$$

Default $\sigma_{\text{read}} = 1 / (2\sqrt{2\ln 2}) \approx 0.42466$, corresponding to $\text{FWHM} = 1.0$ count. Signed values ($Y_i < 0$) are preserved to avoid zero-truncation bias.
