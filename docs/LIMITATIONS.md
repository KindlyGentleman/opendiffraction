# OpenDiffraction Scientific Limitations & Scope

OpenDiffraction 2.0 provides a physics-grounded synthetic benchmark for coherent diffraction imaging and inverse problems. To maintain scientific integrity and prevent algorithmic over-claims, this document explicitly details the physical approximations, machine-learning considerations, and operational boundaries of the dataset.

---

## 1. Physical Approximations and Assumptions

### 1. Thin-Object Approximation (Projection Approximation)
Models F1, F2, F3, and F5 model the specimen as a 2D complex transmission function $T(\mathbf{r}) = \exp(-\mu(\mathbf{r}) + i \phi(\mathbf{r}))$. This projection approximation assumes that:
- The longitudinal thickness of the specimen along the optical axis $\Delta z$ is negligible compared to the depth of focus $\Delta z \ll \lambda / \theta_{\text{max}}^2$.
- Multiple scattering, dynamical diffraction, and beam refraction within the specimen volume are negligible.
For thick materials (exceeding a few hundred nanometers at soft X-ray energies), 3D multislice wave propagation or dynamical scattering theory is required.

### 2. First Born Approximation in 3D Scattering (Model F6)
Model F6 models 3D particle scattering using the continuous Fourier slice projection theorem under the first Born approximation. This assumes:
- The scattering potential is sufficiently weak that the total wavefield inside the specimen equals the incident probe wavefield.
- Absorption shadowing, anomalous dispersion resonances, and internal phase retardations are linearized.

### 3. Scalar Wave Optics
All forward models solve the scalar Helmholtz equation. Vectorial Maxwell electromagnetic effects, X-ray polarization anisotropy (e.g., linear polarization from synchrotrons or magnetic circular dichroism), and cross-polarization scattering are not modeled.

### 4. Spatially Stationary Partial Coherence (Gaussian Schell-Model)
Model F2 simulates spatial coherence damping using the Gaussian Schell-model (GSM). This treats the transverse mutual intensity as spatially stationary ($J(\mathbf{r}_1, \mathbf{r}_2) = \sqrt{I(\mathbf{r}_1) I(\mathbf{r}_2)} g(\mathbf{r}_1 - \mathbf{r}_2)$). While consistent with upstream DPR and typical undulator coherence profiles, real SASE XFEL pulses exhibit shot-to-shot stochastic modal fluctuations and wavefront tilts that deviate from idealized stationary Gaussian statistics.

### 5. Fresnel Near-Field Sampling Bounds (Model F4)
Near-field Fresnel diffraction is simulated using the angular spectrum method. While SI-unit propagation distances and wavelengths are strictly verified, numerical grid sampling restricts the valid propagation distance range to avoid aliasing:
$$z \ge \frac{N \Delta x^2}{\lambda}$$
Violations outside this regime can cause wrap-around boundary reflections.

---

## 2. Machine-Learning and Generalization Caveats

### 1. Synthetic-to-Real Domain Gap
High reconstruction fidelity, high SSIM, or near-zero R-factors achieved on OpenDiffraction **do not prove or imply equal performance on experimental synchrotron or XFEL beamlines**. Experimental beamtimes exhibit complex non-idealities:
- Spatially heterogeneous detector gain calibrations and dark-current drift.
- Fluctuating parasitic scatter from sample delivery nozzles or aerosol injectors.
- Pulse-to-pulse energy jitter and spectral mode fluctuations.
Researchers deploying models to real experimental beamlines should treat OpenDiffraction as a foundational pretraining and benchmarking suite, followed by fine-tuning on experimental calibration runs.

### 2. Intrinsic Phase Ambiguities
Phase retrieval from intensity measurements alone is fundamentally non-unique up to:
- Global phase shifts: $O(\mathbf{r}) \to O(\mathbf{r}) e^{i \phi_0}$.
- Spatial translations: $O(\mathbf{r}) \to O(\mathbf{r} - \mathbf{r}_0)$.
- Twin-image reflections: $O(\mathbf{r}) \to O^*(-\mathbf{r})$.
Direct pixel-by-pixel comparisons without optimal translation and twin-image registration will produce falsely degraded metric scores.

---

## 3. Scope and Prohibited Uses

> [!CAUTION]
> **NON-DIAGNOSTIC NOTICE**: OpenDiffraction is released exclusively for academic, research, and algorithmic evaluation purposes.
> It must NOT be used for:
> 1. Clinical, biomedical, or healthcare diagnosis without independent clinical trials.
> 2. Non-destructive safety inspections of structural engineering components where undetected defects could cause material failure or hazard.
> 3. Safety-critical automated control systems.
