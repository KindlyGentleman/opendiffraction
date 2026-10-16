# OpenDiffraction: Known Limitations & Physics Scope

## 1. Forward Model Assumptions

1. **Fraunhofer Far-Field Approximation**:
   The diffraction patterns are computed under the Fraunhofer far-field approximation (valid when Fresnel number $N_F = a^2 / (\lambda z) \ll 1$). Near-field Fresnel propagation or multi-slice dynamical scattering effects in thick crystals are not modeled.

2. **Kinematical (Single-Scattering) Approximation**:
   Diffraction is modeled as a 2D Fourier transform of projected optical density. Multiple internal scattering within thick 3D macromolecules is neglected.

3. **Gaussian Schell-Model Spatial Coherence**:
   Coherence degradation is modeled via spatial autocorrelation damping under the Gaussian Schell-model. Temporal coherence (longitudinal bandwidth $\Delta \lambda / \lambda$) is neglected, which is standard for monochromatic XFEL and synchrotron pulses.

4. **Quasi-Flat Ewald Sphere**:
   The detector is modeled as a planar screen tangent to the Ewald sphere at low scattering angles ($q_z \approx 0$). High-angle curvature corrections are not included.

---

## 2. Machine Learning & Benchmarking Boundaries

1. **Synthetic vs. Experimental Gap**:
   While detector noise, partial coherence, beam stops, and module gaps are modeled accurately, real experimental XFEL instruments exhibit beam jitter, sample background scattering (e.g. water jets, helium gas), and polarization factors that may require transfer-learning or domain adaptation.

2. **Phase Retrieval Ambiguities**:
   Phase retrieval from a single diffraction intensity is mathematically subject to trivial ambiguities:
   - Global phase shift: $O(\mathbf{r}) \to O(\mathbf{r}) e^{i \phi_0}$
   - Spatial translation: $O(\mathbf{r}) \to O(\mathbf{r} - \mathbf{r}_0)$
   - Twin image / Conjugate inversion: $O(\mathbf{r}) \to O^*(-\mathbf{r})$
   Reconstruction evaluations must align coordinates and account for twin-image inversion before calculating metrics such as PSNR or SSIM.

3. **Target Scaling Convention**:
   The dataset supports both the standard physical target amplitude ($O \cdot \sqrt{s}$) and the DPR training convention (`false_scale=True`, $O \cdot s$). Models trained on one convention must not be evaluated naively on the other without scale adjustment.
