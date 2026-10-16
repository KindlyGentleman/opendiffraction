# Synthetic-to-Real Domain Gap Analysis in Coherent Diffractive Imaging

## 1. Executive Summary

This report provides a rigorous comparative analysis of the domain gap between physics-based synthetic diffraction patterns (OpenDiffraction 3.0) and experimental measurements collected at X-ray Free-Electron Laser (XFEL) and synchrotron beamlines. Identifying and modeling these physical discrepancies is critical for successful transfer learning and zero-shot neural phase retrieval on experimental data.

---

## 2. Quantitative Domain Discrepancies

### 2.1 Radial Intensity Decay and Power Spectral Density $I(q)$
- **Synthetic Physics**: In coherent Fraunhofer scattering from compact specimens with sharp interfaces, the angle-averaged intensity follows the classical Porod power-law decay:
  $$I(q) \propto q^{-4}$$
  OpenDiffraction reproduces this power-law decay across its 48 specimen geometries, with high-frequency speckle oscillations determined by specimen support.
- **Experimental Reality**: Physical beamline patterns exhibit several non-specimen background components superimposed on the Porod decay:
  1. *Low-$q$ Slit Flare*: Parasitic scattering from KB-mirror edges, apertures, and guard slits creates orthogonal streak flares radiating from the beam center.
  2. *Diffuse Gas Scattering*: Residual carrier gas in the interaction chamber contributes a smooth, slowly varying isotropic background.
  3. *Solvent / Buffer Halo*: Liquid jet droplets or ice shells introduce broad diffuse diffraction rings (e.g., the water ring at $q \approx 2.0$ $\text{\AA}^{-1}$).
- **Mitigation in OpenDiffraction**: Profiles `V32` (gas scattering), `V33` (support film halo), `V34` (solvent halo), and `V35` (slit streaks) explicitly simulate these background components.

### 2.2 Speckle Visibility and Beam Coherence
- **Synthetic Physics**: Purely coherent simulations assume infinite longitudinal and transverse coherence length ($\ell_{\text{coh}} = \infty$), yielding maximum speckle contrast:
  $$\beta = \frac{\sigma_I}{\langle I \rangle} = 1.0$$
- **Experimental Reality**: Real SASE (Self-Amplified Spontaneous Emission) XFEL pulses exhibit finite transverse spatial coherence and shot-to-shot spectral bandwidth fluctuations ($\Delta E / E \approx 0.1 - 0.5\%$), reducing speckle visibility ($\beta \approx 0.5 - 0.85$) and washing out high-$q$ interference fringes.
- **Mitigation in OpenDiffraction**: Group 2 profiles (`V14`, `V15`) implement the corrected zero-lag Gaussian Schell-model (GSM) with finite coherence lengths ($\ell_{\text{coh}} = 40 - 150$ px), and Profile `V46` models polychromatic spectral smearing.

### 2.3 Detector Noise and Pixel Response
- **Synthetic Physics**: Modeled as independent, identically distributed (i.i.d.) Poisson photon arrival statistics followed by additive Gaussian electronic readout noise ($\sigma_{\text{read}} \approx 0.42$ ADU).
- **Experimental Reality**:
  1. *Charge Sharing*: Photons absorbed near pixel boundaries induce charge splitting across adjacent pixels, introducing spatial noise correlation.
  2. *Adaptive Multi-Gain Switching*: Advanced detectors (e.g., AGIPD at European XFEL, CSPAD at LCLS) dynamically switch gain stages (high, medium, low) per pixel. Gain boundary miscalibrations introduce non-linear step discontinuities.
  3. *Dark Pedestal Drift*: Temperature fluctuations cause baseline ADU shifts across multi-hour data collection runs.

### 2.4 Detector Mask and Geometry Artifacts
- **Synthetic Physics**: Composite binary masks combining rectangular/circular beam stops, orthogonal module gaps, and holder shadow arms.
- **Experimental Reality**: Physical detectors feature non-coplanar sensor panels, sub-millimeter quadrant misalignments, curved shadow masks from flight tubes, and localized charge blooming around saturated beamstop edges.

---

## 3. Honest Documentation of Model Simplifications

| Physical Phenomenon | Synthetic Implementation (OpenDiffraction 3.0) | Physical Beamline Reality | Impact on Machine Learning |
| :--- | :--- | :--- | :--- |
| **Object Transmission** | 2D projection approximation (F1-F5) | 3D multiple scattering / thick objects | Ewald sphere curvature becomes significant for specimens $>1\,\mu\text{m}$. |
| **Beam Profile** | Flat plane-wave or low-order aberrations (F3) | Spatially non-uniform focused wavefront | Measured diffraction is the convolution of object with focused probe. |
| **Pulse Duration** | Instantaneous static specimen | Femtosecond pulse causing Coulomb explosion | Radiation damage and sample disintegration within pulse envelope. |
| **Fluorescence Background**| Neglected (pure coherent elastic scattering) | Incoherent isotropic fluorescence | Low-frequency background pedestal on high-$Z$ elements. |

---

## 4. Actionable Strategies for Real-World Transfer Learning

1. **Domain Randomization during Training**:
   - Randomize beamstop scale ($12 - 24$ px) and placement jitter ($\pm 4$ px).
   - Augment background with random 2D polynomial surfaces and low-frequency Gaussian noise.
   - Randomize photon budget across three orders of magnitude ($10^4 - 10^7$).
2. **Masked Loss Functions**:
   - Compute reconstruction and consistency losses exclusively over the valid detector support:
     $$\mathcal{L} = \frac{1}{\sum M_i} \sum_{i} M_i \cdot |I_{\text{pred}, i} - I_{\text{meas}, i}|^2$$
3. **Pre-processing Alignment**:
   - Run experimental frames through `CXIExperimentalAdapter` to center the zero-frequency DC peak and apply dark-pedestal subtraction before inference.
4. **Fine-Tuning on Experimental Hits**:
   - Pretrain encoders on the 1,000 OpenDiffraction synthetic samples and fine-tune with self-supervised consistency loss (Task T10/T12) on unlabelled experimental shots from CXIDB id-62 or id-81.
