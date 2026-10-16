# OpenDiffraction 48 Controlled Variation Profiles (V01 - V48)

OpenDiffraction 2.0 systematically controls physical degradation mechanisms across 48 explicitly labeled profiles grouped into 6 scientific domains. This design enables researchers to isolate the sensitivity of phase retrieval algorithms and neural architectures to specific physical phenomena.

---

## Group 1: Real-World Specimens & Structural Morphologies (V01 - V12)

This group evaluates structural reconstruction fidelity across biologically inspired and materials science morphologies.

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V01** | Cell-Like Biological Phantoms | F1 (Fraunhofer) | Outer membrane ($R=22$ px), internal nucleus, organelle inclusions | Tests phase retrieval on hierarchical nested biological structures. |
| **V02** | Virus-Like Particles (VLP) | F1 (Fraunhofer) | Capsid shell ($T_{\text{shell}}=4$ px), internal core ($R_{\text{core}}=12$ px) | Benchmarks spherical hollow shells and inner-density resolution. |
| **V03** | Protein-Complex Aggregates | F1 (Fraunhofer) | 5 to 12 stochastic globular subunits, variable Gaussian radii | Evaluates unstructured asymmetric multi-domain biological macromolecules. |
| **V04** | Core-Shell Nanoparticles | F1 (Fraunhofer) | Dense inner core ($\rho=1.0$), dielectric outer shell ($\rho=0.4$) | Tests contrast sensitivity between core and shell refractive indices. |
| **V05** | Ellipsoidal Nanoparticles | F1 (Fraunhofer) | Aspect ratio $a/b \in [1.5, 4.0]$, arbitrary 2D in-plane orientation | Evaluates anisotropic Fourier speckle orientation recovery. |
| **V06** | Nanorods and Nanowires | F1 (Fraunhofer) | High aspect ratio ($L/W \ge 6.0$), rounded end-caps | Probes streak diffraction signatures and high-spatial-frequency edges. |
| **V07** | Multi-Particle Clusters | F1 (Fraunhofer) | Stochastic oligomeric clusters with non-overlapping contact bounds | Assesses cross-talk and phase stagnation in multi-component objects. |
| **V08** | Porous Matrix / Aerogels | F1 (Fraunhofer) | Spatially correlated Gaussian random field thresholded at porosity 0.65 | Evaluates continuous interconnected porous media without simple convex supports. |
| **V09** | Colloidal Crystals | F1 (Fraunhofer) | Hexagonal and square periodic lattice ordering of spherical colloids | Measures Bragg-like peak intensity modulation and missing-order retrieval. |
| **V10** | Lithographic Test Patterns | F1 (Fraunhofer) | Siemens star sectors, resolution lines, binary zone-plate gratings | Quantifies spatial resolution limits (line pairs per micron). |
| **V11** | Internal Defect Structures | F1 (Fraunhofer) | Bulk material containing micro-void cavities and internal fracture cracks | Benchmarks internal defect and pore detection inside solid matrices. |
| **V12** | Phase-Separated Mixtures | F1 (Fraunhofer) | Bimodal spinodal decomposition textures (strictly OOD split) | Tests out-of-distribution generalization on multi-phase materials. |

---

## Group 2: Illumination & Beam Coherence (V13 - V20)

This group benchmarks algorithmic robustness against source imperfections, wavefront aberrations, and coherence loss.

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V13** | Ideal Plane-Wave Coherence | F1 (Fraunhofer) | Spatial coherence length $l_{\text{coh}} = \infty$, zero wavefront distortion | Clean theoretical baseline for coherent diffraction imaging. |
| **V14** | Moderate Partial Coherence | F2 (GSM) | Gaussian Schell-model mutual intensity, $l_{\text{coh}} = 150$ px | Simulates typical undulator synchrotron spatial coherence blur. |
| **V15** | Low Partial Coherence | F2 (GSM) | Gaussian Schell-model mutual intensity, $l_{\text{coh}} = 40$ px | Severe speckle contrast damping representing low-coherence laboratory sources. |
| **V16** | Anisotropic Partial Coherence | F2 (GSM) | Asymmetric coherence lengths: $l_{\text{coh}, x} = 150$ px, $l_{\text{coh}, y} = 50$ px | Reflects horizontal vs. vertical emittance differences in storage rings. |
| **V17** | Focused Probe Envelope | F3 (Thin Object) | Gaussian probe amplitude envelope ($\sigma_{\text{probe}} = 24$ px) | Evaluates probe truncation and non-uniform object illumination. |
| **V18** | Wavefront Curvature (Defocus)| F3 (Thin Object) | Quadratic phase curvature $\exp(i k r^2 / 2 R_c)$ with defocus $z_{\text{def}} = 50\,\mu$m | Assesses phase retrieval in divergent or convergent wavefront geometries. |
| **V19** | Beam Pointing Jitter | F3 (Thin Object) | Sub-pixel stochastic beam translation ($\sigma_{\text{jitter}} = 1.5$ px) | Simulates optomechanical vibration and beamline stability drift. |
| **V20** | Polychromatic Pink Beam | F5 (Polychromatic) | Spectral bandwidth $\Delta\lambda / \lambda_0 = 1.5\%$, integrated over 9 wavelengths | Evaluates radial speckle elongation across finite energy bandwidths. |

---

## Group 3: Detector Realism & Readout Artifacts (V21 - V30)

This group simulates non-idealities inherent to modern silicon pixelated X-ray detectors (e.g., CSPAD, Jungfrau, PILATUS, EIGER).

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V21** | Baseline Readout Noise | F1 (Fraunhofer) | Gaussian electronics noise $\sigma_{\text{read}} = 0.425$ count ($\text{FWHM} = 1.0$) | Standard DPR-compatible detector readout noise. |
| **V22** | Elevated Readout Noise | F1 (Fraunhofer) | High Gaussian electronics noise $\sigma_{\text{read}} = 2.0$ counts | Tests noise tolerance in high-temperature or uncooled detectors. |
| **V23** | Pure Poisson Counting | F1 (Fraunhofer) | Direct integer Poisson sampling without additive Gaussian readout noise | Idealized photon-counting pixel detectors (e.g., hybrid pixel array detectors). |
| **V24** | Extreme Photon Starvation | F1 (Fraunhofer) | Low incident flux ($10^3$ to $10^4$ total photons across $512 \times 512$) | Tests ultra-low-dose imaging for radiation-sensitive biological samples. |
| **V25** | Dynamic Range Saturation | F1 (Fraunhofer) | Full-well saturation clipping at $I_{\text{max}} = 16{,}383$ (14-bit ADC) | Benchmarks loss of central intense speckles due to pixel saturation. |
| **V26** | ADC Integer Quantization | F1 (Fraunhofer) | Discrete integer rounding and uniform truncation | Tests sensitivity to digitization and low-bit ADCs. |
| **V27** | Hot / Persistent Stuck Pixels | F1 (Fraunhofer) | Fixed isolated pixels pinned to full-scale saturation value | Evaluates algorithmic stability against localized point outliers. |
| **V28** | Dead Pixel Clusters | F1 (Fraunhofer) | Random dead sensor clusters ($R=2$ to $5$ px) returning zero counts | Evaluates 2D spatial inpainting and missing-speckle reconstruction. |
| **V29** | Readout Channel Dropouts | F1 (Fraunhofer) | Complete horizontal and vertical 1-pixel readout line dropouts | Tests 1D streak inpainting along ASIC readout channel boundaries. |
| **V30** | Multi-Tile Module Gaps | F1 (Fraunhofer) | 4-quadrant cross gaps ($W_{\text{gap}} = 8$ px) separating detector modules | Standard modular detector geometry in large-area XFEL cameras. |

---

## Group 4: Background Scattering & Parasitic Light (V31 - V36)

This group incorporates realistic parasitic signals arising from beamline optics, sample delivery systems, and stray radiation.

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V31** | Clean High-Vacuum Baseline | F1 (Fraunhofer) | Parasitic scattering background intensity $I_{\text{bg}} \equiv 0$ | Theoretical pure diffraction reference without ambient scatter. |
| **V32** | Ambient Gas Halo | F1 (Fraunhofer) | Central Lorentzian diffuse halo from residual helium/air scattering | Benchmarks background subtraction and high-angle signal isolation. |
| **V33** | Support Substrate Halo | F1 (Fraunhofer) | Broad diffuse scattering ring from 10 nm amorphous carbon film | Evaluates separation of specimen signal from carrier substrate scattering. |
| **V34** | Liquid Carrier Water Ring | F1 (Fraunhofer) | Diffuse water scattering peak centered at $q \approx 2.0$ Å$^{-1}$ | Critical for liquid-jet single-particle injection experiments. |
| **V35** | Slit Parasitic Flare | F1 (Fraunhofer) | Cross-shaped diffraction flare from upstream scatter-guard slits | Evaluates phase retrieval under directional optical flare contamination. |
| **V36** | Stray Radiation Tracks | F1 (Fraunhofer) | High-energy cosmic ray muon ionization tracks (linear streaks) | Tests robustness against extreme structured outliers and gamma hits. |

---

## Group 5: Experimental Acquisition & Hit Finding (V37 - V42)

This group simulates single-shot pulse-by-pulse acquisition protocols common to serial femtosecond crystallography and single-particle CDI at XFEL facilities.

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V37** | High-Flux Centered Hit | F1 (Fraunhofer) | $10^7$ incident photons, particle centered within the focal spot | High-SNR benchmark frame for optimal iterative convergence. |
| **V38** | Low-Flux Centered Hit | F1 (Fraunhofer) | $5 \times 10^4$ incident photons, weak scattering cross-section | Low-SNR benchmark frame near detection threshold. |
| **V39** | Nominal Centered Hit | F1 (Fraunhofer) | $10^6$ incident photons, zero lateral beam displacement | Standard serial acquisition reference frame. |
| **V40** | Off-Center Marginal Hit | F3 (Thin Object) | Stochastic lateral displacement ($\Delta r \in [10, 25]$ px) | Benchmarks illumination gradient correction for grazing hits. |
| **V41** | Multi-Particle Hit | F1 (Fraunhofer) | Two or more distinct particles simultaneously intersecting the pulse | Tests hit-finding filters and multi-body interference handling. |
| **V42** | Blank Carrier Non-Hit | F1 (Fraunhofer) | Zero particle density, pure solvent carrier background and noise | Benchmark frame for XFEL automated hit-finding classifiers (Task T4). |

---

## Group 6: Advanced Physical Models & Out-of-Distribution Regimes (V43 - V48)

This group benchmarks complex optical interactions, non-Fraunhofer propagation, and compound experimental stress.

| Profile | Name | Forward Model | Key Physical Parameters | Benchmark Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **V43** | Pure Phase Object | F1 (Fraunhofer) | Uniform transmission amplitude ($|O|=1.0$), phase $\phi \in [0, 0.8\pi]$ | Evaluates phase-contrast imaging of unstained low-Z specimens. |
| **V44** | Strong Phase Object | F1 (Fraunhofer) | Non-linear phase depth with topological phase vortex singularity | Tests phase unwrapping and phase vortex recovery. |
| **V45** | Near-Field Fresnel Regime | F4 (Fresnel) | Fresnel number $N_F = a^2 / (\lambda z) \approx 1.2$ in SI units | Benchmarks holography and near-field propagation solvers. |
| **V46** | 3D Euler Scattering | F6 (3D Particle) | Continuous 3D rotation $(\alpha, \beta, \gamma)$ with Born projection | Benchmark for 3D single-particle imaging and orientation classification. |
| **V47** | Complex Coupled Specimen | F1 (Fraunhofer) | Spatially correlated amplitude absorption and phase retardance | Realistic multi-element specimen with both absorption and phase contrast. |
| **V48** | Compound Multi-Defect Stress | F2 (GSM) | Partial coherence + high noise + dead modules + water ring + beamstop | Ultimate stress test combining multiple simultaneous degradations. |
