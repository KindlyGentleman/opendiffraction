# OpenDiffraction 3.0 Physical Model Validation Report

## Forward Physics Models Verification

| Model | Name | Profile Coverage | Target Compatibility | Status |
| :--- | :--- | :--- | :--- | :--- |
| F1 | Coherent Fraunhofer | V01-V13, V21-V42, V48 | DPR `false_scale=True` + Physical | PASS |
| F2 | Partial Coherence (GSM) | V14, V15 | Corrected zero-lag centering | PASS |
| F3 | Wavefront Aberrations | V16, V17, V18, V19 | Defocus, spherical, astigmatism | PASS |
| F4 | Near-field Fresnel | V45 | Holographic near-field propagation | PASS |
| F5 | Polychromatic Spectrum | V20, V46 | Pink beam & broad bandwidth | PASS |
| F6 | 3D Euler Scattering | V47 | 3D particle volume projection | PASS |

## Target Scaling and Normalization

- **DPR Convention**: `target_dpr` $= O(\mathbf{r}) \cdot s$ (where $s = F / \sum I$). Matches upstream DPR neural training targets exactly.
- **Physical Convention**: `target_physical` $= O(\mathbf{r}) \cdot \sqrt{s}$. Preserves true amplitude scaling.
- **Signed Detector Readouts**: Electronic readout noise produces small negative intensities (min: -17.35), accurately reflecting unclipped detector ADUs.
