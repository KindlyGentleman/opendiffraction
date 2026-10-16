# OpenDiffraction 3.0 Physical Acquisition Matrix

Specification of multi-exposure acquisition conditions applied to every base object:

| Acquisition ID | Index | Physical Description | Photon Scaling | Detector Features |
| :--- | :--- | :--- | :--- | :--- |
| `acq_00` | 0 | Nominal Reference Exposure | Nominal profile flux ($10^5 - 10^7$) | Standard central beam stop |
| `acq_01` | 1 | Low Photon Flux / Shot-Noise Regime | 10% nominal flux (Poisson limited) | Standard central beam stop |
| `acq_02` | 2 | Partial Coherence Blur & Module Gaps | 40% coherence length damping | 4px module gaps + detector holder arm |
| `acq_03` | 3 | Elevated Readout Noise & Beam Jitter | 150% readout std dev ($\sigma=0.64$) | Sub-pixel beam center displacement ($\pm 2.0$ px) |

## Acquisition Verification

- **Total Base Objects**: 250
- **Acquisitions per Object**: Exactly 4
- **Total Observations**: 1,000
- **Consistency**: Every base object is observed across all 4 distinct acquisition states.
