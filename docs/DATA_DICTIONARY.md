# OpenDiffraction Data Dictionary

This document provides a comprehensive technical reference for the data structures, array schemas, data types, physical units, and tabular metadata in OpenDiffraction 2.0.

---

## 1. HDF5 Primary Dataset Schema

OpenDiffraction stores paired benchmark samples in HDF5 format (`.h5` files, chunked and compressed with gzip level 4). Each HDF5 file adheres to the following internal hierarchy:

| Dataset Path | Array Shape | Dtype | Units | Physical Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/input` | `(N, 1, 512, 512)` | `float32` | Photons / ADU counts | $(-\infty, +\infty)$ (signed) | Measured degraded diffraction pattern with Poisson noise, readout noise, and beamstop masking. Ready for direct input into neural networks. |
| `/target` | `(N, 1, 64, 64)` | `float32` | Dimensionless (normalized) | $[0.0, 1.0]$ | Ground-truth real-space object transmission density scaled according to DPR convention (`false_scale=True`, $O_{\text{dpr}} = O / \sqrt{\sum |O|^2}$). |
| `/mask` | `(N, 1, 512, 512)` | `bool` | Boolean flag | `{False, True}` | Binary detector validity mask. `True` indicates a valid measured pixel; `False` indicates a missing pixel (beamstop, dead pixel, gap). |
| `/object_amplitude` | `(N, 1, 64, 64)` | `float32` | Transmission magnitude | $[0.0, 1.0]$ | Physical unscaled transmission amplitude $|O(\mathbf{r})|$ before DPR $L_2$ normalization. |
| `/object_phase` | `(N, 1, 64, 64)` | `float32` | Radians | $[-\pi, +\pi]$ | Ground-truth phase shift $\phi(\mathbf{r})$ induced by the object. For pure absorption phantoms, phase is identically zero. |
| `/intensity_ideal` | `(N, 1, 512, 512)` | `float32` | Arbitrary intensity | $[0.0, +\infty)$ | Uncorrupted continuous Fraunhofer diffraction modulus squared $\|A(\mathbf{q})\|^2$ at infinite photon flux. |
| `/intensity_expected`| `(N, 1, 512, 512)` | `float32` | Expected photon count | $[0.0, +\infty)$ | Partially coherent photon arrival rate $\lambda_i$ incorporating coherence blur and incident photon flux before stochastic noise sampling. |
| `/intensity_measured`| `(N, 1, 512, 512)` | `float32` | Measured detector counts| $(-\infty, +\infty)$ (signed) | Complete simulated detector readout after Poisson counting and additive Gaussian electronics noise ($\sigma_{\text{read}} = 0.425$, FWHM = 1.0 count). |
| `/valid_mask` | `(N, 1, 512, 512)` | `bool` | Boolean flag | `{False, True}` | Alias to `/mask` defining active detector channels. |

### Memory and Storage Footprint
- Single sample HDF5 size: $\approx 1.25$ MB compressed ($\approx 6.3$ MB uncompressed).
- Fast random access chunking: Chunks are sized `(1, 1, 512, 512)` for 2D patterns and `(1, 1, 64, 64)` for real-space objects.

---

## 2. CXI / NeXus Experimental Data Schema

For interoperability with synchrotron and XFEL data analysis packages (such as `cxi-view`, `Dragonfly`, and `OnDA`), OpenDiffraction exports standard Coherent X-ray Imaging (CXI) NeXus files (`.cxi` format conforming to NeXus NXdata specifications).

```text
/
└── entry_1: NXentry
    ├── data_1: NXdata
    │   ├── data: float32 [N, 512, 512]         # Measured diffraction pattern
    │   ├── mask: uint32 [N, 512, 512]         # Defect bitmask (0 = valid, 1 = invalid)
    │   └── @signal = "data"
    ├── image_1: NXdata
    │   ├── data: float32 [N, 64, 64]           # Real-space ground-truth amplitude
    │   └── @signal = "data"
    └── instrument_1: NXinstrument
        ├── detector_1: NXdetector
        │   ├── distance: float64               # Sample-to-detector distance [meters]
        │   ├── x_pixel_size: float64           # Horizontal detector pixel pitch [meters]
        │   ├── y_pixel_size: float64           # Vertical detector pixel pitch [meters]
        │   └── description: string             # Detector hardware identifier (e.g., Jungfrau, CSPAD)
        └── source_1: NXsource
            ├── energy: float64                 # Photon energy [Joules]
            ├── wavelength: float64             # Radiation wavelength [meters]
            └── probe: string                   # Source type (e.g., SASE XFEL, Undulator Synchrotron)
```

### CXI Mask Conventions
- In NeXus CXI conventions, the mask array is an integer bitmask where **0 denotes a good pixel** and **non-zero values denote corrupted or masked pixels**.
- Conversion: `cxi_mask = uint32(~hdf5_mask)`.

---

## 3. Metadata Table Schema (`metadata.csv`)

The tabular manifest provides sample-level ground-truth parameters, physical configurations, and partition tags for reproducible benchmarking and filtering:

| Column Header | Data Type | Physical Units | Description & Allowed Values |
| :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Identifier | Unique sample key (format: `{split}_{shard:03d}_{index:05d}`, e.g., `train_000_00042`). |
| `split` | `string` | Categorical | Split partition assignment: `train`, `valid`, `test`, `test_ood`. |
| `group_id` | `string` | Hash / UUID | Base object identifier used for strictly disjoint splitting. All variations of an object share the same `group_id`. |
| `profile_id` | `string` | Categorical | Controlled variation profile code: `V01` through `V48`. |
| `profile_group` | `string` | Categorical | High-level physical grouping: `Specimens`, `Illumination`, `Detectors`, `Backgrounds`, `Acquisition`, `AdvancedPhysics`. |
| `scenario_id` | `string` | Categorical | Real-world acquisition scenario tag: `S1` to `S6` (or `None`). |
| `forward_model` | `string` | Categorical | Forward physics engine: `F1_Fraunhofer`, `F2_PartialCoherence`, `F3_ThinObject`, `F4_Fresnel`, `F5_Polychromatic`, `F6_3DParticle`. |
| `total_photons` | `float64` | Integrated photons | Total photon count incident on the active detector surface. Range: $10^3$ to $10^8$. |
| `coherence_length_px` | `float64` | Pixels | Transverse coherence length $l_{\text{coh}}$ in detector pixels. $\infty$ indicates full spatial coherence. |
| `beamstop_radius_px` | `int32` | Pixels | Radius of the central circular beamstop mask (typically 10 to 45 pixels). |
| `readout_noise_std` | `float64` | Counts | Gaussian readout electronic noise standard deviation $\sigma_{\text{read}}$. Nominal baseline is $0.425$ counts ($\text{FWHM} = 1.0$). |
| `is_hit` | `bool` | Boolean flag | Hit detection label: `True` if specimen is within the beam pulse; `False` for blank solvent carrier exposures. |
| `euler_alpha` | `float64` | Radians | First Euler rotation angle $\alpha \in [0, 2\pi)$ for 3D scattering models (F6). |
| `euler_beta` | `float64` | Radians | Second Euler rotation angle $\beta \in [0, \pi]$ for 3D scattering models (F6). |
| `euler_gamma` | `float64` | Radians | Third Euler rotation angle $\gamma \in [0, 2\pi)$ for 3D scattering models (F6). |
| `wavelength_m` | `float64` | Meters | Illumination nominal wavelength $\lambda_0$ (e.g., $1.2398 \times 10^{-10}$ m for 10 keV X-rays). |
| `detector_distance_m` | `float64` | Meters | Working sample-to-detector distance $z$ in SI meters (typically $0.1$ m to $5.0$ m). |
| `pixel_size_m` | `float64` | Meters | Detector pixel pitch $\Delta x_{\text{det}}$ in SI meters (typically $75 \times 10^{-6}$ m). |
| `oversampling_ratio` | `float64` | Ratio | Linear oversampling factor $\sigma = N_{\text{det}} / N_{\text{obj}} = 512 / 64 = 8.0$. |
| `dynamic_range_bits`| `int32` | Bits | ADC bit depth before saturation (e.g., 14-bit or 16-bit integer bounds). |

---

## 4. Benchmark Task Sub-Manifests (`tasks/`)

Each benchmark directory under `tasks/` contains specialized subsets of `metadata.csv` with standardized input/target pairs and metric targets:

- `tasks/T01_phase_retrieval.csv`: Input `/input`, Target `/target` (supervised phase retrieval).
- `tasks/T02_denoising.csv`: Input `/input`, Target `/intensity_ideal` (Poisson/Gaussian denoising).
- `tasks/T03_inpainting.csv`: Input `/input` masked by `/mask`, Target `/intensity_ideal` (defect restoration).
- `tasks/T04_hit_finding.csv`: Input `/input`, Target `is_hit` (binary classification).
- `tasks/T05_defect_segmentation.csv`: Input `/input`, Target `/mask` (anomaly mask recovery).
- `tasks/T06_morphology_classification.csv`: Input `/input`, Target object family classification.
- `tasks/T07_parameter_estimation.csv`: Input `/input`, Target `total_photons`, `coherence_length_px`.
- `tasks/T08_anomaly_detection.csv`: Standard train split vs. `test_ood` out-of-distribution evaluation.
- `tasks/T09_domain_generalization.csv`: Evaluates models trained on S1 across S2-S6 scenarios.
- `tasks/T10_self_supervised.csv`: Self-supervised representation pretraining pairs.
- `tasks/T11_uncertainty_quantification.csv`: Calibrated pixel-wise confidence intervals.
- `tasks/T12_physics_loss_benchmark.csv`: Gradient-based phase retrieval with physical consistency losses.
