# HDF5 Storage Schema, Target Integrity, and DPR Scaling Audit Report (Gate IV)

**Branch**: `remediation/v3.1-scientific-audit`  
**Date**: 2026-10-17  
**Status**: PASS  
**Addressed Issues**: OD-004 (Target integrity, silent fallback elimination, compact/extended contracts, DPR scaling)

---

## 1. Executive Summary

This audit establishes the storage contracts, target scaling conventions, and task target requirements for OpenDiffraction 3.1.

In OpenDiffraction 3.0:
- When operating in compact storage mode, fields `intensity_expected` and `intensity_ideal` were omitted to conserve disk space. When a user instantiated `OpenDiffractionTorchDataset(task="denoising")` or `task="inpainting"`, a silent fallback substituted the noisy, masked input as its own clean ground truth (`clean_arr = inp_arr`). Supervised models trained under this fallback learned the identity mapping rather than denoising or inpainting.
- Upstream DPR (Lee et al. 2025) scaled target densities as $O_{\text{dpr}} = O \cdot s$ where $s = \Phi / \sum I$, while standard physical wave optics dictates that field amplitude scales with the square root of photon budget: $O_{\text{phys}} = O \cdot \sqrt{s}$.

In this remediation:
1. Eliminated silent fallback in [`torch_dataset.py`](file:///d:/03_Proyek/Diffraction_Dataset/opendiffraction/io/torch_dataset.py), raising an informative `KeyError` unless `self_supervised=True` is explicitly requested.
2. Verified the two formal storage profiles (Compact and Extended) with exact schema guarantees.
3. Validated DPR target scaling ($s \cdot O$) alongside physical amplitude scaling ($\sqrt{s} \cdot O$) and unscaled normalized object ground truth ($O \in [0, 1]$).
4. Defined explicit target contracts for all 12 proposed scientific ML tasks.

---

## 2. Formal Storage Profiles

### 2.1 Profile A: Extended Scientific Mode (`storage_mode="extended"`)
Recommended for local research, reference benchmarks, algorithm diagnostics, and physics verification. Contains complete forward simulation provenance.

| Dataset Key | Shape | Dtype | Physical Meaning |
|---|---|---|---|
| `input` | $(N, 1, 512, 512)$ | `float32` | Strictly masked observed intensity: $I_{\text{raw}} \cdot M$ (zero leakage) |
| `target` | $(N, 1, 64, 64)$ | `float32` | DPR-compatible ground-truth density ($O \cdot s$) |
| `mask` | $(N, 1, 512, 512)$ | `bool` | Detector valid pixel mask ($1 = \text{valid}, 0 = \text{missing}$) |
| `object_amplitude`| $(N, 1, 64, 64)$ | `float32` | True unscaled specimen density / amplitude in $[0, 1]$ |
| `target_physical` | $(N, 1, 64, 64)$ | `float32` | Physical amplitude-scaled density ($O \cdot \sqrt{s}$) |
| `object_phase` | $(N, 1, 64, 64)$ | `float32` | Ground-truth phase in radians (present for complex objects) |
| `intensity_ideal` | $(N, 1, 512, 512)$ | `float32` | Coherent far-field Fraunhofer intensity $|\mathcal{F}\{O\}|^2$ |
| `intensity_expected`| $(N, 1, 512, 512)$| `float32` | Clean physical intensity after coherence damping & background |
| `intensity_raw` | $(N, 1, 512, 512)$ | `float32` | Unmasked detector readout including noise behind beamstop |
| `intensity_observed`| $(N, 1, 512, 512)$| `float32` | Strictly masked observation ($I_{\text{raw}} \cdot M$) |
| `intensity_measured`| $(N, 1, 512, 512)$| `float32` | Backward-compatibility alias returning `intensity_observed` |
| `valid_mask` | $(N, 1, 512, 512)$ | `bool` | Duplicate alias for detector mask |

### 2.2 Profile B: Compact Core Mode (`storage_mode="compact"`)
Optimized for public sharing (Kaggle, Zenodo) and deep learning training where disk storage and network bandwidth are constrained ($\approx 70\%$ smaller file sizes).

| Dataset Key | Shape | Dtype | Physical Meaning |
|---|---|---|---|
| `input` | $(N, 1, 512, 512)$ | `float32` | Strictly masked observed intensity ($I_{\text{raw}} \cdot M$) |
| `target` | $(N, 1, 64, 64)$ | `float32` | DPR-compatible target density ($O \cdot s$) |
| `mask` | $(N, 1, 512, 512)$ | `bool` | Detector valid pixel mask |

Both profiles embed canonical root attributes:
- `generator_version`: Current generator release version (e.g. `3.1.0`)
- `schema_version`: Data schema version (e.g. `3.1`)
- `storage_mode`: `"compact"` or `"extended"`
- `config_fingerprint`: Deterministic SHA-256 hash of task generation parameters
- `num_samples`: Total records in shard

---

## 3. DPR Target Scaling vs Physical Scaling

### 3.1 Upstream DPR Convention (Lee et al. 2025)
In the upstream repository (`sungyun98/DPR`), the training pair is:
- Input: Measured intensity with noise and beamstop $I_{\text{obs}}$.
- Target: Object density scaled linearly by the intensity normalization factor:

$$O_{\text{dpr}} = O \cdot s, \quad s = \frac{\Phi}{\sum_{\mathbf{k}} I_{\text{ideal}}(\mathbf{k})}$$

Because $I \propto |O|^2$, scaling intensity by $s$ means the field amplitude scales by $\sqrt{s}$. Scaling $O$ by $s$ instead of $\sqrt{s}$ introduces a dimensionally non-linear scaling across different photon budgets.

### 3.2 Physical Scaling Convention
Under classical scalar diffraction:

$$|\Psi(\mathbf{k})|^2 = s |\mathcal{F}\{O(\mathbf{r})\}|^2 = |\mathcal{F}\{\sqrt{s} O(\mathbf{r})\}|^2$$

Hence, the physical exit-surface density is:

$$O_{\text{phys}} = O \cdot \sqrt{s}$$

The ratio between conventions is:

$$\frac{O_{\text{dpr}}}{O_{\text{phys}}} = \sqrt{s}$$

OpenDiffraction guarantees:
1. `target` / `target_dpr`: Exactly reproduces upstream DPR scaling for zero-code-change drop-in compatibility with `deeppr.CustomDataset` and pre-trained DPR weights.
2. `target_physical`: Available in extended mode for physically rigorous inverse scattering benchmarks.
3. `object_amplitude`: Raw unscaled object density in $[0, 1]$ for invariant structural metric evaluation.

---

## 4. Task-Specific Target Requirements (T1 to T12)

| Task ID | Task Name | Input Field | Supervised Target Required | Compact Shard Compatible |
|---|---|---|---|---|
| **T1** | Phase Retrieval | `input` | `target` (or `object_amplitude`) | Yes |
| **T2** | Intensity Denoising | `input` | `intensity_expected` (clean intensity) | Extended only (or `self_supervised=True`) |
| **T3** | Detector Inpainting | `input` | `intensity_raw` (unmasked readout) | Extended only (or `self_supervised=True`) |
| **T4** | Hit Classification | `input` | `hit_status` class label (0, 1, 2) | Yes (from CSV manifest) |
| **T5** | Defect Segmentation | `input` | `~mask` binary defect map | Yes |
| **T6** | Mask Super-Resolution | `input` | High-resolution mask / intensity | Companion shard |
| **T7** | Multi-Exposure Fusion | Multi `input` | `intensity_expected` | Extended only |
| **T8** | Anomaly Detection | `input` | `is_ood` binary label | Yes (from CSV manifest) |
| **T9** | Continuous Inversion | `input` | Latent physical parameter vector | Yes (from CSV manifest) |
| **T10**| Domain Transfer | `input` | Target domain representation | Companion shard |
| **T11**| Phase-Only Recovery | `input` | `object_phase` | Extended only |
| **T12**| 3D Orientation Est. | `input` | Euler angles $(\alpha, \beta, \gamma)$ | Yes (from CSV manifest) |

---

## 5. Acceptance Gate IV Sign-Off

All Gate IV criteria are fulfilled:
- Silent fallback in dataset loaders eliminated; explicit exceptions raised.
- Compact and Extended schemas verified with atomic chunked writing.
- DPR and Physical target scalings mathematically and numerically validated.
- 100% test pass rate across target and storage test suites.
