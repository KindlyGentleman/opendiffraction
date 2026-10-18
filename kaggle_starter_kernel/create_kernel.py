import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# OpenDiffraction: Quickstart, EDA & Phase Retrieval Baseline
### Physics-Based Synthetic Coherent Diffraction Patterns for Scientific Machine Learning

**Dataset:** [hamadmuhammadkamal/opendiffraction-synthetic-cdi](https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi)  
**GitHub Repository:** [KindlyGentleman/opendiffraction](https://github.com/KindlyGentleman/opendiffraction)  
**Reference Paper:** Lee et al. (2025), *Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers*, **npj Computational Materials** 11, 68. DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7)

---

## 1. Scientific Background

**Coherent Diffraction Imaging (CDI)** is a lensless microscopy technique used at synchrotron beamlines and X-ray Free-Electron Lasers (XFELs). A coherent X-ray beam illuminates an isolated specimen, producing a far-field continuous diffraction intensity pattern recorded on a pixel detector:

$$I(\\mathbf{q}) = |\\mathcal{F}\\{O(\\mathbf{r})\\}|^2$$

Because detectors only measure intensity $|F|^2$, the Fourier phase is lost (the classic **phase retrieval problem**). Furthermore, experimental patterns suffer from:
1. **Central beamstops and detector tile gaps** (missing low-frequency and dead pixels)
2. **Poisson photon shot noise** ($10^3$ to $10^7$ incident photons)
3. **Additive Gaussian electronic readout noise**
4. **Partial spatial coherence and polychromatic bandwidth**

**OpenDiffraction** provides a fully reproducible, physics-grounded synthetic benchmark spanning 1,200 patterns across 6 forward models (F1–F6) and 12 machine learning tasks (T1–T12).
"""))

# Setup cell
cells.append(nbf.v4.new_code_cell("""import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Robust Kaggle / local environment data directory discovery
candidates = []
if Path('/kaggle/input').exists():
    candidates.extend(list(Path('/kaggle/input').rglob('metadata.csv')))
if Path('..').exists():
    candidates.extend(list(Path('..').rglob('metadata.csv')))
candidates.extend(list(Path('.').rglob('metadata.csv')))

if not candidates:
    raise FileNotFoundError("Could not locate metadata.csv in environment.")

DATA_DIR = candidates[0].parent
print(f'Loading data from: {DATA_DIR.resolve()}')
print('Files available:')
for f in sorted(DATA_DIR.rglob('*')):
    if f.is_file():
        print(f'  - {f.relative_to(DATA_DIR)} ({f.stat().st_size / 1024:.1f} KB)')
"""))

# EDA on metadata.csv
cells.append(nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis: `metadata.csv`

The `metadata.csv` file provides a complete index of all 1,200 diffraction patterns, including split assignments, forward model physics, incident photon budgets, noise parameters, and detector geometry.
"""))

cells.append(nbf.v4.new_code_cell("""metadata_path = DATA_DIR / 'metadata.csv'
df = pd.read_csv(metadata_path)
print(f'Total samples indexed: {len(df)}')
df.head(5)
"""))

cells.append(nbf.v4.new_code_cell("""# Distribution across splits and physical forward models
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=120)

# 1. Dataset splits
split_counts = df['split'].value_counts()
axes[0].bar(split_counts.index, split_counts.values, color=['#0284C7', '#0D9488', '#F59E0B', '#EF4444'])
axes[0].set_title('Dataset Partition Splits', fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].grid(axis='y', alpha=0.3)

# 2. Forward models (F1 - F6)
model_counts = df['density_type'].value_counts()
axes[1].bar(model_counts.index, model_counts.values, color='#6366F1')
axes[1].set_title('Forward Simulation Models (F1 - F6)', fontweight='bold')
axes[1].set_xlabel('Model ID')
axes[1].grid(axis='y', alpha=0.3)

# 3. Photon budget distribution
photon_vals = df['photon_budget'].dropna().astype(float)
axes[2].hist(np.log10(photon_vals), bins=15, color='#10B981', edgecolor='black', alpha=0.7)
axes[2].set_title('Incident Photon Count (log10)', fontweight='bold')
axes[2].set_xlabel('log10(Photons)')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# Loading HDF5 Shards
cells.append(nbf.v4.new_markdown_cell("""## 3. Inspecting HDF5 Shards

Each HDF5 shard contains:
- `intensity_measured`: $(512 \\times 512)$ imperfect recorded detector counts
- `mask`: $(512 \\times 512)$ binary detector mask ($1 = \\text{valid}, 0 = \\text{beamstop/gap}$)
- `target_dpr`: $(64 \\times 64)$ DPR-compatible scaled target object
- `object_amplitude`: $(64 \\times 64)$ physical ground-truth object transmission
"""))

cells.append(nbf.v4.new_code_cell("""# Find available HDF5 shard
h5_files = list(DATA_DIR.rglob('*.h5'))
assert len(h5_files) > 0, 'No HDF5 shards found!'
shard_path = h5_files[0]
print(f'Reading shard: {shard_path.name}')

with h5py.File(shard_path, 'r') as h5f:
    print('Datasets inside shard:')
    for k in h5f.keys():
        print(f'  - {k}: shape={h5f[k].shape}, dtype={h5f[k].dtype}')
    
    # Load first sample and squeeze channel dimensions (C, H, W) -> (H, W)
    idx = 0
    input_key = 'intensity_measured' if 'intensity_measured' in h5f else ('input' if 'input' in h5f else 'intensity')
    intensity = np.squeeze(h5f[input_key][idx])
    mask = np.squeeze(h5f['mask'][idx])
    target_key = 'target_dpr' if 'target_dpr' in h5f else ('target' if 'target' in h5f else 'object_amplitude')
    gt_object = np.squeeze(h5f[target_key][idx])

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)

im0 = axes[0].imshow(np.log10(np.maximum(intensity, 1e-1)), cmap='inferno')
axes[0].set_title(f'Measured Intensity (512x512)\\n[log10 counts]', fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'Binary Detector Mask (512x512)\\n[1=valid, 0=blocked]', fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(gt_object, cmap='viridis')
axes[2].set_title(f'Ground Truth Object (64x64)\\n[Normalized Target]', fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

# Phase Retrieval Demonstration
cells.append(nbf.v4.new_markdown_cell("""## 4. Phase Retrieval Baseline: Error-Reduction (ER) Algorithm

In phase retrieval (Task T1), our goal is to reconstruct the specimen $O(\\mathbf{r})$ given only the measured intensity $|F(\\mathbf{q})|^2$ and the support constraint in real space.

The classic **Gerchberg-Saxton Error-Reduction (ER)** algorithm alternates between real and Fourier space with constraint projections:
1. Compute Fourier transform $F_k = \\mathcal{F}\\{O_k\\}$.
2. Replace modulus where measured: $F_k' = \\sqrt{I_{\\text{measured}}} \\cdot e^{i \\arg(F_k)}$ for valid mask pixels.
3. Inverse Fourier transform $O_k' = \\mathcal{F}^{-1}\\{F_k'\\}$.
4. Enforce real-space support and nonnegativity constraints.
"""))

cells.append(nbf.v4.new_code_cell("""def run_error_reduction(intensity, mask, support_radius=20, n_iter=60):
    \"\"\"Vectorized Gerchberg-Saxton Error Reduction baseline.\"\"\"
    rng = np.random.default_rng(42)
    modulus_meas = np.sqrt(np.maximum(intensity, 0.0))
    H, W = intensity.shape
    
    # Real-space coordinate grid for circular support constraint
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((Y - H // 2)**2 + (X - W // 2)**2)
    support = (dist_from_center <= support_radius).astype(np.float32)
    
    # Initialize with random phase
    init_phase = rng.uniform(-np.pi, np.pi, size=(H, W))
    F_current = modulus_meas * np.exp(1j * init_phase)
    
    errors = []
    for i in range(n_iter):
        # 1. Back to real space
        obj = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(F_current)))
        
        # 2. Real-space projection (support & non-negativity)
        obj_constrained = np.real(obj) * support
        obj_constrained = np.maximum(obj_constrained, 0.0)
        
        # 3. Forward to Fourier space
        F_pred = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(obj_constrained)))
        
        # Monitor reciprocal error (R-factor) on valid pixels
        pred_mod = np.abs(F_pred)
        diff = np.abs(pred_mod - modulus_meas) * mask
        r_factor = np.sum(diff) / np.maximum(np.sum(modulus_meas * mask), 1e-12)
        errors.append(r_factor)
        
        # 4. Modulus replacement on valid mask pixels
        phase = np.angle(F_pred)
        new_mod = np.where(mask > 0.5, modulus_meas, pred_mod)
        F_current = new_mod * np.exp(1j * phase)
        
    final_obj = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(F_current))))
    crop = 32
    ch, cw = H // 2, W // 2
    recon_cropped = final_obj[ch - crop : ch + crop, cw - crop : cw + crop]
    return recon_cropped, errors

recon, err_curve = run_error_reduction(intensity, mask, support_radius=22, n_iter=50)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)

axes[0].plot(range(1, len(err_curve) + 1), err_curve, color='#0284C7', lw=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Reciprocal Error ($R$-factor)')
axes[0].set_title('Convergence Curve', fontweight='bold')
axes[0].grid(True, alpha=0.3)

im1 = axes[1].imshow(recon, cmap='viridis')
axes[1].set_title('Reconstructed Object (ER Baseline)', fontweight='bold')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(gt_object, cmap='viridis')
axes[2].set_title('Ground Truth Object', fontweight='bold')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
"""))

# Benchmark Tasks Summary
cells.append(nbf.v4.new_markdown_cell("""## 5. Summary of OpenDiffraction Benchmark Tasks (T1 – T12)

OpenDiffraction standardizes 12 machine learning benchmarks for scientific inverse problems:

| Task ID | Task Name | Problem Type | Primary Metric |
| :--- | :--- | :--- | :--- |
| **T1** | Phase Retrieval | Inverse Problem | `ALIGNED_MSE` |
| **T2** | Diffraction Pattern Denoising | Image Restoration | `PSNR` |
| **T3** | Detector Inpainting | Spatial Inpainting | `MASKED_PSNR` |
| **T4** | Hit Finding & Classification | Classification | `BALANCED_ACCURACY` |
| **T5** | Detector Defect Segmentation | Semantic Segmentation | `IOU` |
| **T6** | Specimen Morphology Classification | Classification | `TOP1_ACCURACY` |
| **T7** | Physical Parameter Estimation | Regression | `RELATIVE_MAE` |
| **T8** | Out-of-Distribution Anomaly Detection | Anomaly Detection | `AUROC` |
| **T9** | Domain Generalization | Domain Adaptation | `TRANSFER_DEGRADATION_RATIO` |
| **T10** | Self-Supervised Representation Learning | Representation Learning | `LINEAR_PROBE_ACCURACY` |
| **T11** | Uncertainty Quantification | Probabilistic Inverse | `EXPECTED_CALIBRATION_ERROR` |
| **T12** | Forward-Inverse Physics Consistency | Constraint Verification | `R_FACTOR` |

---

### Resources & Links
- **GitHub Repository**: [KindlyGentleman/opendiffraction](https://github.com/KindlyGentleman/opendiffraction)
- **PyPI / Source Code**: Complete simulator engine, modular forward models F1–F6, and baseline models.
- **Citation**: Lee et al., *Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers*, npj Comput. Mater. 11, 68 (2025).
"""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}
with open('kaggle_starter_kernel/starter_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Generated kaggle_starter_kernel/starter_notebook.ipynb with kernelspec')
