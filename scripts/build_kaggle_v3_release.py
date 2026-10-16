"""OpenDiffraction 3.0 Kaggle Release Staging Builder.

Packages the verified 1,000-observation pilot dataset (output_v3) into
kaggle_release_v3/ with standardized metadata, manifests, checksums, and documentation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from typing import Any


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file in binary mode."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_kaggle_v3_release(
    source_dir: str = "output_v3",
    output_dir: str = "kaggle_release_v3",
    username: str = "hamadmuhammadkamal",
    dataset_slug: str = "opendiffraction-synthetic-cdi",
) -> dict[str, Any]:
    print(f"\n[Release Builder] Building Kaggle release package in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)

    # 1. Copy metadata.csv
    src_meta = os.path.join(source_dir, "metadata.csv")
    dst_meta = os.path.join(output_dir, "metadata.csv")
    shutil.copy2(src_meta, dst_meta)
    print(f"Copied metadata.csv ({os.path.getsize(dst_meta)} bytes)")

    # 2. Copy HDF5 shards
    src_data = os.path.join(source_dir, "data")
    dst_data = os.path.join(output_dir, "data")
    shards = sorted([f for f in os.listdir(src_data) if f.endswith(".h5")])
    for s in shards:
        shutil.copy2(os.path.join(src_data, s), os.path.join(dst_data, s))
        print(f"Copied data/{s} ({os.path.getsize(os.path.join(dst_data, s)):,} bytes)")

    # 3. Copy benchmark CXI
    src_cxi = os.path.join(source_dir, "opendiffraction_benchmark.cxi")
    dst_cxi = os.path.join(output_dir, "opendiffraction_benchmark.cxi")
    if os.path.exists(src_cxi):
        shutil.copy2(src_cxi, dst_cxi)
        print(f"Copied opendiffraction_benchmark.cxi ({os.path.getsize(dst_cxi):,} bytes)")

    # 4. Copy tasks manifests
    src_tasks = os.path.join(source_dir, "tasks")
    dst_tasks = os.path.join(output_dir, "tasks")
    if os.path.exists(src_tasks):
        for tf in sorted(os.listdir(src_tasks)):
            if tf.endswith(".csv"):
                shutil.copy2(os.path.join(src_tasks, tf), os.path.join(dst_tasks, tf))
        print(f"Copied {len(os.listdir(dst_tasks))} task manifests")

    # 5. Copy examples figures
    src_examples = os.path.join("kaggle_release", "examples")
    dst_examples = os.path.join(output_dir, "examples")
    if os.path.exists(src_examples):
        for ef in sorted(os.listdir(src_examples)):
            if ef.endswith(".png"):
                shutil.copy2(os.path.join(src_examples, ef), os.path.join(dst_examples, ef))
        print(f"Copied {len(os.listdir(dst_examples))} example figures")

    # 6. Copy standalone loader script
    src_loader = os.path.join("kaggle_release", "scripts", "load_dataset.py")
    dst_loader = os.path.join(output_dir, "scripts", "load_dataset.py")
    if os.path.exists(src_loader):
        shutil.copy2(src_loader, dst_loader)
        print("Copied scripts/load_dataset.py")

    # 7. Write configuration.json
    config = {
        "dataset_name": "OpenDiffraction",
        "version": "0.3.0",
        "sample_count": 1000,
        "base_objects": 250,
        "acquisitions_per_object": 4,
        "object_size": [64, 64],
        "detector_size": [512, 512],
        "profiles": [f"V{i:02d}" for i in range(1, 49)],
        "forward_models": ["F1", "F2", "F3", "F4", "F5", "F6"],
        "target_scaling": "DPR_false_scale_and_physical",
        "partial_coherence_model": "GSM_zero_lag_corrected",
    }
    with open(os.path.join(output_dir, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 8. Copy or write LICENSE and CITATION.cff
    for lf in ["LICENSE", "LICENSE-DATA.md", "CITATION.cff"]:
        src_f = os.path.join("kaggle_release", lf)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(output_dir, lf))

    # 9. Write dataset-metadata.json
    dataset_metadata = {
        "title": "OpenDiffraction: Synthetic CDI Patterns",
        "subtitle": "Physics-based coherent diffraction images with ground truth and benchmarks",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}],
        "keywords": [
            "physics",
            "computer vision",
            "deep learning",
            "benchmark",
            "optics",
            "x-ray",
            "pytorch",
        ],
        "expectedUpdateFrequency": "annually",
        "description": """# OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Imaging Dataset (Version 3.0 Pilot)

**Creator & Principal Maintainer**: OpenDiffraction computational optics and scientific machine learning contributors (Principal Maintainer: [Muhammad Kamal](https://github.com/KindlyGentleman), GitHub: `@KindlyGentleman`, Kaggle: `@hamadmuhammadkamal`)
**GitHub Repository**: [https://github.com/KindlyGentleman/opendiffraction](https://github.com/KindlyGentleman/opendiffraction)
**Foundational Reference Paper**: Lee et al. (2025), *Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers*, npj Computational Materials 11, 68. DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7)

---

## Dataset Scope: Validated 1,000-Observation Training Pilot

This release provides **1,000 verified synthetic diffraction observations** generated from first-principles wave optics:
- **Base Specimens**: Exactly 250 distinct ground truth objects across 6 morphological classes (Biological, Nanomaterials, Optical, Fractals, Lithography, Dense Crystals).
- **Multi-Acquisition Sampling**: Exactly 4 independent physical acquisitions per base object simulating beamline parameter variations (fluence, photon noise, beamstop geometry, partial coherence, defocus).
- **Physical Profiles**: Full representation across all 48 variation profiles (`V01` through `V48`).
- **Forward Optics Models**: Full representation across all 6 forward models:
  - `F1`: Fraunhofer far-field propagation
  - `F2`: GSM partial spatial coherence
  - `F3`: Thin-specimen probe wavefront transmission
  - `F4`: Fresnel near-field propagation
  - `F5`: Polychromatic pink-beam spectral summation
  - `F6`: 3D particle Euler rotation Born projections
- **Partition Splits**: Strict disjoint object-level partitioning with zero data leakage:
  - `train`: 644 observations (161 base objects, 64.4%)
  - `valid`: 140 observations (35 base objects, 14.0%)
  - `test`: 116 observations (29 base objects, 11.6%)
  - `test_ood`: 100 observations (25 base objects, 10.0%)

---

## PyTorch and Python Integration

Load the dataset directly using PyTorch:

```python
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
from torch.utils.data import DataLoader

dataset = OpenDiffractionTorchDataset(
    root="./kaggle_release_v3",
    split="train",
    task="phase_retrieval",
    return_dict=True,
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

for batch in loader:
    inputs = batch["input"]   # [B, 1, 512, 512] float32 measured intensity
    targets = batch["target"] # [B, 1, 64, 64] float32 ground truth object
    masks = batch["mask"]     # [B, 1, 512, 512] bool detector validity mask
    break
```

---

## Standardized Benchmark Tasks

1. **Task 1: Phase Retrieval**: Reconstruct complex specimen density from Fourier modulus under detector imperfections.
2. **Task 2: Diffraction Pattern Denoising**: Recover Poisson-corrupted high-frequency speckles.
3. **Task 3: Detector Inpainting**: Extrapolate missing intensity in central beamstop shadows and inter-module gaps.
4. **Task 4: Hit Finding & Frame Classification**: Distinguish scattering hit frames from blank background shots.

---

## Scientific Limitations and Integrity Notice

1. **Hit Finding Synthetic Gap**: Synthetic non-hit frames contain zero specimen scatter, enabling 100% classification via total energy thresholding. Real beamline data feature gas background and stray scattering that require learned classifiers.
2. **Inpainting Center Shadow**: Classical nearest-neighbor inpainting exhibits high overall PSNR (~36.8 dB) because >98% of pixels are valid, but masked hole MAE is ~750 photons due to forward scattering $|F(0)|^2$ underestimation.
3. **Phase Retrieval Stagnation**: Classical HIO/ER achieves low Fourier residual (R-factor ~0.14) but stagnates in twin-image pseudo-solutions (SSIM ~0.01 - 0.05), demonstrating the necessity of learned priors.

---

## Citation & Provenance

```bibtex
@misc{muhammad_kamal_2026,
  author    = {Muhammad Kamal},
  title     = {OpenDiffraction: Synthetic CDI Patterns},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi}
}
```
""",
    }
    with open(os.path.join(output_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    # 10. Write DATASET_CARD.md and README.md
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(dataset_metadata["description"])

    with open(os.path.join(output_dir, "DATASET_CARD.md"), "w", encoding="utf-8") as f:
        f.write(dataset_metadata["description"])

    # 11. Compute checksums.sha256 for all staged files
    print("\nComputing SHA-256 checksums...")
    checksum_lines = []
    total_bytes = 0
    for root, _, files in os.walk(output_dir):
        for fname in sorted(files):
            if fname == "checksums.sha256":
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, output_dir).replace("\\", "/")
            f_size = os.path.getsize(fpath)
            total_bytes += f_size
            f_hash = compute_sha256(fpath)
            checksum_lines.append(f"{f_hash}  {rel_path}\n")

    with open(os.path.join(output_dir, "checksums.sha256"), "w", encoding="utf-8") as f:
        f.writelines(sorted(checksum_lines))

    print(f"Generated checksums.sha256 with {len(checksum_lines)} entries")
    print(f"Total package payload: {total_bytes:,} bytes ({total_bytes / (1024**3):.4f} GiB)")

    return {
        "total_files": len(checksum_lines),
        "total_bytes": total_bytes,
        "shards": len(shards),
    }


if __name__ == "__main__":
    build_kaggle_v3_release()
