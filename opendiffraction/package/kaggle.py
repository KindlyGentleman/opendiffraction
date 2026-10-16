"""Packaging utilities for Kaggle dataset releases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_kaggle_release(
    source_dir: str,
    release_dir: str,
    username: str = "YOUR_KAGGLE_USERNAME",
) -> dict[str, Any]:
    """Package a verified dataset into a complete, standalone Kaggle deliverable.

    Parameters
    ----------
    source_dir : str
        Directory containing generated data/ and metadata.csv.
    release_dir : str
        Target release directory to create.
    username : str
        Kaggle username placeholder for dataset slug.
    """
    source_dir = os.path.abspath(source_dir)
    release_dir = os.path.abspath(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    # 1. Copy data directory
    src_data = os.path.join(source_dir, "data")
    dst_data = os.path.join(release_dir, "data")
    os.makedirs(dst_data, exist_ok=True)
    if os.path.exists(src_data):
        for f in os.listdir(src_data):
            if f.endswith(".h5"):
                shutil.copy2(os.path.join(src_data, f), os.path.join(dst_data, f))

    # 2. Copy metadata manifest
    src_meta = os.path.join(source_dir, "metadata.csv")
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, os.path.join(release_dir, "metadata.csv"))

    # 3. Create dataset-metadata.json
    metadata = {
        "title": "OpenDiffraction: Synthetic CDI Patterns",
        "subtitle": "Physics-based diffraction images with ground truth and controlled detector imperfections",
        "id": f"{username}/opendiffraction-synthetic-cdi",
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["physics", "computer-vision", "image-processing", "optics", "synthetic-data"],
        "description": "OpenDiffraction is a physics-based synthetic dataset for coherent diffraction imaging (CDI) and phase retrieval benchmarking.",
    }
    with open(os.path.join(release_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 4. Create configuration.json
    config_summary = {
        "dataset_name": "OpenDiffraction",
        "version": "0.1.0",
        "object_size": [64, 64],
        "detector_size": [512, 512],
        "profiles": [f"V{i:02d}" for i in range(1, 13)],
        "target_scaling": "DPR_false_scale_and_physical",
        "partial_coherence_model": "GSM_zero_lag_corrected",
    }
    with open(os.path.join(release_dir, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    # 5. Populate examples directory
    dst_examples = os.path.join(release_dir, "examples")
    os.makedirs(dst_examples, exist_ok=True)
    fig_dir = os.path.join(source_dir, "figures")
    if os.path.exists(fig_dir):
        for img_name in ["figure_3_geometry_diffraction.png", "figure_4_photon_sweep.png", "figure_5_coherence_sweep.png", "kaggle_cover.png"]:
            src_img = os.path.join(fig_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(dst_examples, img_name))

    # 6. Populate scripts directory with standalone loader
    dst_scripts = os.path.join(release_dir, "scripts")
    os.makedirs(dst_scripts, exist_ok=True)
    loader_script = """import h5py
import numpy as np

def load_sample(h5_path, index=0):
    with h5py.File(h5_path, 'r') as f:
        inp = f['input'][index]
        tgt = f['target'][index]
        msk = f['mask'][index]
    return inp, tgt, msk

if __name__ == '__main__':
    print("OpenDiffraction standalone loader ready.")
"""
    with open(os.path.join(dst_scripts, "load_dataset.py"), "w", encoding="utf-8") as f:
        f.write(loader_script)

    # 7. Generate checksums.sha256
    checksums = []
    for root, _, files in os.walk(dst_data):
        for f in files:
            fp = os.path.join(root, f)
            rel_p = os.path.relpath(fp, release_dir)
            sha = calculate_sha256(fp)
            checksums.append(f"{sha}  {rel_p}")

    with open(os.path.join(release_dir, "checksums.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(checksums) + "\n")

    return {
        "status": "packaged",
        "release_dir": release_dir,
        "shards_packaged": len(checksums),
        "cli_init_command": f"kaggle datasets init -p {release_dir}",
        "cli_create_command": f"kaggle datasets create -p {release_dir}",
    }
