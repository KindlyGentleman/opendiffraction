"""Tests for strict manifest enforcement, split validation, and T4A/T4B hit definitions."""

from __future__ import annotations

import csv
import os
import tempfile
import h5py
import numpy as np
import pandas as pd
import pytest

from opendiffraction.io.loader import OpenDiffractionDataset
from opendiffraction.io.torch_dataset import OpenDiffractionTorchDataset
from opendiffraction.tasks.manifests import TASK_DEFINITIONS, build_task_manifest
from benchmarks.eval_phase_retrieval import run_phase_retrieval_benchmark
from benchmarks.eval_denoising import run_denoising_benchmark
from benchmarks.eval_inpainting import run_inpainting_benchmark
from benchmarks.eval_hit_classification import run_hit_classification_benchmark


def create_minimal_dataset(tmp_dir: str) -> str:
    """Helper to generate a minimal valid dataset fixture for unit testing."""
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    shard_path = os.path.join(data_dir, "train_shard_000.h5")
    with h5py.File(shard_path, "w") as f:
        f.create_dataset("input", data=np.ones((2, 1, 512, 512), dtype=np.float32))
        f.create_dataset("target", data=np.ones((2, 1, 64, 64), dtype=np.float32))
        f.create_dataset("mask", data=np.ones((2, 1, 512, 512), dtype=bool))
        f.create_dataset("intensity_expected", data=np.ones((2, 1, 512, 512), dtype=np.float32))

    meta_csv = os.path.join(tmp_dir, "metadata.csv")
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "shard_name", "row_index", "split", "hit_status", "profile_id"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_001",
                "shard_name": "train_shard_000.h5",
                "row_index": "0",
                "split": "train",
                "hit_status": "single_hit",
                "profile_id": "V01",
            }
        )
        writer.writerow(
            {
                "sample_id": "sample_002",
                "shard_name": "train_shard_000.h5",
                "row_index": "1",
                "split": "train",
                "hit_status": "non_hit",
                "profile_id": "V42",
            }
        )
    return tmp_dir


def test_t4a_and_t4b_task_definitions() -> None:
    """T4A and T4B must be distinct registered tasks with correct modalities."""
    assert "T4A" in TASK_DEFINITIONS
    assert "T4B" in TASK_DEFINITIONS
    assert TASK_DEFINITIONS["T4A"]["primary_metric"] == "balanced_accuracy"
    assert TASK_DEFINITIONS["T4B"]["primary_metric"] == "macro_f1"

    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "split": ["valid", "valid"],
            "profile_id": ["V37", "V42"],
            "hit_status": ["single_hit", "non_hit"],
        }
    )
    mf_4a = build_task_manifest(df, "T4A")
    assert "is_hit" in mf_4a.columns
    assert mf_4a["is_hit"].tolist() == [True, False]

    mf_4b = build_task_manifest(df, "T4B")
    assert "hit_status" in mf_4b.columns
    assert len(mf_4b) == 2


def test_missing_manifest_raises_filenotfound() -> None:
    """Evaluators must fail explicitly when manifest path is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_minimal_dataset(tmp_dir)
        bogus_manifest = os.path.join(tmp_dir, "non_existent.csv")

        with pytest.raises(FileNotFoundError):
            run_phase_retrieval_benchmark(tmp_dir, bogus_manifest)

        with pytest.raises(FileNotFoundError):
            run_denoising_benchmark(tmp_dir, bogus_manifest)

        with pytest.raises(FileNotFoundError):
            run_inpainting_benchmark(tmp_dir, bogus_manifest)

        with pytest.raises(FileNotFoundError):
            run_hit_classification_benchmark(tmp_dir, bogus_manifest)


def test_duplicate_sample_ids_raise_value_error() -> None:
    """Duplicate sample_ids in manifests must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_minimal_dataset(tmp_dir)
        dup_manifest = os.path.join(tmp_dir, "dup_manifest.csv")
        pd.DataFrame(
            {
                "sample_id": ["sample_001", "sample_001"],
                "split": ["train", "train"],
            }
        ).to_csv(dup_manifest, index=False)

        with pytest.raises(ValueError, match="duplicate sample"):
            run_denoising_benchmark(tmp_dir, dup_manifest)


def test_missing_sample_id_in_index_raises_key_error() -> None:
    """Manifest referencing non-existent sample_id must raise KeyError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_minimal_dataset(tmp_dir)
        missing_id_manifest = os.path.join(tmp_dir, "missing_id_manifest.csv")
        pd.DataFrame(
            {
                "sample_id": ["sample_999"],
                "split": ["train"],
            }
        ).to_csv(missing_id_manifest, index=False)

        with pytest.raises(KeyError, match="sample_999"):
            run_denoising_benchmark(tmp_dir, missing_id_manifest)


def test_split_mismatch_raises_value_error() -> None:
    """Dataset split filter mismatch with manifest split must fail explicitly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_minimal_dataset(tmp_dir)
        val_manifest = os.path.join(tmp_dir, "val_manifest.csv")
        pd.DataFrame(
            {
                "sample_id": ["sample_001"],
                "split": ["valid"],
            }
        ).to_csv(val_manifest, index=False)

        with pytest.raises(ValueError, match="does not match requested dataset split"):
            OpenDiffractionTorchDataset(tmp_dir, split="train", manifest_path=val_manifest)

        with pytest.raises(ValueError, match="does not match requested dataset split"):
            OpenDiffractionDataset(tmp_dir, split="train", manifest_path=val_manifest)
