"""Tests for OpenDiffraction command-line interface."""

import os
import tempfile
import pytest

from opendiffraction.cli import main


def test_cli_generate_validate_visualize_package():
    """Test full CLI workflow: generate -> validate -> visualize -> package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "dataset")
        reports_dir = os.path.join(tmpdir, "reports")
        figs_dir = os.path.join(tmpdir, "figures")
        pkg_dir = os.path.join(tmpdir, "kaggle_release")

        # 1. Generate quick preset (48 samples)
        rc_gen = main(["generate", "--preset", "quick", "--output", out_dir, "--seed", "42"])
        assert rc_gen == 0
        assert os.path.exists(os.path.join(out_dir, "metadata.csv"))
        assert os.path.exists(os.path.join(out_dir, "data"))

        # 2. Validate
        rc_val = main(["validate", "--dataset", out_dir, "--reports-dir", reports_dir])
        assert rc_val == 0
        assert os.path.exists(os.path.join(reports_dir, "VALIDATION_REPORT.md"))
        assert os.path.exists(os.path.join(reports_dir, "QUALITY_AUDIT.md"))
        assert os.path.exists(os.path.join(reports_dir, "KNOWN_LIMITATIONS.md"))

        # 3. Visualize
        rc_vis = main(["visualize", "--dataset", out_dir, "--output", figs_dir])
        assert rc_vis == 0
        assert os.path.exists(os.path.join(figs_dir, "figure_1_workflow.png"))
        assert os.path.exists(os.path.join(figs_dir, "kaggle_cover.png"))

        # 4. Package for Kaggle
        rc_pkg = main(["package", "--dataset", out_dir, "--output", pkg_dir, "--username", "testuser"])
        assert rc_pkg == 0
        assert os.path.exists(os.path.join(pkg_dir, "dataset-metadata.json"))
        assert os.path.exists(os.path.join(pkg_dir, "metadata.csv"))
        assert os.path.exists(os.path.join(pkg_dir, "checksums.sha256"))
        assert os.path.exists(os.path.join(pkg_dir, "scripts", "load_dataset.py"))
