"""Regression tests for OD-001: Forward-physics dispatcher verification."""

from __future__ import annotations

import numpy as np
import pytest

from opendiffraction.physics.forward import (
    SimulationConfig,
    simulate_diffraction,
    simulate_forward_model,
)
from opendiffraction.generator.pipeline import simulate_single_task


def _make_test_object() -> np.ndarray:
    """Create asymmetric test phantom."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[20:44, 25:35] = 1.0
    obj[30:38, 20:48] = 0.8
    return obj


def test_forward_model_dispatch_f1_to_f6() -> None:
    """Verify simulate_forward_model executes all six models without error."""
    obj = _make_test_object()
    cfg = SimulationConfig(photon_budget=1e6)
    rng = np.random.default_rng(42)

    for mid in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        meta = {}
        if mid == "F3":
            meta["defocus_waves"] = 1.0
        elif mid == "F4":
            meta["fresnel_propagation_distance_m"] = 0.05
        elif mid == "F5":
            meta["spectral_bandwidth_fraction"] = 0.05
        elif mid == "F6":
            meta["euler_angles_rad"] = (0.5, 0.3, 0.2)

        res = simulate_forward_model(
            object_density=obj,
            config=cfg,
            model_id=mid,
            rng=np.random.default_rng(42),
            metadata=meta,
        )
        assert res.forward_model == mid
        assert res.intensity_observed.shape == (512, 512)
        assert res.valid_mask.shape == (512, 512)
        assert np.all(res.intensity_observed[~res.valid_mask] == 0.0)


def test_f3_aberrations_alter_diffraction() -> None:
    """F3 aberrated probe must produce distinct diffraction pattern from unaberrated F1."""
    obj = _make_test_object()
    cfg = SimulationConfig(photon_budget=1e6, readout_std=0.0)

    res_f1 = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F1",
        rng=np.random.default_rng(123),
    )

    res_f3 = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F3",
        rng=np.random.default_rng(123),
        metadata={"defocus_waves": 1.5, "astigmatism_waves": 1.0},
    )

    # Patterns must differ significantly
    rel_diff = np.mean(np.abs(res_f3.intensity_ideal - res_f1.intensity_ideal)) / (
        np.mean(res_f1.intensity_ideal) + 1e-8
    )
    assert rel_diff > 0.1, f"Aberrated F3 was indistinguishable from F1 (rel_diff={rel_diff:.4f})"


def test_f4_nearfield_differs_from_farfield() -> None:
    """F4 Fresnel propagation must produce distinct near-field pattern from F1 Fraunhofer."""
    obj = _make_test_object()
    cfg = SimulationConfig(photon_budget=1e6, readout_std=0.0)

    res_f1 = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F1",
        rng=np.random.default_rng(123),
    )

    res_f4 = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F4",
        rng=np.random.default_rng(123),
        metadata={"fresnel_propagation_distance_m": 0.05},
    )

    rel_diff = np.mean(np.abs(res_f4.intensity_ideal - res_f1.intensity_ideal)) / (
        np.mean(res_f1.intensity_ideal) + 1e-8
    )
    assert rel_diff > 0.2, f"Fresnel F4 was indistinguishable from F1 (rel_diff={rel_diff:.4f})"


def test_f5_polychromatic_broadening() -> None:
    """F5 with broad bandwidth must differ from narrow-band limit."""
    obj = _make_test_object()
    cfg = SimulationConfig(photon_budget=1e6, readout_std=0.0)

    res_mono = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F5",
        rng=np.random.default_rng(123),
        metadata={"spectral_bandwidth_fraction": 0.0},
    )

    res_poly = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F5",
        rng=np.random.default_rng(123),
        metadata={"spectral_bandwidth_fraction": 0.10},
    )

    diff = np.mean(np.abs(res_poly.intensity_ideal - res_mono.intensity_ideal))
    assert diff > 0.0, "Polychromatic broadening had zero effect on ideal intensity."


def test_f6_3d_orientation_dependence() -> None:
    """F6 must produce orientation-dependent diffraction for anisotropic volume."""
    from opendiffraction.objects.object_3d import make_3d_nanorod

    vol = make_3d_nanorod(size=64, length=28.0, radius=5.0)
    cfg = SimulationConfig(photon_budget=1e6, readout_std=0.0)

    res_rot0 = simulate_forward_model(
        object_density=np.zeros((64, 64), dtype=np.float32),
        config=cfg,
        model_id="F6",
        rng=np.random.default_rng(123),
        metadata={"volume_3d": vol, "euler_angles_rad": (0.0, 0.0, 0.0)},
    )

    res_rot90 = simulate_forward_model(
        object_density=np.zeros((64, 64), dtype=np.float32),
        config=cfg,
        model_id="F6",
        rng=np.random.default_rng(123),
        metadata={"volume_3d": vol, "euler_angles_rad": (0.0, np.pi / 2, 0.0)},
    )

    # Projections and diffraction intensities must differ between 0 and 90 degree tilts
    max_proj_diff = float(np.max(np.abs(res_rot0.object_amplitude - res_rot90.object_amplitude)))
    proj_diff = float(np.mean(np.abs(res_rot0.object_amplitude - res_rot90.object_amplitude)))
    inten_diff = float(np.mean(np.abs(res_rot0.intensity_ideal - res_rot90.intensity_ideal)))
    assert max_proj_diff > 0.5, f"Rotated projections were identical (max_diff={max_proj_diff:.4f})."
    assert proj_diff > 0.02, f"Mean projection difference too small (proj_diff={proj_diff:.4f})."
    assert inten_diff > 0.01, f"Diffraction intensities for orthogonal orientations were identical (diff={inten_diff:.4f})."


def test_pipeline_dispatches_declared_profile_model() -> None:
    """Production pipeline simulate_single_task must execute the declared forward model."""
    task_v45 = {
        "global_sample_idx": 0,
        "sample_id": "OD_000000_V45_a0",
        "base_object_id": "obj_V45_00000",
        "scene_id": "scene_V45_00000",
        "acquisition_id": "acq_00",
        "group_id": "obj_V45_00000",
        "split": "test_ood",
        "profile_id": "V45",
        "object_index": 0,
        "acq_index": 0,
        "is_ood": True,
        "object_seed": 1001,
        "acquisition_seed": 1002,
        "group_name": "Advanced Physics",
        "total_objects": 48,
    }

    sample_dict, manifest_rec = simulate_single_task(task_v45, profiles_mode="48")
    assert manifest_rec["forward_model"] == "F4"
    assert sample_dict["intensity_observed"].shape == (512, 512)
    assert np.all(sample_dict["intensity_observed"][~sample_dict["valid_mask"]] == 0.0)
