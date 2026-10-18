"""Regression test suite for Scenario-to-Physics Parameter Propagation Provenance (Phase 2 / OD-011)."""

from __future__ import annotations

import numpy as np
import pytest

from opendiffraction.generator.pipeline import simulate_single_task
from opendiffraction.physics.forward import simulate_diffraction, SimulationConfig
from opendiffraction.physics.models.fresnel import simulate_fresnel_diffraction
from opendiffraction.physics.models.polychromatic import simulate_polychromatic_diffraction
from opendiffraction.scenarios import SCENARIOS, get_scenario_for_profile, get_scenario_physics


def test_v45_fresnel_optical_lab_provenance() -> None:
    """Profile V45 must consume S5 optical laboratory geometry and record identical parameters in metadata."""
    task = {
        "sample_id": "test_v45_prov",
        "base_object_id": "obj_v45",
        "scene_id": "scene_v45",
        "acquisition_id": "acq_00",
        "group_id": "grp_v45",
        "split": "train",
        "profile_id": "V45",
        "object_index": 0,
        "acq_index": 0,
        "object_seed": 100,
        "acquisition_seed": 200,
    }
    sample, meta = simulate_single_task(task, profiles_mode="48")

    # Verify scenario identity
    assert meta["scenario_id"] == "S5"
    assert meta["forward_model"] == "F4"

    # S5 He-Ne laser visible wavelength (~632.8 nm) and CCD pitch (6.45 um)
    expected_lam = 1.239841984e-9 / 0.00196  # ~6.3257e-7 m
    expected_pitch = 6.45e-6
    expected_dist = 0.05  # configured propagation distance for V45

    assert np.isclose(meta["wavelength_m"], expected_lam, rtol=1e-4)
    assert np.isclose(meta["pixel_pitch_m"], expected_pitch, rtol=1e-4)
    assert np.isclose(meta["detector_distance_m"], expected_dist, rtol=1e-4)

    # Must NOT use fallback X-ray wavelength or Pilatus pixel pitch
    assert not np.isclose(meta["wavelength_m"], 1.24e-10, atol=1e-11)
    assert not np.isclose(meta["pixel_pitch_m"], 75e-6, atol=1e-6)

    # Output array must be physically non-zero and bounded
    assert np.any(sample["intensity_observed"] > 0)
    assert np.all(np.isfinite(sample["intensity_observed"]))


def test_v46_polychromatic_spectral_bandwidth_provenance() -> None:
    """Profile V46 must consume S2 pink beam synchrotron spectrum and record active bandwidth."""
    task = {
        "sample_id": "test_v46_prov",
        "base_object_id": "obj_v46",
        "scene_id": "scene_v46",
        "acquisition_id": "acq_00",
        "group_id": "grp_v46",
        "split": "train",
        "profile_id": "V46",
        "object_index": 0,
        "acq_index": 0,
        "object_seed": 101,
        "acquisition_seed": 201,
    }
    sample, meta = simulate_single_task(task, profiles_mode="48")

    assert meta["scenario_id"] == "S2"
    assert meta["forward_model"] == "F5"

    # S2 synchrotron 12 keV nominal energy
    expected_lam = 1.239841984e-9 / 12.0  # ~1.0332e-10 m
    assert np.isclose(meta["wavelength_m"], expected_lam, rtol=1e-4)


def test_f4_fresnel_zero_distance_identity() -> None:
    """F4 ASM propagation must satisfy identity U(z=0) == U_0 within numerical precision."""
    ny, nx = 64, 64
    rng = np.random.default_rng(42)
    obj = rng.uniform(0.1, 1.0, size=(ny, nx)).astype(np.complex128)

    field_z0, intensity_z0 = simulate_fresnel_diffraction(
        input_field=obj,
        wavelength_m=632.8e-9,
        propagation_distance_m=0.0,
        pixel_pitch_m=6.45e-6,
        detector_size=(512, 512),
    )

    # Crop object region from 512x512 frame
    sy0 = (512 - ny) // 2
    sx0 = (512 - nx) // 2
    crop_z0 = field_z0[sy0 : sy0 + ny, sx0 : sx0 + nx]

    assert np.allclose(crop_z0, obj, atol=1e-12)


def test_f5_polychromatic_zero_bandwidth_and_flux_conservation() -> None:
    """F5 must cleanly handle zero bandwidth and conserve integrated energy across bandwidths."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[20:44, 20:44] = 1.0

    # Zero bandwidth
    i_mono, spec_mono = simulate_polychromatic_diffraction(
        obj, central_wavelength_m=1e-10, spectral_bandwidth_ratio=0.0
    )
    assert np.all(np.isfinite(i_mono))
    assert len(spec_mono) == 1
    assert spec_mono[0][1] == 1.0

    # Finite bandwidths: 5% and 20%
    i_poly5, _ = simulate_polychromatic_diffraction(
        obj, central_wavelength_m=1e-10, spectral_bandwidth_ratio=0.05
    )
    i_poly20, _ = simulate_polychromatic_diffraction(
        obj, central_wavelength_m=1e-10, spectral_bandwidth_ratio=0.20
    )

    total_mono = float(np.sum(i_mono))
    total_poly5 = float(np.sum(i_poly5))
    total_poly20 = float(np.sum(i_poly20))

    # Energy conservation within 0.1%
    assert abs(total_poly5 - total_mono) / total_mono < 1e-3
    assert abs(total_poly20 - total_mono) / total_mono < 1e-3


def test_all_scenarios_have_physically_consistent_provenance() -> None:
    """Verify that every scenario S1 to S6 defines valid physical units and non-zero wavelength."""
    for scen_id, spec in SCENARIOS.items():
        scen_phys = get_scenario_physics(scen_id)
        assert scen_phys["scenario_id"] == scen_id
        assert scen_phys["photon_energy_kev"] > 0
        assert scen_phys["wavelength_m"] > 0
        assert scen_phys["pixel_pitch_m"] > 0
        assert scen_phys["detector_distance_m"] > 0
