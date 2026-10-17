"""Regression tests for OD-002: Detector mask leakage elimination."""

from __future__ import annotations

import numpy as np
import pytest

from opendiffraction.physics.forward import SimulationConfig, simulate_forward_model
from opendiffraction.generator.pipeline import simulate_single_task


def test_observed_pixels_strictly_zero_outside_mask() -> None:
    """Every generated observation must contain exactly 0.0 at invalid detector pixels."""
    cfg = SimulationConfig(
        include_beam_stop=True,
        include_module_gaps=True,
        random_pixel_fraction=0.01,
        photon_budget=5e6,
    )
    obj = np.ones((64, 64), dtype=np.float32)

    res = simulate_forward_model(
        object_density=obj,
        config=cfg,
        model_id="F1",
        rng=np.random.default_rng(999),
    )

    invalid_pixels = ~res.valid_mask
    assert np.any(invalid_pixels), "Test mask should contain invalid pixels."

    # Inaccessible pixels in observed measurement MUST be strictly zero
    masked_vals = res.intensity_observed[invalid_pixels]
    assert np.all(masked_vals == 0.0), f"Found {np.count_nonzero(masked_vals)} non-zero leaked pixels in intensity_observed."

    # intensity_raw MUST preserve the underlying detector readout for evaluation
    raw_vals = res.intensity_raw[invalid_pixels]
    assert np.any(raw_vals != 0.0), "intensity_raw should preserve simulated readout behind mask."


def test_holder_arm_strictly_profile_driven() -> None:
    """Holder arm must not be injected spontaneously on acquisition shot 2 unless requested."""
    # V01 is a clean standard profile without holder arm
    for acq_idx in [0, 1, 2, 3]:
        task = {
            "global_sample_idx": acq_idx,
            "sample_id": f"OD_{acq_idx:06d}_V01_a{acq_idx}",
            "base_object_id": "obj_V01_00000",
            "scene_id": "scene_V01_00000",
            "acquisition_id": f"acq_{acq_idx:02d}",
            "group_id": "obj_V01_00000",
            "split": "train",
            "profile_id": "V01",
            "object_index": 0,
            "acq_index": acq_idx,
            "is_ood": False,
            "object_seed": 100,
            "acquisition_seed": 100 + acq_idx * 101,
            "group_name": "Standard",
            "total_objects": 48,
        }
        sample_dict, _ = simulate_single_task(task, profiles_mode="48")
        mask = sample_dict["valid_mask"]
        # V01 has only central beamstop, no holder arm
        # An arm creates a line extending from center to border
        border_pixels_masked = (
            np.any(~mask[0, :])
            or np.any(~mask[-1, :])
            or np.any(~mask[:, 0])
            or np.any(~mask[:, -1])
        )
        assert not border_pixels_masked, f"Spurious holder arm detected on acq_idx={acq_idx} for V01."
