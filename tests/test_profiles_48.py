"""Tests for 48 Variation Profiles (V01 - V48).

Verifies metadata definitions, forward model bindings, parameter sampling,
and simulation output validity for all 48 experimental profiles across 6 groups.
"""

import numpy as np
import pytest

from opendiffraction.variations.profiles_48 import PROFILES_48, sample_profile_48
from opendiffraction.physics.forward import simulate_diffraction


def test_profiles_48_completeness():
    """Verify all 48 profiles are uniquely defined across 6 scientific groups."""
    assert len(PROFILES_48) == 48

    groups = {p.group_name for p in PROFILES_48.values()}
    assert len(groups) == 6
    assert "Real-World Specimens" in groups
    assert "Illumination and Coherence" in groups
    assert "Detector Realism" in groups
    assert "Background and Scattering" in groups
    assert "Acquisition and Hit Finding" in groups
    assert "Advanced Physics and OOD" in groups

    for i in range(1, 49):
        pid = f"V{i:02d}"
        assert pid in PROFILES_48
        spec = PROFILES_48[pid]
        assert spec.profile_id == pid
        assert len(spec.name) > 0
        assert len(spec.description) > 0
        assert len(spec.forward_model) > 0


@pytest.mark.parametrize("pid", [f"V{i:02d}" for i in range(1, 49)])
def test_sample_profile_48_execution(pid):
    """Test generating a physical diffraction sample from each of the 48 profiles."""
    rng = np.random.default_rng(100 + int(pid[1:]))
    cfg, amplitude, phase, meta = sample_profile_48(pid, index=0, total=1, rng=rng)

    assert amplitude.shape == (64, 64)
    assert np.all(amplitude >= 0.0)
    if phase is not None:
        assert phase.shape == (64, 64)

    assert meta["profile_id"] == pid
    assert "group_name" in meta
    assert "forward_model" in meta

    # Run forward simulation
    sim = simulate_diffraction(amplitude, cfg, object_phase=phase, rng=rng)
    assert sim.intensity_measured.shape == (512, 512)
    assert sim.intensity_measured.dtype == np.float32
    assert sim.valid_mask.shape == (512, 512)
    assert np.all(sim.intensity_expected >= 0.0)
    assert np.all(sim.intensity_ideal >= 0.0)
