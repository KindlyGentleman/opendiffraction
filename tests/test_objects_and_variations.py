"""Tests for procedural object generation, determinism, and duplicate detection."""

import numpy as np
import pytest

from opendiffraction.objects.procedural import (
    generate_procedural_object,
    make_basic_geometric,
    make_multiple_scatterers,
    make_particle_blob,
    make_porous_heterogeneous,
    make_structured_pattern,
)
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction


def test_02_deterministic_output_from_fixed_seed():
    """Test 2: Verify deterministic output from fixed seeds."""
    seed = 8888

    # Run 1
    rng1 = np.random.default_rng(seed)
    obj1, supp1 = generate_procedural_object("particle_blob", "textured", size=64, rng=rng1)
    cfg1 = SimulationConfig(photon_budget=3e6, coherence_length=200.0)
    sim1 = simulate_diffraction(obj1, cfg1, rng=rng1)

    # Run 2 with identical seed
    rng2 = np.random.default_rng(seed)
    obj2, supp2 = generate_procedural_object("particle_blob", "textured", size=64, rng=rng2)
    cfg2 = SimulationConfig(photon_budget=3e6, coherence_length=200.0)
    sim2 = simulate_diffraction(obj2, cfg2, rng=rng2)

    # Must be bitwise identical
    np.testing.assert_array_equal(obj1, obj2)
    np.testing.assert_array_equal(supp1, supp2)
    np.testing.assert_array_equal(sim1.intensity_ideal, sim2.intensity_ideal)
    np.testing.assert_array_equal(sim1.intensity_measured, sim2.intensity_measured)
    np.testing.assert_array_equal(sim1.valid_mask, sim2.valid_mask)


def test_13_detect_exact_and_near_duplicate_samples():
    """Test 13: Verify that distinct seeds produce distinct, non-duplicate objects."""
    objects = []
    for s in range(30):
        rng = np.random.default_rng(s)
        fam = ["basic_geometric", "particle_blob", "multiple_scatterer", "porous_heterogeneous", "structured_pattern"][s % 5]
        obj, _ = generate_procedural_object(fam, "textured", size=64, rng=rng)
        objects.append(obj)

    # Pairwise comparison: cosine distance or normalized RMSE
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            diff = np.linalg.norm(objects[i] - objects[j]) / (np.linalg.norm(objects[i]) + 1e-6)
            # Ensure no identical or near-duplicate objects
            assert diff > 0.05, f"Objects {i} and {j} are near-duplicates: relative diff {diff:.4f}"
