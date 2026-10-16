"""Tests for Gaussian Schell-model partial coherence and autocorrelation centering."""

import numpy as np
import pytest

from opendiffraction.objects.procedural import generate_procedural_object
from opendiffraction.physics.coherence import (
    apply_partial_coherence,
    create_coherence_kernel,
)
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction


def test_08_reducing_coherence_decays_speckle_contrast():
    """Test 8: Check that reducing coherence has the expected effect on diffraction contrast."""
    rng = np.random.default_rng(303)
    obj, _ = generate_procedural_object("structured_pattern", "uniform", size=64, rng=rng)

    cfg_base = SimulationConfig(
        photon_budget=1e7,
        readout_std=0.0,
        include_beam_stop=False,
    )

    # Evaluate speckle contrast (std / mean) across decreasing coherence lengths
    lengths = [float("inf"), 300.0, 150.0, 75.0, 30.0]
    contrasts = []

    for l_coh in lengths:
        cfg = SimulationConfig(
            photon_budget=1e7,
            coherence_length=l_coh,
            readout_std=0.0,
            include_beam_stop=False,
        )
        sim = simulate_diffraction(obj, cfg, rng=np.random.default_rng(303))
        # Compute speckle contrast in an annular region away from DC
        I = sim.intensity_expected
        cy, cx = 256, 256
        y, x = np.ogrid[:512, :512]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        annulus = (r >= 30) & (r <= 120)
        I_annulus = I[annulus]
        contrast = float(np.std(I_annulus) / max(np.mean(I_annulus), 1e-6))
        contrasts.append(contrast)

    # Monotonic decrease or equal
    for i in range(len(contrasts) - 1):
        assert (
            contrasts[i] >= contrasts[i + 1] - 1e-4
        ), f"Coherence contrast failed monotonic decay: {contrasts}"


def test_09_centered_autocorrelation_kernel_implementation():
    """Test 9: Verify the centered autocorrelation coherence-kernel implementation."""
    shape = (512, 512)
    l_coh = 200.0

    # Modern corrected kernel
    kernel_modern = create_coherence_kernel(l_coh, shape=shape, legacy=False)

    # Autocorrelation from ifft2 has zero spatial lag at index [0, 0]
    # Modern kernel MUST have maximum 1.0 at [0, 0]
    assert np.isclose(kernel_modern[0, 0], 1.0, atol=1e-7)
    assert np.all(kernel_modern <= 1.0 + 1e-7)
    # The kernel must decay away from [0, 0] (e.g. at [256, 256] it is minimum)
    assert kernel_modern[256, 256] < kernel_modern[0, 0]

    # Contrast with legacy kernel (which was erroneously centered at grid center [256, 256])
    kernel_legacy = create_coherence_kernel(l_coh, shape=shape, legacy=True)
    assert np.isclose(kernel_legacy[256, 256], 1.0, atol=1e-7)
    assert kernel_legacy[0, 0] < 0.5  # Legacy had ~0.44 at zero lag!
