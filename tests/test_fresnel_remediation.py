"""Regression test for OD-003: Fresnel propagation centering and z=0 identity."""

from __future__ import annotations

import numpy as np
import pytest

from opendiffraction.physics.models.fresnel import simulate_fresnel_diffraction


def test_fresnel_zero_distance_identity() -> None:
    """Propagating over z = 0 must return identical field with identical coordinate center."""
    ny, nx = 512, 512
    inp = np.zeros((ny, nx), dtype=np.complex128)
    # Off-center point scatterer to detect coordinate shifts
    py, px = 200, 310
    inp[py, px] = 1.0 + 0.5j

    field_out, intensity_out = simulate_fresnel_diffraction(
        inp,
        wavelength_m=1e-10,
        propagation_distance_m=0.0,
        pixel_pitch_m=75e-6,
        detector_size=(ny, nx),
    )

    peak_y, peak_x = np.unravel_index(np.argmax(intensity_out), intensity_out.shape)
    assert (peak_y, peak_x) == (py, px), f"Expected peak at ({py}, {px}), got ({peak_y}, {peak_x})"
    assert np.allclose(field_out, inp, atol=1e-6), "Field at z=0 did not match input field."


def test_fresnel_propagated_dispersion() -> None:
    """Propagating over z > 0 must broaden point source while conserving integrated power."""
    ny, nx = 512, 512
    inp = np.zeros((ny, nx), dtype=np.complex128)
    cy, cx = ny // 2, nx // 2
    # Smooth Gaussian phantom
    y, x = np.ogrid[:ny, :nx]
    inp = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * 8.0**2)).astype(np.complex128)
    initial_energy = float(np.sum(np.abs(inp) ** 2))

    field_z, intensity_z = simulate_fresnel_diffraction(
        inp,
        wavelength_m=500e-9,  # 500 nm visible light
        propagation_distance_m=0.01,  # 1 cm propagation
        pixel_pitch_m=5e-6,  # 5 um pitch
        detector_size=(ny, nx),
    )

    final_energy = float(np.sum(intensity_z))
    # Energy should be conserved within numerical integration and band-limiting tolerance
    energy_ratio = final_energy / initial_energy
    assert 0.95 <= energy_ratio <= 1.05, f"Energy conservation violated: ratio = {energy_ratio:.4f}"
    # Peak intensity must decrease due to diffractive spreading in the near field
    assert np.max(intensity_z) < np.max(np.abs(inp) ** 2)
