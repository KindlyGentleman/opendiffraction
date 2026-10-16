"""Tests for Modular Forward Physics Models (F1 - F6).

F1: Coherent Fraunhofer Diffraction
F2: Partial Coherence (GSM & Multimode)
F3: Thin-Object Transmission with Probe Aberrations
F4: Fresnel Near-Field Diffraction
F5: Polychromatic Pink-Beam Diffraction
F6: 3D Particle Scattering via Euler Rotated Born Projections
"""

import numpy as np
import pytest

from opendiffraction.physics.models.fraunhofer import simulate_fraunhofer_coherent
from opendiffraction.physics.models.coherence import (
    simulate_partial_coherence_gsm,
    simulate_multimode_coherence,
)
from opendiffraction.physics.models.transmission import (
    generate_probe,
    simulate_thin_object_transmission,
)
from opendiffraction.physics.models.fresnel import simulate_fresnel_diffraction
from opendiffraction.physics.models.polychromatic import simulate_polychromatic_diffraction
from opendiffraction.physics.models.particle_3d import simulate_3d_scattering
from opendiffraction.objects.object_3d import generate_3d_volume


def test_model_f1_fraunhofer_coherent():
    """Test F1 Fraunhofer coherent diffraction forward model."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[24:40, 24:40] = 1.0  # Centered square

    field, intensity = simulate_fraunhofer_coherent(obj, detector_size=(256, 256))
    assert field.shape == (256, 256)
    assert intensity.shape == (256, 256)
    assert np.all(intensity >= 0.0)
    # Peak at center (zero frequency)
    cy, cx = 128, 128
    assert np.argmax(intensity) == cy * 256 + cx


def test_model_f2_partial_coherence():
    """Test F2 partial coherence: GSM kernel and multimode decomposition."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[28:36, 28:36] = 1.0

    _, coherent_int = simulate_fraunhofer_coherent(obj, detector_size=(128, 128))
    part_int = simulate_partial_coherence_gsm(coherent_int, l_coh=15.0)

    assert part_int.shape == coherent_int.shape
    assert np.all(part_int >= 0.0)
    # Partial coherence blurs speckles, reducing peak-to-valley variance
    assert np.std(part_int) < np.std(coherent_int)

    # Multimode test
    multi_int, weights = simulate_multimode_coherence(obj, num_modes=3, detector_size=(128, 128))
    assert multi_int.shape == (128, 128)
    assert len(weights) == 3
    assert np.all(multi_int >= 0.0)


def test_model_f3_probe_transmission():
    """Test F3 thin object transmission with probe aberrations and jitter."""
    specimen = np.ones((64, 64), dtype=np.float32)
    probe = generate_probe(
        size=64,
        fwhm=25.0,
        defocus_rad=0.5,
        astigmatism_rad=0.2,
        jitter_offset=(1.5, -0.5),
    )
    assert probe.shape == (64, 64)
    assert np.iscomplexobj(probe)

    exit_wave, field, intensity = simulate_thin_object_transmission(
        specimen, probe_field=probe, detector_size=(128, 128)
    )
    assert exit_wave.shape == (64, 64)
    assert field.shape == (128, 128)
    assert intensity.shape == (128, 128)
    assert np.all(intensity >= 0.0)


def test_model_f4_fresnel_diffraction():
    """Test F4 Fresnel nearfield propagation model."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[24:40, 24:40] = 1.0

    field, intensity = simulate_fresnel_diffraction(
        obj,
        wavelength_m=1e-10,
        propagation_distance_m=0.1,
        pixel_pitch_m=5e-8,
        detector_size=(128, 128),
    )
    assert field.shape == (128, 128)
    assert intensity.shape == (128, 128)
    assert np.all(intensity >= 0.0)


def test_model_f5_polychromatic():
    """Test F5 polychromatic pink-beam diffraction."""
    obj = np.zeros((64, 64), dtype=np.float32)
    obj[20:44, 20:44] = 1.0

    int_poly, components = simulate_polychromatic_diffraction(
        obj,
        central_wavelength_m=1.24e-10,
        spectral_bandwidth_ratio=0.05,
        num_wavelength_bins=5,
        detector_size=(128, 128),
    )
    assert int_poly.shape == (128, 128)
    assert len(components) == 5
    assert np.all(int_poly >= 0.0)


def test_model_f6_particle_3d():
    """Test F6 3D scattering with Euler rotation and Born projection."""
    vol = generate_3d_volume("ellipsoid", size=32)
    assert vol.shape == (32, 32, 32)

    proj, field, intensity = simulate_3d_scattering(
        vol,
        euler_angles=(np.pi / 4, np.pi / 6, np.pi / 12),
        detector_size=(128, 128),
    )
    assert proj.shape == (32, 32)
    assert field.shape == (128, 128)
    assert intensity.shape == (128, 128)
    assert np.all(intensity >= 0.0)
