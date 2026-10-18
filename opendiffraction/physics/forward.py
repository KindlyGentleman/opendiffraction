"""Physical forward simulation pipeline for coherent and partially coherent X-ray diffraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .coherence import apply_partial_coherence
from .masks import composite_detector_mask
from .noise import (
    FWHM_ONE_SIGMA,
    apply_gaussian_readout_noise,
    apply_poisson_noise,
    apply_saturation_and_defects,
    scale_to_photon_budget,
)


@dataclass
class SimulationConfig:
    """Configuration for single diffraction pattern generation."""

    object_size: tuple[int, int] = (64, 64)
    detector_size: tuple[int, int] = (512, 512)
    photon_budget: float = 3e6
    coherence_length: float = 200.0  # pixels
    readout_std: float = FWHM_ONE_SIGMA  # ~0.42466
    false_scale: bool = True  # DPR training convention
    coherence_legacy: bool = False  # DPR v1.0-legacy toggle
    saturation_threshold: float | None = None
    hot_pixels: list[tuple[int, int]] | None = None
    beam_center_offset: tuple[float, float] = (0.0, 0.0)  # (dy, dx)
    # Mask configuration
    include_beam_stop: bool = True
    beam_stop_hw: int = 16
    beam_stop_hh: int = 16
    beam_stop_offset: tuple[int, int] = (0, 0)
    circular_beam_stop: bool = False
    include_holder_arm: bool = False
    arm_angle_rad: float = 0.0
    arm_width: int = 5
    include_module_gaps: bool = False
    gap_width: int = 4
    random_pixel_fraction: float = 0.0
    num_clusters: int = 0
    cluster_radius_range: tuple[int, int] = (2, 6)
    num_stripes_h: int = 0
    num_stripes_v: int = 0
    irregular_coverage: float = 0.0


@dataclass
class SimulationResult:
    """Complete output record of forward diffraction simulation."""

    object_amplitude: np.ndarray  # Unscaled ground truth (H, W)
    object_phase: np.ndarray | None  # Ground truth phase (H, W) or None
    target_dpr: np.ndarray  # Target scaled per DPR convention (H, W)
    target_physical: np.ndarray  # Target scaled per physical amplitude sqrt(s)
    intensity_ideal: np.ndarray  # Coherent diffraction intensity (DetH, DetW)
    intensity_expected: np.ndarray  # Post-coherence photon arrival rates lambda_i
    intensity_raw: np.ndarray  # Unmasked signed detector readouts
    valid_mask: np.ndarray  # Detector validity mask (bool)
    intensity_observed: np.ndarray  # Detector-accessible measurement (raw * mask)
    scale_factor: float  # Flux normalization scale factor s
    photon_budget: float  # Total photons simulated
    forward_model: str = "F1"  # Executed physical forward model ID

    @property
    def intensity_measured(self) -> np.ndarray:
        """Backward-compatibility alias returning detector-accessible observed intensity."""
        return self.intensity_observed

    @property
    def observed_masked(self) -> np.ndarray:
        """Backward-compatibility alias for intensity_observed."""
        return self.intensity_observed


def pad_object_to_detector(
    obj: np.ndarray,
    detector_size: tuple[int, int] = (512, 512),
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Place object at the center of the detector frame with optional beam displacement.

    Parameters
    ----------
    obj : np.ndarray
        Complex or real object of shape (Ny_obj, Nx_obj).
    detector_size : tuple[int, int]
        Detector frame dimensions (DetH, DetW).
    offset : tuple[float, float]
        Beam displacement (dy, dx) in pixels, applied via illumination wavevector tilt
        so that the diffraction pattern on the detector shifts by (dy, dx).

    Returns
    -------
    frame : np.ndarray
        Padded frame matching detector_size with object centered and wavevector tilted.
    """
    ny_det, nx_det = detector_size
    ny_obj, nx_obj = obj.shape[-2:]

    # Place object strictly centered in detector frame
    y0 = (ny_det - ny_obj) // 2
    x0 = (nx_det - nx_obj) // 2

    frame = np.zeros(detector_size, dtype=np.complex128 if np.iscomplexobj(obj) else np.float64)
    frame[y0 : y0 + ny_obj, x0 : x0 + nx_obj] = obj

    # Apply beam displacement via linear phase ramp (wavevector tilt)
    # so the far-field diffraction pattern is translated by (dy, dx) on the detector.
    dy, dx = float(offset[0]), float(offset[1])
    if abs(dy) > 1e-5 or abs(dx) > 1e-5:
        y_grid, x_grid = np.mgrid[:ny_det, :nx_det]
        y_c = y_grid - (ny_det // 2)
        x_c = x_grid - (nx_det // 2)
        phase_ramp = np.exp(1j * 2.0 * np.pi * (dy * y_c / ny_det + dx * x_c / nx_det))
        frame = (frame * phase_ramp).astype(np.complex128)

    return frame


def compute_coherent_diffraction(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centered 2D Fraunhofer diffraction field and intensity.

    Uses centered 2D FFT (DC at (ny//2, nx//2)).
    """
    field = np.fft.fftshift(np.fft.fft2(frame))
    intensity = np.abs(field) ** 2
    return field, intensity


def simulate_forward_model(
    object_density: np.ndarray,
    config: SimulationConfig,
    model_id: str = "F1",
    rng: np.random.Generator | None = None,
    object_phase: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> SimulationResult:
    """Execute physical forward diffraction under specified model (F1 to F6)."""
    if rng is None:
        rng = np.random.default_rng()
    if metadata is None:
        metadata = {}

    model_key = model_id.upper().strip()

    if object_phase is not None:
        obj_field = object_density * np.exp(1j * object_phase)
    else:
        obj_field = object_density.astype(
            np.complex128 if np.iscomplexobj(object_density) else np.float64
        )

    if model_key == "F1":
        frame = pad_object_to_detector(
            obj_field,
            detector_size=config.detector_size,
            offset=config.beam_center_offset,
        )
        _, intensity_ideal = compute_coherent_diffraction(frame)
        if np.isfinite(config.coherence_length):
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=config.coherence_length,
                legacy=config.coherence_legacy,
            )
        else:
            intensity_coh = intensity_ideal.copy()

    elif model_key == "F2":
        coherence_mode = metadata.get("coherence_mode", "gsm")
        if coherence_mode == "multimode":
            from .models.coherence import simulate_multimode_coherence

            total_intensity, _ = simulate_multimode_coherence(
                obj_field,
                num_modes=int(metadata.get("num_modes", 4)),
                mode_decay_ratio=float(metadata.get("mode_decay_ratio", 0.5)),
                detector_size=config.detector_size,
                rng=rng,
            )
            intensity_ideal = total_intensity
            intensity_coh = total_intensity
        else:
            frame = pad_object_to_detector(
                obj_field,
                detector_size=config.detector_size,
                offset=config.beam_center_offset,
            )
            _, intensity_ideal = compute_coherent_diffraction(frame)
            l_coh = float(metadata.get("coherence_length", config.coherence_length))
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=l_coh,
                legacy=config.coherence_legacy,
            )

    elif model_key == "F3":
        from .models.transmission import generate_probe

        defocus_waves = float(metadata.get("defocus_waves", 0.0))
        spherical_aberration_waves = float(metadata.get("spherical_aberration_waves", 0.0))
        astigmatism_waves = float(metadata.get("astigmatism_waves", 0.0))
        astigmatism_angle = float(metadata.get("astigmatism_angle", 0.0))
        coma_waves = float(metadata.get("coma_waves", 0.0))
        jitter_offset = metadata.get("jitter_offset", config.beam_center_offset)
        probe_fwhm = float(metadata.get("probe_fwhm", 30.0))

        probe = generate_probe(
            size=object_density.shape[0],
            fwhm=probe_fwhm,
            defocus_rad=defocus_waves * 2.0 * np.pi,
            spherical_aberration_rad=spherical_aberration_waves * 2.0 * np.pi,
            astigmatism_rad=astigmatism_waves * 2.0 * np.pi,
            astigmatism_angle=astigmatism_angle,
            coma_rad=coma_waves * 2.0 * np.pi,
            jitter_offset=jitter_offset,
        )
        exit_wave = (probe * obj_field).astype(np.complex128)
        frame = pad_object_to_detector(
            exit_wave,
            detector_size=config.detector_size,
            offset=(0.0, 0.0),
        )
        _, intensity_ideal = compute_coherent_diffraction(frame)
        if np.isfinite(config.coherence_length):
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=config.coherence_length,
                legacy=config.coherence_legacy,
            )
        else:
            intensity_coh = intensity_ideal.copy()

    elif model_key == "F4":
        from .models.fresnel import simulate_fresnel_diffraction

        distance_m = float(
            metadata.get(
                "fresnel_propagation_distance_m",
                metadata.get("detector_distance_m", 0.05),
            )
        )
        wavelength_m = float(metadata.get("wavelength_m", 1.24e-10))
        pixel_pitch_m = float(metadata.get("pixel_pitch_m", 75e-6))

        _, intensity_ideal = simulate_fresnel_diffraction(
            input_field=obj_field,
            wavelength_m=wavelength_m,
            propagation_distance_m=distance_m,
            pixel_pitch_m=pixel_pitch_m,
            detector_size=config.detector_size,
        )
        if np.isfinite(config.coherence_length):
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=config.coherence_length,
                legacy=config.coherence_legacy,
            )
        else:
            intensity_coh = intensity_ideal.copy()

    elif model_key == "F5":
        from .models.polychromatic import simulate_polychromatic_diffraction

        bw_ratio = float(metadata.get("spectral_bandwidth_fraction", 0.05))
        central_lam = float(metadata.get("wavelength_m", 1.24e-10))
        n_bins = int(metadata.get("num_wavelength_bins", 7))

        frame = pad_object_to_detector(
            obj_field,
            detector_size=config.detector_size,
            offset=config.beam_center_offset,
        )
        if bw_ratio <= 1e-4:
            _, intensity_ideal = compute_coherent_diffraction(frame)
        else:
            intensity_ideal, _ = simulate_polychromatic_diffraction(
                frame,
                central_wavelength_m=central_lam,
                spectral_bandwidth_ratio=bw_ratio,
                num_wavelength_bins=n_bins,
                detector_size=config.detector_size,
            )
        if np.isfinite(config.coherence_length):
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=config.coherence_length,
                legacy=config.coherence_legacy,
            )
        else:
            intensity_coh = intensity_ideal.copy()

    elif model_key == "F6":
        from .models.particle_3d import simulate_3d_scattering
        from opendiffraction.objects.object_3d import generate_3d_volume

        if "volume_3d" in metadata:
            vol_3d = metadata["volume_3d"]
        else:
            vol_3d = generate_3d_volume(
                shape_type="cluster",
                size=object_density.shape[0],
                rng=rng,
            )

        euler_angles = metadata.get("euler_angles_rad", (0.0, 0.0, 0.0))
        proj_2d, _, intensity_ideal = simulate_3d_scattering(
            volume_3d=vol_3d,
            euler_angles=euler_angles,
            detector_size=config.detector_size,
        )
        object_density = proj_2d
        if np.isfinite(config.coherence_length):
            intensity_coh = apply_partial_coherence(
                intensity_ideal,
                l_coh=config.coherence_length,
                legacy=config.coherence_legacy,
            )
        else:
            intensity_coh = intensity_ideal.copy()

    else:
        raise ValueError(
            f"Unsupported forward model: {model_id}. Supported models are F1 through F6."
        )

    # 3. Incorporate background scattering if configured
    bg_type = metadata.get("background_type", None)
    if bg_type in ("gas_scattering", "support_film_halo", "solvent_halo", "slit_streaks"):
        from .backgrounds import (
            generate_gas_scattering_background,
            generate_parasitic_beamline_scattering,
            generate_solvent_halo_background,
            generate_support_film_background,
        )

        norm_factor = max(1.0, float(config.photon_budget) / 512.0)
        if bg_type == "gas_scattering":
            bg = generate_gas_scattering_background(
                config.detector_size, peak_intensity=0.015 * norm_factor
            )
        elif bg_type == "support_film_halo":
            bg = generate_support_film_background(
                config.detector_size, peak_intensity=0.010 * norm_factor
            )
        elif bg_type == "solvent_halo":
            bg = generate_solvent_halo_background(
                config.detector_size, peak_intensity=0.012 * norm_factor
            )
        elif bg_type == "slit_streaks":
            bg = generate_parasitic_beamline_scattering(
                config.detector_size,
                streak_intensity=0.020 * norm_factor,
                rng=rng,
            )
        else:
            bg = np.zeros(config.detector_size, dtype=np.float32)

        intensity_coh = np.maximum(intensity_coh + bg, 0.0)

    # 5. Photon budget normalization
    intensity_expected, scale = scale_to_photon_budget(
        intensity_coh,
        photon_budget=config.photon_budget,
    )

    # 6. Target ground-truth scalings
    # DPR convention: target = obj * scale (false_scale=True)
    target_dpr = (object_density * scale).astype(np.float32)
    # Physical convention: target = obj * sqrt(scale)
    target_physical = (object_density * np.sqrt(scale)).astype(np.float32)

    # 7. Detector noise realizations
    # Poisson shot noise
    detected_photons = apply_poisson_noise(intensity_expected, rng=rng)

    # Gaussian readout noise
    readout = apply_gaussian_readout_noise(
        detected_photons,
        readout_std=config.readout_std,
        rng=rng,
    )

    # Saturation and hot pixels
    intensity_raw, _ = apply_saturation_and_defects(
        readout,
        saturation_threshold=config.saturation_threshold,
        hot_pixel_coords=config.hot_pixels,
    )

    if bg_type == "cosmic_ray":
        from .backgrounds import generate_cosmic_ray_hit

        cosmic = generate_cosmic_ray_hit(config.detector_size, num_hits=3, rng=rng)
        intensity_raw += cosmic

    # 7. Detector validity mask
    holder_arm = config.include_holder_arm or metadata.get("holder_arm", False)
    arm_angle = float(metadata.get("arm_angle_rad", config.arm_angle_rad))
    arm_width = int(metadata.get("arm_width", config.arm_width))

    mask = composite_detector_mask(
        shape=config.detector_size,
        include_beam_stop=config.include_beam_stop,
        beam_stop_hw=config.beam_stop_hw,
        beam_stop_hh=config.beam_stop_hh,
        beam_stop_offset=config.beam_stop_offset,
        circular_beam_stop=config.circular_beam_stop,
        include_holder_arm=holder_arm,
        arm_angle_rad=arm_angle,
        arm_width=arm_width,
        include_module_gaps=config.include_module_gaps,
        gap_width=config.gap_width,
        random_pixel_fraction=config.random_pixel_fraction,
        num_clusters=config.num_clusters,
        cluster_radius_range=config.cluster_radius_range,
        num_stripes_h=config.num_stripes_h,
        num_stripes_v=config.num_stripes_v,
        irregular_coverage=config.irregular_coverage,
        rng=rng,
    )

    # 8. Strictly masked observation with zero leakage guarantee
    intensity_observed = np.where(mask, intensity_raw, 0.0).astype(np.float32)

    return SimulationResult(
        object_amplitude=object_density.astype(np.float32),
        object_phase=object_phase.astype(np.float32) if object_phase is not None else None,
        target_dpr=target_dpr,
        target_physical=target_physical,
        intensity_ideal=intensity_ideal.astype(np.float32),
        intensity_expected=intensity_expected.astype(np.float32),
        intensity_raw=intensity_raw.astype(np.float32),
        valid_mask=mask.astype(bool),
        intensity_observed=intensity_observed,
        scale_factor=float(scale),
        photon_budget=float(config.photon_budget),
        forward_model=model_key,
    )


def simulate_diffraction(
    object_density: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
    object_phase: np.ndarray | None = None,
    forward_model: str = "F1",
    metadata: dict[str, Any] | None = None,
) -> SimulationResult:
    """Execute complete forward physical diffraction simulation."""
    if forward_model == "F1" and metadata is not None and "forward_model" in metadata:
        forward_model = str(metadata["forward_model"])

    return simulate_forward_model(
        object_density=object_density,
        config=config,
        model_id=forward_model,
        rng=rng,
        object_phase=object_phase,
        metadata=metadata,
    )
