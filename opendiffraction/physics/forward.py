"""Physical forward simulation pipeline for coherent and partially coherent X-ray diffraction."""

from __future__ import annotations

from dataclasses import dataclass
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
    intensity_measured: np.ndarray  # Noisy signed detector readouts
    valid_mask: np.ndarray  # Detector validity mask (bool)
    observed_masked: np.ndarray  # measured * mask
    scale_factor: float  # Flux normalization scale factor s
    photon_budget: float  # Total photons simulated


def pad_object_to_detector(
    obj: np.ndarray,
    detector_size: tuple[int, int] = (512, 512),
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Place object at the center of the detector frame with optional sub-pixel shift.

    Parameters
    ----------
    obj : np.ndarray
        Complex or real object of shape (Ny_obj, Nx_obj).
    detector_size : tuple[int, int]
        Detector frame dimensions (DetH, DetW).
    offset : tuple[float, float]
        Center displacement (dy, dx) in pixels.

    Returns
    -------
    frame : np.ndarray
        Padded frame matching detector_size with object centered.
    """
    ny_det, nx_det = detector_size
    ny_obj, nx_obj = obj.shape[-2:]

    # Center placement
    y0 = (ny_det - ny_obj) // 2
    x0 = (nx_det - nx_obj) // 2

    # Integer part of displacement
    dy_int, dx_int = int(round(offset[0])), int(round(offset[1]))
    y0_adj = y0 + dy_int
    x0_adj = x0 + dx_int

    frame = np.zeros(detector_size, dtype=obj.dtype)
    sy0 = max(0, y0_adj)
    sy1 = min(ny_det, y0_adj + ny_obj)
    sx0 = max(0, x0_adj)
    sx1 = min(nx_det, x0_adj + nx_obj)

    oy0 = max(0, -y0_adj)
    oy1 = oy0 + (sy1 - sy0)
    ox0 = max(0, -x0_adj)
    ox1 = ox0 + (sx1 - sx0)

    if sy1 > sy0 and sx1 > sx0:
        frame[sy0:sy1, sx0:sx1] = obj[oy0:oy1, ox0:ox1]

    # Sub-pixel shift via Fourier phase ramp if subpixel residue exists
    rem_y = offset[0] - dy_int
    rem_x = offset[1] - dx_int
    if abs(rem_y) > 1e-4 or abs(rem_x) > 1e-4:
        ky = np.fft.fftfreq(ny_det)
        kx = np.fft.fftfreq(nx_det)
        kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
        phase_ramp = np.exp(-2j * np.pi * (kyy * rem_y + kxx * rem_x))
        F_frame = np.fft.fft2(frame)
        frame = np.fft.ifft2(F_frame * phase_ramp)
        if np.isrealobj(obj):
            frame = np.real(frame)

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


def simulate_diffraction(
    object_density: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
    object_phase: np.ndarray | None = None,
) -> SimulationResult:
    """Execute complete forward physical diffraction simulation.

    Parameters
    ----------
    object_density : np.ndarray
        2D real array of transmission density / amplitude, shape (64, 64).
    config : SimulationConfig
        All physical and detector parameters.
    rng : np.random.Generator or None
        NumPy random generator for reproducible stochastic noise.
    object_phase : np.ndarray or None
        Optional phase field for complex-valued objects.

    Returns
    -------
    SimulationResult
        Dataclass containing all ground truth and observed diffraction fields.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Prepare complex or real object transmission
    if object_phase is not None:
        obj_field = object_density * np.exp(1j * object_phase)
    else:
        obj_field = object_density.astype(np.float64)

    # 2. Pad object into detector frame (with optional beam offset)
    frame = pad_object_to_detector(
        obj_field,
        detector_size=config.detector_size,
        offset=config.beam_center_offset,
    )

    # 3. Coherent Fraunhofer diffraction
    _, intensity_ideal = compute_coherent_diffraction(frame)

    # 4. Partial coherence via Gaussian Schell-model
    intensity_coh = apply_partial_coherence(
        intensity_ideal,
        l_coh=config.coherence_length,
        legacy=config.coherence_legacy,
    )

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
    intensity_measured, _ = apply_saturation_and_defects(
        readout,
        saturation_threshold=config.saturation_threshold,
        hot_pixel_coords=config.hot_pixels,
    )

    # 8. Detector mask
    mask = composite_detector_mask(
        shape=config.detector_size,
        include_beam_stop=config.include_beam_stop,
        beam_stop_hw=config.beam_stop_hw,
        beam_stop_hh=config.beam_stop_hh,
        beam_stop_offset=config.beam_stop_offset,
        circular_beam_stop=config.circular_beam_stop,
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

    observed_masked = intensity_measured * mask.astype(np.float32)

    return SimulationResult(
        object_amplitude=object_density.astype(np.float32),
        object_phase=object_phase.astype(np.float32) if object_phase is not None else None,
        target_dpr=target_dpr,
        target_physical=target_physical,
        intensity_ideal=intensity_ideal.astype(np.float32),
        intensity_expected=intensity_expected.astype(np.float32),
        intensity_measured=intensity_measured.astype(np.float32),
        valid_mask=mask.astype(bool),
        observed_masked=observed_masked.astype(np.float32),
        scale_factor=float(scale),
        photon_budget=float(config.photon_budget),
    )
