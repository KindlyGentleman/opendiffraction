"""Physical simulation modules for OpenDiffraction."""

from .coherence import apply_partial_coherence, create_coherence_kernel
from .forward import (
    SimulationConfig,
    SimulationResult,
    compute_coherent_diffraction,
    pad_object_to_detector,
    simulate_diffraction,
)
from .masks import (
    composite_detector_mask,
    create_centered_beam_stop,
    create_dead_pixels_and_clusters,
    create_irregular_patches,
    create_module_gaps,
    create_stripes,
)
from .noise import (
    FWHM_ONE_SIGMA,
    apply_gaussian_readout_noise,
    apply_poisson_noise,
    apply_saturation_and_defects,
    scale_to_photon_budget,
)

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "simulate_diffraction",
    "compute_coherent_diffraction",
    "pad_object_to_detector",
    "apply_partial_coherence",
    "create_coherence_kernel",
    "scale_to_photon_budget",
    "apply_poisson_noise",
    "apply_gaussian_readout_noise",
    "apply_saturation_and_defects",
    "composite_detector_mask",
    "create_centered_beam_stop",
    "create_module_gaps",
    "create_dead_pixels_and_clusters",
    "create_stripes",
    "create_irregular_patches",
    "FWHM_ONE_SIGMA",
]
