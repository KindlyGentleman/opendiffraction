"""Machine-readable specifications for the 12 controlled experimental variation profiles (V01 - V12).

Each profile specifies which physical variables are swept and which are held constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

from opendiffraction.objects.procedural import DensityType, ShapeFamily
from opendiffraction.physics.forward import SimulationConfig
from opendiffraction.physics.noise import FWHM_ONE_SIGMA


@dataclass
class VariationSpec:
    """Specification of an experimental profile."""

    profile_id: str
    name: str
    description: str
    scientific_purpose: str
    sweep_parameter: str
    fixed_parameters: dict[str, Any]
    is_ood: bool = False


PROFILE_SPECS: dict[str, VariationSpec] = {
    "V01": VariationSpec(
        profile_id="V01",
        name="Clean Coherent Diffraction",
        description="Ideal coherent Fraunhofer diffraction without detector defects or stochastic noise.",
        scientific_purpose="Establish theoretical upper reconstruction limits under perfect observations.",
        sweep_parameter="object_geometry",
        fixed_parameters={
            "coherence_length": float("inf"),
            "photon_budget": 1e9,  # Effectively infinite
            "readout_std": 0.0,
            "include_beam_stop": False,
            "random_pixel_fraction": 0.0,
            "noise_free": True,
        },
    ),
    "V02": VariationSpec(
        profile_id="V02",
        name="Geometry Diversity",
        description="Comprehensive sweep across geometric, blob, multi-scatterer, and lattice shapes.",
        scientific_purpose="Benchmark shape generalization under fixed standard acquisition conditions.",
        sweep_parameter="shape_family_and_subtype",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
            "density_type": "uniform",
        },
    ),
    "V03": VariationSpec(
        profile_id="V03",
        name="Density Heterogeneity",
        description="Objects with uniform, binary, graded, textured, and piecewise constant densities.",
        scientific_purpose="Evaluate continuous amplitude reconstruction fidelity and internal contrast recovery.",
        sweep_parameter="density_type",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V04": VariationSpec(
        profile_id="V04",
        name="Photon Starvation",
        description="Photon budget systematically swept over 6 orders of magnitude: 1e3 to 1e8 photons.",
        scientific_purpose="Quantify reconstruction breakdown across low-flux to high-flux regimes.",
        sweep_parameter="photon_budget",
        fixed_parameters={
            "coherence_length": 200.0,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V05": VariationSpec(
        profile_id="V05",
        name="Partial Coherence",
        description="Coherence length swept from fully coherent (inf) down to 25 pixels.",
        scientific_purpose="Quantify speckle contrast degradation from reduced spatial coherence.",
        sweep_parameter="coherence_length",
        fixed_parameters={
            "photon_budget": 5e6,
            "readout_std": 0.0,  # Zero readout to isolate coherence blur
            "include_beam_stop": True,
            "beam_stop_hw": 12,
            "beam_stop_hh": 12,
        },
    ),
    "V06": VariationSpec(
        profile_id="V06",
        name="Readout Noise",
        description="Gaussian detector readout noise standard deviation varied from 0.0 to 3.0 photons.",
        scientific_purpose="Disentangle additive detector electronics noise from multiplicative Poisson noise.",
        sweep_parameter="readout_std",
        fixed_parameters={
            "photon_budget": 1e7,  # High flux to isolate readout noise
            "coherence_length": 200.0,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V07": VariationSpec(
        profile_id="V07",
        name="Beam-Stop Geometry",
        description="Central beam-stop dimensions swept from 4x4 to 48x48 px with centered and shifted positions.",
        scientific_purpose="Evaluate missing low-spatial-frequency information loss.",
        sweep_parameter="beam_stop_dimensions_and_offsets",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
        },
    ),
    "V08": VariationSpec(
        profile_id="V08",
        name="Irregular Missing Detector Pixels",
        description="Procedural detector defects: random dead pixels, clusters, stripes, module gaps, irregular patches.",
        scientific_purpose="Benchmark inpainting and phase retrieval under incomplete detector geometries.",
        sweep_parameter="defect_type_and_coverage",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V09": VariationSpec(
        profile_id="V09",
        name="Saturation and Hot Pixels",
        description="Full-well detector saturation thresholds and isolated hot pixel defects.",
        scientific_purpose="Evaluate robustness against sensor nonlinearities and high-flux clipping.",
        sweep_parameter="saturation_threshold_and_hot_pixels",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 1e7,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V10": VariationSpec(
        profile_id="V10",
        name="Beam-Center and Alignment Errors",
        description="Sub-pixel and integer detector alignment shifts (dy, dx).",
        scientific_purpose="Evaluate sensitivity to beam-center miscalibration.",
        sweep_parameter="beam_center_offset",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "beam_stop_hw": 16,
            "beam_stop_hh": 16,
        },
    ),
    "V11": VariationSpec(
        profile_id="V11",
        name="Combined Realistic Degradation",
        description="Simultaneous photon noise, finite coherence, readout noise, beam stop, gaps, and dead pixels.",
        scientific_purpose="Challenging composite benchmark reflecting real-world synchrotron/XFEL conditions.",
        sweep_parameter="stochastic_composite_realizations",
        fixed_parameters={
            "coherence_length": 200.0,
            "photon_budget": 3e6,
            "readout_std": FWHM_ONE_SIGMA,
            "include_beam_stop": True,
            "include_module_gaps": True,
            "gap_width": 4,
            "random_pixel_fraction": 0.005,
            "num_clusters": 2,
            "irregular_coverage": 0.08,
        },
    ),
    "V12": VariationSpec(
        profile_id="V12",
        name="Out-of-Distribution Stress Tests",
        description="Extreme adverse conditions, held-out dense fractal/void geometries, and severe photon starvation.",
        scientific_purpose="Strict out-of-distribution evaluation for model failure mode analysis.",
        sweep_parameter="extreme_multivariate_stresses",
        fixed_parameters={},
        is_ood=True,
    ),
}


def sample_profile_config(
    profile_id: str,
    index_within_profile: int,
    total_in_profile: int,
    rng: np.random.Generator,
) -> tuple[SimulationConfig, ShapeFamily, DensityType, str]:
    """Sample simulation config and object properties for a given profile."""
    # Default baseline configuration
    cfg = SimulationConfig()
    shape_family: ShapeFamily = "basic_geometric"
    density_type: DensityType = "uniform"
    shape_subtype = "circle"

    # Shape and density pools
    all_families: list[ShapeFamily] = [
        "basic_geometric",
        "particle_blob",
        "multiple_scatterer",
        "porous_heterogeneous",
        "structured_pattern",
    ]
    all_densities: list[DensityType] = [
        "uniform",
        "binary",
        "graded",
        "textured",
        "piecewise_constant",
    ]

    if profile_id == "V01":
        # Clean coherent diffraction
        cfg.coherence_length = float("inf")
        cfg.photon_budget = 1e8
        cfg.readout_std = 0.0
        cfg.include_beam_stop = False
        shape_family = all_families[index_within_profile % len(all_families)]
        density_type = all_densities[index_within_profile % len(all_densities)]

    elif profile_id == "V02":
        # Geometry diversity sweep
        cfg.coherence_length = 200.0
        cfg.photon_budget = 3e6
        cfg.readout_std = FWHM_ONE_SIGMA
        cfg.include_beam_stop = True
        shape_family = all_families[index_within_profile % len(all_families)]
        density_type = "uniform"

    elif profile_id == "V03":
        # Density heterogeneity sweep
        cfg.coherence_length = 200.0
        cfg.photon_budget = 3e6
        cfg.readout_std = FWHM_ONE_SIGMA
        cfg.include_beam_stop = True
        shape_family = "particle_blob"
        density_type = all_densities[index_within_profile % len(all_densities)]

    elif profile_id == "V04":
        # Photon starvation sweep: 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
        budgets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        cfg.photon_budget = budgets[index_within_profile % len(budgets)]
        cfg.coherence_length = 200.0
        cfg.readout_std = FWHM_ONE_SIGMA
        cfg.include_beam_stop = True
        shape_family = "particle_blob"
        density_type = "uniform"

    elif profile_id == "V05":
        # Partial coherence sweep: inf, 300, 200, 150, 100, 50, 25
        coherences = [float("inf"), 300.0, 200.0, 150.0, 100.0, 50.0, 25.0]
        cfg.coherence_length = coherences[index_within_profile % len(coherences)]
        cfg.photon_budget = 5e6
        cfg.readout_std = 0.0  # Zero readout to isolate fringe decay
        cfg.include_beam_stop = True
        shape_family = "structured_pattern"
        density_type = "uniform"

    elif profile_id == "V06":
        # Readout noise sweep
        readout_stds = [0.0, 0.2, FWHM_ONE_SIGMA, 0.8, 1.5, 3.0]
        cfg.readout_std = readout_stds[index_within_profile % len(readout_stds)]
        cfg.photon_budget = 1e7
        cfg.coherence_length = 200.0
        cfg.include_beam_stop = True
        shape_family = "basic_geometric"
        density_type = "textured"

    elif profile_id == "V07":
        # Beam-stop size and offset sweep
        beam_stops = [
            (8, 8, 0, 0),
            (16, 16, 0, 0),
            (24, 24, 0, 0),
            (32, 32, 0, 0),
            (16, 16, 4, -4),
            (20, 20, -6, 6),
        ]
        hw, hh, oy, ox = beam_stops[index_within_profile % len(beam_stops)]
        cfg.include_beam_stop = True
        cfg.beam_stop_hw = hw
        cfg.beam_stop_hh = hh
        cfg.beam_stop_offset = (oy, ox)
        cfg.photon_budget = 3e6
        cfg.coherence_length = 200.0
        cfg.readout_std = FWHM_ONE_SIGMA
        shape_family = "multiple_scatterer"
        density_type = "uniform"

    elif profile_id == "V08":
        # Irregular missing detector pixels
        cfg.include_beam_stop = True
        cfg.beam_stop_hw = 16
        cfg.beam_stop_hh = 16
        cfg.photon_budget = 3e6
        cfg.coherence_length = 200.0
        cfg.readout_std = FWHM_ONE_SIGMA
        mode = index_within_profile % 4
        if mode == 0:
            cfg.random_pixel_fraction = 0.02
        elif mode == 1:
            cfg.num_clusters = 4
        elif mode == 2:
            cfg.num_stripes_h = 2
            cfg.num_stripes_v = 2
        else:
            cfg.include_module_gaps = True
            cfg.irregular_coverage = 0.12
        shape_family = "porous_heterogeneous"
        density_type = "binary"

    elif profile_id == "V09":
        # Saturation and hot pixels
        sat_thresholds = [2e4, 5e4, 1e5, 5e5]
        cfg.saturation_threshold = sat_thresholds[index_within_profile % len(sat_thresholds)]
        cfg.hot_pixels = [(256 + 15, 256 + 20), (256 - 30, 256 + 40)]
        cfg.photon_budget = 1e7
        cfg.coherence_length = 200.0
        cfg.readout_std = FWHM_ONE_SIGMA
        shape_family = "basic_geometric"
        density_type = "graded"

    elif profile_id == "V10":
        # Beam center and alignment errors
        offsets = [(0.0, 0.0), (-2.0, 3.0), (1.5, -2.5), (4.0, -4.0), (-3.5, -1.5)]
        cfg.beam_center_offset = offsets[index_within_profile % len(offsets)]
        cfg.photon_budget = 3e6
        cfg.coherence_length = 200.0
        cfg.readout_std = FWHM_ONE_SIGMA
        cfg.include_beam_stop = True
        shape_family = "particle_blob"
        density_type = "textured"

    elif profile_id == "V11":
        # Combined realistic degradation
        cfg.photon_budget = float(rng.uniform(1e6, 5e6))
        cfg.coherence_length = float(rng.uniform(150.0, 250.0))
        cfg.readout_std = FWHM_ONE_SIGMA
        cfg.include_beam_stop = True
        cfg.beam_stop_hw = int(rng.integers(12, 20))
        cfg.beam_stop_hh = int(rng.integers(12, 20))
        cfg.include_module_gaps = True
        cfg.random_pixel_fraction = 0.003
        cfg.num_clusters = 2
        cfg.irregular_coverage = 0.06
        shape_family = all_families[index_within_profile % len(all_families)]
        density_type = all_densities[index_within_profile % len(all_densities)]

    elif profile_id == "V12":
        # Out-of-distribution stress test
        cfg.photon_budget = 500.0 if index_within_profile % 2 == 0 else 5e7
        cfg.coherence_length = 30.0  # Ultra-low coherence
        cfg.readout_std = 2.5
        cfg.include_beam_stop = True
        cfg.beam_stop_hw = 36
        cfg.beam_stop_hh = 36
        cfg.include_module_gaps = True
        cfg.gap_width = 8
        cfg.irregular_coverage = 0.20
        shape_family = "structured_pattern"
        shape_subtype = "sparse_lattice"
        density_type = "textured"

    return cfg, shape_family, density_type, shape_subtype
