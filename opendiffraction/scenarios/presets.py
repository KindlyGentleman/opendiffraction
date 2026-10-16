"""Acquisition Scenarios (S1 - S6) representing distinct experimental beamline setups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from opendiffraction.physics.forward import SimulationConfig
from opendiffraction.physics.noise import FWHM_ONE_SIGMA


@dataclass
class ScenarioSpec:
    """Specification of an experimental acquisition scenario."""

    scenario_id: str
    name: str
    facility_type: str  # e.g. "XFEL", "Synchrotron", "Laboratory"
    description: str
    scientific_application: str
    associated_profiles: list[str]
    nominal_photon_energy_kev: float
    detector_pixel_pitch_um: float
    sample_detector_distance_m: float
    default_parameters: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        scenario_id="S1",
        name="XFEL Single-Particle Imaging",
        facility_type="X-ray Free-Electron Laser (XFEL)",
        description="Femtosecond pulses interacting with injected aerosolized or liquid-jet particles.",
        scientific_application="Single-shot atomic/nanoscale CDI prior to radiation damage (diffract-before-destroy).",
        associated_profiles=["V01", "V02", "V03", "V34", "V37", "V38", "V41", "V42"],
        nominal_photon_energy_kev=8.0,  # ~1.55 Angstrom
        detector_pixel_pitch_um=110.0,  # CSPAD / MPCCD style
        sample_detector_distance_m=0.5,
        default_parameters={
            "coherence_length": 300.0,
            "readout_std": FWHM_ONE_SIGMA,
            "include_module_gaps": True,
            "gap_width": 4,
            "solvent_ring": True,
        },
    ),
    "S2": ScenarioSpec(
        scenario_id="S2",
        name="Synchrotron Pink-Beam CDI",
        facility_type="Synchrotron Storage Ring",
        description="High-repetition-rate undulator beamline with moderate spatial coherence and slit flare.",
        scientific_application="High-resolution imaging of extended nanostructures and material assemblies.",
        associated_profiles=["V04", "V06", "V10", "V14", "V20", "V26", "V35"],
        nominal_photon_energy_kev=12.0,  # ~1.03 Angstrom
        detector_pixel_pitch_um=75.0,  # Pilatus / Eiger style
        sample_detector_distance_m=2.0,
        default_parameters={
            "coherence_length": 120.0,
            "readout_std": 0.0,  # Photon counting
            "include_beam_stop": True,
            "holder_arm": True,
            "slit_streaks": True,
        },
    ),
    "S3": ScenarioSpec(
        scenario_id="S3",
        name="Nanomaterials & Defect Inspection",
        facility_type="Synchrotron / Laboratory X-ray",
        description="Targeted CDI of isolated core-shell nanoparticles, nanorods, and cracked matrices.",
        scientific_application="Nondestructive material metrology, void tracking, and strain/defect reconstruction.",
        associated_profiles=["V04", "V05", "V06", "V07", "V08", "V11", "V23"],
        nominal_photon_energy_kev=9.5,
        detector_pixel_pitch_um=55.0,
        sample_detector_distance_m=1.2,
        default_parameters={
            "coherence_length": 250.0,
            "photon_budget": 5e6,
            "readout_std": FWHM_ONE_SIGMA,
            "circular_beam_stop": True,
        },
    ),
    "S4": ScenarioSpec(
        scenario_id="S4",
        name="Biological CDI & Soft Matter",
        facility_type="Synchrotron / XFEL",
        description="Low-dose imaging of cells, organelles, and viruses on thin carbon or SiN support films.",
        scientific_application="Label-free cellular ultrastructure imaging in the water window or tender X-ray regime.",
        associated_profiles=["V01", "V02", "V03", "V33", "V38", "V43"],
        nominal_photon_energy_kev=2.5,  # Soft / tender X-ray
        detector_pixel_pitch_um=30.0,
        sample_detector_distance_m=0.8,
        default_parameters={
            "coherence_length": 180.0,
            "photon_budget": 2e5,  # Dose-sensitive
            "support_film_halo": True,
        },
    ),
    "S5": ScenarioSpec(
        scenario_id="S5",
        name="Educational & Optical Lab Bench",
        facility_type="Optical Laser Laboratory",
        description="Visible light laser illumination (He-Ne / diode) on transmission masks and pinholes.",
        scientific_application="Teaching Fourier optics, ptychography baselines, and phase retrieval fundamentals.",
        associated_profiles=["V10", "V13", "V21", "V45"],
        nominal_photon_energy_kev=0.00196,  # 632.8 nm He-Ne
        detector_pixel_pitch_um=6.45,  # Standard CCD
        sample_detector_distance_m=0.15,
        default_parameters={
            "coherence_length": float("inf"),
            "photon_budget": 1e8,
            "readout_std": 0.5,
            "include_beam_stop": False,
        },
    ),
    "S6": ScenarioSpec(
        scenario_id="S6",
        name="Detector Calibration & QA Benchmark",
        facility_type="Metrology Beamline",
        description="Systematic sensor characterization: flat fields, gain sweeps, dead cluster mapping.",
        scientific_application="Automated detector defect identification, dark-current drift tracking, and linearity audit.",
        associated_profiles=["V24", "V27", "V28", "V29", "V30", "V36"],
        nominal_photon_energy_kev=10.0,
        detector_pixel_pitch_um=75.0,
        sample_detector_distance_m=1.0,
        default_parameters={
            "gain_nonuniformity": 0.04,
            "baseline_drift": True,
            "saturation_sweep": True,
        },
    ),
}


def sample_scenario_config(
    scenario_id: str,
    rng: np.random.Generator | None = None,
) -> tuple[SimulationConfig, dict[str, Any]]:
    """Sample a simulation configuration tuned for a specific experimental scenario."""
    if rng is None:
        rng = np.random.default_rng()

    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario ID: {scenario_id}. Choose S1 to S6.")

    spec = SCENARIOS[scenario_id]
    cfg = SimulationConfig()
    meta: dict[str, Any] = {
        "scenario_id": spec.scenario_id,
        "scenario_name": spec.name,
        "facility_type": spec.facility_type,
        "photon_energy_kev": spec.nominal_photon_energy_kev,
        "pixel_pitch_um": spec.detector_pixel_pitch_um,
        "distance_m": spec.sample_detector_distance_m,
    }

    if scenario_id == "S1":
        cfg.coherence_length = 300.0
        cfg.photon_budget = float(rng.choice([1e5, 5e5, 2e6, 1e7]))
        cfg.include_module_gaps = True
        cfg.gap_width = 4
    elif scenario_id == "S2":
        cfg.coherence_length = float(rng.uniform(100.0, 160.0))
        cfg.photon_budget = 4e6
        cfg.include_beam_stop = True
    elif scenario_id == "S3":
        cfg.coherence_length = 250.0
        cfg.photon_budget = 5e6
        cfg.circular_beam_stop = True
    elif scenario_id == "S4":
        cfg.coherence_length = 180.0
        cfg.photon_budget = float(rng.uniform(1e5, 8e5))
    elif scenario_id == "S5":
        cfg.coherence_length = float("inf")
        cfg.photon_budget = 1e8
        cfg.include_beam_stop = False
    elif scenario_id == "S6":
        cfg.readout_std = 2.0
        cfg.random_pixel_fraction = 0.01
        cfg.num_clusters = 4
        cfg.include_module_gaps = True

    return cfg, meta
