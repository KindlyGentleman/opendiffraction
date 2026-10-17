"""Comprehensive registry and specification of all 48 Variation Profiles (V01 - V48).

Organized into 6 Physical Groups:
- Group 1 (V01 - V12): Real-World Specimens
- Group 2 (V13 - V20): Illumination and Coherence
- Group 3 (V21 - V30): Detector Realism
- Group 4 (V31 - V36): Background and Scattering
- Group 5 (V37 - V42): Acquisition and Hit Finding
- Group 6 (V43 - V48): Advanced Physics and Out-of-Distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from opendiffraction.objects.complex_objects import generate_phase_field
from opendiffraction.objects.object_3d import generate_3d_volume
from opendiffraction.objects.procedural import (
    generate_procedural_object,
    make_multiple_scatterers,
)
from opendiffraction.objects.specimens import (
    make_cell_phantom,
    make_cluster_assembly,
    make_colloidal_crystal,
    make_core_shell_nanoparticle,
    make_crack_void,
    make_ellipsoidal_nanoparticle,
    make_lithographic_pattern,
    make_mixed_density,
    make_nanorod,
    make_porous_matrix,
    make_protein_aggregate,
    make_virus_particle,
)
from opendiffraction.physics.forward import SimulationConfig
from opendiffraction.physics.noise import FWHM_ONE_SIGMA


@dataclass
class Profile48Spec:
    """Detailed physical profile specification for V01 - V48."""

    profile_id: str
    name: str
    group_id: int
    group_name: str
    forward_model: str  # "F1", "F2", "F3", "F4", "F5", "F6"
    description: str
    scientific_purpose: str
    primary_tasks: list[str]  # e.g. ["T1_phase_retrieval", "T4_hit_classification"]
    is_ood: bool = False
    is_dpr_compatible: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


PROFILES_48: dict[str, Profile48Spec] = {
    # ---------------------------------------------------------
    # Group 1: Real-World Specimens (V01 - V12)
    # ---------------------------------------------------------
    "V01": Profile48Spec(
        profile_id="V01",
        name="Cell-Like Biological Phantoms",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Projected biological cell phantom with outer membrane, nucleus, and vesicles.",
        scientific_purpose="Benchmark biological CDI reconstruction on smooth multi-compartment structures.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),
    "V02": Profile48Spec(
        profile_id="V02",
        name="Virus-Like Particles",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Spherical/shell-like viral capsid with radial surface spike proteins.",
        scientific_purpose="Single-particle imaging and symmetry-driven reconstruction benchmarking.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification", "T7_parameter_estimation"],
    ),
    "V03": Profile48Spec(
        profile_id="V03",
        name="Protein-Complex Aggregates",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Multi-subunit synthetic proxy of macromolecular protein complexes.",
        scientific_purpose="Evaluate resolution of heterogeneous multi-domain biological complexes.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),
    "V04": Profile48Spec(
        profile_id="V04",
        name="Core-Shell Nanoparticles",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Concentric nanoparticle with high-Z core and low-Z shell.",
        scientific_purpose="Quantitative contrast retrieval across step-index material boundaries.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V05": Profile48Spec(
        profile_id="V05",
        name="Ellipsoidal Nanoparticles",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Anisotropic ellipsoidal nanoparticles with varying aspect ratios and orientations.",
        scientific_purpose="Reconstruct anisotropic scatterers and determine particle orientation.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V06": Profile48Spec(
        profile_id="V06",
        name="Nanorods and Nanowires",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="High-aspect-ratio 1D nanorods with hemispherical end caps and curvature.",
        scientific_purpose="High-frequency streak analysis and aspect-ratio quantification.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),
    "V07": Profile48Spec(
        profile_id="V07",
        name="Nanoparticle Clusters",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Fractal-like agglomerates of touching spherical nanoparticles.",
        scientific_purpose="Interference fringes from multiple interacting sub-scatterers.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),
    "V08": Profile48Spec(
        profile_id="V08",
        name="Porous Materials / Aerogels",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Random porous spinodal or aerogel network with interconnected pores.",
        scientific_purpose="Speckle analysis and pore size distribution estimation.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V09": Profile48Spec(
        profile_id="V09",
        name="Colloidal Crystal Assemblies",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Ordered hexagonal close-packed arrays of monodisperse colloidal spheres.",
        scientific_purpose="Bragg peak detection alongside diffuse form factor scattering.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),
    "V10": Profile48Spec(
        profile_id="V10",
        name="Lithographic Test Patterns",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Orthogonal grating lines and resolution Siemens-star features.",
        scientific_purpose="Quantitative MTF / spatial resolution verification via knife-edge metrics.",
        primary_tasks=["T1_phase_retrieval", "T12_consistency_check"],
    ),
    "V11": Profile48Spec(
        profile_id="V11",
        name="Defect Structures (Cracks/Voids)",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Solid matrix containing interior voids and branching micro-cracks.",
        scientific_purpose="Nondestructive flaw detection and high-contrast anomaly localization.",
        primary_tasks=["T1_phase_retrieval", "T8_anomaly_detection"],
    ),
    "V12": Profile48Spec(
        profile_id="V12",
        name="Multi-Component Mixed Density",
        group_id=1,
        group_name="Real-World Specimens",
        forward_model="F1",
        description="Heterogeneous specimen with mixed low-Z and high-Z phases.",
        scientific_purpose="Segment multi-phase materials from continuous CDI density maps.",
        primary_tasks=["T1_phase_retrieval", "T6_morphology_classification"],
    ),

    # ---------------------------------------------------------
    # Group 2: Illumination & Coherence (V13 - V20)
    # ---------------------------------------------------------
    "V13": Profile48Spec(
        profile_id="V13",
        name="Fully Coherent Plane Wave",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F1",
        description="Ideal monochromatic plane wave with infinite spatial coherence.",
        scientific_purpose="Theoretical upper bound for phase retrieval under ideal illumination.",
        primary_tasks=["T1_phase_retrieval", "T12_consistency_check"],
    ),
    "V14": Profile48Spec(
        profile_id="V14",
        name="Moderate Partial Coherence",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F2",
        description="Gaussian Schell-model partial coherence with coherence length 150 px.",
        scientific_purpose="Evaluate speckle contrast reduction in typical synchrotron beams.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V15": Profile48Spec(
        profile_id="V15",
        name="Strong Partial Coherence",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F2",
        description="Severe partial coherence (coherence length 40 px) blurring high-q speckles.",
        scientific_purpose="Phase retrieval degradation under degraded beamline coherence.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
    ),
    "V16": Profile48Spec(
        profile_id="V16",
        name="Focused Curved Wavefront",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F3",
        description="Converging illumination with parabolic phase curvature (defocus).",
        scientific_purpose="Probe wavefront curvature effects on diffraction intensity.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V17": Profile48Spec(
        profile_id="V17",
        name="Aberrated Beam (Spherical)",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F3",
        description="Illumination with fourth-order radial spherical wavefront aberration.",
        scientific_purpose="Reconstruction robustness against KB-mirror figure errors.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
    ),
    "V18": Profile48Spec(
        profile_id="V18",
        name="Astigmatic Illumination",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F3",
        description="Cylindrical wavefront curvature with asymmetric focal lengths.",
        scientific_purpose="Probe asymmetry and orthogonal phase aberration recovery.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V19": Profile48Spec(
        profile_id="V19",
        name="Pointing Jitter / Wavefront Tilt",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F3",
        description="Sub-pixel transverse beam pointing instability and linear phase tilt.",
        scientific_purpose="Vibration-induced pattern shifting and speckle blurring analysis.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
    ),
    "V20": Profile48Spec(
        profile_id="V20",
        name="Polychromatic Pink Beam",
        group_id=2,
        group_name="Illumination and Coherence",
        forward_model="F5",
        description="Narrow-bandwidth polychromatic beam (1.5% dE/E FWHM spectral distribution).",
        scientific_purpose="Radial speckle elongation in pink-beam synchrotron imaging.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
    ),

    # ---------------------------------------------------------
    # Group 3: Detector Realism (V21 - V30)
    # ---------------------------------------------------------
    "V21": Profile48Spec(
        profile_id="V21",
        name="Ideal Noise-Free Detector",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Infinite dynamic range, zero dark noise, 100% fill factor.",
        scientific_purpose="Isolate algorithm performance from sensor imperfections.",
        primary_tasks=["T1_phase_retrieval"],
    ),
    "V22": Profile48Spec(
        profile_id="V22",
        name="Pure Poisson Shot Noise",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Quantum photon counting with Poisson statistics and zero electronic noise.",
        scientific_purpose="Photon-limited statistical phase retrieval benchmarks.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V23": Profile48Spec(
        profile_id="V23",
        name="Mixed Poisson-Gaussian Noise",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Poisson quantum noise combined with Gaussian electronic readout noise.",
        scientific_purpose="Standard pixel detector physics matching Pilatus and Jungfrau.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V24": Profile48Spec(
        profile_id="V24",
        name="Severe Readout Noise",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="High electronic readout noise (sigma = 2.5 photons) overwhelming faint speckles.",
        scientific_purpose="Benchmark low-dose sensor denoising and phase estimation.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V25": Profile48Spec(
        profile_id="V25",
        name="Central Circular Beam Stop",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Circular central beam stop blocking low-q direct transmission.",
        scientific_purpose="Evaluate missing low-spatial-frequency phase retrieval limits.",
        primary_tasks=["T1_phase_retrieval", "T3_inpainting"],
    ),
    "V26": Profile48Spec(
        profile_id="V26",
        name="Beam Stop with Support Arm",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Central beam stop supported by a rigid diagonal tungsten shadow arm.",
        scientific_purpose="Asymmetric continuous missing data inpainting and reconstruction.",
        primary_tasks=["T1_phase_retrieval", "T3_inpainting"],
    ),
    "V27": Profile48Spec(
        profile_id="V27",
        name="Multi-Panel Module Gaps",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Tiled sensor geometry with inactive horizontal and vertical module gaps.",
        scientific_purpose="Multi-panel gap inpainting matching CSPAD and AGIPD sensors.",
        primary_tasks=["T1_phase_retrieval", "T3_inpainting"],
    ),
    "V28": Profile48Spec(
        profile_id="V28",
        name="Dead and Hot Pixel Defects",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Random non-responsive dead pixels and saturated hot pixel clusters.",
        scientific_purpose="Defect detection, mask estimation, and robust phase retrieval.",
        primary_tasks=["T3_inpainting", "T5_defect_segmentation"],
    ),
    "V29": Profile48Spec(
        profile_id="V29",
        name="Full-Well Saturation",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Sensor dynamic-range saturation clipping high-intensity low-q rings.",
        scientific_purpose="High dynamic range clipping recovery and saturation inpainting.",
        primary_tasks=["T1_phase_retrieval", "T3_inpainting"],
    ),
    "V30": Profile48Spec(
        profile_id="V30",
        name="Combined Realistic Detector",
        group_id=3,
        group_name="Detector Realism",
        forward_model="F1",
        description="Simultaneous Poisson noise, readout noise, beam stop, arm, gaps, and dead pixels.",
        scientific_purpose="Comprehensive realistic detector benchmark reflecting experimental data.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising", "T3_inpainting"],
    ),

    # ---------------------------------------------------------
    # Group 4: Background & Scattering (V31 - V36)
    # ---------------------------------------------------------
    "V31": Profile48Spec(
        profile_id="V31",
        name="Background-Free Vacuum",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="High-vacuum environment with zero parasitic background scattering.",
        scientific_purpose="Baseline for background subtraction and contrast evaluation.",
        primary_tasks=["T1_phase_retrieval"],
    ),
    "V32": Profile48Spec(
        profile_id="V32",
        name="Diffuse Gas Scattering",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="Isotropic low-q Lorentzian background from residual chamber gas.",
        scientific_purpose="Diffuse background modeling and low-q baseline estimation.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V33": Profile48Spec(
        profile_id="V33",
        name="Support Membrane Halo",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="Diffuse scattering ring from amorphous carbon or SiN support film.",
        scientific_purpose="Separate sample diffraction from support membrane scattering.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V34": Profile48Spec(
        profile_id="V34",
        name="Solvent / Water Ring",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="Wide-angle diffuse water scattering ring from aerosol/liquid jet injection.",
        scientific_purpose="Background subtraction for solution/liquid jet single particle imaging.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising"],
    ),
    "V35": Profile48Spec(
        profile_id="V35",
        name="Aperture Slit Streaks",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="Orthogonal high-intensity streaks from beam-defining slit knife edges.",
        scientific_purpose="Anisotropic parasitic background filtering and artifact removal.",
        primary_tasks=["T1_phase_retrieval", "T2_denoising", "T8_anomaly_detection"],
    ),
    "V36": Profile48Spec(
        profile_id="V36",
        name="Cosmic Ray Event Artifacts",
        group_id=4,
        group_name="Background and Scattering",
        forward_model="F1",
        description="High-energy cosmic ray or radioactive decay track across detector panels.",
        scientific_purpose="Outlier artifact identification and transient strike removal.",
        primary_tasks=["T5_defect_segmentation", "T8_anomaly_detection"],
    ),

    # ---------------------------------------------------------
    # Group 5: Acquisition & Hit Finding (V37 - V42)
    # ---------------------------------------------------------
    "V37": Profile48Spec(
        profile_id="V37",
        name="High-Flux Hit",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Optimal beam-sample intersection with high incident photon flux (1e7).",
        scientific_purpose="High signal-to-noise benchmark for single-shot phase retrieval.",
        primary_tasks=["T1_phase_retrieval", "T4_hit_classification"],
    ),
    "V38": Profile48Spec(
        profile_id="V38",
        name="Photon-Starved Hit",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Faint single-particle hit at low flux (5e4 photons) near detection limit.",
        scientific_purpose="Low-dose hit finding and ultra-low-photon phase retrieval.",
        primary_tasks=["T1_phase_retrieval", "T4_hit_classification"],
    ),
    "V39": Profile48Spec(
        profile_id="V39",
        name="Centered Beam Intersection",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Specimen perfectly centered in the beam with symmetric diffraction.",
        scientific_purpose="Reference class for spatial alignment and center-of-mass metrics.",
        primary_tasks=["T1_phase_retrieval", "T4_hit_classification"],
    ),
    "V40": Profile48Spec(
        profile_id="V40",
        name="Off-Center Intersection",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Specimen displaced towards probe fringe (5 to 15 px translation).",
        scientific_purpose="Transverse position estimation and edge-illumination effects.",
        primary_tasks=["T1_phase_retrieval", "T4_hit_classification", "T7_parameter_estimation"],
    ),
    "V41": Profile48Spec(
        profile_id="V41",
        name="Multi-Particle Collision",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Two or three distinct specimens simultaneously illuminated in the focal volume.",
        scientific_purpose="Multi-particle hit veto and non-single-particle classification.",
        primary_tasks=["T4_hit_classification", "T6_morphology_classification", "T8_anomaly_detection"],
    ),
    "V42": Profile48Spec(
        profile_id="V42",
        name="Empty Jet Non-Hit",
        group_id=5,
        group_name="Acquisition and Hit Finding",
        forward_model="F1",
        description="Beamshot through liquid jet/vacuum with no particle present (pure background).",
        scientific_purpose="High-throughput hit-finding rejection of non-hits at XFELs.",
        primary_tasks=["T4_hit_classification", "T8_anomaly_detection"],
    ),

    # ---------------------------------------------------------
    # Group 6: Advanced Physics & OOD (V43 - V48)
    # ---------------------------------------------------------
    "V43": Profile48Spec(
        profile_id="V43",
        name="Weak Phase Specimen",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F1",
        description="Thin specimen with uniform transmission amplitude and subtle phase shifts (<0.5 rad).",
        scientific_purpose="Direct phase contrast retrieval under weak-scattering assumptions.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
        is_ood=True,
        is_dpr_compatible=False,
    ),
    "V44": Profile48Spec(
        profile_id="V44",
        name="Strong Complex Object",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F1",
        description="Complex-valued transmission with coupled amplitude variations and vortex phase.",
        scientific_purpose="Generalized complex phase retrieval without nonnegativity constraints.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
        is_ood=True,
        is_dpr_compatible=False,
    ),
    "V45": Profile48Spec(
        profile_id="V45",
        name="Near-Field Fresnel Regime",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F4",
        description="Fresnel diffraction under near-field propagation (Fresnel number ~ 2.0).",
        scientific_purpose="Near-field holographic phase retrieval and regime boundary testing.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
        is_ood=True,
        is_dpr_compatible=False,
    ),
    "V46": Profile48Spec(
        profile_id="V46",
        name="Broadband Polychromatic",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F5",
        description="Wide-spectrum illumination with 8% spectral bandwidth causing radial smearing.",
        scientific_purpose="Phase retrieval under severe chromatic speckle dispersion.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization"],
        is_ood=True,
        is_dpr_compatible=False,
    ),
    "V47": Profile48Spec(
        profile_id="V47",
        name="3D Particle Euler Scattering",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F6",
        description="3D scattering volume projected at arbitrary 3D Euler angles via Fourier slice theorem.",
        scientific_purpose="Validate 3D single-particle reconstruction and orientation estimation.",
        primary_tasks=["T1_phase_retrieval", "T7_parameter_estimation"],
        is_ood=False,
        is_dpr_compatible=True,
    ),
    "V48": Profile48Spec(
        profile_id="V48",
        name="Compound Adversarial Stress Test",
        group_id=6,
        group_name="Advanced Physics and OOD",
        forward_model="F1",
        description="Combined low photon budget, partial coherence, slit flare, gaps, and translation offset.",
        scientific_purpose="Extreme stress test for inverse problem robustness and uncertainty quantification.",
        primary_tasks=["T1_phase_retrieval", "T9_domain_generalization", "T11_uncertainty_quantification"],
        is_ood=True,
        is_dpr_compatible=True,
    ),
}


def _make_obj(family: str = "particle_blob", size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    density, _ = generate_procedural_object(family=family, size=size, rng=rng)
    return density


def sample_profile_48(
    profile_id: str,
    index: int,
    total: int,
    rng: np.random.Generator,
) -> tuple[SimulationConfig, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Sample simulation configuration and object for any of the 48 profiles.

    Parameters
    ----------
    profile_id : str
        Profile identifier (e.g. "V01" to "V48").
    index : int
        Sample index within this profile.
    total : int
        Total samples to be drawn for this profile.
    rng : np.random.Generator
        Deterministic random generator.

    Returns
    -------
    cfg : SimulationConfig
        Physics forward simulation parameters.
    amplitude : np.ndarray
        2D unscaled ground truth object density (64, 64).
    phase : np.ndarray or None
        2D ground truth phase in radians (64, 64) or None for real objects.
    metadata : dict[str, Any]
        Profile-specific scientific metadata attributes.
    """
    if profile_id not in PROFILES_48:
        raise ValueError(f"Unknown profile ID: {profile_id}. Must be V01 through V48.")

    spec = PROFILES_48[profile_id]
    cfg = SimulationConfig()
    metadata: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_name": spec.name,

        "group_id": spec.group_id,
        "group_name": spec.group_name,
        "forward_model": spec.forward_model,
        "is_ood": spec.is_ood,
        "is_dpr_compatible": spec.is_dpr_compatible,
    }

    phase: np.ndarray | None = None

    # -------------------------------------------------------------
    # Group 1: Real-World Specimens (V01 - V12)
    # -------------------------------------------------------------
    if profile_id == "V01":
        amplitude = make_cell_phantom(64, rng=rng)
    elif profile_id == "V02":
        amplitude = make_virus_particle(64, rng=rng)
    elif profile_id == "V03":
        amplitude = make_protein_aggregate(64, rng=rng)
    elif profile_id == "V04":
        amplitude = make_core_shell_nanoparticle(64, rng=rng)
    elif profile_id == "V05":
        amplitude = make_ellipsoidal_nanoparticle(64, rng=rng)
    elif profile_id == "V06":
        amplitude = make_nanorod(64, rng=rng)
    elif profile_id == "V07":
        amplitude = make_cluster_assembly(64, rng=rng)
    elif profile_id == "V08":
        amplitude = make_porous_matrix(64, rng=rng)
    elif profile_id == "V09":
        amplitude = make_colloidal_crystal(64, rng=rng)
    elif profile_id == "V10":
        amplitude = make_lithographic_pattern(64, rng=rng)
    elif profile_id == "V11":
        amplitude = make_crack_void(64, rng=rng)
    elif profile_id == "V12":
        amplitude = make_mixed_density(64, rng=rng)

    # -------------------------------------------------------------
    # Group 2: Illumination & Coherence (V13 - V20)
    # -------------------------------------------------------------
    elif profile_id == "V13":
        # Fully coherent plane wave
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.coherence_length = float("inf")
    elif profile_id == "V14":
        # Moderate partial coherence
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.coherence_length = 150.0
    elif profile_id == "V15":
        # Strongly partially coherent
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.coherence_length = 40.0
    elif profile_id in ("V16", "V17", "V18", "V19"):
        amplitude = _make_obj("multiple_scatterer", 64, rng=rng)
        if profile_id == "V16":
            metadata["defocus_waves"] = float(rng.uniform(0.5, 2.0))
        elif profile_id == "V17":
            metadata["spherical_aberration_waves"] = float(rng.uniform(0.5, 1.5))
        elif profile_id == "V18":
            metadata["astigmatism_waves"] = float(rng.uniform(0.4, 1.2))
        elif profile_id == "V19":
            jitter_y = float(rng.uniform(-2.0, 2.0))
            jitter_x = float(rng.uniform(-2.0, 2.0))
            cfg.beam_center_offset = (jitter_y, jitter_x)
            metadata["jitter_offset"] = (jitter_y, jitter_x)
    elif profile_id == "V20":
        # Polychromatic pink beam
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        metadata["spectral_bandwidth_fraction"] = 0.015

    # -------------------------------------------------------------
    # Group 3: Detector Realism (V21 - V30)
    # -------------------------------------------------------------
    elif profile_id == "V21":
        amplitude = _make_obj("basic_geometric", 64, rng=rng)
        cfg.readout_std = 0.0
        cfg.include_beam_stop = False
    elif profile_id == "V22":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.readout_std = 0.0
    elif profile_id == "V23":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.readout_std = FWHM_ONE_SIGMA
    elif profile_id == "V24":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.readout_std = 2.5
    elif profile_id == "V25":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.circular_beam_stop = True
        cfg.beam_stop_hw = 16
        cfg.beam_stop_hh = 16
    elif profile_id == "V26":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.include_beam_stop = True
        metadata["holder_arm"] = True
    elif profile_id == "V27":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.include_module_gaps = True
        cfg.gap_width = 6
    elif profile_id == "V28":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.random_pixel_fraction = 0.008
        cfg.num_clusters = 3
    elif profile_id == "V29":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.photon_budget = 1e7
        cfg.saturation_threshold = 2.5e4
    elif profile_id == "V30":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.include_module_gaps = True
        cfg.random_pixel_fraction = 0.005
        cfg.num_clusters = 2
        metadata["holder_arm"] = True

    # -------------------------------------------------------------
    # Group 4: Background & Scattering (V31 - V36)
    # -------------------------------------------------------------
    elif profile_id == "V31":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
    elif profile_id in ("V32", "V33", "V34", "V35", "V36"):
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        metadata["background_type"] = {
            "V32": "gas_scattering",
            "V33": "support_film_halo",
            "V34": "solvent_halo",
            "V35": "slit_streaks",
            "V36": "cosmic_ray",
        }[profile_id]

    # -------------------------------------------------------------
    # Group 5: Acquisition & Hit Finding (V37 - V42)
    # -------------------------------------------------------------
    elif profile_id == "V37":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.photon_budget = 1e7
        metadata["hit_status"] = "single_hit"
    elif profile_id == "V38":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.photon_budget = 5e4
        metadata["hit_status"] = "single_hit"
    elif profile_id == "V39":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        cfg.beam_center_offset = (0.0, 0.0)
        metadata["hit_status"] = "single_hit"
    elif profile_id == "V40":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        dy = float(rng.uniform(-8.0, 8.0))
        dx = float(rng.uniform(-8.0, 8.0))
        cfg.beam_center_offset = (dy, dx)
        metadata["hit_status"] = "off_center_hit"
        metadata["offset"] = (dy, dx)
    elif profile_id == "V41":
        # Multi-particle hit: 2 or 3 scatterers
        amplitude = make_multiple_scatterers(subtype="sparse", size=64, rng=rng)
        metadata["hit_status"] = "multi_particle_hit"
    elif profile_id == "V42":
        # Blank background non-hit
        amplitude = np.zeros((64, 64), dtype=np.float32)
        metadata["hit_status"] = "non_hit"

    # -------------------------------------------------------------
    # Group 6: Advanced Physics & OOD (V43 - V48)
    # -------------------------------------------------------------
    elif profile_id == "V43":
        # Pure phase object: amplitude is constant 1 across support
        support = _make_obj("particle_blob", 64, rng=rng) > 0.1
        amplitude = support.astype(np.float32)
        phase = generate_phase_field(size=64, phase_type="smooth", max_phase=0.4, rng=rng) * support
    elif profile_id == "V44":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        phase = generate_phase_field(size=64, phase_type="vortex", max_phase=np.pi, rng=rng)
    elif profile_id == "V45":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        metadata["fresnel_propagation_distance_m"] = 0.05
    elif profile_id == "V46":
        amplitude = _make_obj("particle_blob", 64, rng=rng)
        metadata["spectral_bandwidth_fraction"] = 0.08
    elif profile_id == "V47":
        # 3D particle scattering
        vol = generate_3d_volume(shape_type="cluster", size=64, rng=rng)
        angles = (
            float(rng.uniform(0, 2 * np.pi)),
            float(rng.uniform(0, np.pi)),
            float(rng.uniform(0, 2 * np.pi)),
        )
        metadata["euler_angles_rad"] = angles
        metadata["volume_3d"] = vol
        from opendiffraction.physics.models.particle_3d import project_volume_to_2d

        amplitude = project_volume_to_2d(vol, angles=angles)
    elif profile_id == "V48":
        amplitude = make_crack_void(64, rng=rng)
        cfg.photon_budget = 1e5
        cfg.coherence_length = 50.0
        cfg.include_module_gaps = True
        cfg.beam_center_offset = (float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)))
        metadata["is_adversarial_stress"] = True

    return cfg, amplitude, phase, metadata
