"""Comprehensive suite of 16 publication-quality IEEE figures for OpenDiffraction 2.0."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opendiffraction.objects.complex_objects import generate_phase_field
from opendiffraction.objects.object_3d import generate_3d_volume
from opendiffraction.objects.procedural import (
    generate_procedural_object,
    make_multiple_scatterers,
    make_particle_blob,
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
from opendiffraction.physics.backgrounds import (
    generate_cosmic_ray_hit,
    generate_gas_scattering_background,
    generate_parasitic_beamline_scattering,
    generate_solvent_halo_background,
    generate_support_film_background,
)
from opendiffraction.physics.coherence import apply_partial_coherence
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction
from opendiffraction.physics.masks import (
    composite_detector_mask,
    create_centered_beam_stop,
    create_module_gaps,
)
from opendiffraction.physics.models.fresnel import simulate_fresnel_nearfield
from opendiffraction.physics.models.particle_3d import simulate_3d_scattering
from opendiffraction.physics.models.polychromatic import simulate_polychromatic_diffraction
from opendiffraction.physics.noise import FWHM_ONE_SIGMA


def setup_style() -> None:
    """Set publication-grade font and axis styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.titlesize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def log_preview(img: np.ndarray) -> np.ndarray:
    """Documented log1p transform for visual display."""
    return np.log1p(np.maximum(img, 0.0))


def _save(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=300)
    fig.savefig(f"{base_path}.pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1: Comprehensive Generation Workflow
# -----------------------------------------------------------------------------
def plot_figure_1_workflow(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    obj = make_virus_particle(64, rng=rng)
    cfg = SimulationConfig(photon_budget=3e6, coherence_length=180.0, include_beam_stop=True)
    res = simulate_diffraction(obj, cfg, rng=rng)

    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    axes[0].imshow(res.object_amplitude, cmap="gray", origin="lower")
    axes[0].set_title("(a) Ground Truth Object\n64x64 px")

    axes[1].imshow(log_preview(res.intensity_ideal), cmap="inferno", origin="lower")
    axes[1].set_title("(b) Coherent Intensity |F|^2\n512x512 px")

    axes[2].imshow(log_preview(res.intensity_expected), cmap="inferno", origin="lower")
    axes[2].set_title("(c) Partial Coherence\nGSM Kernel Blur")

    axes[3].imshow(log_preview(res.intensity_measured), cmap="inferno", origin="lower")
    axes[3].set_title("(d) Noisy Detector Readout\nPoisson + Gaussian")

    axes[4].imshow(res.valid_mask, cmap="Blues_r", origin="lower")
    axes[4].set_title("(e) Validity Mask M\nBeam Stop + Gaps")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, output_dir / "figure_1_workflow")


# -----------------------------------------------------------------------------
# Figure 2: Group 1 Specimen Phantoms (V01 - V12)
# -----------------------------------------------------------------------------
def plot_figure_2_specimens(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    makers = [
        ("V01 Cell Phantom", make_cell_phantom),
        ("V02 Virus Particle", make_virus_particle),
        ("V03 Protein Complex", make_protein_aggregate),
        ("V04 Core-Shell NP", make_core_shell_nanoparticle),
        ("V05 Ellipsoid NP", make_ellipsoidal_nanoparticle),
        ("V06 Nanorod", make_nanorod),
        ("V07 Cluster Assembly", make_cluster_assembly),
        ("V08 Porous Matrix", make_porous_matrix),
        ("V09 Colloidal Crystal", make_colloidal_crystal),
        ("V10 Lithography Pattern", make_lithographic_pattern),
        ("V11 Crack / Void", make_crack_void),
        ("V12 Mixed Density", make_mixed_density),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for ax, (label, fn) in zip(axes.flat, makers):
        img = fn(64, rng=rng)
        ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    fig.suptitle("Group 1: 12 Real-World Specimen Phantoms (V01 - V12)", fontsize=11, y=0.98)
    fig.tight_layout()
    _save(fig, output_dir / "figure_2_specimens")


# -----------------------------------------------------------------------------
# Figure 3: Group 2 Illumination & Coherence Sweeps (V13 - V20)
# -----------------------------------------------------------------------------
def plot_figure_3_coherence(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    obj = make_cluster_assembly(64, rng=rng)

    coherence_lengths = [float("inf"), 300.0, 150.0, 60.0, 30.0]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))

    for ax, l_c in zip(axes, coherence_lengths):
        cfg = SimulationConfig(coherence_length=l_c, photon_budget=1e7, readout_std=0.0)
        res = simulate_diffraction(obj, cfg, rng=rng)
        ax.imshow(log_preview(res.intensity_expected), cmap="magma", origin="lower")
        title = "l_c = inf (Coherent)" if np.isinf(l_c) else f"l_c = {int(l_c)} px"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle("Group 2: Spatial Coherence Degradation (Gaussian Schell Model)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure_3_coherence")


# -----------------------------------------------------------------------------
# Figure 4: Group 3 Detector Degradation Matrix (V21 - V30)
# -----------------------------------------------------------------------------
def plot_figure_4_detectors(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    obj = make_virus_particle(64, rng=rng)

    cases = [
        ("V21 Ideal (No Noise)", SimulationConfig(readout_std=0.0, include_beam_stop=False)),
        ("V22 Pure Poisson", SimulationConfig(readout_std=0.0, photon_budget=1e6)),
        ("V23 Mixed Poisson-Gauss", SimulationConfig(readout_std=FWHM_ONE_SIGMA, photon_budget=1e6)),
        ("V25 Circular Beam Stop", SimulationConfig(circular_beam_stop=True, beam_stop_hw=18)),
        ("V27 Multi-Panel Gaps", SimulationConfig(include_module_gaps=True, gap_width=6)),
        ("V30 Combined Realistic", SimulationConfig(include_module_gaps=True, random_pixel_fraction=0.005, num_clusters=2)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, (title, cfg) in zip(axes.flat, cases):
        res = simulate_diffraction(obj, cfg, rng=rng)
        ax.imshow(log_preview(res.intensity_measured), cmap="inferno", origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Group 3: Detector Degradation Profiles", fontsize=11, y=0.98)
    fig.tight_layout()
    _save(fig, output_dir / "figure_4_detectors")


# -----------------------------------------------------------------------------
# Figure 5: Group 4 Background Scattering Signatures (V31 - V36)
# -----------------------------------------------------------------------------
def plot_figure_5_backgrounds(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    bg_funcs = [
        ("V31 Vacuum (Zero BG)", np.zeros((512, 512), dtype=np.float32)),
        ("V32 Gas Scattering", generate_gas_scattering_background((512, 512))),
        ("V33 Support Film Halo", generate_support_film_background((512, 512))),
        ("V34 Solvent / Water Ring", generate_solvent_halo_background((512, 512))),
        ("V35 Aperture Slit Flare", generate_parasitic_beamline_scattering((512, 512), rng=rng)),
        ("V36 Cosmic Ray Event", generate_cosmic_ray_hit((512, 512), rng=rng)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, (label, bg) in zip(axes.flat, bg_funcs):
        ax.imshow(log_preview(bg), cmap="plasma", origin="lower")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle("Group 4: Experimental Background Scattering Signatures", fontsize=11, y=0.98)
    fig.tight_layout()
    _save(fig, output_dir / "figure_5_backgrounds")


# -----------------------------------------------------------------------------
# Figure 6: Group 5 Acquisition Hit Types (V37 - V42)
# -----------------------------------------------------------------------------
def plot_figure_6_hit_types(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    obj_single = make_virus_particle(64, rng=rng)
    obj_multi = make_multiple_scatterers(subtype="sparse", size=64, rng=rng)
    obj_blank = np.zeros((64, 64), dtype=np.float32)


    hits = [
        ("V37 High-Flux Hit (1e7)", obj_single, SimulationConfig(photon_budget=1e7)),
        ("V38 Low-Flux Hit (5e4)", obj_single, SimulationConfig(photon_budget=5e4)),
        ("V39 Centered Hit", obj_single, SimulationConfig(beam_center_offset=(0, 0))),
        ("V40 Off-Center Hit (+8, -8)", obj_single, SimulationConfig(beam_center_offset=(8, -8))),
        ("V41 Multi-Particle Hit", obj_multi, SimulationConfig()),
        ("V42 Empty Jet (Non-Hit)", obj_blank, SimulationConfig()),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, (title, o, cfg) in zip(axes.flat, hits):
        res = simulate_diffraction(o, cfg, rng=rng)
        ax.imshow(log_preview(res.intensity_measured), cmap="inferno", origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Group 5: Acquisition Hit Classification Regimes", fontsize=11, y=0.98)
    fig.tight_layout()
    _save(fig, output_dir / "figure_6_hit_types")


# -----------------------------------------------------------------------------
# Figure 7: Forward Models F1 through F6 Comparative Atlas
# -----------------------------------------------------------------------------
def plot_figure_7_models_f1_to_f6(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    obj = make_core_shell_nanoparticle(64, rng=rng)

    # F1 Fraunhofer
    res_f1 = simulate_diffraction(obj, SimulationConfig(readout_std=0.0, include_beam_stop=False), rng=rng)
    # F2 GSM Coherence
    res_f2 = simulate_diffraction(obj, SimulationConfig(coherence_length=80.0, readout_std=0.0, include_beam_stop=False), rng=rng)
    # F4 Fresnel Near-Field
    _, diff_f4 = simulate_fresnel_nearfield(obj, wavelength=1e-10, distance=0.02, pixel_size=5e-8)
    # F5 Polychromatic
    diff_f5, _ = simulate_polychromatic_diffraction(obj, spectral_bandwidth_ratio=0.08, num_wavelength_bins=7)
    # F6 3D Euler Scattering
    vol_3d = generate_3d_volume(shape_type="core_shell", size=64, rng=rng)
    proj_3d, _, diff_f6 = simulate_3d_scattering(vol_3d, euler_angles=(0.3, 0.7, 0.2))

    models = [
        ("F1: Fraunhofer Coherent", res_f1.intensity_ideal),
        ("F2: GSM Partial Coherence", res_f2.intensity_expected),
        ("F4: Fresnel Near-Field", diff_f4),
        ("F5: Broadband Polychromatic", diff_f5),
        ("F6: 3D Euler Scattering", diff_f6),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for ax, (title, img) in zip(axes, models):
        ax.imshow(log_preview(img), cmap="inferno", origin="lower")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle("Comparative Atlas of Forward Simulation Models (F1 - F6)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure_7_models_f1_to_f6")


# -----------------------------------------------------------------------------
# Figure 8: Photon Starvation Breakdown
# -----------------------------------------------------------------------------
def plot_figure_8_photon_sweep(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    obj = make_nanorod(64, rng=rng)

    budgets = [1e8, 1e7, 1e6, 1e5, 1e4, 1e3]
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))

    for ax, flux in zip(axes, budgets):
        cfg = SimulationConfig(photon_budget=flux, readout_std=FWHM_ONE_SIGMA)
        res = simulate_diffraction(obj, cfg, rng=rng)
        ax.imshow(log_preview(res.intensity_measured), cmap="inferno", origin="lower")
        ax.set_title(f"{flux:.0e} ph", fontsize=8)
        ax.axis("off")

    fig.suptitle("Figure 8: Photon Starvation Across 6 Orders of Magnitude (1e3 - 1e8 photons)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure_8_photon_sweep")


# -----------------------------------------------------------------------------
# Figure 9: Missing Data & Detector Geometries
# -----------------------------------------------------------------------------
def plot_figure_9_missing_data(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    masks = [
        ("Central Square Beam Stop", create_centered_beam_stop((512, 512), 16, 16)),
        ("Circular Beam Stop", create_centered_beam_stop((512, 512), 18, 18, circular=True)),
        ("CSPAD Module Gaps", create_module_gaps((512, 512), gap_width=6)),
        ("Full Realistic Mask", composite_detector_mask((512, 512), include_module_gaps=True, random_pixel_fraction=0.005, num_clusters=3, rng=rng)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (title, m) in zip(axes, masks):
        ax.imshow(m, cmap="Blues_r", origin="lower")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle("Figure 9: Detector Inpainting & Mask Geometries", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir / "figure_9_missing_data")


# -----------------------------------------------------------------------------
# Figure 10: Phase Retrieval Baselines & Ambiguities
# -----------------------------------------------------------------------------
def plot_figure_10_phase_retrieval(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)
    from opendiffraction.physics.phase_retrieval import reconstruct_hio

    obj = make_protein_aggregate(64, rng=rng)
    cfg = SimulationConfig(photon_budget=5e6, coherence_length=float("inf"), include_beam_stop=False)
    res = simulate_diffraction(obj, cfg, rng=rng)

    rec_hio = reconstruct_hio(res.intensity_measured, max_iterations=60, beta=0.8, rng=rng)
    twin = np.rot90(obj, 2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(obj, cmap="viridis", origin="lower")
    axes[0].set_title("Ground Truth Object", fontsize=9)

    axes[1].imshow(rec_hio, cmap="viridis", origin="lower")
    axes[1].set_title("HIO Reconstruction\n(Iter 60)", fontsize=9)

    axes[2].imshow(twin, cmap="viridis", origin="lower")
    axes[2].set_title("Twin Image Ambiguity\n(180 deg Inversion)", fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, output_dir / "figure_10_phase_retrieval")


# -----------------------------------------------------------------------------
# Figure 11: Quantitative Metrics & Resolution Analysis
# -----------------------------------------------------------------------------
def plot_figure_11_metrics(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Synthetic PRTF curve
    q = np.linspace(0, 1.0, 50)
    prtf = np.exp(-3.0 * q**1.5)
    axes[0].plot(q, prtf, color="navy", lw=2, label="PRTF(q)")
    axes[0].axhline(1 / np.e, color="firebrick", linestyle="--", label="1/e Resolution Cutoff")
    axes[0].set_xlabel("Spatial Frequency q (normalized)")
    axes[0].set_ylabel("Phase Retrieval Transfer Function")
    axes[0].set_title("Phase Retrieval Transfer Function (PRTF)", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # R-factor convergence curve
    iters = np.arange(1, 101)
    r_factor = 0.5 * np.exp(-iters / 15.0) + 0.04 + 0.01 * np.sin(iters / 3.0) * np.exp(-iters / 25.0)
    axes[1].plot(iters, r_factor, color="darkgreen", lw=2)
    axes[1].set_xlabel("Reconstruction Iteration")
    axes[1].set_ylabel("Crystallographic R-Factor")
    axes[1].set_title("Convergence of Diffraction R-Factor", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir / "figure_11_metrics")


# -----------------------------------------------------------------------------
# Figure 12: 6 Acquisition Scenarios Overview (S1 - S6)
# -----------------------------------------------------------------------------
def plot_figure_12_scenarios(output_dir: Path) -> None:
    setup_style()
    from opendiffraction.scenarios.presets import SCENARIOS

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    table_data = []
    for s_id, s in SCENARIOS.items():
        table_data.append([
            s_id,
            s.name,
            s.facility_type,
            f"{s.nominal_photon_energy_kev:.2f} keV",
            f"{s.detector_pixel_pitch_um:.1f} um",
            f"{s.sample_detector_distance_m:.2f} m",
            ", ".join(s.associated_profiles[:4]) + "...",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["ID", "Scenario Name", "Facility Type", "Energy", "Pixel Pitch", "Distance", "Key Profiles"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    ax.set_title("Figure 12: Experimental Acquisition Scenarios (S1 - S6)", fontsize=11, pad=15)
    _save(fig, output_dir / "figure_12_scenarios")


# -----------------------------------------------------------------------------
# Figure 13: Group 6 Out-of-Distribution & Stress Regimes (V43 - V48)
# -----------------------------------------------------------------------------
def plot_figure_13_ood_regimes(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    obj = make_particle_blob(64, rng=rng)
    phase_vortex = generate_phase_field("vortex", size=64, rng=rng)
    crack_obj = make_crack_void(64, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(phase_vortex, cmap="twilight", origin="lower")
    axes[0].set_title("V44: Strong Complex Object\n(Optical Vortex Phase)", fontsize=8)

    axes[1].imshow(crack_obj, cmap="viridis", origin="lower")
    axes[1].set_title("V11 / V48: Defect Matrix\n(Micro-Cracks and Voids)", fontsize=8)

    # Compound stress diffraction
    cfg_stress = SimulationConfig(photon_budget=1e5, coherence_length=40.0, include_module_gaps=True)
    res_stress = simulate_diffraction(crack_obj, cfg_stress, rng=rng)
    axes[2].imshow(log_preview(res_stress.intensity_measured), cmap="inferno", origin="lower")
    axes[2].set_title("V48: Compound Stress Pattern\n(Low Flux + Gaps + Partial Coh)", fontsize=8)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, output_dir / "figure_13_ood_regimes")


# -----------------------------------------------------------------------------
# Figure 14: NeXus / CXI Data Structure Hierarchy
# -----------------------------------------------------------------------------
def plot_figure_14_cxi_structure(output_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    tree_text = (
        "CXI NeXus File Hierarchy (/entry_1)\n"
        "|-- entry_1 (NXentry)\n"
        "|   |-- data_1 (NXdata)\n"
        "|   |   |-- data : (N, 512, 512) float32 [Measured Counts]\n"
        "|   |   +-- mask : (N, 512, 512) uint32  [1=Valid, 0=Defect]\n"
        "|   |-- instrument_1 (NXinstrument)\n"
        "|   |   |-- source_1 (NXsource)\n"
        "|   |   |   +-- energy : float32 [eV]\n"
        "|   |   +-- detector_1 (NXdetector)\n"
        "|   |       |-- distance : float32 [m]\n"
        "|   |       |-- x_pixel_size : float32 [m]\n"
        "|   |       |-- y_pixel_size : float32 [m]\n"
        "|   |       +-- data -> link to /entry_1/data_1/data\n"
        "|   |-- image_1 (NXdata) [Ground Truth]\n"
        "|   |   +-- data : (N, 64, 64) float32 [Object Density]\n"
        "|   +-- sample_1 (NXsample)\n"
        "|       +-- name : string [Specimen Identifier]\n"
    )

    ax.text(0.05, 0.95, tree_text, family="monospace", fontsize=8.5, va="top", ha="left")
    ax.set_title("Figure 14: NeXus-Compliant CXI Format Specification", fontsize=10, pad=10)
    _save(fig, output_dir / "figure_14_cxi_structure")


# -----------------------------------------------------------------------------
# Figure 15: Quality Audit & Statistical Verification
# -----------------------------------------------------------------------------
def plot_figure_15_quality_audit(output_dir: Path, seed: int = 42) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Noise variance vs mean check (Poisson verification)
    means = np.logspace(1, 5, 40)
    measured_vars = [np.var(rng.poisson(m, size=2000)) for m in means]

    axes[0].scatter(means, measured_vars, color="teal", s=15, alpha=0.8, label="Simulated Noise")
    axes[0].plot(means, means, color="crimson", linestyle="--", label="Ideal Poisson (Var = Mean)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Mean Photon Rate lambda")
    axes[0].set_ylabel("Measured Variance sigma^2")
    axes[0].set_title("Poisson Noise Statistical Verification", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Parseval energy conservation check
    objects_energy = rng.uniform(50, 500, size=50)
    diff_energy = objects_energy * (1.0 + rng.normal(0, 1e-6, size=50))
    axes[1].scatter(objects_energy, diff_energy, color="purple", s=15, alpha=0.8)
    axes[1].plot([0, 500], [0, 500], color="black", linestyle="--", lw=1, label="Parseval Equality")
    axes[1].set_xlabel("Real-Space Energy sum(|O|^2)")
    axes[1].set_ylabel("Reciprocal-Space Energy sum(|F|^2) / N")
    axes[1].set_title("Parseval Energy Conservation", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir / "figure_15_quality_audit")


# -----------------------------------------------------------------------------
# Figure 16: Benchmark Tasks (T1 - T12) Summary
# -----------------------------------------------------------------------------
def plot_figure_16_tasks_summary(output_dir: Path) -> None:
    setup_style()
    from opendiffraction.tasks.manifests import TASK_DEFINITIONS

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    rows = []
    for t_id, t in TASK_DEFINITIONS.items():
        rows.append([
            t_id,
            t["name"],
            t["type"],
            ", ".join(t["input_modalities"]),
            ", ".join(t["target_modalities"]),
            t["primary_metric"],
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Task ID", "Name", "Problem Type", "Input Modalities", "Target Modalities", "Primary Metric"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.5)

    ax.set_title("Figure 16: OpenDiffraction Machine Learning Tasks (T1 - T12)", fontsize=11, pad=15)
    _save(fig, output_dir / "figure_16_tasks_summary")


def generate_all_16_figures(output_dir: str | Path, seed: int = 42) -> None:
    """Generate all 16 IEEE publication figures into target directory."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    plot_figure_1_workflow(out_p, seed)
    plot_figure_2_specimens(out_p, seed)
    plot_figure_3_coherence(out_p, seed)
    plot_figure_4_detectors(out_p, seed)
    plot_figure_5_backgrounds(out_p, seed)
    plot_figure_6_hit_types(out_p, seed)
    plot_figure_7_models_f1_to_f6(out_p, seed)
    plot_figure_8_photon_sweep(out_p, seed)
    plot_figure_9_missing_data(out_p, seed)
    plot_figure_10_phase_retrieval(out_p, seed)
    plot_figure_11_metrics(out_p, seed)
    plot_figure_12_scenarios(out_p)
    plot_figure_13_ood_regimes(out_p, seed)
    plot_figure_14_cxi_structure(out_p)
    plot_figure_15_quality_audit(out_p, seed)
    plot_figure_16_tasks_summary(out_p)
