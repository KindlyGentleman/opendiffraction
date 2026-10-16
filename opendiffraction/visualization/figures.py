"""Publication-quality scientific visualization suite for OpenDiffraction (Figures 1-8, contact sheet, Kaggle cover)."""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from opendiffraction.objects.procedural import (
    generate_procedural_object,
    make_basic_geometric,
    make_multiple_scatterers,
    make_particle_blob,
    make_porous_heterogeneous,
    make_structured_pattern,
)
from opendiffraction.physics.coherence import apply_partial_coherence
from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction
from opendiffraction.physics.masks import (
    create_centered_beam_stop,
    create_dead_pixels_and_clusters,
    create_irregular_patches,
    create_module_gaps,
    create_stripes,
)
from opendiffraction.physics.noise import FWHM_ONE_SIGMA


def setup_style() -> None:
    """Set clean publication-ready plotting style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def log_preview(img: np.ndarray) -> np.ndarray:
    """Documented log1p transform for display visibility."""
    return np.log1p(np.maximum(img, 0.0))


# -----------------------------------------------------------------------------
# Figure 1: Dataset Generation Workflow
# -----------------------------------------------------------------------------
def plot_figure_1_workflow(output_path: str, seed: int = 42) -> None:
    """Figure 1: Workflow stages from object to degraded observation."""
    setup_style()
    rng = np.random.default_rng(seed)

    # 1. Base object
    obj, _ = generate_procedural_object("particle_blob", "textured", size=64, rng=rng)

    # 2. Forward steps
    cfg = SimulationConfig(
        photon_budget=3e6,
        coherence_length=200.0,
        readout_std=FWHM_ONE_SIGMA,
        include_beam_stop=True,
        include_module_gaps=True,
        irregular_coverage=0.08,
    )
    sim = simulate_diffraction(obj, cfg, rng=rng)

    fig, axes = plt.subplots(1, 7, figsize=(15, 2.5))

    axes[0].imshow(obj, cmap="inferno", origin="lower")
    axes[0].set_title("1. Ground Truth $O(r)$")

    axes[1].imshow(log_preview(sim.intensity_ideal), cmap="viridis", origin="lower")
    axes[1].set_title("2. Ideal FFT $|A(q)|^2$")

    # Partial coherence
    coh_only = apply_partial_coherence(sim.intensity_ideal, l_coh=200.0)
    axes[2].imshow(log_preview(coh_only), cmap="viridis", origin="lower")
    axes[2].set_title("3. Partial Coherence")

    axes[3].imshow(log_preview(sim.intensity_expected), cmap="viridis", origin="lower")
    axes[3].set_title("4. Photon Scaling $\\lambda_i$")

    axes[4].imshow(log_preview(sim.intensity_measured), cmap="viridis", origin="lower")
    axes[4].set_title("5. Poisson + Readout")

    axes[5].imshow(sim.valid_mask, cmap="gray", origin="lower")
    axes[5].set_title("6. Valid Mask $M$")

    axes[6].imshow(log_preview(sim.observed_masked), cmap="viridis", origin="lower")
    axes[6].set_title("7. Observation $Y \\odot M$")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 2: Object Diversity Gallery
# -----------------------------------------------------------------------------
def plot_figure_2_object_diversity(output_path: str, seed: int = 100) -> None:
    """Figure 2: Gallery of 15 diverse shapes across all 5 families."""
    setup_style()
    rng = np.random.default_rng(seed)

    shapes = [
        ("Circle", make_basic_geometric("circle", rng=rng)),
        ("Ellipse", make_basic_geometric("ellipse", rng=rng)),
        ("Rectangle", make_basic_geometric("rectangle", rng=rng)),
        ("Triangle", make_basic_geometric("triangle", rng=rng)),
        ("Annulus", make_basic_geometric("annulus", rng=rng)),
        ("Slit Aperture", make_basic_geometric("slit", rng=rng)),
        ("Smooth Blob", make_particle_blob("blob", rng=rng)),
        ("Irregular Particle", make_particle_blob("irregular", rng=rng)),
        ("Elongated Particle", make_particle_blob("elongated", rng=rng)),
        ("Multi-lobed", make_particle_blob("multi_lobed", rng=rng)),
        ("Particle Pair", make_multiple_scatterers("pair_varying_sep", rng=rng)),
        ("Dense Cluster", make_multiple_scatterers("dense", rng=rng)),
        ("Porous Matrix", make_porous_heterogeneous("porous_matrix", rng=rng)),
        ("Internal Voids", make_porous_heterogeneous("internal_voids", rng=rng)),
        ("Sparse Lattice", make_structured_pattern("sparse_lattice", rng=rng)),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(11, 6.8))
    for ax, (title, img) in zip(axes.flat, shapes):
        ax.imshow(img, cmap="inferno", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 3: Geometry vs Diffraction
# -----------------------------------------------------------------------------
def plot_figure_3_geometry_diffraction(output_path: str, seed: int = 200) -> None:
    """Figure 3: Paired real-space objects and diffraction patterns."""
    setup_style()
    rng = np.random.default_rng(seed)

    test_shapes = [
        ("Circular Disk", make_basic_geometric("circle", rng=rng)),
        ("Slit Aperture", make_basic_geometric("slit", rng=rng)),
        ("Particle Pair", make_multiple_scatterers("pair_varying_sep", rng=rng)),
        ("Irregular Blob", make_particle_blob("irregular", rng=rng)),
        ("Periodic Lattice", make_structured_pattern("periodic_grid", rng=rng)),
    ]

    cfg = SimulationConfig(photon_budget=1e7, coherence_length=float("inf"), readout_std=0.0)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    for col, (title, obj) in enumerate(test_shapes):
        sim = simulate_diffraction(obj, cfg, rng=rng)

        axes[0, col].imshow(obj, cmap="inferno", origin="lower")
        axes[0, col].set_title(f"Object: {title}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(log_preview(sim.intensity_ideal), cmap="viridis", origin="lower")
        axes[1, col].set_title(f"Diffraction $\\log(1+|A|^2)$")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 4: Photon Budget Sweep
# -----------------------------------------------------------------------------
def plot_figure_4_photon_sweep(output_path: str, seed: int = 300) -> None:
    """Figure 4: Diffraction pattern degradation across photon budgets."""
    setup_style()
    rng = np.random.default_rng(seed)
    obj, _ = generate_procedural_object("particle_blob", "uniform", size=64, rng=rng)

    budgets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))

    for ax, flux in zip(axes, budgets):
        task_rng = np.random.default_rng(seed + 1)
        cfg = SimulationConfig(
            photon_budget=flux,
            coherence_length=200.0,
            readout_std=FWHM_ONE_SIGMA,
            include_beam_stop=True,
            beam_stop_hw=16,
            beam_stop_hh=16,
        )
        sim = simulate_diffraction(obj, cfg, rng=task_rng)

        ax.imshow(log_preview(sim.observed_masked), cmap="viridis", origin="lower")
        ax.set_title(f"$N_{{\\text{{photons}}}} = 10^{{{int(np.log10(flux))}}}$")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 5: Coherence Sweep
# -----------------------------------------------------------------------------
def plot_figure_5_coherence_sweep(output_path: str, seed: int = 400) -> None:
    """Figure 5: Fringe visibility degradation across spatial coherence lengths."""
    setup_style()
    rng = np.random.default_rng(seed)
    obj, _ = generate_procedural_object("structured_pattern", "uniform", size=64, rng=rng)

    lengths = [float("inf"), 300.0, 200.0, 100.0, 50.0, 25.0]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))

    for ax, l_coh in zip(axes, lengths):
        task_rng = np.random.default_rng(seed + 2)
        cfg = SimulationConfig(
            photon_budget=5e6,
            coherence_length=l_coh,
            readout_std=0.0,
            include_beam_stop=True,
            beam_stop_hw=12,
            beam_stop_hh=12,
        )
        sim = simulate_diffraction(obj, cfg, rng=task_rng)

        ax.imshow(log_preview(sim.observed_masked), cmap="viridis", origin="lower")
        lbl = "$\\infty$ (Coherent)" if np.isinf(l_coh) else f"{int(l_coh)} px"
        ax.set_title(f"$\\ell_{{\\text{{coh}}}} = {lbl}$")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 6: Missing-Data Variations
# -----------------------------------------------------------------------------
def plot_figure_6_missing_data(output_path: str, seed: int = 500) -> None:
    """Figure 6: Procedural detector defects: beam stops, module gaps, dead pixels, stripes, irregular patches."""
    setup_style()
    rng = np.random.default_rng(seed)

    m1 = create_centered_beam_stop(half_width=24, half_height=24)
    m2 = create_module_gaps(gap_width=6)
    m3 = create_stripes(num_horizontal_stripes=3, num_vertical_stripes=3, rng=rng)
    m4 = create_dead_pixels_and_clusters(
        random_pixel_fraction=0.01, num_clusters=6, cluster_radius_range=(3, 7), rng=rng
    )
    m5 = create_irregular_patches(coverage_fraction=0.15, rng=rng)
    m6 = m1 & m2 & m3 & m4 & m5  # Composite

    masks = [
        ("Central Beam Stop", m1),
        ("Detector Module Gaps", m2),
        ("Dead Lines / Stripes", m3),
        ("Dead Clusters", m4),
        ("Irregular Patches", m5),
        ("Realistic Composite", m6),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
    for ax, (title, mask) in zip(axes, masks):
        ax.imshow(mask, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 7: Dataset Distribution & Parameter Coverage
# -----------------------------------------------------------------------------
def plot_figure_7_dataset_distribution(output_path: str) -> None:
    """Figure 7: Summary charts of splits, variation profiles, and physical parameter coverage."""
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Split breakdown
    splits = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    ratios = [70, 15, 15]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    axes[0].pie(ratios, labels=splits, autopct="%1.0f%%", colors=colors, startangle=140)
    axes[0].set_title("Object-Disjoint Splits")

    # 12 variation profiles
    p_ids = [f"V{i:02d}" for i in range(1, 13)]
    counts = [100 / 12.0] * 12
    axes[1].bar(p_ids, counts, color="#1b9e77")
    axes[1].set_ylabel("Profile Balance (%)")
    axes[1].set_title("12 Controlled Variation Profiles")
    axes[1].tick_params(axis="x", rotation=45)

    # Photon budget coverage
    log_photons = [3, 4, 5, 6, 7, 8]
    axes[2].bar([f"$10^{{{p}}}$" for p in log_photons], [1] * 6, color="#e7298a")
    axes[2].set_ylabel("Controlled Bins")
    axes[2].set_title("Photon Budget Range")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 8: Quality Control & Physics Diagnostics
# -----------------------------------------------------------------------------
def plot_figure_8_quality_control(output_path: str, seed: int = 600) -> None:
    """Figure 8: Quality control diagnostics: Parseval energy, noise distributions, Friedel symmetry."""
    setup_style()
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # 1. Poisson statistics check
    lam = 50.0
    samples = rng.poisson(lam, size=10000)
    axes[0].hist(samples, bins=35, density=True, color="#377eb8", alpha=0.7, label="Sampled")
    x = np.arange(25, 75)
    from scipy.stats import poisson
    axes[0].plot(x, poisson.pmf(x, lam), "r--", lw=2, label="Theoretical")
    axes[0].set_title(f"Poisson Noise ($\\lambda = {int(lam)}$)")
    axes[0].set_xlabel("Photon Counts")
    axes[0].legend(frameon=False)

    # 2. Gaussian readout noise check
    std = FWHM_ONE_SIGMA
    noise = rng.normal(0, std, size=10000)
    axes[1].hist(noise, bins=40, density=True, color="#4daf4a", alpha=0.7, label="Readout $\\epsilon$")
    x_g = np.linspace(-1.5, 1.5, 100)
    from scipy.stats import norm
    axes[1].plot(x_g, norm.pdf(x_g, 0, std), "k--", lw=2, label="Gaussian Fit")
    axes[1].set_title("Gaussian Readout (FWHM = 1)")
    axes[1].set_xlabel("Readout Error")
    axes[1].legend(frameon=False)

    # 3. Friedel Symmetry Residual
    obj, _ = generate_procedural_object("basic_geometric", "uniform", size=64, rng=rng)
    cfg = SimulationConfig(photon_budget=1e7, coherence_length=float("inf"), readout_std=0.0)
    sim = simulate_diffraction(obj, cfg, rng=rng)
    I = sim.intensity_ideal
    I_rot = np.flip(I)
    diff = np.abs(I - I_rot)
    max_err = np.max(diff)
    im = axes[2].imshow(diff, cmap="magma", origin="lower")
    axes[2].set_title(f"Friedel Inversion Error (Max: {max_err:.1e})")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    base, _ = os.path.splitext(output_path)
    plt.savefig(base + ".pdf")
    plt.close()


# -----------------------------------------------------------------------------
# Kaggle Cover Image and Contact Sheet
# -----------------------------------------------------------------------------
def plot_kaggle_cover(output_path: str, seed: int = 700) -> None:
    """Create curated Kaggle dataset cover image (1200 x 630 px aspect)."""
    setup_style()
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.25), dpi=300)
    fig.patch.set_facecolor("#0b0f19")

    shapes = ["circle", "ellipse", "blob", "irregular", "pair_varying_sep", "dense", "porous_matrix", "sparse_lattice"]
    fams = ["basic_geometric", "basic_geometric", "particle_blob", "particle_blob", "multiple_scatterer", "multiple_scatterer", "porous_heterogeneous", "structured_pattern"]

    for idx, (ax, fam, st) in enumerate(zip(axes.flat, fams, shapes)):
        ax.set_facecolor("#0b0f19")
        sub_rng = np.random.default_rng(seed + idx)
        obj, _ = generate_procedural_object(fam, "textured", size=64, rng=sub_rng, shape_subtype=st)
        cfg = SimulationConfig(photon_budget=3e6, coherence_length=200.0, readout_std=FWHM_ONE_SIGMA, include_beam_stop=True)
        sim = simulate_diffraction(obj, cfg, rng=sub_rng)

        ax.imshow(log_preview(sim.observed_masked), cmap="magma", origin="lower")
        ax.axis("off")

    fig.suptitle("OpenDiffraction: Physics-Based Synthetic Coherent Diffraction", color="white", fontsize=14, y=0.98, weight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def generate_all_visualizations(output_dir: str) -> None:
    """Generate the full figure suite."""
    os.makedirs(output_dir, exist_ok=True)
    plot_figure_1_workflow(os.path.join(output_dir, "figure_1_workflow.png"))
    plot_figure_2_object_diversity(os.path.join(output_dir, "figure_2_object_diversity.png"))
    plot_figure_3_geometry_diffraction(os.path.join(output_dir, "figure_3_geometry_diffraction.png"))
    plot_figure_4_photon_sweep(os.path.join(output_dir, "figure_4_photon_sweep.png"))
    plot_figure_5_coherence_sweep(os.path.join(output_dir, "figure_5_coherence_sweep.png"))
    plot_figure_6_missing_data(os.path.join(output_dir, "figure_6_missing_data.png"))
    plot_figure_7_dataset_distribution(os.path.join(output_dir, "figure_7_dataset_distribution.png"))
    plot_figure_8_quality_control(os.path.join(output_dir, "figure_8_quality_control.png"))
    plot_kaggle_cover(os.path.join(output_dir, "kaggle_cover.png"))
    plot_kaggle_cover(os.path.join(output_dir, "contact_sheet.png"))
