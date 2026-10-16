"""Parameter sweep animation and GIF generation for OpenDiffraction."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from opendiffraction.objects.object_3d import generate_3d_volume
from opendiffraction.physics.coherence import apply_partial_coherence
from opendiffraction.physics.forward import simulate_diffraction
from opendiffraction.physics.models.particle_3d import simulate_3d_scattering


def generate_rotation_sweep_gif(
    output_path: str | Path,
    n_frames: int = 24,
    fps: int = 8,
    seed: int = 42,
) -> Path:
    """Generate an animated GIF of a 3D nanoparticle rotating around its vertical axis.

    Visualizes the 2D projected density and its corresponding far-field diffraction pattern.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    vol = generate_3d_volume(shape_type="cluster", size=64, rng=rng)

    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor("white")

    # Initial frame
    proj, _, diff = simulate_3d_scattering(vol, euler_angles=(0.0, float(angles[0]), 0.0))
    im_obj = axes[0].imshow(proj, cmap="viridis", origin="lower")
    axes[0].set_title("3D Projected Density", fontsize=10)
    axes[0].axis("off")

    im_diff = axes[1].imshow(np.log1p(diff), cmap="magma", origin="lower")
    axes[1].set_title(f"Diffraction Intensity (log1p)\nEuler Beta = {angles[0]:.2f} rad", fontsize=10)
    axes[1].axis("off")

    fig.tight_layout()

    def update(frame_idx: int) -> list:
        beta = float(angles[frame_idx])
        p, _, d = simulate_3d_scattering(vol, euler_angles=(0.0, beta, 0.0))
        im_obj.set_data(p)
        im_diff.set_data(np.log1p(d))
        axes[1].set_title(f"Diffraction Intensity (log1p)\nEuler Beta = {np.degrees(beta):.1f} deg", fontsize=10)
        return [im_obj, im_diff]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_p), writer=writer)
    plt.close(fig)

    return out_p


def generate_photon_sweep_gif(
    output_path: str | Path,
    n_frames: int = 20,
    fps: int = 5,
    seed: int = 42,
) -> Path:
    """Generate an animated GIF showing progressive photon starvation from 1e8 down to 1e3 photons."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    from opendiffraction.objects.specimens import make_virus_particle
    from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction

    rng = np.random.default_rng(seed)
    obj = make_virus_particle(64, rng=rng)

    budgets = np.logspace(8, 3, n_frames)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    cfg = SimulationConfig(photon_budget=float(budgets[0]), include_beam_stop=True)
    res = simulate_diffraction(obj, cfg, rng=np.random.default_rng(seed))

    im = ax.imshow(np.log1p(np.maximum(res.intensity_measured, 0)), cmap="inferno", origin="lower")
    title = ax.set_title(f"Photon Budget: {budgets[0]:.1e}", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    def update(frame_idx: int) -> list:
        b = float(budgets[frame_idx])
        c = SimulationConfig(photon_budget=b, include_beam_stop=True)
        r = simulate_diffraction(obj, c, rng=np.random.default_rng(seed + frame_idx))
        im.set_data(np.log1p(np.maximum(r.intensity_measured, 0)))
        title.set_text(f"Photon Budget: {b:.2e} photons")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_p), writer=writer)
    plt.close(fig)

    return out_p
