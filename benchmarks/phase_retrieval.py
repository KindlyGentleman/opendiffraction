"""Benchmark B1: Classical Phase Retrieval Baseline (HIO and Error Reduction)."""

from __future__ import annotations

import numpy as np
from opendiffraction.physics.phase_retrieval import hybrid_input_output, error_reduction
from opendiffraction.tasks.metrics import aligned_ssim, aligned_mse, compute_r_factor, compute_prtf


def evaluate_phase_retrieval(
    measured_intensity: np.ndarray,
    ground_truth_object: np.ndarray,
    support_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    num_hio_iters: int = 40,
    num_er_iters: int = 20,
    beta: float = 0.8,
    seed: int = 42,
) -> dict[str, float]:
    """Execute standard HIO -> ER reconstruction pipeline and compute metrics.

    Parameters
    ----------
    measured_intensity : np.ndarray
        Observed diffraction pattern (512, 512).
    ground_truth_object : np.ndarray
        True transmission density (64, 64).
    support_mask : np.ndarray or None
        Real-space support constraint. If None, uses centered disc.
    valid_mask : np.ndarray or None
        Valid pixel detector mask.
    num_hio_iters : int
        Iterations of Hybrid Input-Output.
    num_er_iters : int
        Iterations of Error Reduction.
    beta : float
        HIO feedback constant.
    seed : int
        Random seed for initial phase.

    Returns
    -------
    dict[str, float]
        Dictionary of aligned SSIM, aligned MSE, and final R-factor.
    """
    rng = np.random.default_rng(seed)
    modulus = np.sqrt(np.maximum(measured_intensity, 0.0))

    if support_mask is None:
        ny, nx = measured_intensity.shape
        cy, cx = ny // 2, nx // 2
        y, x = np.ogrid[:ny, :nx]
        support_mask = (y - cy) ** 2 + (x - cx) ** 2 <= 28.0**2

    # Step 1: HIO exploration with mask-aware Fourier constraint
    hio_recon, _ = hybrid_input_output(
        modulus,
        support_mask,
        valid_mask=valid_mask,
        num_iterations=num_hio_iters,
        beta=beta,
        rng=rng,
    )

    # Step 2: ER refinement with mask-aware Fourier constraint
    initial_phase = np.angle(np.fft.fft2(np.fft.ifftshift(hio_recon)))
    er_recon, er_errors = error_reduction(
        modulus,
        support_mask,
        valid_mask=valid_mask,
        num_iterations=num_er_iters,
        initial_phase=initial_phase,
        rng=rng,
    )

    # Crop to object window (64, 64)
    ny, nx = er_recon.shape
    cy, cx = ny // 2, nx // 2
    cropped = er_recon[cy - 32 : cy + 32, cx - 32 : cx + 32]
    p_max = float(np.max(cropped))
    if p_max > 0:
        cropped = (cropped / p_max).astype(np.float32)

    # Scale-invariant normalization of ground-truth target
    gt_max = float(np.max(ground_truth_object))
    if gt_max > 0:
        gt_norm = (ground_truth_object / gt_max).astype(np.float32)
    else:
        gt_norm = ground_truth_object.astype(np.float32)

    ssim_val = aligned_ssim(cropped, gt_norm)
    mse_val = aligned_mse(cropped, gt_norm)
    r_factor = er_errors[-1] if er_errors else 0.0

    return {
        "aligned_ssim": float(ssim_val),
        "aligned_mse": float(mse_val),
        "final_r_factor": float(r_factor),
    }


if __name__ == "__main__":
    from opendiffraction.objects.specimens import make_virus_particle
    from opendiffraction.physics.forward import SimulationConfig, simulate_diffraction

    rng = np.random.default_rng(42)
    obj = make_virus_particle(64, rng=rng)
    cfg = SimulationConfig(photon_budget=5e6, coherence_length=float("inf"), include_beam_stop=False)
    sim = simulate_diffraction(obj, cfg, rng=rng)

    metrics = evaluate_phase_retrieval(sim.intensity_measured, obj, seed=42)
    print("Phase Retrieval Baseline Results:", metrics)
