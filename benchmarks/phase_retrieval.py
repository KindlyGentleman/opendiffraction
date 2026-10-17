"""Benchmark B1: Classical Phase Retrieval Baseline (HIO and Error Reduction)."""

from __future__ import annotations

import numpy as np
from opendiffraction.physics.phase_retrieval import hybrid_input_output, error_reduction
from opendiffraction.tasks.metrics import aligned_ssim, aligned_mse, compute_r_factor, compute_prtf


def compute_support_regime(
    detector_shape: tuple[int, int],
    regime: str = "disc",
    ground_truth_object: np.ndarray | None = None,
    measured_intensity: np.ndarray | None = None,
    disc_radius: float = 28.0,
    box_half_size: int = 32,
    dilation_px: int = 3,
) -> np.ndarray:
    """Construct real-space support constraint across benchmark regimes.

    Regimes:
    - 'disc': Centered circular disc (default radius 28.0 px).
    - 'box': Centered square bounding box (default 64x64 px).
    - 'oracle': Ground-truth object binarized and dilated by dilation_px.
    - 'autocorrelation': Thresholded inverse Fourier transform of measured intensity.
    """
    ny, nx = detector_shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]

    if regime == "disc":
        return ((y - cy) ** 2 + (x - cx) ** 2 <= disc_radius**2).astype(bool)
    elif regime == "box":
        mask = np.zeros(detector_shape, dtype=bool)
        mask[cy - box_half_size : cy + box_half_size, cx - box_half_size : cx + box_half_size] = True
        return mask
    elif regime == "oracle":
        if ground_truth_object is None:
            return ((y - cy) ** 2 + (x - cx) ** 2 <= disc_radius**2).astype(bool)
        gt_max = float(np.max(ground_truth_object))
        gt_bin = ground_truth_object > (0.05 * gt_max) if gt_max > 0 else (ground_truth_object > 0)
        from scipy.ndimage import binary_dilation

        dilated = binary_dilation(gt_bin, iterations=dilation_px) if dilation_px > 0 else gt_bin
        gy, gx = dilated.shape
        mask = np.zeros(detector_shape, dtype=bool)
        mask[cy - gy // 2 : cy - gy // 2 + gy, cx - gx // 2 : cx - gx // 2 + gx] = dilated
        return mask
    elif regime == "autocorrelation":
        if measured_intensity is None:
            return ((y - cy) ** 2 + (x - cx) ** 2 <= disc_radius**2).astype(bool)
        autocorr = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measured_intensity))))
        thresh = 0.05 * float(np.max(autocorr))
        return (autocorr > thresh).astype(bool)
    else:
        raise ValueError(
            f"Unknown support regime '{regime}'. Choose 'disc', 'box', 'oracle', or 'autocorrelation'."
        )


def evaluate_phase_retrieval(
    measured_intensity: np.ndarray,
    ground_truth_object: np.ndarray,
    support_mask: np.ndarray | None = None,
    support_regime: str = "disc",
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
        Real-space support constraint. If None, derived via support_regime.
    support_regime : str
        Support constraint mode ('disc', 'box', 'oracle', 'autocorrelation').
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
        support_mask = compute_support_regime(
            detector_shape=measured_intensity.shape,
            regime=support_regime,
            ground_truth_object=ground_truth_object,
            measured_intensity=measured_intensity,
        )

    # Step 1: HIO exploration with mask-aware Fourier constraint
    hio_recon, _ = hybrid_input_output(
        modulus,
        support_mask,
        valid_mask=valid_mask,
        num_iterations=num_hio_iters,
        beta=beta,
        rng=rng,
    )

    # Step 2: ER refinement initialized directly with HIO real-space iterate
    # (eliminating coordinate conversion and uncentered Fourier phase scrambling)
    er_recon, er_errors = error_reduction(
        modulus,
        support_mask,
        valid_mask=valid_mask,
        num_iterations=num_er_iters,
        initial_object=hio_recon,
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
        "reconstructed": cropped,
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
