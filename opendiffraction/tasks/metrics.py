"""Scientific machine-learning benchmark evaluation metrics for phase retrieval and inverse imaging."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift


def align_twin_and_translation(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Resolve 2D phase-retrieval ambiguities: twin image (180 deg) and sub-pixel translation.

    In CDI without support asymmetry, reconstruction has inherent twin (conjugate reflection)
    and translation symmetries. This function tests both standard and rotated candidates,
    computes cross-correlation for sub-pixel shift, and returns the best-aligned candidate.
    """
    candidates = [reconstructed, np.rot90(reconstructed, 2)]
    best_candidate = candidates[0]
    best_corr = -float("inf")
    best_shift = (0.0, 0.0)

    try:
        from skimage.registration import phase_cross_correlation

        for cand in candidates:
            sh, _, _ = phase_cross_correlation(ground_truth, cand, upsample_factor=10)
            al = shift(cand, shift=sh, mode="wrap", order=1)
            corr_val = float(np.sum(al * ground_truth))
            if corr_val > best_corr:
                best_corr = corr_val
                best_candidate = cand
                best_shift = tuple(sh)
    except ImportError:
        for cand in candidates:
            # Cross correlation via FFT
            F_cand = np.fft.fft2(cand)
            F_gt = np.fft.fft2(ground_truth)
            xcorr = np.fft.ifft2(F_cand * np.conj(F_gt)).real
            max_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)

            sh_y = max_idx[0] if max_idx[0] < cand.shape[0] // 2 else max_idx[0] - cand.shape[0]
            sh_x = max_idx[1] if max_idx[1] < cand.shape[1] // 2 else max_idx[1] - cand.shape[1]

            corr_val = xcorr[max_idx]
            if corr_val > best_corr:
                best_corr = corr_val
                best_candidate = cand
                best_shift = (-sh_y, -sh_x)

    aligned = shift(best_candidate, shift=best_shift, mode="wrap", order=1)
    return aligned.astype(np.float32)


def aligned_mse(reconstructed: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Squared Error (MSE) after resolving twin image and spatial shift."""
    aligned = align_twin_and_translation(reconstructed, ground_truth)
    return float(np.mean((aligned - ground_truth) ** 2))


def ssim_2d(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
) -> float:
    """Compute standard 2D Structural Similarity Index (SSIM).

    Uses skimage.metrics.structural_similarity when scikit-image is available,
    with a fallback to standard 11x11 local sliding window filter (Wang et al., 2004).
    """
    try:
        from skimage.metrics import structural_similarity

        w = min(win_size, min(img1.shape))
        if w % 2 == 0:
            w -= 1
        return float(
            structural_similarity(
                img1.astype(np.float64),
                img2.astype(np.float64),
                data_range=float(data_range),
                win_size=max(w, 3),
            )
        )
    except ImportError:
        pass

    from scipy.ndimage import uniform_filter

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    w = min(win_size, min(img1.shape))
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)

    a1 = img1.astype(np.float64)
    a2 = img2.astype(np.float64)

    mu1 = uniform_filter(a1, size=w)
    mu2 = uniform_filter(a2, size=w)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(a1 * a1, size=w) - mu1_sq
    sigma2_sq = uniform_filter(a2 * a2, size=w) - mu2_sq
    sigma12 = uniform_filter(a1 * a2, size=w) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def aligned_ssim(reconstructed: np.ndarray, ground_truth: np.ndarray) -> float:
    """SSIM after resolving twin image and spatial shift."""
    aligned = align_twin_and_translation(reconstructed, ground_truth)
    d_range = float(max(np.ptp(ground_truth), 1e-4))
    return ssim_2d(aligned, ground_truth, data_range=d_range)


def compute_r_factor(
    simulated_intensity: np.ndarray,
    measured_intensity: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Crystallographic / CDI diffraction R-factor on amplitudes.

    R = sum(|sqrt(I_meas) - scale * sqrt(I_sim)|) / sum(sqrt(I_meas))
    evaluated strictly over valid (unmasked) pixels.
    """
    if mask is None:
        mask = np.ones_like(measured_intensity, dtype=bool)

    # Valid non-negative intensities
    valid = mask & (measured_intensity >= 0) & (simulated_intensity >= 0)
    if not np.any(valid):
        return 1.0

    amp_meas = np.sqrt(measured_intensity[valid])
    amp_sim = np.sqrt(simulated_intensity[valid])

    # Optimal scaling factor
    denom_scale = np.sum(amp_sim**2)
    scale = np.sum(amp_meas * amp_sim) / (denom_scale + 1e-12) if denom_scale > 0 else 1.0

    r_num = np.sum(np.abs(amp_meas - scale * amp_sim))
    r_den = np.sum(amp_meas)
    return float(r_num / (r_den + 1e-12))


def compute_prtf(
    reconstructions: list[np.ndarray],
    ground_truth: np.ndarray | None = None,
    radial_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase Retrieval Transfer Function (PRTF) as a function of spatial frequency q.

    PRTF(q) = |<F_rec(q)>| / <|F_rec(q)|>
    Measures reproducible phase consistency across independent reconstruction starts.
    """
    if len(reconstructions) == 0:
        return np.zeros(radial_bins), np.zeros(radial_bins)

    ny, nx = reconstructions[0].shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = min(cy, cx)

    # FFT of all reconstructions after twin alignment
    ref = ground_truth if ground_truth is not None else reconstructions[0]
    aligned_ffts = []
    for rec in reconstructions:
        al = align_twin_and_translation(rec, ref)
        aligned_ffts.append(np.fft.fftshift(np.fft.fft2(al)))

    stack = np.stack(aligned_ffts, axis=0)
    mean_complex = np.mean(stack, axis=0)
    mean_modulus = np.mean(np.abs(stack), axis=0)

    # Azimuthal radial averaging
    bin_edges = np.linspace(0, max_r, radial_bins + 1)
    q_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prtf_vals = np.zeros(radial_bins, dtype=np.float32)

    for i in range(radial_bins):
        in_bin = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(in_bin):
            num = np.sum(np.abs(mean_complex[in_bin]))
            den = np.sum(mean_modulus[in_bin])
            prtf_vals[i] = num / (den + 1e-12)

    return q_centers, prtf_vals


def psnr(clean: np.ndarray, noisy: np.ndarray, max_val: float | None = None) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((clean - noisy) ** 2)
    if mse <= 1e-12:
        return 100.0
    if max_val is None:
        max_val = float(np.max(clean))
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def intersection_over_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union (IoU / Jaccard Index) for binary defect segmentation."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def relative_mae(pred: float | np.ndarray, target: float | np.ndarray) -> float:
    """Relative Mean Absolute Error."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    denom = np.abs(t)
    denom[denom == 0] = 1e-8
    return float(np.mean(np.abs(p - t) / denom))
