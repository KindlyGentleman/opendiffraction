"""Scientific machine-learning benchmark evaluation metrics for phase retrieval and inverse imaging."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift


def fourier_shift_2d(img: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Sub-pixel translation via Fourier shift theorem (exact for band-limited periodic signals)."""
    if abs(shift_y) < 1e-7 and abs(shift_x) < 1e-7:
        return img.copy()
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    phase = np.exp(-2j * np.pi * (ky * shift_y + kx * shift_x))
    shifted = np.real(np.fft.ifft2(np.fft.fft2(img) * phase))
    return shifted


def align_twin_and_translation(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Resolve 2D phase-retrieval ambiguities: twin image (180 deg) and sub-pixel translation.

    In CDI without support asymmetry, reconstruction has inherent twin (conjugate reflection)
    and translation symmetries. This function tests both standard and rotated candidates,
    computes cross-correlation for sub-pixel shift, and returns the best-aligned candidate
    using exact Fourier-domain subpixel translation.
    """
    candidates = [reconstructed, np.rot90(reconstructed, 2)]
    best_candidate = candidates[0]
    best_corr = -float("inf")
    best_shift = (0.0, 0.0)

    try:
        from skimage.registration import phase_cross_correlation

        for cand in candidates:
            sh, _, _ = phase_cross_correlation(ground_truth, cand, upsample_factor=50)
            sh_tuple = (float(sh[0]), float(sh[1]))
            al = fourier_shift_2d(cand, sh_tuple[0], sh_tuple[1])
            corr_val = float(np.sum(al * ground_truth))
            if corr_val > best_corr:
                best_corr = corr_val
                best_candidate = cand
                best_shift = sh_tuple
    except ImportError:
        for cand in candidates:
            # Cross correlation via FFT
            F_cand = np.fft.fft2(cand)
            F_gt = np.fft.fft2(ground_truth)
            xcorr = np.fft.ifft2(F_cand * np.conj(F_gt)).real
            max_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            y0, x0 = int(max_idx[0]), int(max_idx[1])
            ny, nx = cand.shape

            # Sub-pixel parabolic refinement around cross-correlation peak
            ym1, yp1 = (y0 - 1) % ny, (y0 + 1) % ny
            xm1, xp1 = (x0 - 1) % nx, (x0 + 1) % nx
            denom_y = xcorr[ym1, x0] - 2 * xcorr[y0, x0] + xcorr[yp1, x0]
            denom_x = xcorr[y0, xm1] - 2 * xcorr[y0, x0] + xcorr[y0, xp1]
            dy_sub = 0.5 * (xcorr[ym1, x0] - xcorr[yp1, x0]) / denom_y if abs(denom_y) > 1e-12 else 0.0
            dx_sub = 0.5 * (xcorr[y0, xm1] - xcorr[y0, xp1]) / denom_x if abs(denom_x) > 1e-12 else 0.0

            sh_y = float(y0) + dy_sub
            sh_x = float(x0) + dx_sub
            if sh_y >= ny // 2:
                sh_y -= ny
            if sh_x >= nx // 2:
                sh_x -= nx

            est_sh = (-sh_y, -sh_x)
            al = fourier_shift_2d(cand, est_sh[0], est_sh[1])
            corr_val = float(np.sum(al * ground_truth))
            if corr_val > best_corr:
                best_corr = corr_val
                best_candidate = cand
                best_shift = est_sh

    aligned = fourier_shift_2d(best_candidate, best_shift[0], best_shift[1])
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

    NP = w**2
    cov_norm = NP / (NP - 1) if NP > 1 else 1.0

    sigma1_sq = cov_norm * (uniform_filter(a1 * a1, size=w) - mu1_sq)
    sigma2_sq = cov_norm * (uniform_filter(a2 * a2, size=w) - mu2_sq)
    sigma12 = cov_norm * (uniform_filter(a1 * a2, size=w) - mu1_mu2)

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)

    pad = (w - 1) // 2
    if pad > 0 and ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        return float(np.mean(ssim_map[pad:-pad, pad:-pad]))
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
