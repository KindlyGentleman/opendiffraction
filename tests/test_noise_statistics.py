"""Statistical tests for Poisson shot noise and Gaussian readout noise."""

import numpy as np
import pytest
from scipy.stats import kstest, norm

from opendiffraction.physics.noise import (
    FWHM_ONE_SIGMA,
    apply_gaussian_readout_noise,
    apply_poisson_noise,
)


def test_06_poisson_mean_and_variance_statistics():
    """Test 6: Test that photon counts follow their expected mean and variance statistically."""
    rng = np.random.default_rng(606)

    # Test array of varying arrival rates
    lambda_rates = np.array([2.0, 10.0, 50.0, 200.0, 1000.0])
    N_samples = 50000

    # Draw N_samples for each rate
    sampled = np.array([apply_poisson_noise(np.full(N_samples, lam), rng=rng) for lam in lambda_rates])

    sample_means = np.mean(sampled, axis=1)
    sample_vars = np.var(sampled, axis=1, ddof=1)

    # Relative errors for Poisson distribution (mean == variance == lambda)
    # Standard error of the mean is sqrt(lambda / N) -> relative error is 1 / sqrt(lambda * N)
    for i, lam in enumerate(lambda_rates):
        rel_mean_err = abs(sample_means[i] - lam) / lam
        rel_var_err = abs(sample_vars[i] - lam) / lam
        assert rel_mean_err < 0.05, f"Poisson mean error {rel_mean_err} for lambda={lam}"
        assert rel_var_err < 0.05, f"Poisson variance error {rel_var_err} for lambda={lam}"


def test_07_gaussian_readout_noise_statistics():
    """Test 7: Verify Gaussian detector-noise statistics."""
    rng = np.random.default_rng(707)
    target_sigma = FWHM_ONE_SIGMA  # ~0.42466
    N_samples = 100000

    zeros = np.zeros(N_samples)
    readout = apply_gaussian_readout_noise(zeros, readout_std=target_sigma, rng=rng)

    mean_err = float(np.mean(readout))
    std_measured = float(np.std(readout, ddof=1))

    # Mean must be within 3 standard errors of mean (3 * sigma / sqrt(N))
    sem = target_sigma / np.sqrt(N_samples)
    assert abs(mean_err) < 3 * sem, f"Mean readout error: {mean_err}"

    # Std must match within 2%
    rel_std_err = abs(std_measured - target_sigma) / target_sigma
    assert rel_std_err < 0.02, f"Readout std relative error: {rel_std_err}"

    # Kolmogorov-Smirnov test against normal CDF
    _, p_value = kstest(readout / target_sigma, "norm")
    assert p_value > 0.01, f"KS test failed for Gaussian readout noise: p={p_value}"
