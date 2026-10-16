## Description of Changes

Please provide a concise description of what this PR changes, fixes, or adds to OpenDiffraction.

## Type of Change
- [ ] Physics forward model update (F1 - F6)
- [ ] Variation profile modification (V01 - V48)
- [ ] Bug fix in simulation or preprocessing
- [ ] Benchmark task or metric evaluation
- [ ] Documentation update
- [ ] Performance optimization

## Scientific & Physical Validation
- [ ] Forward model changes adhere to Fourier optics and analytical limits.
- [ ] Partial coherence changes maintain zero-lag autocorrelation centering.
- [ ] Grouped splitting preserves object disjointness (no data leakage).
- [ ] All unit tests pass: `python -m pytest tests/`.
- [ ] Quick generation verified: `python -m opendiffraction generate --preset quick`.

## Checklist
- [ ] Documentation updated in `docs/` or `README.md`.
- [ ] New unit tests added in `tests/`.
- [ ] No proprietary or unlicensed datasets introduced.
- [ ] No API keys, credentials, or large binary HDF5 files committed.
