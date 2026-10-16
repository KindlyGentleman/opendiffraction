# Dataset Card: OpenDiffraction 3.0

## Dataset Summary

OpenDiffraction 3.0 is a standardized, physics-validated synthetic diffraction dataset designed for machine learning in coherent diffractive imaging (CDI) and X-ray free-electron laser (XFEL) science. It provides 1,000 multi-exposure observations across 48 physical profiles and 6 forward models.

## Dataset Structure

- **Format**: Chunked HDF5 (`gzip` compression level 4) + CSV manifests.
- **Observations**: 1,000 diffraction patterns (512x512) paired with ground-truth specimens (64x64) and masks (512x512).
- **Base Objects**: 250 procedurally simulated biological, nanoparticle, and solid-state specimens.
- **Acquisitions per Object**: 4 physical exposure variations.
- **Splits**: Train (64.4%), Validation (14.0%), Test (11.6%), Test Out-of-Distribution (10.0%).
- **Zero Group Leakage**: Guaranteed disjoint base object IDs between train and test.
