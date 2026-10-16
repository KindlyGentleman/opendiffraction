# OpenDiffraction 3.0 Benchmark Failure Case Analysis

## 1. Phase Retrieval Failure Modes (HIO/ER)

Classical iterative projection algorithms fail across several distinct structural regimes:

1. **Twin-Image Stagnation**: Without an asymmetric support mask, HIO frequently stagnates in a superposition of the true object and its inverted complex conjugate reflection $O^*(-\mathbf{r})$.
2. **Low-Photon Truncation (Profile V38)**: Under low photon budgets ($10^4$ photons), shot noise destroys high-frequency speckle fringes, causing ER to reconstruct a featureless low-pass blob.
3. **Partial Coherence Blur (Profiles V14, V15)**: Partial spatial coherence acts as a low-pass filter on the diffraction intensity, violating the coherent Fraunhofer modulus constraint and leading to high crystallographic R-factors (>0.30).

## 2. Inpainting Failure Modes

- **Central Beamstop Shadow**: Nearest-neighbor extrapolation propagates boundary intensity horizontally/vertically into the central hole, completely missing the exponential rise of forward central scattering $|F(0)|^2$.
