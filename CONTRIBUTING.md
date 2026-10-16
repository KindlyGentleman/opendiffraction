# Contributing to OpenDiffraction

Thank you for your interest in contributing to OpenDiffraction. We welcome contributions from computational physicists, optical scientists, machine learning researchers, and scientific software engineers.

---

## Code of Conduct

All contributors and participants are expected to adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How to Contribute

### 1. Reporting Bugs & Dataset Anomalies
- Use the appropriate GitHub Issue template:
  - **Bug Report**: For software or pipeline crashes.
  - **Dataset Issue**: For invalid numerical values or metadata inconsistencies.
  - **Scientific Model Correction**: For forward model physics adjustments.

### 2. Developing New Features
1. Fork the repository and create a branch from `main`.
2. Set up the development environment:
   ```bash
   git clone https://github.com/<your-username>/opendiffraction.git
   cd opendiffraction
   pip install -e .
   pip install pytest pytest-cov ruff
   ```
3. Implement your changes following established Fourier optics conventions.
4. Run the full test suite before submitting:
   ```bash
   python -m pytest tests/
   ```
5. Check code formatting:
   ```bash
   ruff check opendiffraction/ tests/
   ```

---

## Core Scientific Guidelines

- **Analytical rigor**: All forward models (F1 - F6) must be physically grounded and tested against known analytical limits.
- **Autocorrelation zero-lag centering**: Partial coherence kernels in GSM simulations must be centered at `[0, 0]` in autocorrelation lag space.
- **Group-disjoint splitting**: Base objects must never leak between train, valid, and test partitions.
- **Zero proprietary assets**: Do not add third-party copyrighted datasets or images. All phantoms must be procedurally generated or licensed under compatible open terms.

---

## License Notice

By contributing to OpenDiffraction, you agree that your code contributions will be licensed under the [BSD 2-Clause License](LICENSE), and any generated data specifications will be licensed under [CC BY 4.0](LICENSE-DATA.md).
