# OpenDiffraction Reproducibility Guide

OpenDiffraction 2.0 is designed from the ground up for strict scientific reproducibility. Every simulated frame, object geometry, noise realization, and benchmark split is deterministically generated from immutable seeds, requiring zero proprietary software or hardware accelerators.

---

## 1. Deterministic Random Number Generation

To guarantee bit-exact or numerically bounded reproducibility across platforms:

1. **Independent Generator State**: All stochastic procedures utilize NumPy's modern `Generator` API via `np.random.default_rng(seed)`. Global random state (`np.random.seed()`) is never used.
2. **Hierarchical Deterministic Sub-Seeding**: Every sample index $i$ derives an isolated seed:
   $$\text{seed}_i = (\text{base\_seed} \times 1000003 + i) \pmod{2^{31} - 1}$$
   This ensures that generating sample $i$ produces identical results whether executed serially, in parallel multiprocessing pools, or in distributed batches.
3. **Partition Invariance**: The splitting algorithm partitions base object identifiers (`group_id`) using a cryptographic SHA-256 hash modulo 100:
   $$\text{hash\_val} = \text{int}(\text{SHA256}(\text{group\_id})[:8], 16) \pmod{100}$$
   - $0 \le \text{hash\_val} < 70$: `train` (70%)
   - $70 \le \text{hash\_val} < 85$: `valid` (15%)
   - $85 \le \text{hash\_val} < 100$: `test` (15%)
   This mathematical partitioning guarantees zero object leakage across splits regardless of sample ordering.

---

## 2. Hardware and Environment Requirements

OpenDiffraction runs natively on commodity hardware without requiring GPU acceleration:

- **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+), macOS (12+), Windows 10/11.
- **Python Version**: Python 3.10, 3.11, 3.12, or 3.13.
- **Memory (RAM)**: Minimum 4 GB; 8 GB+ recommended for large-batch multiprocessing.
- **Disk Storage**:
  - Code and benchmark manifests: $< 50$ MB.
  - Quick benchmark package (48 samples): $\approx 48$ MB.
  - Scaled production packages (10,000 samples): $\approx 10$ GB.
- **Architecture**: x86_64 or ARM64 (Apple Silicon).

### Core Dependencies (`pyproject.toml`)
```toml
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.8.0",
    "h5py>=3.7.0",
    "pandas>=1.4.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.64.0",
    "pyyaml>=6.0",
]
```

---

## 3. Step-by-Step Reproduction Workflow

Follow these steps to reproduce the dataset, test suite, and baseline benchmarks from scratch:

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/KindlyGentleman/opendiffraction.git
cd opendiffraction

# Create and activate a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Step 2: Execute Complete Test Suite
Verify that all 75 scientific unit and integration tests pass on your host:
```bash
pytest tests/ -v
```
Expected output: `75 passed in ~30s (100% success)`.

### Step 3: Run Baseline Benchmarks
Run the classical iterative phase retrieval (HIO/ER) and benchmark evaluation:
```bash
python scripts/generate_phase8_reports.py
```

### Step 4: Generate Benchmark Release Dataset
Generate the reproducible 48-sample Quick-Release benchmark dataset using deterministic seed 42:
```bash
python -m opendiffraction.cli generate --preset quick --output ./reproduced_dataset --seed 42
```

### Step 5: Validate Data Integrity and Checksums
Audit HDF5 structure, array dtypes, CXI compliance, and verification rules:
```bash
python -m opendiffraction.cli validate --dataset ./reproduced_dataset --reports-dir ./reproduced_reports
python scripts/audit_release.py --strict
```
Expected output: `ALL 20 RELEASE AUDIT CHECKS PASSED: 100% SUCCESS`.

---

## 4. Cross-Platform Numerical Invariance

- **FFT Tolerances**: NumPy's FFT implementations across different BLAS/FFTW backends (Intel MKL, OpenBLAS, Apple Accelerate) exhibit numerical variations bounded by $|\Delta I| / I < 10^{-6}$ for single-precision `float32`.
- **Operating System Portability**: File paths, HDF5 internal keys, and CSV line endings (`\n`) are strictly normalized for cross-platform interoperability between POSIX and Windows environments.
