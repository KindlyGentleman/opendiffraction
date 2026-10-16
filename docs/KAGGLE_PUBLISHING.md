# Kaggle Dataset Publishing & Maintenance Guide

This document describes the automated packaging, validation, publishing, and maintenance procedure for releasing OpenDiffraction to the Kaggle dataset platform.

---

## 1. Release Package Architecture

The automated packaging pipeline organizes the dataset into the self-contained directory `kaggle_release/` with the following structure:

```text
kaggle_release/
├── dataset-metadata.json               # Kaggle dataset configuration and licensing schema
├── README.md                           # Public Kaggle dataset description and quickstart
├── DATASET_CARD.md                     # Comprehensive scientific dataset card
├── DATASHEET.md                        # Gebru et al. standard datasheet
├── CITATION.cff                        # Machine-readable citation metadata
├── LICENSE                             # BSD-2-Clause code license
├── LICENSE-DATA.md                     # CC BY 4.0 dataset license
├── metadata.csv                        # Unified sample-level physical parameters
├── checksums.sha256                    # Cryptographic SHA-256 manifest
├── opendiffraction_benchmark.cxi       # NeXus CXI experimental format archive
├── data/
│   ├── train_000.h5                    # Training partition (70%)
│   ├── valid_000.h5                    # Validation partition (15%)
│   ├── test_000.h5                     # In-distribution test partition (15%)
│   └── test_ood_000.h5                 # Held-out out-of-distribution test partition
├── tasks/                              # Task-specific manifests (T1 to T12)
│   ├── T01_phase_retrieval.csv
│   ├── T02_denoising.csv
│   └── ...
└── examples/                           # Standalone data-loading scripts
    ├── load_dataset.py
    └── pytorch_dataset.py
```

---

## 2. Kaggle Authentication Setup

Before publishing to Kaggle, authenticate the Kaggle CLI.

### Option A: Modern Browser OAuth (Recommended)
Run:
```bash
kaggle auth login
```
This launches a browser window to securely authorize your local Kaggle CLI token without exposing API keys.

### Option B: Kaggle API Token (`kaggle.json`)
1. Log in to [Kaggle](https://www.kaggle.com/), go to **Account Settings**, and click **Create New Token**.
2. Save the downloaded `kaggle.json` file to the standard configuration path:
   - **Linux / macOS**: `~/.kaggle/kaggle.json`
   - **Windows**: `%USERPROFILE%\.kaggle\kaggle.json`
3. Secure file permissions (Linux/macOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Option C: Environment Variables
For headless or automated CI/CD environments:
```bash
export KAGGLE_USERNAME="<your-kaggle-username>"
export KAGGLE_KEY="<your-kaggle-api-key>"
```

---

## 3. Packaging and Pre-Publication Verification

Run the automated preparation and verification scripts locally before upload:

### Step 1: Package Release Bundle
```bash
python scripts/package_kaggle.py --input-dir output_quick --output-dir kaggle_release
```

### Step 2: Validate Data Integrity
Audit array dimensions, DPR conventions, and CXI compliance:
```bash
python scripts/validate_release.py kaggle_release
```

### Step 3: Verify Cryptographic Checksums
Confirm that all files match the generated `checksums.sha256` manifest:
```bash
python scripts/verify_download.py kaggle_release
```

---

## 4. Dataset Publishing Commands

> [!CAUTION]
> Remote dataset publication is an irreversible, externally visible action. Do not run publishing commands without explicit confirmation.

### Initial Publication (First Release)
To publish the dataset for the first time:
```bash
# Preview dataset metadata
cat kaggle_release/dataset-metadata.json

# Create public dataset on Kaggle
kaggle datasets create -p kaggle_release --public
```

### Updating an Existing Dataset (Version Bump)
To push an updated version:
```bash
kaggle datasets version -p kaggle_release -m "Release OpenDiffraction v2.0.0"
```

---

## 5. Post-Publication Verification

After upload completes, verify that Kaggle processed and published the dataset:

```bash
# Check dataset processing status
kaggle datasets status <kaggle-username>/opendiffraction-synthetic-cdi

# Test remote download and verification
kaggle datasets download -d <kaggle-username>/opendiffraction-synthetic-cdi --unzip -p ./kaggle_download_test
python scripts/verify_download.py ./kaggle_download_test
```
