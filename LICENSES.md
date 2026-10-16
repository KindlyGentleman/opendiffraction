# OpenDiffraction Dual Licensing Structure

The OpenDiffraction project maintains a clear and explicit separation between its open-source **software codebase** and its **scientific dataset artifacts**.

---

## 1. Summary of Applicable Licenses

| Category | Applicable License | Governing File | Scope of Coverage |
| :--- | :--- | :--- | :--- |
| **Software Source Code** | **BSD 2-Clause License** | [`LICENSE`](LICENSE) | All Python code in `opendiffraction/`, generator pipelines, validation tools, CLI, tests, and utility scripts. |
| **Scientific Dataset & Metadata** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** | [`LICENSE-DATA.md`](LICENSE-DATA.md) | All generated HDF5 shards (`data/*.h5`), CXI files (`*.cxi`), tabular metadata (`metadata.csv`), task manifests (`tasks/*.csv`), and original figures (`assets/*.png`, `output/figures/*`). |
| **Documentation & Reports** | **CC BY 4.0** | [`LICENSE-DATA.md`](LICENSE-DATA.md) | README files, dataset cards, datasheets, reports, and documentation guides. |
| **Example & Starter Notebooks** | **BSD 2-Clause License** | [`LICENSE`](LICENSE) | All Jupyter notebooks in `notebooks/` and `kaggle_starter_kernel/`. |

---

## 2. Upstream Software Attribution

Portions of the simulation algorithms and data-loading structures are inspired by and adapted from the open-source **Deep Phase Retrieval (DPR)** project:

- **Upstream Repository**: `https://github.com/sungyun98/DPR`
- **Copyright**: Copyright (c) 2024, sungyun98.
- **License**: BSD 2-Clause License.
- **Upstream Paper**: Lee et al. (2025), *Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers*, **npj Computational Materials** 11, 68. DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).

Full third-party notices and disclaimers are preserved in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSE`](LICENSE).

---

## 3. How to Cite

When using OpenDiffraction software or datasets, please provide attribution to **Muhammad Kamal** as detailed in [`CITATION.md`](CITATION.md) and cite the foundational Lee et al. (2025) paper.
