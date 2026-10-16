# OpenDiffraction Comprehensive Release Inventory

**Audit Date**: October 16, 2026  
**Auditor**: Independent Scientific Reproducibility Auditor  
**Total Scanned Files**: 183  

---

## File Classification Summary

| Disposition | Meaning |
| :--- | :--- |
| **KEEP** | Essential scientific code, validated data array, or unique publication document. |
| **COMPRESS** | High-value visual asset that can be safely optimized for web transfer without loss of readability. |
| **DEDUPLICATE** | Redundant file identical to another primary asset. |
| **MOVE** | File belonging in a more appropriate directory hierarchy. |
| **REGENERATE** | Manifest, checksum, or metadata needing regeneration from canonical data. |
| **DEPRECATE** | Obsolete or superseded configuration/documentation. |
| **INVESTIGATE**| File requiring scientific or legal clarification. |

---

## Inventory Table

| Path | Size (KB) | SHA-256 (Prefix) | Tracked | Kaggle | License | Purpose | Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `.github/dependabot.yml` | 0.2 | `ac559cc7a6...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/ISSUE_TEMPLATE/bug_report.md` | 0.5 | `64f2cbfbb8...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/ISSUE_TEMPLATE/dataset_issue.md` | 0.5 | `f2547a1d20...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/ISSUE_TEMPLATE/feature_request.md` | 0.5 | `a1b4e3e155...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/ISSUE_TEMPLATE/model_correction.md` | 0.5 | `0c09abf6d3...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/PULL_REQUEST_TEMPLATE.md` | 1.0 | `08503fcadd...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/workflows/build-package.yml` | 0.5 | `407770fe28...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/workflows/lint.yml` | 0.5 | `63088a9fa4...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/workflows/prepare-release.yml` | 1.1 | `08555ff27c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.github/workflows/tests.yml` | 1.1 | `11ef664d56...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.gitignore` | 0.6 | `0fbd126bbf...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pre-commit-config.yaml` | 0.4 | `297044db6e...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pytest_cache/.gitignore` | 0.0 | `3ed731b65d...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pytest_cache/CACHEDIR.TAG` | 0.2 | `37dc88ef9a...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pytest_cache/README.md` | 0.3 | `73fd6fccdd...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pytest_cache/v/cache/lastfailed` | 0.0 | `44136fa355...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `.pytest_cache/v/cache/nodeids` | 5.3 | `209955e287...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `assets/figure_10_phase_retrieval.png` | 56.0 | `6d2a615914...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **KEEP** |
| `assets/figure_16_tasks_summary.png` | 494.3 | `a4d49a3e2f...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **KEEP** |
| `assets/figure_1_workflow.png` | 536.2 | `b1acd832d9...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **KEEP** |
| `assets/figure_2_specimens.png` | 124.0 | `b30fa2972f...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **KEEP** |
| `assets/figure_7_models_f1_to_f6.png` | 423.3 | `13f7afc955...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **KEEP** |
| `assets/kaggle_cover.png` | 2230.6 | `adf96b69f2...` | YES | NO | CC BY 4.0 (Figures) | Scientific visualization | **COMPRESS** |
| `benchmarks/denoising.py` | 2.2 | `d9209b7b1c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `benchmarks/evaluate.py` | 2.8 | `d851a4f5fe...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `benchmarks/hit_detection.py` | 2.2 | `2d55313612...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `benchmarks/missing_pixel_recovery.py` | 1.7 | `2c06185ad0...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `benchmarks/phase_retrieval.py` | 3.1 | `fc476540bc...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `CHANGELOG.md` | 1.8 | `c6902ee30c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `CITATION.cff` | 1.2 | `3ff3a950b2...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `CODE_OF_CONDUCT.md` | 2.1 | `8520011304...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/full.yaml` | 0.4 | `cbf80c7b41...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/optical_lab.yaml` | 0.4 | `116908bf79...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/quick.yaml` | 0.3 | `2ff5996059...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/starter.yaml` | 0.4 | `17929d0fd4...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/synchrotron.yaml` | 0.4 | `97af5fd553...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `configs/xfel.yaml` | 0.4 | `00fbc3b68a...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `CONTRIBUTING.md` | 2.0 | `34afbd9f46...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `DATASET_CARD.md` | 12.7 | `bf5dcf26f1...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `DATASHEET.md` | 5.5 | `6ed0c04c8f...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/BENCHMARKS.md` | 6.8 | `15eb6300ec...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/DATA_DICTIONARY.md` | 8.7 | `b584a7669b...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/DATASET_CARD.md` | 12.7 | `bf5dcf26f1...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/DPR_COMPATIBILITY.md` | 8.9 | `123fce6a0c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/KAGGLE_PUBLISHING.md` | 4.4 | `d48ca3fa98...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/LICENSING.md` | 3.7 | `06439ae390...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/LIMITATIONS.md` | 4.4 | `76aff8bd16...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/PHYSICS.md` | 5.9 | `7c27fa298f...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/REPRODUCIBILITY.md` | 4.4 | `f4be511267...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `docs/VARIATIONS.md` | 11.3 | `dc1b570233...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `examples/load_dataset.py` | 1.6 | `8db5c68d39...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `examples/minimal_training.py` | 2.1 | `589baf85c5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `examples/pytorch_dataset.py` | 2.6 | `e3169de261...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `examples/visualize_samples.py` | 1.3 | `ebe26595e9...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/checksums.sha256` | 0.8 | `7eb83bfd6d...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/CITATION.cff` | 1.2 | `3ff3a950b2...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/configuration.json` | 0.4 | `a3c5df66f1...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/data/test_000.h5` | 11342.3 | `60c55a1e70...` | NO | YES | CC BY 4.0 (Dataset) | HDF5 scientific array container | **KEEP** |
| `kaggle_release/data/test_ood_000.h5` | 18942.2 | `bf3f8371b2...` | NO | YES | CC BY 4.0 (Dataset) | HDF5 scientific array container | **KEEP** |
| `kaggle_release/data/train_000.h5` | 126304.6 | `431b0b4fd7...` | NO | YES | CC BY 4.0 (Dataset) | HDF5 scientific array container | **KEEP** |
| `kaggle_release/data/valid_000.h5` | 18923.0 | `f91b449988...` | NO | YES | CC BY 4.0 (Dataset) | HDF5 scientific array container | **KEEP** |
| `kaggle_release/dataset-metadata.json` | 10.2 | `98f5194eca...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/DATASET_CARD.md` | 12.7 | `bf5dcf26f1...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/DATASHEET.md` | 5.5 | `6ed0c04c8f...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/examples/figure_10_phase_retrieval.png` | 56.0 | `6d2a615914...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_16_tasks_summary.png` | 494.3 | `a4d49a3e2f...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_1_workflow.png` | 536.2 | `b1acd832d9...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_2_specimens.png` | 124.0 | `b30fa2972f...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_3_geometry_diffraction.png` | 3108.9 | `f197453c9c...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_4_photon_sweep.png` | 6039.0 | `973b59f511...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_5_coherence_sweep.png` | 4480.2 | `5f737f0907...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/figure_7_models_f1_to_f6.png` | 423.3 | `13f7afc955...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **DEDUPLICATE** |
| `kaggle_release/examples/kaggle_cover.png` | 8087.5 | `e7c054c91b...` | YES | YES | CC BY 4.0 (Figures) | Scientific visualization | **COMPRESS** |
| `kaggle_release/LICENSE` | 2.7 | `90468793a5...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/metadata.csv` | 10.0 | `1909dcdd1b...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/opendiffraction_benchmark.cxi` | 4811.4 | `f3798b3915...` | NO | YES | CC BY 4.0 (CXI Data) | CXIDB NeXus-compliant scientific container | **KEEP** |
| `kaggle_release/README.md` | 8.7 | `c23ecb3793...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/scripts/load_dataset.py` | 0.3 | `95a03dfe9e...` | YES | YES | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_release/tasks/T10_manifest.csv` | 11.9 | `5bf97c2bd3...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T11_manifest.csv` | 11.3 | `61d24bac10...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T12_manifest.csv` | 11.7 | `d0b36c58a6...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T1_manifest.csv` | 10.7 | `31fb4a9f16...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T2_manifest.csv` | 10.2 | `8937d9ddd6...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T3_manifest.csv` | 9.8 | `9761c1b76d...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T4_manifest.csv` | 1.8 | `e59b4433e8...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T5_manifest.csv` | 1.2 | `145f7651c3...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T6_manifest.csv` | 3.1 | `626ada863a...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T7_manifest.csv` | 10.2 | `d96ae74d14...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T8_manifest.csv` | 11.8 | `ec31aa4c55...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_release/tasks/T9_manifest.csv` | 11.0 | `a16a3f45b8...` | YES | YES | CC BY 4.0 (Metadata/Manifest) | Tabular metadata or task manifest | **KEEP** |
| `kaggle_starter_kernel/create_kernel.py` | 11.8 | `84ba2fbd7c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_starter_kernel/kernel-metadata.json` | 0.5 | `7d739f0077...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `kaggle_starter_kernel/starter_notebook.ipynb` | 14.2 | `f62dd469dc...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `LICENSE` | 2.7 | `90468793a5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `LICENSE-DATA.md` | 0.9 | `51e6f56223...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `notebooks/01_explore_opendiffraction.ipynb` | 6.2 | `80100b7fd8...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `notebooks/02_phase_retrieval_baseline.ipynb` | 6.2 | `3940cf4b40...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `notebooks/03_denoising.ipynb` | 2.1 | `57a2798444...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `notebooks/04_hit_detection.ipynb` | 2.0 | `e1ed098b59...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `notebooks/05_model_training.ipynb` | 1.4 | `2fbee2ce4b...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `notebooks/06_kaggle_starter.ipynb` | 14.2 | `f62dd469dc...` | YES | NO | BSD-2-Clause (Notebook) | Interactive benchmark or demonstration | **KEEP** |
| `opendiffraction/__init__.py` | 0.3 | `daa2054310...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/__main__.py` | 0.1 | `18f5793dc5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/cli.py` | 8.7 | `3b0b0dafbc...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/generator/__init__.py` | 0.2 | `d381427075...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/generator/pipeline.py` | 12.3 | `4814a04b1b...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/generator/splits.py` | 1.1 | `0235ec5af8...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/generator/storage.py` | 7.2 | `bb6ce13171...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/io/__init__.py` | 0.3 | `39bfce4d10...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/io/cxi.py` | 8.1 | `520174e705...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/io/loader.py` | 4.2 | `9fee1312a8...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/__init__.py` | 1.6 | `58a9be9bb6...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/adapters.py` | 1.4 | `653b328370...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/complex_objects.py` | 2.8 | `5d445d9d88...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/object_3d.py` | 4.9 | `7bed53afc3...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/procedural.py` | 15.2 | `7e680d2d09...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/objects/specimens.py` | 12.3 | `c925dc4b76...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/package/__init__.py` | 0.2 | `8658d0cb6d...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/package/kaggle.py` | 4.6 | `d1db0affa1...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/__init__.py` | 1.2 | `46952fb610...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/backgrounds.py` | 4.1 | `807ec442a4...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/coherence.py` | 3.0 | `eb3e0879a0...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/detectors.py` | 3.5 | `3b9b76ce7f...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/forward.py` | 8.4 | `e965c38d99...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/masks.py` | 8.5 | `f140dc7ff8...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/__init__.py` | 0.7 | `8d83bc0422...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/coherence.py` | 3.3 | `7ab4019a26...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/fraunhofer.py` | 2.4 | `2290be63be...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/fresnel.py` | 3.7 | `2b4f3f69af...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/particle_3d.py` | 3.5 | `6cf718bd46...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/polychromatic.py` | 3.3 | `6729961dcc...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/models/transmission.py` | 3.3 | `4f37cb5ec4...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/noise.py` | 3.9 | `f93f72a551...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/physics/phase_retrieval.py` | 5.0 | `896238f9d4...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/scenarios/__init__.py` | 0.2 | `6931bf3cb5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/scenarios/presets.py` | 7.2 | `6d1b9b01d3...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/tasks/__init__.py` | 0.5 | `b3ba7cfc82...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/tasks/manifests.py` | 7.4 | `295a7587ea...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/tasks/metrics.py` | 6.2 | `2862afecb5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/validation/__init__.py` | 0.3 | `4f33176cca...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/validation/auditor.py` | 6.1 | `679dbb4827...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/validation/reports.py` | 7.5 | `e2d1ef545e...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/variations/__init__.py` | 0.3 | `c7a4b5320d...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/variations/profiles.py` | 14.4 | `6fa624411c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/variations/profiles_48.py` | 34.4 | `754cdfa81f...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/visualization/__init__.py` | 0.5 | `0b4803b73e...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/visualization/animation.py` | 3.8 | `3f80b1f8f9...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/visualization/figures.py` | 17.9 | `8457a7da3e...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction/visualization/figures_16.py` | 27.8 | `3c2a3a9312...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/dependency_links.txt` | 0.0 | `01ba4719c8...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/entry_points.txt` | 0.1 | `85df7e4019...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/PKG-INFO` | 8.0 | `4675e47f9b...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/requires.txt` | 0.1 | `eaf79dfdda...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/SOURCES.txt` | 2.3 | `d5ec91ea79...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `opendiffraction.egg-info/top_level.txt` | 0.0 | `f32a6ecaab...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `pyproject.toml` | 0.8 | `80f2ac0c76...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `README.md` | 9.4 | `a2a561a801...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `reports/KNOWN_LIMITATIONS.md` | 2.3 | `134a20f476...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `reports/PREPUBLICATION_AUDIT.md` | 8.7 | `bb4cbd2db7...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `reports/PUBLICATION_STATE.md` | 4.0 | `747e460566...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `reports/QUALITY_AUDIT.md` | 2.1 | `13e7f0ca0a...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `reports/VALIDATION_REPORT.md` | 2.0 | `980624679e...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `requirements.txt` | 0.1 | `b0e1a94206...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/audit_downloaded_kaggle.py` | 4.6 | `c8831fab1d...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/audit_repo_files.py` | 1.1 | `68441d8956...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/generate_release.py` | 0.4 | `6919a18657...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/generate_release_inventory.py` | 4.4 | `1a96efc9f0...` | NO | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/package_kaggle.py` | 0.4 | `ff689dc025...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/publish_kaggle.py` | 3.8 | `61ff4ff26a...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/validate_release.py` | 0.4 | `2814ba2e0b...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `scripts/verify_download.py` | 1.7 | `be6886f123...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `SECURITY.md` | 0.7 | `946a6eec5b...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/__init__.py` | 0.0 | `fcc687c92a...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_cli.py` | 1.9 | `33995bf337...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_coherence.py` | 2.8 | `f41d620711...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_cxi.py` | 1.8 | `f7370ee38f...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_dpr_compatibility.py` | 3.3 | `8b820f7a79...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_f1_to_f6.py` | 4.4 | `b4b853c26c...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_noise_statistics.py` | 2.3 | `4a34cacc38...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_objects_and_variations.py` | 2.3 | `b91b4c81c5...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_physics_forward.py` | 4.9 | `e9cbfc38ab...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_profiles_48.py` | 2.1 | `f2c223706b...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
| `tests/test_storage_and_splits.py` | 4.1 | `a3c8791103...` | YES | NO | BSD-2-Clause (Software) | Core source code / asset | **KEEP** |
