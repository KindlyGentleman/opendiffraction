# Citation Guide for OpenDiffraction

If you use OpenDiffraction in scientific research, academic publications, machine learning benchmarks, or educational curricula, please cite the project and the foundational scientific literature as described below.

---

## 1. How to Cite the Dataset

To cite the synthetic dataset itself:

### BibTeX
```bibtex
@misc{kamal2026opendiffraction_dataset,
  author       = {Kamal, Muhammad},
  title        = {OpenDiffraction: Physics-Based Synthetic Diffraction Imaging Dataset},
  year         = {2026},
  version      = {2.0.0},
  publisher    = {Kaggle},
  url          = {https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi},
  note         = {Open-access synthetic coherent diffraction imaging dataset and inverse benchmark}
}
```

### APA Style
> Kamal, M. (2026). *OpenDiffraction: Physics-Based Synthetic Diffraction Imaging Dataset* (Version 2.0.0) [Data set]. Kaggle. https://www.kaggle.com/datasets/hamadmuhammadkamal/opendiffraction-synthetic-cdi

---

## 2. How to Cite the Software & Benchmark Suite

To cite the OpenDiffraction Python data-generation engine, physical forward models, or benchmark protocol:

### BibTeX
```bibtex
@software{kamal2026opendiffraction_code,
  author       = {Kamal, Muhammad},
  title        = {OpenDiffraction: Python Simulation Engine for Coherent Diffraction Imaging and Phase Retrieval Benchmarks},
  year         = {2026},
  version      = {2.0.0},
  publisher    = {GitHub},
  url          = {https://github.com/KindlyGentleman/opendiffraction}
}
```

### APA Style
> Kamal, M. (2026). *OpenDiffraction: Python Simulation Engine for Coherent Diffraction Imaging and Phase Retrieval Benchmarks* (Version 2.0.0) [Computer software]. GitHub. https://github.com/KindlyGentleman/opendiffraction

---

## 3. Foundational Scientific DPR Paper Citation

OpenDiffraction's forward modeling and benchmark design are inspired by the foundational Deep Phase Retrieval (DPR) paper. When publishing results on this dataset, please cite all seven original authors:

### BibTeX
```bibtex
@article{lee2025npjcompumats,
  author    = {Lee, Sung Yun and Cho, Do Hyung and Jung, Chulho and Sung, Daeho and Nam, Daewoong and Kim, Sangsoo and Song, Changyong},
  title     = {Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers},
  journal   = {npj Computational Materials},
  volume    = {11},
  number    = {1},
  pages     = {68},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41524-025-01569-7},
  url       = {https://doi.org/10.1038/s41524-025-01569-7}
}
```

### APA Style
> Lee, S. Y., Cho, D. H., Jung, C., Sung, D., Nam, D., Kim, S., & Song, C. (2025). Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers. *npj Computational Materials*, 11(1), 68. https://doi.org/10.1038/s41524-025-01569-7

---

## 4. Versioning & Archival Policy

- **Software Releases**: Versioned via Semantic Versioning (`MAJOR.MINOR.PATCH`) in `pyproject.toml` and Git tags.
- **Dataset Versions**: Versioned sequentially on Kaggle (e.g., Version 1, 2, 3...). Every release is permanently pinned with cryptographic SHA-256 manifests in `checksums.sha256`.
