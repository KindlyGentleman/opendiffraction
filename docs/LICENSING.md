# OpenDiffraction Licensing and Attribution

OpenDiffraction 2.0 adopts an open, dual-license model to facilitate maximum academic and commercial research utility while protecting scientific provenance.

---

## 1. Dual-License Framework

### Software License: BSD 2-Clause
All source code, command-line utilities, benchmark implementations, tests, and generation scripts in this repository are licensed under the **BSD 2-Clause "Simplified" License**.

- You are free to copy, modify, redistribute, and integrate the code into academic or commercial software.
- Redistribution must retain the copyright notice, list of conditions, and disclaimer.
- See [`LICENSE`](file:///d:/03_Proyek/Diffraction_Dataset/LICENSE) in the repository root for the full legal text.

### Data & Media License: Creative Commons Attribution 4.0 International (CC BY 4.0)
All generated benchmark datasets, HDF5 archives, CXI NeXus files, metadata tables, figures, diagrams, animations, and documentation are licensed under **CC BY 4.0**.

- You are free to share (copy and redistribute in any medium or format) and adapt (remix, transform, and build upon) the material for any purpose, including commercial applications.
- You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- See [`LICENSE-DATA.md`](file:///d:/03_Proyek/Diffraction_Dataset/LICENSE-DATA.md) for complete license details.

---

## 2. Relationship and Attribution to Upstream DPR

OpenDiffraction was inspired by the Deep Phase Retrieval (DPR) framework:
- **Upstream Paper**: Lee et al., *"Deep-learning real-time phase retrieval of imperfect diffraction patterns from X-ray free-electron lasers"*, *npj Computational Materials* 11, 68 (2025). DOI: [10.1038/s41524-025-01569-7](https://doi.org/10.1038/s41524-025-01569-7).
- **Upstream Repository**: [https://github.com/sungyun98/DPR](https://github.com/sungyun98/DPR) (BSD 2-Clause License).

### Key Distinctions from Upstream DPR
1. **Independent Implementation**: OpenDiffraction is a standalone, clean-room software architecture developed independently. It does not wrap, fork, or modify upstream DPR code.
2. **Zero Proprietary Data Reuse**: None of the training samples, experimental beamline files, or pre-trained weights from DPR are bundled or redistributed. All samples are 100% procedurally synthesized.
3. **Corrected Physical Model**: OpenDiffraction adheres to the corrected Gaussian Schell-model partial coherence formulation (centered zero-lag autocorrelation damping) rather than the legacy formulation. For technical details, see [`DPR_COMPATIBILITY.md`](file:///d:/03_Proyek/Diffraction_Dataset/docs/DPR_COMPATIBILITY.md).

---

## 3. How to Cite OpenDiffraction

If you use OpenDiffraction in scientific publications, academic theses, software projects, or Kaggle competitions, please cite both this dataset and the foundational DPR publication:

### BibTeX Citation for OpenDiffraction
```bibtex
@dataset{opendiffraction2026,
  author       = {Muhammad and OpenDiffraction Contributors},
  title        = {{OpenDiffraction: Physics-Based Synthetic Coherent Diffraction Patterns for Phase Retrieval and Image Reconstruction}},
  year         = {2026},
  publisher    = {Kaggle / GitHub},
  version      = {2.0.0},
  url          = {https://github.com/KindlyGentleman/opendiffraction}
}
```

### BibTeX Citation for Foundational DPR Paper
```bibtex
@article{lee2025deep,
  author    = {Lee, Sungyun and others},
  title     = {Deep-learning real-time phase retrieval of imperfect diffraction patterns from {X}-ray free-electron lasers},
  journal   = {npj Computational Materials},
  volume    = {11},
  number    = {1},
  pages     = {68},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41524-025-01569-7}
}
```
