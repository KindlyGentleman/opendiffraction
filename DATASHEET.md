# Datasheet for Dataset: OpenDiffraction

Following the recommended format from Gebru et al. (2021), *"Datasheets for Datasets"*, Communications of the ACM.

---

## 1. Motivation

- **For what purpose was the dataset created?**  
  To provide an open, transparent, fully reproducible, physics-based benchmark dataset for coherent diffraction imaging (CDI), phase retrieval, and detector defect inpainting without proprietary third-party image licensing restrictions.
- **Who created the dataset?**  
  OpenDiffraction computational optics and scientific machine learning contributors.
- **Who funded the creation of the dataset?**  
  Open research initiative with self-hosted computational resources.

---

## 2. Composition

- **What do the instances that comprise the dataset represent?**  
  Each instance consists of a paired 2D synthetic ground-truth transmission object ($64 \times 64$) and its corresponding degraded Fraunhofer diffraction pattern ($512 \times 512$), accompanied by an acquisition validity mask and simulation metadata.
- **How many instances are there in total?**  
  Configurable via presets: `quick` (48 samples, 1 per profile), `starter` (2,400 samples, 50 per profile), and `full` (9,600 samples, 200 per profile), or domain-specific scenarios (XFEL, synchrotron, optical lab).
- **Does the dataset contain all possible instances or is it a sample?**  
  It is a comprehensive sample across 48 controlled variation profiles (V01 - V48) spanning 6 scientific groups, 6 modular forward physics models (F1 - F6), 6 experimental scenarios (S1 - S6), and 12 scientific machine learning benchmark tasks (T1 - T12).
- **Is what is being measured representative?**  
  The dataset samples cover 6 orders of magnitude in photon budget ($10^3$ to $10^8$ photons), spatial coherence lengths from infinity down to 25 pixels, diverse beam-stop and detector defect geometries, parasitic backgrounds (gas, carbon film, solvent ring, slit streaks, cosmic rays), probe aberrations, and SI-unit Fresnel and pink-beam regimes.
- **Is any information missing?**  
  Temporal femtosecond pulse shape jitter and dynamical multi-slice scattering in thick 3D crystals are deliberately approximated by the kinematic Born projection and spectral integration models based on documented physical assumptions.
- **Does the dataset contain data that might be considered confidential or sensitive?**  
  No. All data are entirely synthetic and procedural.
- **Does the dataset contain data that relates to people?**  
  No.

---

## 3. Collection Process

- **How were the data associated with each instance acquired?**  
  Synthesized by numerical simulation using NumPy Fourier optics pipelines and physical noise sampling models.
- **What mechanisms or procedures were used to collect the data?**  
  Procedural shape synthesis (geometric, smooth Fourier blobs, multi-scatterers, porous structures, structured lattices) followed by centered 2D Fast Fourier Transform, Gaussian Schell-model partial coherence filtering, Poisson shot noise generation, and Gaussian readout noise addition.
- **Who was involved in the data collection process?**  
  Automated numerical scripts executed on local CPU hardware.
- **Over what timeframe were the data collected?**  
  Generated deterministically on demand from fixed random seeds.

---

## 4. Preprocessing, Cleaning, and Labeling

- **Was any preprocessing/cleaning/labeling done on the data?**  
  Yes. Raw simulated intensities are scaled to target photon fluxes. Real-space targets are provided in both true physical amplitude and DPR `false_scale=True` neural network conventions.
- **Was the "raw" data saved in addition to the preprocessed data?**  
  Yes. The unscaled object amplitude, uncorrupted ideal coherent intensity, expected partially coherent photon arrival rate, and signed noisy detector readouts are stored separately in the extended scientific HDF5 dataset.
- **Is the software used to preprocess/clean/label the data available?**  
  Yes. The complete source code is provided in the `opendiffraction` package.

---

## 5. Uses

- **Has the dataset been used for any tasks already?**  
  Benchmarked with classical phase retrieval algorithms (Error Reduction and Hybrid Input-Output) and evaluated for compatibility with the Deep Phase Retrieval (DPR) network architecture.
- **Is there a repository that links to papers or systems that use the dataset?**  
  Source repository documentation in `docs/DPR_COMPATIBILITY.md`.
- **What tasks should the dataset NOT be used for?**  
  The dataset must not be used as proof of performance for safety-critical real-world diagnostic applications without experimental beamtime verification.

---

## 6. Distribution

- **How is the dataset distributed?**  
  Publicly accessible via Kaggle datasets and reproducible via the local generator CLI.
- **When was the dataset released?**  
  2026.
- **What license is the dataset distributed under?**  
  Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Have any third parties imposed intellectual property constraints?**  
  No. The procedural generation pipeline has zero dependencies on external copyrighted image datasets (EMNIST, CIFAR-100, or NVIDIA masks).

---

## 7. Maintenance

- **Who is supporting/hosting/maintaining the dataset?**  
  OpenDiffraction open-source repository maintainers.
- **How can the owner/curator/manager of the dataset be contacted?**  
  Via GitHub issues and repository channels.
- **Will the dataset be updated?**  
  Yes, subsequent versioned releases (e.g. 3D CDI, ptychography, or experimental beamtime transfer extensions) will follow semantic versioning.
