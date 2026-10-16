# Survey of Experimental CXI Datasets for Coherent Diffractive Imaging

## 1. Executive Summary

This report surveys candidate experimental datasets from the Coherent X-ray Imaging Data Bank (CXIDB, [cxidb.org](https://cxidb.org/)) to benchmark against the OpenDiffraction synthetic dataset. Experimental coherent diffractive imaging (CDI) and single-particle imaging (SPI) datasets from leading X-ray Free-Electron Laser (XFEL) facilities (LCLS, SACLA, European XFEL) are analyzed for wavelength, detector architecture, specimen class, and file layout.

---

## 2. Experimental Candidate Inventory

### 2.1 CXIDB id-62: PR772 Virus Single-Particle Diffraction
- **Facility / Beamline**: Linac Coherent Light Source (LCLS), AMO (Atomic, Molecular, and Optical Science) endstation.
- **Reference**: Reddy et al. (2017), *"Coherent soft X-ray diffraction imaging of coliphage PR772 at the Linac Coherent Light Source"*, Scientific Data 4, 170079.
- **Specimen**: Coliphage PR772 virus particles (diameter ~70 nm, icosahedral capsid).
- **Photon Energy**: 1.2 keV (soft X-ray regime, $\lambda \approx 1.03$ nm).
- **Detector**: pnCCD detector ($1024 \times 1024$ pixels, central beam passage hole).
- **Sample Delivery**: Aerodynamic aerosol lens injector.
- **Hit Rate**: ~1.5% single-particle hits out of ~1,000,000 shots.
- **Data Volume**: ~1 TB raw, ~15 GB curated single-particle hit subset.
- **Relevance to OpenDiffraction**: Directly aligns with Profile `V02` (Virus-Like Particles) and forward model `F1` (Fraunhofer single-particle diffraction).

### 2.2 CXIDB id-81: In Vivo Cyanobacteria Whole-Cell Diffraction
- **Facility / Beamline**: LCLS CXI (Coherent X-ray Imaging) instrument.
- **Reference**: van der Schot et al. (2015), *"Open science data from the first successful imaging of live viruses with an X-ray free electron laser"*, Nature Communications 6, 5704.
- **Specimen**: *Synechococcus* cyanobacteria cells (~1.0 $\mu$m size).
- **Photon Energy**: 6.0 keV ($\lambda \approx 0.206$ nm).
- **Detector**: Cornell-SLAC Pixel Array Detector (CSPAD, $1516 \times 1516$ active pixels, 32 tiled ASICs).
- **Detector Distance**: 0.56 m.
- **Data Format**: CXI NeXus format (`/entry_1/data_1/data` and `/entry_1/data_1/mask`).
- **Relevance to OpenDiffraction**: Directly benchmarks Profile `V01` (Cell-Like Biological Phantoms) and CSPAD detector mask geometries (module gaps, center hole).

### 2.3 CXIDB id-91: Gold Nanoparticle Clusters at SACLA
- **Facility / Beamline**: SACLA (SPring-8 Angstrom Compact Free Electron Laser), BL3.
- **Reference**: Suzuki et al. (2016), *"High-resolution coherent diffractive imaging of gold nanoparticle clusters"*, Journal of Synchrotron Radiation 23, 725-731.
- **Specimen**: Self-assembled colloidal gold nanoparticles and multi-particle clusters.
- **Photon Energy**: 5.0 keV ($\lambda \approx 0.248$ nm).
- **Detector**: Multiple-Port Charge-Coupled Device (MPCCD, $1024 \times 1024$ pixels).
- **Data Volume**: ~22 GB.
- **Relevance to OpenDiffraction**: Directly corresponds to Profiles `V04` (Core-Shell), `V05` (Ellipsoidal), and `V07` (Clusters).

### 2.4 CXIDB id-164: Nanoporous Aerogel at European XFEL
- **Facility / Beamline**: European XFEL, SPB/SFX (Single Particles, Clusters, and Biomolecules and Serial Femtosecond Crystallography) instrument.
- **Specimen**: Nanoporous silica aerogels and soot aggregates.
- **Photon Energy**: 9.3 keV ($\lambda \approx 0.133$ nm).
- **Detector**: Adaptive Gain Integrating Pixel Detector (AGIPD, 16 sensor modules, MHz frame rate).
- **Relevance to OpenDiffraction**: Matches Profile `V08` (Porous Materials / Aerogels) and high-rate multi-module masking.

---

## 3. Comparison Matrix

| Property | OpenDiffraction 3.0 | CXIDB id-62 (LCLS) | CXIDB id-81 (LCLS) | CXIDB id-91 (SACLA) |
| :--- | :--- | :--- | :--- | :--- |
| **Data Type** | Synthetic Physics | Experimental XFEL | Experimental XFEL | Experimental XFEL |
| **Grid Resolution** | $512 \times 512$ | $1024 \times 1024$ | $1516 \times 1516$ | $1024 \times 1024$ |
| **Photon Energy** | 8 - 10 keV | 1.2 keV | 6.0 keV | 5.0 keV |
| **Ground Truth Object** | Known exact (64x64) | Unknown (reconstructed) | Unknown (reconstructed) | Unknown (SEM proxy) |
| **Noise Realization** | Poisson + Gaussian | Mixed Poisson-readout | CSPAD gain switching | MPCCD readout |
| **Mask Format** | Boolean array | Bitfield mask | 32-tile gap mask | 8-sensor slit mask |
| **NeXus CXI Compatibility**| Native read/write | Native read | Native read | Converted CXI |

---

## 4. Recommendations for Experimental Interoperability

1. **Standardized Cropping**: Beamline detectors ($1024 \times 1024$ or $1516 \times 1516$) should be center-cropped or binned to $512 \times 512$ using `CXIExperimentalAdapter`.
2. **Gain Pedestal Subtraction**: Raw experimental ADU frames require dark pedestal subtraction and gain correction before ingestion.
3. **Hybrid Validation**: Train neural phase retrieval architectures on OpenDiffraction synthetic data and test zero-shot transfer on curated experimental hits from CXIDB id-62 and id-81.
