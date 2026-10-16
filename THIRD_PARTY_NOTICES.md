# Third-Party Notices and Acknowledgments

OpenDiffraction incorporates, adapts, or references algorithms and data structures from upstream open-source software and scientific research.

---

## 1. Deep Phase Retrieval (DPR)

- **Source Code**: https://github.com/sungyun98/DPR
- **Referenced Upstream Commit**: `1952ee743a4913f1ae5fd3ca51a006c53709e8ae`
- **Copyright**: Copyright (c) 2024, sungyun98. All rights reserved.
- **License**: BSD 2-Clause License
- **Nature of Usage**:
  - The Gaussian Schell-model (GSM) zero-lag centered autocorrelation damping implementation is mathematically modeled after DPR's corrected spatial coherence kernel (`deeppr.dataset.GenerateDiffraction`).
  - HDF5 tensor dataset conventions (`/input`, `/target`, `/mask`) follow DPR training and evaluation conventions.
  - The DPR `false_scale=True` target normalization convention ($O_{\text{dpr}} = O \cdot s$, where $s$ is the photon budget intensity scaling factor) is implemented as a target modality.

### DPR Upstream License Text:
```text
BSD 2-Clause License

Copyright (c) 2024, sungyun98

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 2. Upstream DPR Third-Party Dependencies

The upstream DPR repository includes third-party modules under respective open-source licenses:
- `deeppr/weightedpartialconv2d.py`: Derived from NVIDIA/partialconv (`models/partialconv2d.py`), BSD 3-Clause, Copyright (c) 2018 NVIDIA Corporation.
- `deeppr/loss.py`: Adapted from NVIDIA/partialconv (`models/loss.py`), BSD 3-Clause, Copyright (c) 2019 NVIDIA Corporation.
- `deeppr/ffc.py`: Modified from pkumivision/FFC (`model_zoo/ffc.py`), Apache 2.0, Copyright 2022 Lu Chi; and advimman/lama, Apache 2.0, Copyright 2021 Samsung Research.
- `deeppr/asam.py`: Copied from SamsungLabs/ASAM, Apache 2.0, Copyright 2021 Samsung Research.

*Note: OpenDiffraction is a standalone synthetic data generation and benchmarking package and does not distribute or bundle pretrained neural network weight files (`param_dpr1.pt`, VGG19).*

---

## 3. Coherent X-ray Imaging Data Bank (CXIDB)

- **Standard Reference**: https://cxidb.org/cxi.html
- **Reference Publication**: Maia, F. R. N. C. (2012), "The Coherent X-ray Imaging Data Bank", *Nature Methods* 9, 854–855. DOI: [10.1038/nmeth.2110](https://doi.org/10.1038/nmeth.2110).
- **Usage**: File structure and metadata attribute naming in `opendiffraction_benchmark.cxi` conform to CXIDB/NeXus specifications.
