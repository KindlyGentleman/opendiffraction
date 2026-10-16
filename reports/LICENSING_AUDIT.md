# OpenDiffraction Legal Licensing Audit Report

**Audit Date**: October 16, 2026  
**Auditor**: Independent Open-Source Licensing Reviewer  
**Audit Scope**: Root `LICENSE`, `LICENSE-DATA.md`, `LICENSES.md`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, Kaggle dataset metadata, and upstream DPR terms.

---

## 1. Dual Licensing Resolution

| Issue Identified | Initial State | Corrected State | Legal Rationale |
| :--- | :--- | :--- | :--- |
| **Mixed Root License** | Single `LICENSE` file conflated BSD 2-Clause and CC BY 4.0 terms. | Strict separation: `LICENSE` holds canonical BSD 2-Clause for software; `LICENSE-DATA.md` holds CC BY 4.0 for data. | Ensures unambiguous legal interpretation by standard license scanners and enterprise compliance teams. |
| **Copyright Notice** | Ambiguous `"Copyright (c) 2026, OpenDiffraction Contributors."` | `"Copyright (c) 2026 Muhammad Kamal. All rights reserved."` | Explicitly attributes individual authorship and stewardship to project creator Muhammad Kamal. |
| **Upstream BSD Notice** | Generic informal mention in license appendix. | Explicit upstream BSD 2-Clause notice with copyright to `sungyun98 (2024)` in `LICENSE` and `THIRD_PARTY_NOTICES.md`. | Fully complies with Clause 1 and 2 of upstream DPR BSD 2-Clause license terms. |
| **Kaggle License Tag** | Set to `"Attribution 4.0 International (CC BY 4.0)"`. | Verified matching CC BY 4.0 identifier supported by Kaggle (`CC-BY-4.0`). | Aligns with Kaggle automated platform validation heuristics. |
| **Commercial Terms** | No non-commercial restrictions. | Retained standard CC BY 4.0 without non-commercial clauses. | Maximizes open-science utility and prevents downstream license incompatibility. |

---

## 2. Third-Party Intellectual Property & Provenance Verification

- **Code Provenance**: Confirmed that OpenDiffraction is an independent implementation written in modern Python. The zero-lag GSM coherence kernel and `false_scale=True` target conventions are mathematical implementations adapted with explicit attribution.
- **Third-Party Model Weights**: Confirmed that no binary neural network weights from upstream DPR or NVIDIA/Samsung are bundled with the software or dataset.
- **Experimental Data Rights**: Confirmed that 100% of the published data arrays are procedurally synthesized by the OpenDiffraction simulation engine from mathematical phantoms and noise models; no proprietary experimental synchrotron or XFEL beamline data files are redistributed.

---

## 3. Compliance Verdict

**STATUS: PASS / FULLY COMPLIANT**  
Both software and dataset licenses are verified, legally sound, and strictly scoped.
