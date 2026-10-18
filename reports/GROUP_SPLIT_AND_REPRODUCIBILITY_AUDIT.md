# Gate V Audit Report: Object Identity, Group Splits, and Replay Reproducibility

**Date:** 2026-10-17  
**Repository:** `KindlyGentleman/opendiffraction`  
**Target Version:** OpenDiffraction 3.1.0  
**Branch:** `remediation/v3.1-scientific-audit`  
**Status:** PASS  

---

## 1. Executive Summary

This audit assesses the scientific rigor and data hygiene of OpenDiffraction's object generation, multi-acquisition grouping, dataset splitting, and seed-level reproducibility. In inverse problems and scientific machine learning, benchmark credibility requires two non-negotiable guarantees:
1. **Zero Data Leakage:** All multiple acquisitions (different photon budgets, coherence lengths, beam displacements, noise realizations) of the same underlying physical specimen must reside within the same dataset split. No object may ever appear in both training and evaluation partitions.
2. **Deterministic Replay:** Any measurement, intermediate field, or ground truth target in an HDF5 shard must be exactly reproducible bit-for-bit from its recorded metadata seeds and generation parameters.

Both properties were audited, verified, and reinforced with automated regression tests.

---

## 2. Object Identity Invariance Across Multi-Acquisition Shots

### 2.1 The Multi-Acquisition Concept
OpenDiffraction models high-throughput XFEL and synchrotron serial crystallography where a single specimen family or synthetic particle may be probed under multiple acquisition conditions:
- **Acquisition 0 (Reference):** Standard configured flux and nominal coherence.
- **Acquisition 1 (Attenuated Flux):** 10x reduced photon budget ($0.1 \times \text{flux}$), maximizing shot noise.
- **Acquisition 2 (Damped Coherence):** Spatial coherence length reduced by 40% ($0.6 \times l_{\text{coh}}$).
- **Acquisition 3 (Perturbed Readout & Alignment):** Readout noise increased by 50% ($1.5 \times \sigma_{\text{readout}}$) with random beam subpixel displacements ($\pm 2.0$ px).

For physical validity, the underlying specimen's unscaled spatial transmission function:
$$O(r) = A(r) \exp(i \phi(r))$$
must remain strictly identical across all 4 acquisition shots.

### 2.2 Identification and Fix of Profile V47 Discrepancy
During comprehensive scanning across all 48 profiles, 47 profiles exhibited exact bitwise identity across shots 0 through 3 ($\max |O_{\text{acq}} - O_0| = 0.0$). However, Profile V47 (3D Particle Euler Scattering under forward model F6) failed invariance:
- **Root Cause:** In `profiles_48.py`, `sample_profile_48` generated a 3D volume `vol` using the specimen random generator `obj_rng`, but did not persist `vol` into `metadata["volume_3d"]`. When `simulate_single_task` subsequently dispatched forward model F6 in `simulate_forward_model`, `forward.py` detected the missing key and re-generated a new 3D cluster using `acq_rng` (which varies per acquisition shot). Consequently, each acquisition shot probed a completely different 3D particle.
- **Remediation:** In `opendiffraction/variations/profiles_48.py`, `metadata["volume_3d"] = vol` was stored explicitly, and the projected 2D transmission `amplitude` was computed using `project_volume_to_2d(vol, angles=angles)`.
- **Validation Result:** After remediation, all 48 profiles satisfy:
$$\max_{a \in \{1, 2, 3\}} \|O_a - O_0\|_\infty = 0.0 \quad \text{and} \quad \max_{a \in \{1, 2, 3\}} \|\phi_a - \phi_0\|_\infty = 0.0$$

---

## 3. Group-Disjoint Dataset Partitioning and Zero-Leakage Guarantee

### 3.1 Splitting Mechanism
Dataset partitioning is implemented in `opendiffraction/generator/splits.py` via `assign_group_to_split(group_id, seed, train_ratio, valid_ratio, is_ood)`.
- The assignment uses a SHA-256 digest of the concatenated seed and `group_id`:
  $$\text{hash\_val} = \text{SHA256}(\text{seed} \parallel \text{group\_id}) \pmod{10000}$$
  $$\text{frac} = \frac{\text{hash\_val}}{10000}$$
- `group_id` is set to `base_object_id` (e.g., `obj_V01_00000`).
- Because all 4 acquisitions of an object share the identical `group_id`, they map deterministically to the exact same split.

### 3.2 Partition Disjointness Verification
A full master task plan was generated for 192 samples (48 unique objects x 4 acquisitions). Object sets were extracted per split:
- **Train partition:** $S_{\text{train}}$
- **Validation partition:** $S_{\text{valid}}$
- **In-distribution Test partition:** $S_{\text{test}}$
- **Out-of-Distribution Test partition:** $S_{\text{test\_ood}}$

Pairwise set intersections were computed:
$$S_{\text{train}} \cap S_{\text{valid}} = \emptyset$$
$$S_{\text{train}} \cap S_{\text{test}} = \emptyset$$
$$S_{\text{train}} \cap S_{\text{test\_ood}} = \emptyset$$
$$S_{\text{valid}} \cap S_{\text{test}} = \emptyset$$
$$S_{\text{valid}} \cap S_{\text{test\_ood}} = \emptyset$$
$$S_{\text{test}} \cap S_{\text{test\_ood}} = \emptyset$$

Zero object leakage is mathematically guaranteed and empirically verified.

### 3.3 Out-of-Distribution (OOD) Isolation
Profiles designated with `is_ood=True` (Profiles V43, V44, V45, V46, and V48) test extreme physical regimes: pure phase objects, vortex phase singularities, near-field Fresnel propagation, broadband polychromatic illumination, and compound stress conditions.
- `assign_group_to_split` routes `is_ood=True` unconditionally to `test_ood`.
- No OOD specimen or acquisition ever contaminates training or validation sets.

---

## 4. Deterministic Replay from Manifest Metadata

### 4.1 Replay Contract
Each row in `metadata.csv` preserves all seeds and physical parameters necessary to reconstitute the simulation:
- `object_seed`: Seed for specimen geometry, material density, and phase.
- `acquisition_seed`: Seed for photon arrivals, readout noise, and beam wander.
- `profile_id`: Variation profile specification (V01 - V48).
- `object_index` and `acquisition_id`: Specimen and shot numbering.

### 4.2 Replay Verification Against Stored HDF5 Shards
A complete generation run was executed into a temporary directory writing extended HDF5 shards. Samples were randomly selected and re-simulated solely from `metadata.csv` parameters. Re-simulated arrays were compared against stored HDF5 datasets:

| Dataset Field | Shape | Stored vs Replayed Discrepancy | Result |
|---|---|---|---|
| `intensity_observed` | $(512, 512)$ | $\max |I_{\text{replay}} - I_{\text{stored}}| = 0.0$ | PASS |
| `intensity_raw` | $(512, 512)$ | $\max |I_{\text{replay}} - I_{\text{stored}}| = 0.0$ | PASS |
| `valid_mask` | $(512, 512)$ | $\sum (M_{\text{replay}} \oplus M_{\text{stored}}) = 0$ | PASS |
| `object_amplitude` | $(64, 64)$ | $\max |O_{\text{replay}} - O_{\text{stored}}| = 0.0$ | PASS |

Every field achieves bitwise exact replication.

---

## 5. Automated Regression Test Suite

All findings and contracts are permanently guarded by automated unit tests in `tests/test_splits_and_identity_remediation.py`:
- `test_object_identity_invariance_across_acquisitions`: Asserts $\max |O_a - O_0| = 0$ across multiple acquisitions for diverse physical profiles.
- `test_group_disjoint_splitting_zero_leakage`: Asserts pairwise null intersections across all partition object sets.
- `test_ood_profile_assignment_isolation`: Asserts strict isolation of OOD profiles to `test_ood`.
- `test_deterministic_replay_from_manifest`: Generates an on-disk dataset, parses `metadata.csv`, and validates bitwise identical re-simulation against stored HDF5 arrays.

All 4 tests pass with 100% success rate (20.03s execution time).

---

## 6. Gate V Sign-off

- **Object Identity Invariance:** PASS (100% across all 48 profiles).
- **Group-Disjoint Splitting:** PASS (0 object leakage).
- **OOD Isolation:** PASS (Strictly isolated in `test_ood`).
- **Deterministic Replay:** PASS (Bitwise exact match across all stored modalities).
- **Gate V Status:** **PASS**
