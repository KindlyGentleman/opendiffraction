# OpenDiffraction Remediation Tasks

Branch: `remediation/v3.1-scientific-audit`
Target: OpenDiffraction 3.1.0 Master Scientific Remediation

## Prioritized Issue Register (OD-001 through OD-030)

| ID | Sev | Category | Description | Status |
|:---:|:---:|:---|:---|:---:|
| **OD-001** | P0 | physics_dispatch | F1-F6 model labels dispatched through production physics | VERIFIED |
| **OD-002** | P0 | masking_leakage | Detector input strictly zeroed out behind occlusions | VERIFIED |
| **OD-003** | P0 | target_fallback | Explicit KeyError when clean supervised targets are missing | VERIFIED |
| **OD-004** | P0 | compact_modalities | Explicit storage schemas for compact vs extended tasks | VERIFIED |
| **OD-005** | P0 | manifest_enforcement | Benchmark loaders enforce exact manifest sample_id mapping | VERIFIED |
| **OD-006** | P1 | fresnel_propagation | Fresnel propagation spatial centering and zero-distance identity | VERIFIED |
| **OD-007** | P1 | beam_displacement | Beam displacement shifts detector center via wavevector tilt | VERIFIED |
| **OD-008** | P1 | profile_effects | Profile parameters (spherical aberration, coma) modulate probe | VERIFIED |
| **OD-009** | P1 | background_scattering | Background halos, flare, and cosmic hits connected to production | VERIFIED |
| **OD-010** | P1 | f6_euler_orientation | 3D Euler volume rotation invariant across shots | VERIFIED |
| **OD-011** | P1 | manifest_physics | Geometry metadata synchronized with active scenarios S1-S6 | VERIFIED |
| **OD-012** | P1 | dpr_scaling_contract | DPR target scaling distinguished from physical amplitude | VERIFIED |
| **OD-013** | P1 | hio_er_conventions | Fourier coordinate and phase transfer between HIO and ER | VERIFIED |
| **OD-014** | P1 | missing_data_constraints | Masked Fourier pixels unconstrained during modulus updates | VERIFIED |
| **OD-015** | P1 | r_factor_definition | Standard crystallographic Fourier R-factor implementation | VERIFIED |
| **OD-016** | P1 | ssim_windowing | Standard local windowed SSIM vs global scalar SSIM | VERIFIED |
| **OD-017** | P1 | invalid_correlation | Replace target vs clip(target) with reconstruction correlation | VERIFIED |
| **OD-018** | P1 | hio_support_assumptions | Separate support information regimes (oracle, loose, estimated) | VERIFIED |
| **OD-019** | P1 | silent_mask_replacement | Remove silent mask reset when valid coverage < 20% | VERIFIED |
| **OD-020** | P1 | quick_profile_coverage | Quick preset exercises all 48 profiles (1 per profile) | VERIFIED |
| **OD-021** | P1 | memory_safety | IncrementalShardWriter streaming (<120 MiB RAM peak) | VERIFIED |
| **OD-022** | P1 | resume_validation | SHA-256 config hash and sample checksum verification | VERIFIED |
| **OD-023** | P1 | hit_finding_balance | Balanced 4-class Hit Finding with 3-sigma noise floor gating | VERIFIED |
| **OD-024** | P1 | localized_metrics | Inpainting evaluated with Masked Region MAE and RMSE | VERIFIED |
| **OD-025** | P2 | preset_consistency | Harmonize sample counts across CLI, YAML, and README | VERIFIED |
| **OD-026** | P2 | release_consistency | Version 3.1.0 synchronized across metadata and documentation | VERIFIED |
| **OD-027** | P2 | ci_coverage | CI executes complete unit and regression test suite | VERIFIED |
| **OD-028** | P2 | physics_unit_compatibility | Mathematical oracles (Airy, Parseval, Friedel, Fresnel) | VERIFIED |
| **OD-029** | P2 | task_scope_truthfulness | Truthful scope: T1-T4 benchmarked, T5-T12 manifest roadmaps | VERIFIED |
| **OD-030** | P2 | cxi_experimental_scope | Explicit disclosure of synthetic format vs real beamtime | VERIFIED |

---

## Remediation Roadmap and Acceptance Phases

### Phase A: Fail-First Defect Reproduction
- [x] Reproduce OD-001 (F1-F6 dispatch bypass).
- [x] Reproduce OD-002 (detector mask raw flux leakage).
- [x] Reproduce OD-003 (clean target silent fallback).
- [x] Reproduce OD-006 (Fresnel ASM real-space coordinate inversion).
- [x] Implement fail-first test for OD-005 (manifest sample_id alignment).
- [x] Implement fail-first test for OD-007 (beam center displacement vs Fourier invariance).
- [x] Implement fail-first test for OD-008 (spherical aberration in probe).
- [x] Implement fail-first test for OD-013 (HIO to ER phase transfer).
- [x] Implement fail-first test for OD-016 (windowed vs global SSIM).
- [x] Implement fail-first test for OD-017 (reconstruction correlation diagnostic).
- [x] Implement fail-first test for OD-019 (severe mask silent reset).
- [x] Implement fail-first test for OD-020 (quick preset profile coverage).

### Phase B: Forward-Model Pipeline Revalidation
- [x] Verify F1 Fraunhofer centering, Parseval conservation, and Friedel symmetry.
- [x] Verify F2 partial coherence GSM lag centering and mode expansion.
- [x] Verify F4 Fresnel ASM propagation and z=0 identity.
- [x] Verify F5 polychromatic pink-beam spectral integration.
- [x] Verify F6 3D volume Euler rotation and transmission projection.

### Phase C: 48 Profiles and Six Scenarios Verification
- [x] Repair OD-007: Implement true diffraction shift on detector or incident beam tilt.
- [x] Repair OD-008: Add spherical aberration $W \propto r^4$ to `generate_probe` for V17.
- [x] Repair OD-011: Connect S1-S6 scenarios to pipeline metadata dynamically.
- [x] Repair OD-020: Ensure `--preset quick` exercises all 48 profiles (V01-V48).

### Phase D: Detector Observability, Masks and Storage
- [x] Enforce zero leakage barrier (`intensity_observed[~mask] == 0.0`).
- [x] Repair OD-019: Remove silent mask reset when coverage < 20% in `masks.py`.
- [x] Enforce explicit target contracts for Compact and Extended HDF5 storage.

### Phase E: Object Identity, Splits and Reproducibility
- [x] Ensure base object identity invariance across acquisition shots.
- [x] Ensure 100% group-disjoint splits across train, validation, test, and test_ood.
- [x] Validate bitwise deterministic replay from metadata seeds.

### Phase F: Phase Retrieval and Evaluation Metrics
- [x] Repair OD-013: Align Fourier phase coordinates and state transfer between HIO and ER.
- [x] Repair OD-016: Implement standard local windowed SSIM in `metrics.py`.
- [x] Repair OD-017: Compute Pearson correlation on reconstructed object in `eval_phase_retrieval.py`.
- [x] Repair OD-018: Add explicit support regimes (disc, box, oracle, autocorrelation).

### Phase G: Benchmark Evaluation Rebuilding
- [x] Repair OD-005: Enforce strict `sample_id` mapping in loaders and benchmark scripts.
- [x] Re-evaluate T1-T4 baselines on balanced manifests with 3-sigma readout noise gating.
- [x] Maintain task scope matrix for T5-T12.

### Phase H: Resource Safety and Crash Recovery
- [x] Verify streaming `IncrementalShardWriter` peak RAM < 120 MiB.
- [x] Verify SHA-256 configuration hash validation during shard resume.

### Phase I: Figure and Visualization Provenance
- [x] Regenerate Figures A through G from fixed seeds (300 DPI PNG + vector PDF).

### Phase J: Independent Adversarial Re-Audit
- [x] Validate 5 independent mathematical oracles.
- [x] Document synthetic format vs real experimental CXI boundary.

### Phase K: Release Consistency, CI and Documentation
- [x] Repair OD-025: Reconcile CLI, YAML, and README sample counts and version 3.1.0.
- [x] Repair OD-027: Update `.github/workflows/tests.yml` to run full test suite.
- [x] Update CHANGELOG, DATASET_CARD, and README for v3.1.0.
