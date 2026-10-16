# Storage Baseline and Capacity Projections

**Date**: 2026-10-16  
**Auditor**: Scientific Data Engineer and Storage Architect  
**Project**: OpenDiffraction 3.0  
**Target Release**: 10,000 Diffraction Observations

---

## 1. Array Dimensions and Raw Data Volumes

Each diffraction observation consists of multi-dimensional arrays representing physical specimen properties and detector readouts:

| Array Field | Dimensions | Data Type | Raw Size per Sample (Bytes) | Raw Size per Sample (KiB/MiB) |
|---|---|---|---|---|
| `input` / `intensity_measured` | `(1, 512, 512)` | `float32` | 1,048,576 | 1.00 MiB (1,024 KiB) |
| `mask` / `valid_mask` | `(1, 512, 512)` | `bool` | 262,144 | 256.00 KiB (0.25 MiB) |
| `target` / `target_dpr` | `(1, 64, 64)` | `float32` | 16,384 | 16.00 KiB |
| `target_physical` | `(1, 64, 64)` | `float32` | 16,384 | 16.00 KiB |
| `object_amplitude` | `(1, 64, 64)` | `float32` | 16,384 | 16.00 KiB |
| `object_phase` | `(1, 64, 64)` | `float32` | 16,384 | 16.00 KiB |
| `intensity_ideal` | `(1, 512, 512)` | `float32` | 1,048,576 | 1.00 MiB (1,024 KiB) |
| `intensity_expected` | `(1, 512, 512)` | `float32` | 1,048,576 | 1.00 MiB (1,024 KiB) |

---

## 2. Storage Modes Specification

To balance machine-learning training efficiency with full scientific auditability, OpenDiffraction implements two complementary storage modes:

### 2.1 Mode 1: Compact Training Mode
Optimized for standard deep learning pipelines (DPR, CNN, Vision Transformer):
- **Stored Datasets**: `input` (512x512 float32), `mask` (512x512 bool), `target` (64x64 float32).
- **Uncompressed Footprint per Sample**: 1,327,104 bytes (~1.27 MiB).
- **Compressed Footprint per Sample (gzip level 4)**: ~380 to 450 KiB (compression ratio ~2.8x to 3.3x).
- **Total Uncompressed Footprint (10,000 samples)**: ~12.36 GiB.
- **Estimated Release Disk Footprint (10,000 samples)**: **3.8 to 4.5 GiB**.

### 2.2 Mode 2: Extended Scientific Mode
Retains all intermediate physical states for inverse problem research and calibration:
- **Stored Datasets**: All compact arrays plus `intensity_ideal`, `intensity_expected`, `object_amplitude`, `object_phase`, `target_physical`.
- **Uncompressed Footprint per Sample**: 3,457,024 bytes (~3.30 MiB).
- **Compressed Footprint per Sample (gzip level 4)**: ~850 to 1,100 KiB.
- **Total Uncompressed Footprint (10,000 samples)**: ~32.20 GiB.
- **Estimated Release Disk Footprint (10,000 samples)**: **8.5 to 11.0 GiB**.

---

## 3. Compression Filter Comparison Strategy

In Phase 2, a benchmark across four candidate HDF5 filter configurations will be evaluated on representative samples:
1. `gzip` (level 1): High write throughput, moderate compression.
2. `gzip` (level 4): Recommended balanced trade-off between read speed and storage size.
3. `lzf`: Built-in HDF5 LZF filter, minimal CPU overhead, lower compression ratio.
4. `none`: Uncompressed raw baseline for measuring pure I/O speed.

---

## 4. Shard Layout and Access Latency

- **Target Shard Size**: 400 to 500 samples per shard.
  - At 500 samples per shard, a compact shard is ~200 MiB, fitting comfortably in RAM during generation.
  - Total shards for 10,000 samples: ~20 to 25 HDF5 files across splits (`train`, `valid`, `test`, `test_ood`).
- **HDF5 Chunk Shape**: `(1, 1, H, W)` per dataset chunk.
  - Chunking on individual sample slice `(1, 1, 512, 512)` allows constant-time $O(1)$ random access to any sample without decompressing surrounding samples in the shard.
- **Read Throughput Requirement**: Over 50 samples/second for PyTorch DataLoader on CPU with zero worker contention.
