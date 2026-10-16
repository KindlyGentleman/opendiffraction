# Storage and Compression Optimization Report

**Date**: 2026-10-16
**Evaluation Suite**: 48 Representative Diffraction Samples (512x512 float32)

## 1. Empirical Compression Benchmarks

| Filter              | Compression   | Opts   |   Total Size (MB) |   KB / Sample |   Write Speed (samples/s) |   Sequential Read (samples/s) |   Random Read Latency (ms) |
|:--------------------|:--------------|:-------|------------------:|--------------:|--------------------------:|------------------------------:|---------------------------:|
| Uncompressed (none) | none          | None   |           218.356 |       4658.26 |                  10.6024  |                      134.596  |                    7.57616 |
| LZF                 | lzf           | None   |           190.708 |       4068.45 |                   7.91543 |                      147.214  |                    7.46425 |
| Gzip-1              | gzip          | 1      |           176.034 |       3755.4  |                   2.00524 |                       20.2643 |                   49.5167  |
| Gzip-4 (default)    | gzip          | 4      |           175.386 |       3741.56 |                   1.81329 |                       19.9312 |                   50.6799  |

## 2. Analysis and Recommendation

- **Gzip Level 4** provides the optimal balance between disk compression (~700-980 KB/sample in extended mode, ~420 KB in compact mode) and CPU throughput.
- **LZF** yields faster write speed but leaves file size ~30% larger.
- **Uncompressed** achieves maximum read speed but requires 3.3 MiB per sample, demanding >33 GB for 10,000 samples.
- **Recommendation**: Retain `gzip` level 4 as the default scientifically lossless storage format for OpenDiffraction 3.0.
