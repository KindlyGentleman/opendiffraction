# Resume Safety and Interruption Audit

**Date**: 2026-10-16  
**Auditor**: Reproducibility and Data Reliability Auditor  
**Project**: OpenDiffraction 3.0  
**Status**: AUDIT COMPLETE (Critical Vulnerability Identified and Solution Designed)

---

## 1. Objective

This audit investigates the transactional safety, idempotency, and crash resilience of `generate_dataset()` in `opendiffraction/generator/pipeline.py` and `DatasetWriter` in `opendiffraction/generator/storage.py`.

---

## 2. Failure Mode Analysis of Existing Implementation

### 2.1 The Dynamic Shard Naming Vulnerability
In the v2.0 pipeline (`opendiffraction/generator/pipeline.py`, lines 181-186):
```python
split_buckets = {}
for t in pending_tasks:
  split_buckets.setdefault(t["split"], []).append(t)

for split_name, split_tasks in split_buckets.items():
  n_shards = int(np.ceil(len(split_tasks) / shard_size))
  for sh_idx in range(n_shards):
    shard_tasks = split_tasks[sh_idx * shard_size : (sh_idx + 1) * shard_size]
    shard_name = f"{split_name}_{sh_idx:03d}.h5"
    shard_path = os.path.join(data_dir, shard_name)
```

**Scenario: Interruption During Generation**
1. Assume split `train` has 1,440 tasks and `shard_size = 480`.
   - Planned shards: `train_000.h5` (tasks 0 to 479), `train_001.h5` (tasks 480 to 959), `train_002.h5` (tasks 960 to 1439).
2. The generator completes `train_000.h5` and records tasks 0 to 479 in `progress.json`.
3. Process is interrupted (e.g. system reboot, SIGINT, or crash).
4. The user resumes generation (`resume=True`).
5. `completed_ids` contains 480 samples.
6. `pending_tasks` for `train` contains only the remaining 960 tasks (tasks 480 to 1439).
7. `split_tasks` has length 960.
8. Outer loop computes `n_shards = ceil(960 / 480) = 2`.
9. First iteration: `sh_idx = 0`.
   - `shard_name = f"train_{0:03d}.h5"` -> `"train_000.h5"`.
10. `writer.write_shard_atomic()` is called with `shard_path = ".../train_000.h5"`!
11. **Catastrophe**: The previously completed `train_000.h5` (tasks 0 to 479) is **completely overwritten** with tasks 480 to 959!
12. Tasks 0 to 479 are lost forever from the HDF5 archive, while remaining marked as completed in `progress.json`.
13. `metadata.csv` now contains duplicate or misaligned rows where row index 0 to 479 points to the overwritten file.

### 2.2 Blind Appending to `metadata.csv`
In `DatasetWriter.append_manifest_rows()`:
- Rows are appended directly via `open(self.metadata_csv, "a")`.
- If an interruption occurs after writing the HDF5 shard but before progress is updated, or during a resumed run, identical sample records are appended again.
- This creates duplicate rows in `metadata.csv`, corrupting dataset length calculations and random-access lookups.

### 2.3 Unsafe `--no-resume` Flag Behavior
In `pipeline.py`:
```python
completed_ids = writer.get_completed_samples() if resume else set()
```
If the user passes `--no-resume`:
- It does not check if the output directory is empty.
- It begins writing `train_000.h5` over existing files.
- It appends new rows onto the existing `metadata.csv` without truncating it.
- Prior data is partially clobbered and mixed with new data.

---

## 3. Required Engineering Solution for OpenDiffraction 3.0

To guarantee crash-safety, idempotency, and deterministic resume, the following architectural rules must be enforced:

### Rule 1: Immutable Master Shard Plan
Shard names must be assigned during the **master plan construction**, before filtering completed samples:
- Every task in a split is assigned a fixed, immutable `(shard_id, row_index)` based on its position in the complete split master list:
  $$\text{shard\_index} = \lfloor i_{\text{split}} / \text{shard\_size} \rfloor, \quad \text{row\_index} = i_{\text{split}} \pmod{\text{shard\_size}}$$
  $$\text{shard\_name} = f"{split}_{shard\_index:03d}.h5"$$
- On resume, shard names remain strictly invariant. Task #500 will always belong to `train_001.h5` at `row_index = 20`, regardless of how many interruptions occur.

### Rule 2: Shard-Level Atomic Transactions
Generation work must proceed shard by shard:
1. For each planned shard `shard_name`:
   - Check if `shard_path` exists on disk AND is verified complete (valid HDF5, contains expected sample count, all sample IDs match manifest).
   - If already valid and complete, skip the entire shard simulation.
2. If incomplete:
   - Simulate all tasks belonging to that shard.
   - Write to `shard_path + ".tmp"`.
   - Flush and close file.
   - Atomically rename `shard_path + ".tmp"` to `shard_path`.
   - Atomically write or update the shard manifest entry.
   - Update `progress.json`.

### Rule 3: Transactional Manifest Tracking
- Store per-shard manifests: `manifests/shards/{shard_name}.csv`.
- Compile `metadata.csv` from verified shard manifests, or maintain an in-memory set of unique sample IDs so duplicates are never written.

### Rule 4: Explicit Overwrite Policy
- If `--no-resume` is passed and the output directory contains existing data or metadata:
  - Fail immediately with a clear error unless an explicit `--overwrite` flag is supplied.
  - Never silently destroy or interleave data.

---

## 4. Acceptance Criteria
1. Interruption after any number of completed shards can be resumed with zero data loss.
2. Resumed generation produces identical shard contents and identical metadata records.
3. No duplicate sample IDs exist in `metadata.csv`.
4. All row indices in `metadata.csv` correspond exactly to the dataset index inside the specified HDF5 shard.
