import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

audit_dir = Path('audit_download')
print('=== AUDIT OF DOWNLOADED KAGGLE DATASET ===')

# 1. Inspect metadata.csv
meta_path = audit_dir / 'metadata.csv'
df = pd.read_csv(meta_path)
print(f'Metadata rows: {len(df)}')
print(f'Unique sample_ids in metadata: {df["sample_id"].nunique()}')
print('Metadata splits distribution:')
print(df['split'].value_counts())
print('Metadata variation families:')
print(df['variation_family'].value_counts().to_dict())
print('Metadata shard_name references:')
print(df['shard_name'].value_counts().to_dict())

# 2. Inspect actual HDF5 shards in data/
data_dir = audit_dir / 'data'
shards = sorted(list(data_dir.glob('*.h5')))
print(f'\nHDF5 shards found in data/: {len(shards)}')

shard_info = []

for s in shards:
    with h5py.File(s, 'r') as f:
        keys = list(f.keys())
        first_key = keys[0]
        n_samples = f[first_key].shape[0]
        print(f'\nShard: {s.name}')
        print(f'  Total sample count: {n_samples}')
        print('  Datasets:')
        for k in keys:
            ds = f[k]
            print(f'    - {k}: shape={ds.shape}, dtype={ds.dtype}')
            
        shard_info.append({
            'shard_name': s.name,
            'num_samples': n_samples,
            'datasets': ', '.join(keys),
            'filesize_bytes': s.stat().st_size
        })

total_shard_samples = sum(x['num_samples'] for x in shard_info)
print(f'\nTotal samples across all HDF5 shards: {total_shard_samples}')

# Check metadata row resolution to shard and row_index
print('\nVerifying metadata row-to-shard resolution:')
resolves = 0
unresolved = 0
for i, row in df.iterrows():
    shard_name = row['shard_name']
    row_idx = int(row['row_index'])
    s_path = data_dir / shard_name
    if not s_path.exists():
        unresolved += 1
        continue
    with h5py.File(s_path, 'r') as f:
        first_key = list(f.keys())[0]
        if row_idx >= f[first_key].shape[0]:
            unresolved += 1
        else:
            resolves += 1

print(f'  Resolved rows: {resolves} / {len(df)}')
print(f'  Unresolved rows: {unresolved}')

# 3. Check every shard row correspondence to metadata
print('\nVerifying every HDF5 shard row has a metadata entry:')
shard_row_coverage = {}
for s in shards:
    s_name = s.name
    with h5py.File(s, 'r') as f:
        n = f[list(f.keys())[0]].shape[0]
    sub_df = df[df['shard_name'] == s_name]
    indices_in_meta = set(sub_df['row_index'].values)
    expected_indices = set(range(n))
    missing_in_meta = expected_indices - indices_in_meta
    shard_row_coverage[s_name] = {
        'total_in_shard': n,
        'in_meta': len(sub_df),
        'missing_in_meta': len(missing_in_meta)
    }
    print(f'  Shard {s_name}: {n} rows, {len(sub_df)} in metadata, missing={len(missing_in_meta)}')

# 4. Inspect CXI file
cxi_path = audit_dir / 'opendiffraction_benchmark.cxi'
if cxi_path.exists():
    print(f'\nCXI file exists: {cxi_path.name} ({cxi_path.stat().st_size / 1024:.1f} KB)')
    with h5py.File(cxi_path, 'r') as f:
        print('  CXI top-level keys:', list(f.keys()))
        def walk_cxi(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'    Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}')
        f.visititems(walk_cxi)
else:
    print('CXI file does not exist!')

# 5. Check tasks manifests
tasks_dir = audit_dir / 'tasks'
if tasks_dir.exists():
    tasks = sorted(list(tasks_dir.glob('*.csv')))
    print(f'\nTask manifests found: {len(tasks)}')
    for t in tasks:
        tdf = pd.read_csv(t)
        print(f'  - {t.name}: {len(tdf)} rows, columns={list(tdf.columns)}')

# 6. Check checksums.sha256
chk_path = audit_dir / 'checksums.sha256'
print(f'\nVerifying checksums.sha256:')
if chk_path.exists():
    with open(chk_path, 'r') as f:
        chk_lines = [line.strip().split() for line in f if line.strip()]
    print(f'  Checksum entries in file: {len(chk_lines)}')
    matched = 0
    mismatched = 0
    missing = 0
    for h, p in chk_lines:
        fp = audit_dir / p
        if not fp.exists():
            missing += 1
            print(f'    MISSING: {p}')
        else:
            hasher = hashlib.sha256()
            with open(fp, 'rb') as f_in:
                while chunk := f_in.read(1024*1024):
                    hasher.update(chunk)
            calc_h = hasher.hexdigest().upper()
            if calc_h == h.upper():
                matched += 1
            else:
                mismatched += 1
                print(f'    MISMATCH: {p} (expected {h[:8]}..., got {calc_h[:8]}...)')
    print(f'  Result: matched={matched}, mismatched={mismatched}, missing={missing}')
