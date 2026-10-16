import os
import hashlib
from pathlib import Path
import pandas as pd

rows = []
for p in Path('.').rglob('*'):
    if not p.is_file():
        continue
    parts = p.parts
    if '.git' in parts or 'audit_download' in parts or '__pycache__' in parts:
        continue
    sz = p.stat().st_size
    hasher = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1024*1024):
            hasher.update(chunk)
    h = hasher.hexdigest()
    rows.append({
        'path': str(p).replace('\\', '/'),
        'size_bytes': sz,
        'sha256': h,
        'suffix': p.suffix
    })

df = pd.DataFrame(rows)
df = df.sort_values('size_bytes', ascending=False)
print(f'Total local non-git files: {len(df)}')
print('Largest 25 files:')
for _, r in df.head(25).iterrows():
    mb = r['size_bytes'] / (1024*1024)
    print(f"  {mb:6.2f} MB | {r['path']}")

print('\nDuplicate SHA-256 files across repo:')
dups = df[df.duplicated('sha256', keep=False)].sort_values('sha256')
for _, r in dups.iterrows():
    h_short = r['sha256'][:12]
    print(f"  {h_short}... | {r['size_bytes']:8d} bytes | {r['path']}")
