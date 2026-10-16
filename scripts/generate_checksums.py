"""Regenerate kaggle_release/checksums.sha256 with SHA-256 for all files."""

import hashlib
import os

RELEASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kaggle_release"))
CHECKSUM_FILE = os.path.join(RELEASE_DIR, "checksums.sha256")

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    entries = []
    for root, dirs, files in os.walk(RELEASE_DIR):
        dirs.sort()
        for file in sorted(files):
            if file == "checksums.sha256":
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, RELEASE_DIR).replace("\\", "/")
            sha = compute_sha256(full_path)
            entries.append(f"{sha}  {rel_path}")

    entries.sort(key=lambda x: x.split("  ")[1])

    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(entries) + "\n")

    print(f"Generated {len(entries)} checksums in {CHECKSUM_FILE}")

if __name__ == "__main__":
    main()
