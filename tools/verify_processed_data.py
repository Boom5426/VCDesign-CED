#!/usr/bin/env python3
"""Verify a downloaded processed-data snapshot against its SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing the downloaded DATA_MANIFEST.json")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "DATA_MANIFEST.json").read_text())
    if manifest.get("schema") != "VCDESIGN_PROCESSED_DATA_V1":
        raise ValueError("unrecognized processed-data manifest schema")
    seen: set[str] = set()
    failures: list[str] = []
    for entry in manifest["files"]:
        relative = entry["path"]
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in seen:
            raise ValueError(f"invalid or duplicate manifest path: {relative}")
        seen.add(relative)
        candidate = root / path
        if not candidate.is_file() or candidate.is_symlink():
            failures.append(f"missing or symlink: {relative}")
            continue
        if candidate.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {relative}")
            continue
        if digest(candidate) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"verified {len(seen) - len(failures)}/{len(seen)} processed-data files")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
