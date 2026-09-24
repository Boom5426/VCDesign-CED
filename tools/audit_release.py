#!/usr/bin/env python3
"""Check the staged snapshot without changing any file."""
from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "provenance" / "SOURCE_MAP.tsv"
PACKAGE_HASHES = ROOT / "provenance" / "PACKAGE_SHA256SUMS"
IDENTITY = re.compile(
    r"(?:(?<![A-Za-z0-9])/(?:home|data|Users)/[^/\s]+/|"
    r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\."
    r"(?:com|edu|org|net|io|gov|ac\.[A-Za-z]{2})\b|git@github\.com)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".py", ".json", ".csv", ".tsv", ".md", ".tex", ".txt", ".sh"}
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "inputs", "outputs", "dist", "build"}


def release_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info")
               for part in relative.parts):
            continue
        if path == PACKAGE_HASHES or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink in release: {relative}")
        if path.is_file():
            files.add(relative.as_posix())
    return files


def main() -> int:
    checked = 0
    errors: list[str] = []
    with MAP.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = ROOT / row["release_path"]
            if not path.is_file():
                errors.append(f"missing: {row['release_path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                errors.append(f"changed: {row['release_path']}")
            checked += 1
    package_checked = 0
    if PACKAGE_HASHES.is_file():
        recorded = set()
        for line in PACKAGE_HASHES.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            if relative in recorded or relative.startswith("/") or ".." in Path(relative).parts:
                errors.append(f"invalid manifest path: {relative}")
                continue
            recorded.add(relative)
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"missing from package: {relative}")
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"changed package file: {relative}")
            package_checked += 1
        for relative in sorted(release_files() - recorded):
            errors.append(f"unlisted package file: {relative}")
        for relative in sorted(recorded - release_files()):
            errors.append(f"manifest-only package file: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        try:
            content = path.read_text()
        except UnicodeError:
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if IDENTITY.search(line):
                errors.append(f"identity or machine path: {path.relative_to(ROOT)}:{number}")
    for path in ROOT.rglob("*.pptx"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith((".xml", ".rels")):
                    continue
                content = archive.read(member).decode("utf-8", errors="replace")
                if IDENTITY.search(content):
                    errors.append(f"embedded identity or machine path: {path.relative_to(ROOT)}:{member}")
    for error in errors:
        print(f"FAIL {error}")
    print(f"verified {checked} source snapshots and {package_checked} package files; "
          f"{len(errors)} audit findings")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
