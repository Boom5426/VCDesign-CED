#!/usr/bin/env python3
"""Build and verify a portable VCDesign code package."""
from __future__ import annotations

import argparse
import hashlib
import re
import stat
import zipfile
from pathlib import Path


ARCHIVE_ROOT = "VCDesign_code_package"
ROOT_FILES = (
    "CODE_PACKAGE_README.md",
    "PROJECT_STRUCTURE.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-cpu.txt",
)
TREE_PATTERNS = (
    "configs/**/*.json",
    "examples/**/*.py",
    "src/**/*.py",
)
TOOL_FILES = ("tools/verify_processed_data.py",)
FORBIDDEN_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "inputs",
    "outputs",
    "records",
}
FORBIDDEN_SUFFIXES = {".h5ad", ".npy", ".npz", ".parquet", ".pt", ".pyc"}
IDENTITY_PATTERNS = (
    re.compile(rb"Boom5426", re.IGNORECASE),
    re.compile(rb"runs in the ``boom`` env", re.IGNORECASE),
    re.compile(rb"/home/[A-Za-z0-9_.-]+/"),
    re.compile(rb"[A-Za-z]:\\\\Users\\\\[^\\\\]+\\\\", re.IGNORECASE),
)


def _payloads(root: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for relative in ROOT_FILES:
        source = root / relative
        target = "README.md" if relative == "CODE_PACKAGE_README.md" else relative
        data = source.read_bytes()
        if relative == "PROJECT_STRUCTURE.md":
            data = data.replace(
                b"[REPRODUCE.md](REPRODUCE.md)", b"[README.md](README.md)"
            )
        payloads[target] = data
    for pattern in TREE_PATTERNS:
        for source in sorted(root.glob(pattern)):
            if source.is_file():
                payloads[source.relative_to(root).as_posix()] = source.read_bytes()
    for relative in TOOL_FILES:
        payloads[relative] = (root / relative).read_bytes()
    return payloads


def _audit_payloads(payloads: dict[str, bytes]) -> None:
    if not payloads:
        raise ValueError("the package contains no files")
    for relative, data in payloads.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {relative}")
        if FORBIDDEN_PARTS.intersection(path.parts):
            raise ValueError(f"forbidden directory in package: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden generated or data file: {relative}")
        for pattern in IDENTITY_PATTERNS:
            if pattern.search(data):
                raise ValueError(f"identity or machine-specific path found in {relative}")


def _manifest(payloads: dict[str, bytes]) -> bytes:
    lines = [
        f"{hashlib.sha256(data).hexdigest()}  {relative}"
        for relative, data in sorted(payloads.items())
    ]
    return ("\n".join(lines) + "\n").encode()


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        f"{ARCHIVE_ROOT}/{relative}", date_time=(2026, 9, 24, 0, 0, 0)
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    payloads = _payloads(root)
    _audit_payloads(payloads)
    payloads["CODE_PACKAGE_MANIFEST.sha256"] = _manifest(payloads)
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative, data in sorted(payloads.items()):
            archive.writestr(_zip_info(relative), data)


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("archive is empty")
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")
        prefix = f"{ARCHIVE_ROOT}/"
        payloads: dict[str, bytes] = {}
        for member in members:
            if not member.filename.startswith(prefix):
                raise ValueError(f"path outside archive root: {member.filename}")
            relative = member.filename[len(prefix):]
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive contains a symlink: {relative}")
            payloads[relative] = archive.read(member)
        manifest = payloads.pop("CODE_PACKAGE_MANIFEST.sha256", None)
        if manifest is None:
            raise ValueError("archive manifest is missing")
        _audit_payloads(payloads)
        if manifest != _manifest(payloads):
            raise ValueError("archive manifest does not match its payload")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New .zip path; existing files are never overwritten",
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip":
        parser.error("--output must end in .zip")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    build(root, output)
    verify(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"built and verified {output}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
