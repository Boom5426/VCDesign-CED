#!/usr/bin/env python3
"""Phase 0: record exactly which bytes this programme was built from.

The two raw H5ADs are the only external input the four-context mission takes, and
everything downstream is a deterministic function of them plus frozen code.  The
manifest is therefore the provenance root: original filename, accession, source URL,
download timestamp, byte size and SHA-256.

The size is also checked against the size the server advertised, because a ranged
parallel download that loses a chunk produces a file that opens and reads and is
simply wrong in the middle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from . import contract as fc


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl"
EXPECTED_BYTES = {
    "GSE264667_hepg2_raw_singlecell_01.h5ad": 5614460941,
    "GSE264667_jurkat_raw_singlecell_01.h5ad": 9366490264,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(directory: Path) -> dict:
    record: dict = {
        "schema": "VCDESIGN_FOUR_CONTEXT_V1_ASSET_MANIFEST",
        "accession": fc.ACCESSION, "series_title": fc.ACCESSION_TITLE,
        "perturbation_type": fc.PERTURBATION_TYPE,
        "fastq_reprocessing_performed": False,
        "fastq_reprocessing_note": "the public H5ADs opened and read, so nothing was reprocessed",
        "files": {},
    }
    for name, expected in EXPECTED_BYTES.items():
        path = directory / name
        if not path.exists():
            record["files"][name] = {"present": False}
            continue
        size = path.stat().st_size
        record["files"][name] = {
            "present": True, "original_filename": name,
            "source_url": f"{BASE_URL}/{name}",
            "accession": fc.ACCESSION,
            "context": next((k for k, v in fc.NEW_CONTEXT_SOURCES.items() if v == name), None),
            "bytes": size, "bytes_advertised_by_server": expected,
            "size_matches_server": size == expected,
            "sha256": sha256(path),
            "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime,
                                                    tz=timezone.utc).isoformat(),
            "download_method": "eight ranged HTTPS workers, reassembled in order",
        }
    record["all_files_complete"] = all(
        entry.get("size_matches_server") for entry in record["files"].values())
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version()}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 asset manifest")
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    record = run(directory)
    destination = directory / f"{fc.ACCESSION}_ASSET_MANIFEST.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["all_files_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
