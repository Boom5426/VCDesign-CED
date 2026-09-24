"""Frozen asset and checkpoint resolution for the alignment audit.

Every path lives in a JSON config rather than in code, so a run records exactly
which frozen artifacts it read and a different machine can be pointed at a
different mirror without editing a module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..open_vocab_dual_encoder_v1.data import sha256


BASE_ASSET_NAMES = (
    "freeze", "cache", "cache_ids", "cache_manifest", "string", "source_goal_manifest",
    "source_states", "goal_states", "mapkg", "mapkg_manifest", "candidate_knowledge",
    "candidate_knowledge_freeze",
)
MODALITIES = ("STRING", "MAPKG", "TEXT")


@dataclass(frozen=True)
class AuditConfig:
    """Resolved frozen inputs for one alignment-audit run."""

    assets: dict[str, str]
    checkpoints: dict[str, Any]
    cached_scores: dict[str, str]

    @staticmethod
    def load(path: Path) -> "AuditConfig":
        payload = json.loads(Path(path).read_text())
        missing = [name for name in BASE_ASSET_NAMES if name not in payload.get("assets", {})]
        if missing:
            raise KeyError(f"alignment audit config is missing frozen assets: {sorted(missing)}")
        return AuditConfig(
            assets=dict(payload["assets"]),
            checkpoints=dict(payload.get("checkpoints", {})),
            cached_scores=dict(payload.get("cached_scores", {})),
        )

    def asset_arguments(self) -> dict[str, str]:
        return {name: self.assets[name] for name in BASE_ASSET_NAMES}

    def provenance(self) -> dict[str, Any]:
        """SHA-256 of every frozen input actually read, recorded in result.json."""
        record: dict[str, Any] = {"assets": {}, "cached_scores": {}, "checkpoints": {}}
        for name, value in self.assets.items():
            path = Path(value)
            record["assets"][name] = {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
        for name, value in self.cached_scores.items():
            path = Path(value)
            record["cached_scores"][name] = {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
        for name, value in self.checkpoints.items():
            path = Path(value if isinstance(value, str) else value.get("path", ""))
            record["checkpoints"][name] = {
                "path": str(path), "sha256": sha256(path) if path.is_file() else None,
                "detail": value if isinstance(value, dict) else None,
            }
        return record
