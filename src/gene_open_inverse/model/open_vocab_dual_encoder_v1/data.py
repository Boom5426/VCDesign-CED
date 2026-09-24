"""Readers for the frozen response cache and legal STRING descriptors only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .model import STRING_WITH_MASK_DIM, TRANSITION_DIM


ROLE_NAMES = ("G_fit", "G_select", "G_check")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _normal_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 0.0) or not np.isfinite(norm).all():
        raise ValueError("frozen response cache contains a non-finite or zero-norm response")
    return values / norm


@dataclass(frozen=True)
class RoleArrays:
    """Model inputs and evaluator-only utilities for one immutable role."""

    ids: np.ndarray
    transitions: np.ndarray  # [2, queries, 8248]; this is the legal query input.
    candidate_descriptors: np.ndarray  # [candidates, 513]; no response is included.
    present: np.ndarray


class FrozenTransitionAssets:
    """Validate and adapt the V2 frozen cache without modifying it.

    The cache is never passed to the model as candidate data.  Methods that
    form cross-view utility are explicitly evaluator/training-label helpers;
    their returned arrays must not be fed to a model forward call.
    """

    def __init__(
        self,
        *,
        freeze: str | Path,
        cache: str | Path,
        cache_ids: str | Path,
        cache_manifest: str | Path,
        string: str | Path,
    ) -> None:
        self.paths = {
            "freeze": Path(freeze), "cache": Path(cache), "cache_ids": Path(cache_ids),
            "cache_manifest": Path(cache_manifest), "string": Path(string),
        }
        self.freeze = _read_json(self.paths["freeze"])
        self.cache_manifest = _read_json(self.paths["cache_manifest"])
        if self.cache_manifest.get("status") != "FROZEN_SHARED_RESPONSE_CACHE":
            raise RuntimeError("response cache is not marked frozen")
        if self.cache_manifest.get("all_models_must_use_this_cache") is not True:
            raise RuntimeError("response cache does not carry the all-models lock")
        if self.cache_manifest.get("cache_sha256") != sha256(self.paths["cache"]):
            raise RuntimeError("frozen response-cache hash mismatch")
        if self.cache_manifest.get("cache_ids_sha256") != sha256(self.paths["cache_ids"]):
            raise RuntimeError("frozen response-cache ID hash mismatch")
        for key in ("source_sha256", "metadata_sha256"):
            frozen_value = self.freeze.get(key)
            cache_value = self.cache_manifest.get(key)
            if frozen_value is not None and cache_value is not None and frozen_value != cache_value:
                raise RuntimeError(f"frozen {key} does not match the response-cache manifest")
        self.cache_ids = np.load(self.paths["cache_ids"], allow_pickle=False).astype(str)
        self.cache = np.load(self.paths["cache"], mmap_mode="r", allow_pickle=False)
        if self.cache.shape != (2, 9205, TRANSITION_DIM):
            raise RuntimeError(f"unexpected frozen response-cache shape {self.cache.shape}")
        if self.cache_ids.shape != (9205,):
            raise RuntimeError("frozen response-cache ID axis has the wrong length")
        if len(np.unique(self.cache_ids)) != len(self.cache_ids):
            raise RuntimeError("frozen response-cache IDs are not unique")
        self.cache_index = {value: index for index, value in enumerate(self.cache_ids.tolist())}
        roles = self.freeze.get("roles")
        if not isinstance(roles, Mapping) or set(ROLE_NAMES) - set(roles):
            raise RuntimeError("frozen role artifact does not contain G_fit/G_select/G_check")
        self.roles = {role: np.asarray(roles[role], dtype=str) for role in ROLE_NAMES}
        expected_counts = {"G_fit": 6445, "G_select": 1380, "G_check": 1380}
        if {role: len(ids) for role, ids in self.roles.items()} != expected_counts:
            raise RuntimeError("frozen role counts differ from the approved V2 partition")
        joined = np.concatenate(tuple(self.roles.values()))
        if len(joined) != 9205 or len(np.unique(joined)) != 9205 or set(joined) != set(self.cache_ids):
            raise RuntimeError("frozen roles do not partition the frozen response-cache identity axis")
        self.role_indices = {
            role: np.asarray([self.cache_index[value] for value in ids], dtype=np.int64)
            for role, ids in self.roles.items()
        }
        self._load_string()

    def _load_string(self) -> None:
        expected_sha = self.freeze.get("string_sha256")
        if expected_sha is not None and expected_sha != sha256(self.paths["string"]):
            raise RuntimeError("frozen STRING asset hash mismatch")
        with np.load(self.paths["string"], allow_pickle=False) as archive:
            if set(("genes", "Z", "degree")) - set(archive.files):
                raise RuntimeError("STRING asset is missing genes, Z, or degree")
            genes = archive["genes"].astype(str)
            z = archive["Z"].astype(np.float32)
            degree = archive["degree"]
        if z.ndim != 2 or z.shape[1] != 512 or len(genes) != len(z) or len(degree) != len(z):
            raise RuntimeError("STRING asset is not a 512-D candidate descriptor table")
        self.string_index = {value: index for index, value in enumerate(genes.tolist())}
        self.string_z = z
        self.string_degree = degree
        raw, present = self._raw_string(self.roles["G_fit"])
        if not present.any():
            raise RuntimeError("no finite STRING descriptors are present in G_fit")
        self.string_mean = raw[present].mean(axis=0, dtype=np.float64).astype(np.float32)
        self.string_std = np.maximum(raw[present].std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)

    def _raw_string(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros((len(ids), 512), dtype=np.float32)
        present = np.zeros(len(ids), dtype=bool)
        for out_index, identity in enumerate(ids.tolist()):
            source_index = self.string_index.get(identity)
            if source_index is None:
                continue
            candidate = self.string_z[source_index]
            if self.string_degree[source_index] > 0 and np.isfinite(candidate).all() and np.linalg.norm(candidate) > 0.0:
                raw[out_index] = candidate
                present[out_index] = True
        return raw, present

    def role_arrays(self, role: str) -> RoleArrays:
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown frozen role {role!r}")
        ids = self.roles[role]
        raw, present = self._raw_string(ids)
        standardized = (raw - self.string_mean) / self.string_std
        standardized[~present] = 0.0
        descriptor = np.concatenate((standardized, present[:, None].astype(np.float32)), axis=1)
        if descriptor.shape[1] != STRING_WITH_MASK_DIM or not np.isfinite(descriptor).all():
            raise RuntimeError("invalid STRING-plus-mask candidate descriptor")
        transition = np.asarray(self.cache[:, self.role_indices[role], :], dtype=np.float32)
        if not np.isfinite(transition).all():
            raise RuntimeError("frozen transition query input contains non-finite values")
        return RoleArrays(ids=ids.copy(), transitions=transition, candidate_descriptors=descriptor, present=present)

    @staticmethod
    def cross_view_utility(role: RoleArrays, query_view: int, query_indices: np.ndarray | None = None) -> np.ndarray:
        """Return frozen evaluator utility, never a model input.

        An A query is graded against B-view candidate responses; a B query is
        graded against A.  This is precisely the old cache evaluator's
        cross-view cosine definition, restricted to the requested role pool.
        """
        if query_view not in (0, 1):
            raise ValueError("query_view must be 0 (A) or 1 (B)")
        if query_indices is None:
            query_indices = np.arange(len(role.ids), dtype=np.int64)
        query_indices = np.asarray(query_indices, dtype=np.int64)
        opposite = 1 - query_view
        candidate = _normal_rows(role.transitions[opposite])
        query = candidate[query_indices]
        return (query @ candidate.T).astype(np.float32)

    @staticmethod
    def measurement_margin(role: RoleArrays, query_indices: np.ndarray | None = None) -> np.ndarray:
        """Median A/B utility disagreement for each query, with no invented threshold."""
        if query_indices is None:
            query_indices = np.arange(len(role.ids), dtype=np.int64)
        query_indices = np.asarray(query_indices, dtype=np.int64)
        a = FrozenTransitionAssets.cross_view_utility(role, 0, query_indices)
        b = FrozenTransitionAssets.cross_view_utility(role, 1, query_indices)
        return np.median(np.abs(a - b), axis=1).astype(np.float32)

    def asset_record(self) -> dict:
        role_record = {}
        for role, ids in self.roles.items():
            _, present = self._raw_string(ids)
            role_record[role] = {
                "count": int(len(ids)), "identity_axis_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
                "string_present": int(present.sum()), "string_zero_fallback": int((~present).sum()),
            }
        return {
            "freeze": {"path": str(self.paths["freeze"]), "sha256": sha256(self.paths["freeze"])},
            "cache": {"path": str(self.paths["cache"]), "sha256": sha256(self.paths["cache"]), "shape": list(self.cache.shape)},
            "cache_ids": {"path": str(self.paths["cache_ids"]), "sha256": sha256(self.paths["cache_ids"])},
            "cache_manifest": {"path": str(self.paths["cache_manifest"]), "sha256": sha256(self.paths["cache_manifest"])},
            "string": {"path": str(self.paths["string"]), "sha256": sha256(self.paths["string"]), "dim": 512},
            "roles": role_record,
            "utility": {"definition": "cross-view cosine over frozen response cache", "higher_is_better": True,
                        "missing_value_policy": "non-finite cache values fail preflight; no candidate is dropped for missing STRING"},
            "views": {"axis": ["A", "B"], "supervision": "A query uses B utility and B query uses A utility", "cache_recipe": self.cache_manifest.get("response_recipe")},
            "query_definition": "control-centred transition r_q = x_g - x_s; no claim of a single global absolute source control",
        }
