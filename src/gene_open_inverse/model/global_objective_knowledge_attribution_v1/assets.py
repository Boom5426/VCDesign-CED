"""Role-limited frozen assets for global knowledge-attribution V1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..open_vocab_dual_encoder_v1.data import FrozenTransitionAssets, RoleArrays, sha256


@dataclass(frozen=True)
class AttributionRoleArrays:
    base: RoleArrays
    source: np.ndarray
    goal: np.ndarray
    capabilities: dict[str, np.ndarray]
    capability_present: dict[str, np.ndarray]

    @property
    def ids(self) -> np.ndarray:
        return self.base.ids


class GlobalAttributionAssets:
    """Read only selected public modalities and only G_fit/G_select roles."""

    def __init__(
        self, *, modalities: Iterable[str], freeze: str | Path, cache: str | Path, cache_ids: str | Path,
        cache_manifest: str | Path, string: str | Path, source_goal_manifest: str | Path,
        source_states: str | Path, goal_states: str | Path, mapkg: str | Path | None = None,
        mapkg_manifest: str | Path | None = None, candidate_knowledge: str | Path | None = None,
        candidate_knowledge_freeze: str | Path | None = None,
    ) -> None:
        self.modalities = tuple(modalities)
        if self.modalities not in (("STRING",), ("STRING", "MAPKG", "TEXT")):
            raise ValueError("only the frozen STRING or STRING+MAPKG+TEXT factorial paths are legal")
        self.base = FrozenTransitionAssets(
            freeze=freeze, cache=cache, cache_ids=cache_ids, cache_manifest=cache_manifest, string=string,
        )
        self.paths = {
            "source_goal_manifest": Path(source_goal_manifest), "source_states": Path(source_states),
            "goal_states": Path(goal_states),
        }
        self.source_goal_manifest = json.loads(self.paths["source_goal_manifest"].read_text())
        if self.source_goal_manifest.get("status") != "FROZEN_SOURCE_GOAL_ASSETS_COMPLETE":
            raise RuntimeError("source/goal assets are not completed")
        if self.source_goal_manifest.get("frozen_delta_cache", {}).get("sha256") != sha256(cache):
            raise RuntimeError("source/goal assets do not bind the supplied frozen delta cache")
        if not self.source_goal_manifest.get("delta_invariant", {}).get("all_elements_close"):
            raise RuntimeError("source/goal delta invariant is not certified")
        self.source = np.load(self.paths["source_states"], mmap_mode="r", allow_pickle=False)
        self.goal = np.load(self.paths["goal_states"], mmap_mode="r", allow_pickle=False)
        if self.source.shape != self.base.cache.shape or self.goal.shape != self.base.cache.shape:
            raise RuntimeError("source/goal arrays do not align with frozen cache")
        self.values, self.present = self._string_values()
        if len(self.modalities) > 1:
            if None in (mapkg, mapkg_manifest, candidate_knowledge, candidate_knowledge_freeze):
                raise ValueError("multi-knowledge mode requires all frozen MAPKG/TEXT asset paths")
            self.paths.update({
                "mapkg": Path(mapkg), "mapkg_manifest": Path(mapkg_manifest),
                "candidate_knowledge": Path(candidate_knowledge),
                "candidate_knowledge_freeze": Path(candidate_knowledge_freeze),
            })
            self._load_multi()

    def _string_values(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        raw, present = self.base._raw_string(self.base.cache_ids)
        standardized = (raw - self.base.string_mean) / self.base.string_std
        standardized[~present] = 0.0
        return {"STRING": standardized.astype(np.float32)}, {"STRING": present.astype(bool)}

    @staticmethod
    def _finite_present(values: np.ndarray, present: np.ndarray) -> np.ndarray:
        return np.asarray(present, dtype=bool) & np.isfinite(values).all(axis=1) & (np.linalg.norm(values, axis=1) > 0.0)

    def _map_to_cache(self, universe: np.ndarray, values: np.ndarray, present: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
        if values.shape != (len(universe), dim) or present.shape != (len(universe),):
            raise RuntimeError("public capability archive shape is invalid")
        lookup = {item: index for index, item in enumerate(universe.astype(str).tolist())}
        result = np.zeros((len(self.base.cache_ids), dim), dtype=np.float32)
        mapped = np.zeros(len(result), dtype=bool)
        legal = self._finite_present(values, present)
        for row, identity in enumerate(self.base.cache_ids.tolist()):
            index = lookup.get(identity)
            if index is not None and legal[index]:
                result[row] = values[index].astype(np.float32, copy=False)
                mapped[row] = True
        return result, mapped

    def _standardize_fit(self, raw: np.ndarray, present: np.ndarray) -> np.ndarray:
        fit = self.base.role_indices["G_fit"]
        if not present[fit].any():
            raise RuntimeError("public modality has no present G_fit row")
        rows = raw[fit][present[fit]]
        mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(rows.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
        output = (raw - mean) / std
        output[~present] = 0.0
        if not np.isfinite(output).all():
            raise RuntimeError("standardized public modality is non-finite")
        return output.astype(np.float32)

    def _load_multi(self) -> None:
        mapkg_manifest = json.loads(self.paths["mapkg_manifest"].read_text())
        governance = mapkg_manifest.get("governance", {})
        if mapkg_manifest.get("status") != "STATIC_FEATURES_COMPLETE_NO_RESPONSE_READ" or governance.get("perturbation_response_read") is not False or governance.get("relevance_read") is not False:
            raise RuntimeError("MAPKG asset fails its response-independent provenance contract")
        candidate_freeze = json.loads(self.paths["candidate_knowledge_freeze"].read_text())
        governance = candidate_freeze.get("governance", {})
        if candidate_freeze.get("status") != "FROZEN_BEFORE_RESPONSE_OR_G_SELECT_READ" or governance.get("response_or_relevance_read_before_freeze") is not False:
            raise RuntimeError("TEXT asset fails its response-independent provenance contract")
        if candidate_freeze.get("sha256", {}).get("features") != sha256(self.paths["candidate_knowledge"]):
            raise RuntimeError("TEXT feature archive hash differs from its frozen manifest")
        with np.load(self.paths["mapkg"], allow_pickle=False) as archive:
            mapkg_raw, mapkg_present = self._map_to_cache(archive["universe"], archive["MAPKG"].astype(np.float32), archive["MAPKG_present"], 1024)
        with np.load(self.paths["candidate_knowledge"], allow_pickle=False) as archive:
            text_raw, text_present = self._map_to_cache(archive["universe"], archive["TEXT"].astype(np.float32), archive["TEXT_present"], 384)
        self.values["MAPKG"] = self._standardize_fit(mapkg_raw, mapkg_present)
        self.values["TEXT"] = self._standardize_fit(text_raw, text_present)
        self.present["MAPKG"] = mapkg_present
        self.present["TEXT"] = text_present

    @property
    def capability_dims(self) -> dict[str, int]:
        return {name: int(self.values[name].shape[1]) for name in self.modalities}

    def role_arrays(self, role: str) -> AttributionRoleArrays:
        if role not in ("G_fit", "G_select"):
            raise ValueError("Global attribution V1 permits G_fit/G_select only; G_check is inaccessible")
        base = self.base.role_arrays(role)
        indices = self.base.role_indices[role]
        source = np.asarray(self.source[:, indices, :], dtype=np.float32)
        goal = np.asarray(self.goal[:, indices, :], dtype=np.float32)
        if not np.allclose(goal - source, base.transitions, atol=1e-5, rtol=1e-5):
            raise RuntimeError("source/goal delta invariant failed for legal role")
        return AttributionRoleArrays(
            base=base, source=source, goal=goal,
            capabilities={name: self.values[name][indices].copy() for name in self.modalities},
            capability_present={name: self.present[name][indices].copy() for name in self.modalities},
        )

    def asset_record(self) -> dict:
        base_paths = {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in self.base.paths.items()
        }
        base_paths["cache"]["shape"] = list(self.base.cache.shape)
        base_paths["string"]["dim"] = 512
        return {
            "base": base_paths,
            "modalities": list(self.modalities),
            "source_goal_manifest": {"path": str(self.paths["source_goal_manifest"].resolve()), "sha256": sha256(self.paths["source_goal_manifest"])},
            "source_states": {"path": str(self.paths["source_states"].resolve()), "sha256": sha256(self.paths["source_states"])},
            "goal_states": {"path": str(self.paths["goal_states"].resolve()), "sha256": sha256(self.paths["goal_states"])},
            "roles_materialized": ["G_fit", "G_select"],
            "check_access": "forbidden",
        }
