"""Everything a fold needs, assembled once: contexts, features, base scores, utility.

Keeping this separate from the fold logic matters for one reason.  The base ranker,
the candidate features and the graded utility are properties of a context, not of a
fold, so they are computed once per context and reused by every arm and every atlas
size.  If they were rebuilt per arm, a difference between two arms could come from a
rebuild rather than from the arm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..candidate_effect_distillation_v1 import contract as ced
from ..candidate_effect_distillation_v1 import predictor as pr
from ..external_context_v1.transfer import external_capabilities
from ..final_clean_model_v1.model import CleanAttributionModel
from ..final_clean_model_v1.scoring import strict_legal_dims
from . import contract as fc
from . import evaluate as ev


@dataclass
class ContextPack:
    """One context reduced to G_4, with its fixed base scores and graded utility."""

    name: str
    ids: np.ndarray                  # [pool] candidate identities, the usable axis
    pool_rows: np.ndarray            # positions of the pool inside the context axis
    query_rows: np.ndarray           # positions of the queries inside the context axis
    self_positions: np.ndarray       # where each query sits inside the pool, or -1
    responses: np.ndarray            # [2, pool, |G_4|]
    sources: np.ndarray              # [2, pool, |G_4|] matched control reference per identity
    query_responses: np.ndarray      # [2, queries, |G_4|]
    query_sources: np.ndarray        # [2, queries, |G_4|] the state a caller would hold
    features: np.ndarray             # [pool, static width]
    present: np.ndarray              # [pool] bool, any legal modality
    base: list                       # two [queries, pool] float64 matrices
    utility: list                    # two [queries, pool] float64 matrices
    fit_mask: np.ndarray             # [pool] bool, identities this context may train on
    verification: dict


def anchor_bridge(genes: np.ndarray, anchor_genes: np.ndarray) -> dict:
    """Where each G_4 gene sits on the frozen anchor axis the base ranker expects."""
    anchor_genes = np.asarray(anchor_genes, dtype=str)
    index = {name: position for position, name in enumerate(anchor_genes.tolist())}
    missing = [name for name in np.asarray(genes, dtype=str).tolist() if name not in index]
    if missing:
        raise KeyError(f"{len(missing)} G_4 genes are absent from the anchor axis")
    return {"positions": np.asarray([index[name] for name in np.asarray(genes, dtype=str).tolist()],
                                    dtype=np.int64), "width": int(anchor_genes.size)}


def embed(values: np.ndarray, bridge: dict) -> np.ndarray:
    """Place G_4 values on the anchor axis and zero everything else.

    Every context, the anchor included, is embedded the same way, so the base ranker
    sees the same number of informative coordinates everywhere.  Leaving K562 on its
    full 8248 genes while the others carry 7000 would make the base stronger in one
    context for a reason that has nothing to do with the question.
    """
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros(values.shape[:-1] + (bridge["width"],), dtype=np.float32)
    output[..., bridge["positions"]] = values
    return output


def load_base_model(lock_path: str | Path, assets, device: torch.device):
    """The frozen CLEAN_BASE scorer.  Loaded read only; never trained here."""
    lock = json.loads(Path(lock_path).read_text())
    model = CleanAttributionModel(strict_legal_dims(assets)).to(device)
    state = torch.load(lock["selected_checkpoints"]["CLEAN_BASE"]["checkpoint"],
                       map_location=device, weights_only=False)["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, lock


def pack_context(context, assets, model, bridge, device: torch.device,
                 batch: int = 64) -> ContextPack:
    """Reduce one context to the objects every fold reads."""
    pool_rows = np.flatnonzero(context.usable)
    query_rows = context.query_rows
    if query_rows.size == 0:
        raise RuntimeError(f"{context.name} has no eligible evaluation query")
    position = {int(value): index for index, value in enumerate(pool_rows.tolist())}
    self_positions = np.asarray([position.get(int(q), -1) for q in query_rows.tolist()],
                                dtype=np.int64)
    if (self_positions < 0).any():
        raise RuntimeError("an evaluation query is not inside its own context's candidate pool")

    ids = context.ids[pool_rows]
    responses = np.ascontiguousarray(context.responses[:, pool_rows])
    sources = np.ascontiguousarray(context.sources[:, pool_rows])
    capabilities, present_by_name, verification = external_capabilities(assets, ids)
    features = pr.static_features(
        SimpleNamespace(capabilities=capabilities, capability_present=present_by_name),
        ced.PRIMARY_MODALITIES)

    query_index = self_positions
    query_responses = np.ascontiguousarray(responses[:, query_index])
    query_sources = np.ascontiguousarray(sources[:, query_index])
    utility = [np.ascontiguousarray(ev.cross_view_utility(responses, view)[query_index])
               for view in (0, 1)]

    descriptors = {name: torch.from_numpy(capabilities[name]).to(device)
                   for name in ced.PRIMARY_MODALITIES}
    masks = {name: torch.from_numpy(present_by_name[name]).to(device)
             for name in ced.PRIMARY_MODALITIES}
    base = []
    with torch.no_grad():
        encoding = model.encode_candidates(descriptors, masks)
        for view in (0, 1):
            source = torch.from_numpy(embed(sources[view][query_index], bridge)).to(device)
            goal = torch.from_numpy(embed((sources[view] + responses[view])[query_index],
                                          bridge)).to(device)
            rows = []
            for start in range(0, query_index.size, batch):
                stop = start + batch
                rows.append(model.score_embeddings(
                    model.encode_query(source[start:stop], goal[start:stop]),
                    encoding).float().cpu())
            base.append(torch.cat(rows).numpy().astype(np.float64))
    del descriptors, masks, encoding
    torch.cuda.empty_cache()

    fit_mask = context.train_ok[pool_rows]
    return ContextPack(name=context.name, ids=ids, pool_rows=pool_rows, query_rows=query_rows,
                       self_positions=self_positions, responses=responses, sources=sources,
                       query_responses=query_responses, query_sources=query_sources,
                       features=features.values,
                       present=features.present_any, base=base, utility=utility,
                       fit_mask=fit_mask, verification=verification)


def training_pool(packs: dict, names: tuple[str, ...]) -> dict:
    """Concatenate the C_fit rows of the named contexts into one training pool."""
    features, responses, present, keys = [], [], [], []
    for name in names:
        pack = packs[name]
        rows = np.flatnonzero(pack.fit_mask)
        features.append(pack.features[rows])
        responses.append(ev.consensus(pack.responses[:, rows]))
        present.append(pack.present[rows])
        keys.append(np.asarray([f"{name}|{value}" for value in pack.ids[rows].tolist()], dtype=str))
    return {"features": np.concatenate(features), "responses": np.concatenate(responses),
            "present": np.concatenate(present), "keys": np.concatenate(keys),
            "contexts": tuple(names)}


def effect_arms(pack: ContextPack, effect: np.ndarray) -> list:
    """Fused scores for one predicted effect, at the frozen unit fusion weight."""
    return [ev.fuse(pack.base[view], ev.effect_cosine(pack.query_responses, effect, view))
            for view in (0, 1)]


def save_pack(pack: ContextPack, directory: str | Path) -> None:
    """Cache everything a later phase reads, so the base ranker runs exactly once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{pack.name}.npz", ids=pack.ids, pool_rows=pack.pool_rows,
             query_rows=pack.query_rows, self_positions=pack.self_positions,
             responses=pack.responses, sources=pack.sources,
             query_responses=pack.query_responses, query_sources=pack.query_sources,
             features=pack.features, present=pack.present, fit_mask=pack.fit_mask,
             base_0=pack.base[0], base_1=pack.base[1],
             utility_0=pack.utility[0], utility_1=pack.utility[1])
    (directory / f"{pack.name}_verification.json").write_text(
        json.dumps(pack.verification, indent=2, sort_keys=True) + "\n")


def load_pack(name: str, directory: str | Path) -> ContextPack:
    directory = Path(directory)
    with np.load(directory / f"{name}.npz", allow_pickle=False) as store:
        return ContextPack(
            name=name, ids=store["ids"].astype(str), pool_rows=store["pool_rows"],
            query_rows=store["query_rows"], self_positions=store["self_positions"],
            responses=store["responses"], sources=store["sources"],
            query_responses=store["query_responses"], query_sources=store["query_sources"],
            features=store["features"], present=store["present"],
            base=[store["base_0"], store["base_1"]],
            utility=[store["utility_0"], store["utility_1"]],
            fit_mask=store["fit_mask"],
            verification=json.loads((directory / f"{name}_verification.json").read_text()))
