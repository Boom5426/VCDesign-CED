"""One uniform view of all four contexts, so nothing downstream special-cases a dataset.

RPE1, HepG2 and Jurkat come from ``build.py``, a single code path.  K562 comes from
its frozen assets instead, because the anchor lock forbids rebuilding it.  That
asymmetry is real and is declared rather than hidden; what makes it safe is that the
anchor's source states carry exactly the definition ``build.py`` implements, namely
the view-specific perturbation-cell-incidence weighted mean of batch-specific
normalized control pseudobulks, and the anchor's manifest records that
``x_goal - x_source`` reproduces the frozen response cache to 4.2e-06.

Who trains on what.  Outside the anchor there is no split: in a leave-one-context-out
fold the held context supplies no training row at all, so nothing it is graded on
could have fitted the predictor.  The anchor is the exception, because the shared
base ranker was trained on its ``G_fit`` identities: K562 trains on ``G_fit`` and is
queried on ``G_select``, so the base is never evaluated in sample.  ``G_check`` is
dropped from every array this programme can reach.  ``G_select`` is a selection
split and not a blind one, so K562 absolute numbers are optimistic for the base; the
quantities reported here are paired differences under that same base.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contract as fc


ANCHOR_TRAIN_ROLE = "G_fit"
ANCHOR_QUERY_ROLE = "G_select"
ANCHOR_SEALED_ROLE = "G_check"


@dataclass(frozen=True)
class Context:
    """One context, already reduced to whatever gene axis the caller asked for."""

    name: str
    ids: np.ndarray            # [n] identity symbols
    genes: np.ndarray          # [g] gene symbols
    responses: np.ndarray      # [2, n, g] float32, perturbed minus matched control
    sources: np.ndarray        # [2, n, g] float32, the matched control reference itself
    usable: np.ndarray         # [n] bool, depth gate passed in both views
    eligible: np.ndarray       # [n] bool, QUERY_ELIGIBILITY_V1
    train_ok: np.ndarray       # [n] bool, may supply a training row
    query_ok: np.ndarray       # [n] bool, may be an evaluation query

    @property
    def fit_rows(self) -> np.ndarray:
        return np.flatnonzero(self.usable & self.train_ok)

    @property
    def query_rows(self) -> np.ndarray:
        return np.flatnonzero(self.usable & self.query_ok & self.eligible)

    def summary(self) -> dict:
        return {"context": self.name, "identities": int(self.ids.size), "genes": int(self.genes.size),
                "usable": int(self.usable.sum()), "eligible": int(self.eligible.sum()),
                "training_rows": int(self.fit_rows.size), "queries": int(self.query_rows.size),
                "train_rule": fc.TRAIN_RULE.get(self.name), "query_rule": fc.QUERY_RULE.get(self.name)}


def _restrict(context: Context, genes: np.ndarray) -> Context:
    """Reduce to a target gene axis, in the target's order."""
    genes = np.asarray(genes, dtype=str)
    position = {name: index for index, name in enumerate(context.genes.tolist())}
    missing = [name for name in genes.tolist() if name not in position]
    if missing:
        raise KeyError(f"{context.name} does not measure {len(missing)} of the requested genes")
    columns = np.asarray([position[name] for name in genes.tolist()], dtype=np.int64)
    return Context(name=context.name, ids=context.ids, genes=genes,
                   responses=np.ascontiguousarray(context.responses[:, :, columns]),
                   sources=np.ascontiguousarray(context.sources[:, :, columns]),
                   usable=context.usable, eligible=context.eligible,
                   train_ok=context.train_ok, query_ok=context.query_ok)


def load_built(name: str, directory: str | Path, genes: np.ndarray | None = None) -> Context:
    """A context produced by ``four_context_v1.build``."""
    directory = Path(directory)
    ids = np.asarray([str(v) for v in np.load(directory / "identity_axis.npy", allow_pickle=True)])
    context = Context(
        name=name, ids=ids,
        genes=np.asarray([str(v) for v in np.load(directory / "gene_axis.npy", allow_pickle=True)]),
        responses=np.load(directory / "side_responses.npy"),
        sources=np.load(directory / "side_sources.npy"),
        usable=np.load(directory / "usable.npy"),
        eligible=np.load(directory / "eligible.npy"),
        train_ok=np.ones(ids.size, dtype=bool), query_ok=np.ones(ids.size, dtype=bool))
    return context if genes is None else _restrict(context, genes)


def load_anchor(assets, source_states: str | Path, anchor_genes: str | Path,
                eligibility: str | Path, genes: np.ndarray | None = None) -> Context:
    """K562 from the frozen lock.  Reads no ``G_check`` response and no checkpoint."""
    base = assets.base if hasattr(assets, "base") else assets
    ids = np.asarray([str(v) for v in base.cache_ids])
    responses = np.asarray(base.cache, dtype=np.float32)
    sources = np.load(source_states, mmap_mode="r")
    if sources.shape != responses.shape:
        raise RuntimeError("anchor source states and response cache disagree in shape")

    train_ok = np.zeros(ids.size, dtype=bool)
    query_ok = np.zeros(ids.size, dtype=bool)
    train_ok[base.role_indices[ANCHOR_TRAIN_ROLE]] = True
    query_ok[base.role_indices[ANCHOR_QUERY_ROLE]] = True
    reachable = train_ok | query_ok
    if reachable[base.role_indices[ANCHOR_SEALED_ROLE]].any():
        raise RuntimeError("a sealed G_check identity became reachable")

    eligibility = Path(eligibility)
    eligible = np.zeros(ids.size, dtype=bool)
    for role in ("G_fit", "G_select"):
        vector = np.load(eligibility / f"{role}_eligible.npy")
        index = base.role_indices[role]
        if vector.shape != index.shape:
            raise RuntimeError(f"frozen {role} eligibility vector does not match the role axis")
        eligible[index] = vector

    # G_check rows are dropped from every array the programme can reach.
    keep = reachable
    context = Context(
        name=fc.ANCHOR_CONTEXT, ids=ids[keep],
        genes=np.asarray([str(v) for v in np.load(anchor_genes, allow_pickle=True)]),
        responses=np.ascontiguousarray(responses[:, keep]),
        sources=np.ascontiguousarray(np.asarray(sources[:, keep], dtype=np.float32)),
        usable=np.ones(int(keep.sum()), dtype=bool),
        eligible=eligible[keep], train_ok=train_ok[keep], query_ok=query_ok[keep])
    if context.genes.size != context.responses.shape[2]:
        raise RuntimeError("anchor gene axis does not match the response width")
    return context if genes is None else _restrict(context, genes)


def common_gene_axis(contexts: dict[str, Context]) -> np.ndarray:
    """``G_4``: the intersection of every measured axis, taken once."""
    axes = [np.asarray(context.genes, dtype=str) for context in contexts.values()]
    common = axes[0]
    for axis in axes[1:]:
        common = np.intersect1d(common, axis)
    return np.sort(common)
