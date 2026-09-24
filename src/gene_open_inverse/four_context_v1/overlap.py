"""Phase 5: the candidate overlap matrix and the strata every result is split by.

Two different things get called generalization and they are never added together
here.  A candidate whose response was measured in a training context had its own
effect supervised, so evaluating it in a held context tests whether the *context*
transfers.  A candidate that no training context measured tests whether the
*candidate* mapping transfers as well.  ``COMMON4`` is the subset measured
everywhere, which is the only population on which a context residual can be
estimated at all.

Membership is a property of which identities were measured where.  It never
depends on a score.
"""
from __future__ import annotations

import numpy as np

from . import contract as fc


def supervised_identities(contexts: dict, names: tuple[str, ...]) -> np.ndarray:
    """Identities whose response actually enters effect-distillation supervision."""
    pools = [contexts[name].ids[contexts[name].fit_rows] for name in names]
    return np.unique(np.concatenate(pools)) if pools else np.zeros(0, dtype=str)


def measured_everywhere(contexts: dict) -> np.ndarray:
    """Identities with a usable response in all four contexts."""
    pools = [context.ids[context.usable] for context in contexts.values()]
    common = pools[0]
    for pool in pools[1:]:
        common = np.intersect1d(common, pool)
    return common


def strata_masks(candidate_ids: np.ndarray, seen: np.ndarray, common4: np.ndarray) -> dict:
    """One boolean mask per stratum over a held context's candidate axis."""
    candidate_ids = np.asarray(candidate_ids, dtype=str)
    in_seen = np.isin(candidate_ids, seen)
    return {"ALL": np.ones(candidate_ids.size, dtype=bool),
            "SEEN_CANDIDATE": in_seen,
            "UNSEEN_CANDIDATE": ~in_seen,
            "COMMON4": np.isin(candidate_ids, common4)}


def overlap_matrix(contexts: dict) -> dict:
    """Pairwise and cumulative identity overlap across the four contexts."""
    usable = {name: set(context.ids[context.usable].tolist()) for name, context in contexts.items()}
    record = {"usable_identities": {name: len(pool) for name, pool in usable.items()},
              "pairwise": {}, "measured_in_exactly_k_contexts": {}}
    for left in fc.CONTEXTS:
        for right in fc.CONTEXTS:
            if left < right:
                shared = usable[left] & usable[right]
                record["pairwise"][f"{left}|{right}"] = {
                    "shared": len(shared),
                    "jaccard": len(shared) / max(len(usable[left] | usable[right]), 1)}
    counter: dict[str, int] = {}
    for name in set().union(*usable.values()):
        counter[name] = sum(name in pool for pool in usable.values())
    for k in (1, 2, 3, 4):
        record["measured_in_exactly_k_contexts"][str(k)] = int(sum(v == k for v in counter.values()))
    record["measured_in_all_four"] = record["measured_in_exactly_k_contexts"]["4"]
    return record


def fold_strata(contexts: dict, held: str) -> dict:
    """Everything the LOCO fold needs to split its results, computed from data alone."""
    training = tuple(name for name in fc.CONTEXTS if name != held)
    seen = supervised_identities(contexts, training)
    common4 = measured_everywhere(contexts)
    context = contexts[held]
    candidates = context.ids[context.usable]
    masks = strata_masks(candidates, seen, common4)
    return {"held": held, "training": training,
            "candidate_ids": candidates, "masks": masks,
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "supervised_identity_count": int(seen.size),
            "common4_identity_count": int(common4.size)}
