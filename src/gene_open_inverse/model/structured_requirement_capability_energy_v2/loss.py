"""New top-set objective while importing the frozen pairwise loss unchanged."""
from __future__ import annotations

import torch


def gt_top20_positive_set_loss(
    scores: torch.Tensor,
    utility: torch.Tensor,
    self_indices: torch.Tensor,
    *,
    positive_k: int = 20,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean logsumexp(all legal) minus logsumexp(GT top-20 legal)."""
    if scores.ndim != 2 or utility.shape != scores.shape:
        raise ValueError("scores and utility must align as [query,candidate]")
    if self_indices.shape != (len(scores),):
        raise ValueError("self_indices must have one entry per query")
    terms: list[torch.Tensor] = []
    for row in range(len(scores)):
        valid = torch.isfinite(utility[row]).clone()
        own = int(self_indices[row])
        if 0 <= own < scores.shape[1]:
            valid[own] = False
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        if len(candidates) < positive_k:
            raise RuntimeError("not enough legal candidates for GT top-20 positive set")
        ordered = candidates[torch.argsort(utility[row, candidates], descending=True, stable=True)]
        positive = ordered[:positive_k]
        terms.append(torch.logsumexp(scores[row, candidates], dim=0) - torch.logsumexp(scores[row, positive], dim=0))
    loss = torch.stack(terms).mean()
    return loss, {"valid_queries": float(len(terms)), "positive_k": float(positive_k)}


def combined_v2_loss(pair_loss: torch.Tensor, set_loss: torch.Tensor, *, stage: str) -> torch.Tensor:
    if stage == "global":
        return pair_loss
    if stage == "conditional":
        return pair_loss + 0.5 * set_loss
    raise ValueError("stage must be global or conditional")
