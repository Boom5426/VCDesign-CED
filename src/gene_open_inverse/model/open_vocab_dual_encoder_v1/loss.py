"""Top-focused utility-gap-weighted pairwise loss for the dual encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PairwiseLossDiagnostics:
    loss: float
    pair_count: int
    mean_pairs_per_query: float
    discarded_by_margin_fraction: float
    hard_negative_fraction: float
    mean_gap: float
    mean_weight: float
    top20_pairs: int
    top64_pairs: int
    top128_pairs: int
    other_pairs: int


def top_focused_pairwise_loss(
    scores: torch.Tensor,
    utility: torch.Tensor,
    self_indices: torch.Tensor,
    margins: torch.Tensor,
    *,
    generator: torch.Generator,
    gt_top_k: int = 128,
    predicted_top_k: int = 128,
    random_k: int = 256,
    max_pairs_per_query: int = 4096,
) -> tuple[torch.Tensor, PairwiseLossDiagnostics]:
    """Compute the prescribed loss without materialising a full N² candidate grid.

    ``utility`` is explicitly a frozen GT label tensor and remains outside the
    model forward path.  ``self_indices`` masks the generating intervention in
    both pair construction and loss reduction.
    """
    if scores.ndim != 2 or utility.shape != scores.shape:
        raise ValueError("scores and utility must have identical [query, candidate] shapes")
    batch_size, candidate_count = scores.shape
    if self_indices.shape != (batch_size,) or margins.shape != (batch_size,):
        raise ValueError("self_indices and margins must have one value per query")
    total_weighted_loss = scores.sum() * 0.0
    total_weight = scores.new_zeros(())
    pair_count = discarded = hard = 0
    gap_sum = weight_sum = 0.0
    strata = [0, 0, 0, 0]
    for q in range(batch_size):
        valid = torch.isfinite(utility[q])
        own = int(self_indices[q].item())
        if 0 <= own < candidate_count:
            valid[own] = False
        eligible = torch.nonzero(valid, as_tuple=False).flatten()
        if eligible.numel() < 2:
            continue
        gt_order = eligible[torch.argsort(utility[q, eligible], descending=True, stable=True)]
        predicted_order = eligible[torch.argsort(scores[q, eligible].detach(), descending=True, stable=True)]
        random_order = eligible[torch.randperm(eligible.numel(), generator=generator, device=eligible.device)]
        subset = torch.unique(torch.cat((gt_order[:gt_top_k], predicted_order[:predicted_top_k], random_order[:random_k])), sorted=False)
        if subset.numel() < 2:
            continue
        values = utility[q, subset]
        gap = values[:, None] - values[None, :]
        preferred, worse = torch.nonzero(gap > margins[q], as_tuple=True)
        possible = int(subset.numel() * (subset.numel() - 1))
        discarded += max(possible - int(preferred.numel()), 0)
        if preferred.numel() == 0:
            continue
        if preferred.numel() > max_pairs_per_query:
            chosen = torch.randperm(preferred.numel(), generator=generator, device=preferred.device)[:max_pairs_per_query]
            preferred, worse = preferred[chosen], worse[chosen]
        better_idx, worse_idx = subset[preferred], subset[worse]
        pair_gap = utility[q, better_idx] - utility[q, worse_idx]
        eligible_values = utility[q, eligible]
        iqr = torch.quantile(eligible_values, 0.75) - torch.quantile(eligible_values, 0.25)
        gap_weight = (pair_gap / iqr.clamp_min(1e-12)).clamp(0.25, 4.0)
        ranks = torch.empty(candidate_count, dtype=torch.long, device=scores.device)
        ranks.fill_(candidate_count + 1)
        ranks[gt_order] = torch.arange(1, gt_order.numel() + 1, device=scores.device)
        better_rank = ranks[better_idx]
        top_weight = torch.where(better_rank <= 20, 4.0, torch.where(better_rank <= 64, 2.0, torch.where(better_rank <= 128, 1.0, 0.5)))
        weight = gap_weight * top_weight
        total_weighted_loss = total_weighted_loss + (weight * F.softplus(scores[q, worse_idx] - scores[q, better_idx])).sum()
        total_weight = total_weight + weight.sum()
        pair_count += int(weight.numel())
        gap_sum += float(pair_gap.detach().sum().item())
        weight_sum += float(weight.detach().sum().item())
        predicted_set = set(predicted_order[:predicted_top_k].detach().cpu().tolist())
        hard += sum(int(item) in predicted_set for item in worse_idx.detach().cpu().tolist())
        strata[0] += int((better_rank <= 20).sum().item())
        strata[1] += int(((better_rank > 20) & (better_rank <= 64)).sum().item())
        strata[2] += int(((better_rank > 64) & (better_rank <= 128)).sum().item())
        strata[3] += int((better_rank > 128).sum().item())
    loss = total_weighted_loss / total_weight.clamp_min(1e-12)
    diagnostics = PairwiseLossDiagnostics(
        loss=float(loss.detach().item()), pair_count=pair_count,
        mean_pairs_per_query=pair_count / max(batch_size, 1),
        discarded_by_margin_fraction=discarded / max(discarded + pair_count, 1),
        hard_negative_fraction=hard / max(pair_count, 1),
        mean_gap=gap_sum / max(pair_count, 1), mean_weight=weight_sum / max(pair_count, 1),
        top20_pairs=strata[0], top64_pairs=strata[1], top128_pairs=strata[2], other_pairs=strata[3],
    )
    return loss, diagnostics
