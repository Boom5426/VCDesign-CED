"""The two scorers under test, on identical frozen inputs.

``CURRENT_SCORER_PROBE`` is the deployed architecture: start and target are fused
by a FiLM modulation into one 256-d query embedding, candidates are pooled into
one 256-d embedding, and the score is a scaled cosine between them.  Two
consequences matter for this phase.  The candidate is never allowed to change how
start and target combine, because the fusion happens before the candidate is
seen, so there is no genuine three-way interaction.  And both sides are
L2-normalized, so the architecture cannot express an additive candidate prior at
all; any candidate-only component it shows has to be carried by embedding
direction.

``TRILINEAR_CONDITIONAL_PROBE`` adds exactly the three things that absence
predicts should matter, and nothing else:

1. an explicit rank-``R`` trilinear term ``sum_r a_r f_s(s)_r f_g(g)_r f_c(c)_r``,
   a CP factorization of a genuine start by target by candidate tensor;
2. a transition-by-candidate bilinear residual on ``g - s``, kept separate from
   the trilinear path so a purely transition-driven effect does not have to be
   routed through the three-way term;
3. an additive candidate-prior branch that reads the candidate alone.

The prior branch is separable at evaluation, so "full minus candidate prior" can
be read off the model rather than estimated.  Both an empirical column-mean prior
(defined identically for both probes) and this explicit branch are reported; the
empirical one is the comparable measure.

Both probes train their own parameters from scratch on the same frozen inputs
with the same loss, optimizer and step budget.  Neither inherits trained weights,
because a comparison where one side starts ten epochs ahead would measure the
head start rather than the interaction form.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..global_objective_knowledge_attribution_v1.model import GlobalObjectiveKnowledgeAttributionModel
from ..structured_requirement_capability_energy_v2.gene_tokenizer import STATE_DIM


TRILINEAR_RANK = 128
RESIDUAL_DIM = 256
STATE_HIDDEN = 512


class CurrentScorerProbe(nn.Module):
    """The deployed scorer, freshly initialized."""

    name = "CURRENT_SCORER_PROBE"

    def __init__(self, modality_dims: Mapping[str, int], dropout: float = 0.10) -> None:
        super().__init__()
        self.model = GlobalObjectiveKnowledgeAttributionModel(modality_dims, dropout=dropout)

    def forward(self, source, goal, descriptors, masks) -> torch.Tensor:
        return self.model(source, goal, descriptors, masks)

    def candidate_prior(self, descriptors, masks) -> torch.Tensor | None:
        """This architecture has no additive candidate-only term to expose."""
        return None


class _StateEncoder(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, STATE_HIDDEN), nn.LayerNorm(STATE_HIDDEN),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _CandidateEncoder(nn.Module):
    """Masked pooling over the frozen modalities, without an output normalization.

    The absent output normalization is deliberate: the prior branch needs the
    candidate representation to carry magnitude.
    """

    def __init__(self, modality_dims: Mapping[str, int], dropout: float) -> None:
        super().__init__()
        self.modality_names = tuple(modality_dims)
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(int(dim)), nn.Linear(int(dim), 512), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(512, 256), nn.LayerNorm(256))
            for name, dim in modality_dims.items()
        })
        self.modality_embedding = nn.Embedding(len(self.modality_names), 256)
        self.unknown_token = nn.Parameter(torch.zeros(256))
        self.projection = nn.Sequential(nn.Linear(256, STATE_HIDDEN), nn.LayerNorm(STATE_HIDDEN))
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.unknown_token, std=0.02)

    def forward(self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(descriptors) != set(self.modality_names):
            raise ValueError("descriptors must equal the frozen modalities")
        tokens, present = [], []
        for index, name in enumerate(self.modality_names):
            tokens.append(self.adapters[name](descriptors[name]) + self.modality_embedding.weight[index])
            present.append(masks[name].bool().reshape(-1))
        token = torch.stack(tokens, dim=1)
        mask = torch.stack(present, dim=1)
        neither = ~mask.any(dim=1)
        if neither.any():
            token = torch.cat((token, self.unknown_token[None, None].expand(len(token), 1, -1)), dim=1)
            mask = torch.cat((mask, neither[:, None]), dim=1)
        pooled = (token * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(token.dtype)
        return self.projection(pooled)


class TrilinearConditionalProbe(nn.Module):
    """Explicit three-way interaction, transition residual and candidate prior."""

    name = "TRILINEAR_CONDITIONAL_PROBE"

    def __init__(self, modality_dims: Mapping[str, int], dropout: float = 0.10,
                 rank: int = TRILINEAR_RANK, residual_dim: int = RESIDUAL_DIM) -> None:
        super().__init__()
        self.source_encoder = _StateEncoder(dropout)
        self.target_encoder = _StateEncoder(dropout)
        self.transition_encoder = _StateEncoder(dropout)
        self.candidate_encoder = _CandidateEncoder(modality_dims, dropout)
        self.source_factor = nn.Sequential(nn.Linear(STATE_HIDDEN, rank), nn.LayerNorm(rank))
        self.target_factor = nn.Sequential(nn.Linear(STATE_HIDDEN, rank), nn.LayerNorm(rank))
        self.candidate_factor = nn.Sequential(nn.Linear(STATE_HIDDEN, rank), nn.LayerNorm(rank))
        self.core = nn.Parameter(torch.full((rank,), 1.0 / rank))
        self.transition_residual = nn.Sequential(nn.Linear(STATE_HIDDEN, residual_dim), nn.LayerNorm(residual_dim))
        self.candidate_residual = nn.Sequential(nn.Linear(STATE_HIDDEN, residual_dim), nn.LayerNorm(residual_dim))
        self.prior_head = nn.Sequential(nn.Linear(STATE_HIDDEN, 256), nn.GELU(), nn.Linear(256, 1))
        self.term_scale = nn.Parameter(torch.zeros(3))
        self.residual_dim = residual_dim

    def _candidate_parts(self, descriptors, masks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.candidate_encoder(descriptors, masks)
        return (self.candidate_factor(embedding), self.candidate_residual(embedding),
                self.prior_head(embedding).reshape(-1))

    def forward(self, source, goal, descriptors, masks) -> torch.Tensor:
        if source.ndim != 2 or source.shape != goal.shape or source.shape[-1] != STATE_DIM:
            raise ValueError("source and goal must be [B, STATE_DIM]")
        factor_c, residual_c, prior = self._candidate_parts(descriptors, masks)
        factor_s = self.source_factor(self.source_encoder(source))
        factor_g = self.target_factor(self.target_encoder(goal))
        residual_t = self.transition_residual(self.transition_encoder(goal - source))
        scale = self.term_scale.exp()
        trilinear = ((factor_s * factor_g) * self.core) @ factor_c.T
        bilinear = (residual_t @ residual_c.T) / float(self.residual_dim) ** 0.5
        return scale[0] * trilinear + scale[1] * bilinear + scale[2] * prior[None, :]

    def candidate_prior(self, descriptors, masks) -> torch.Tensor:
        """The additive candidate-only branch, read on its own."""
        return self.term_scale.exp()[2] * self._candidate_parts(descriptors, masks)[2]


PROBES = {"CURRENT_SCORER_PROBE": CurrentScorerProbe, "TRILINEAR_CONDITIONAL_PROBE": TrilinearConditionalProbe}
