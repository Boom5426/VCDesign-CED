"""Candidate-independent dual encoder for frozen control-centred transitions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TRANSITION_DIM = 8248
STRING_WITH_MASK_DIM = 513
EMBEDDING_DIM = 512


class FixedControlTransitionDualEncoder(nn.Module):
    """Score a transition query against any number of STRING descriptors.

    ``transition`` is the frozen control-centred response ``r_q = x_g - x_s``.
    The model has no source-state branch because no paired absolute source-state
    asset is part of this study.  Candidate descriptors are the 512-D STRING
    feature followed by its one-dimensional present mask.  In particular, this
    module contains no candidate IDs, candidate-count-shaped parameters,
    response-prediction heads, priors, or auxiliary heads.
    """

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(TRANSITION_DIM, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, EMBEDDING_DIM),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(STRING_WITH_MASK_DIM, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, EMBEDDING_DIM),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @staticmethod
    def _normalized(raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(raw, p=2.0, dim=-1, eps=1e-12)

    def encode_transition(self, transition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized query embedding and its pre-normalization norm."""
        if transition.ndim != 2 or transition.shape[-1] != TRANSITION_DIM:
            raise ValueError("transition must have shape [batch, 8248]")
        raw = self.query_encoder(transition)
        return self._normalized(raw), raw.norm(dim=-1)

    def encode_candidates(self, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized embeddings for a dynamic candidate descriptor set."""
        if descriptors.ndim != 2 or descriptors.shape[-1] != STRING_WITH_MASK_DIM:
            raise ValueError("candidate descriptors must have shape [candidates, 513]")
        raw = self.candidate_encoder(descriptors)
        return self._normalized(raw), raw.norm(dim=-1)

    def score_embeddings(self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Parallel matrix score, invariant to the rest of the candidate pool."""
        if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
            raise ValueError("embeddings must be matrices")
        if query_embeddings.shape[-1] != EMBEDDING_DIM or candidate_embeddings.shape[-1] != EMBEDDING_DIM:
            raise ValueError("embedding width must be 512")
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * query_embeddings @ candidate_embeddings.T

    def forward(self, transition: torch.Tensor, candidate_descriptors: torch.Tensor) -> torch.Tensor:
        """Return ``scores[query, candidate]`` without outcome inputs."""
        query, _ = self.encode_transition(transition)
        candidate, _ = self.encode_candidates(candidate_descriptors)
        return self.score_embeddings(query, candidate)
