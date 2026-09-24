"""Gene-level desired-transition tokenizer with source-context FiLM."""
from __future__ import annotations

import torch
from torch import nn


STATE_DIM = 8248
GENE_TOKEN_DIM = 64


class GeneTransitionTokenizer(nn.Module):
    """Tokenize [delta, source, abs(delta)] without candidate-side identity."""

    def __init__(self, state_dim: int = STATE_DIM, token_dim: int = GENE_TOKEN_DIM) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.local_mlp = nn.Sequential(
            nn.Linear(3, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim),
        )
        self.assay_gene_embedding = nn.Embedding(state_dim, token_dim)
        self.context_modulation = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 2 * token_dim),
        )
        nn.init.normal_(self.assay_gene_embedding.weight, std=0.02)
        final = self.context_modulation[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias[:token_dim].fill_(1.0)
            final.bias[token_dim:].zero_()

    def forward(
        self, source: torch.Tensor, delta: torch.Tensor, source_context: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim != 2 or source.shape != delta.shape:
            raise ValueError("source and delta must have identical [B,G] shapes")
        if source.shape[-1] != self.state_dim or source_context.shape != (len(source), 512):
            raise ValueError("unexpected expression or source-context shape")
        local = torch.stack((delta, source, delta.abs()), dim=-1)
        value = self.local_mlp(local)
        identity = self.assay_gene_embedding.weight[None]
        gamma, beta = self.context_modulation(source_context).chunk(2, dim=-1)
        return gamma[:, None, :] * (value + identity) + beta[:, None, :]
