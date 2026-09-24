"""The incumbent architecture with one correction: no learnable unknown token.

Everything else is inherited literally from
``GlobalObjectiveKnowledgeAttributionModel`` so that the only difference between
the final model and the incumbent is the one this phase intends: a candidate with
no legal modality contributes exactly zero instead of a shared learned token that
the previous phase measured sitting at the top of every goal-blind ordering.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F

from ..model.global_objective_knowledge_attribution_v1.model import (
    GlobalCapabilityEncoder, GlobalCapabilityEncoding, GlobalObjectiveKnowledgeAttributionModel,
)
from . import contract as fc


class CleanCapabilityEncoder(GlobalCapabilityEncoder):
    """Masked pooling with a neutral, non-learnable fallback."""

    def __init__(self, modality_dims: Mapping[str, int], dropout: float = 0.10) -> None:
        super().__init__(modality_dims, dropout)
        del self.unknown_token

    def encode_modality_embeddings(self, descriptors, masks):
        raise NotImplementedError(
            "the clean encoder has no unknown token, so modality-restricted banks are undefined")

    def forward(self, descriptors: Mapping[str, torch.Tensor],
                masks: Mapping[str, torch.Tensor]) -> GlobalCapabilityEncoding:
        tokens, present = self._modality_tokens(descriptors, masks)
        denominator = present.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        pooled = (tokens * present[..., None]).sum(dim=1) / denominator
        embedding = F.normalize(self.global_projection(pooled), dim=-1, eps=1e-12)
        # A candidate with no evidence scores exactly the neutral value against every
        # query, because the score is an inner product against this embedding.
        nothing = ~present.any(dim=1)
        embedding = torch.where(nothing[:, None], torch.zeros_like(embedding), embedding)
        return GlobalCapabilityEncoding(embedding=embedding, token_mask=present,
                                        modality_names=self.modality_names)


class CleanAttributionModel(GlobalObjectiveKnowledgeAttributionModel):
    """The incumbent model with the clean capability encoder."""

    def __init__(self, modality_dims: Mapping[str, int], *, dropout: float = 0.10) -> None:
        super().__init__(modality_dims, dropout=dropout)
        self.capability_encoder = CleanCapabilityEncoder(modality_dims, dropout)

    def encode_modality_candidates(self, descriptors, masks):
        raise NotImplementedError("undefined without an unknown token")


def all_missing_mask(role) -> torch.Tensor:
    """Candidates with no legal modality present, in the order of the candidate axis."""
    import numpy as np

    present = None
    for name in fc.PRIMARY_MODALITIES:
        flag = np.asarray(role.capability_present[name], dtype=bool)
        present = flag if present is None else (present | flag)
    return torch.from_numpy(~present)
