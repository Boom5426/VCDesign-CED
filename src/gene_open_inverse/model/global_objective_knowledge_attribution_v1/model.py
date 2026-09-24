"""Exact SRC-Energy-V2 global retrieval pathway, without conditional modules."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..structured_requirement_capability_energy_v2.gene_tokenizer import STATE_DIM
from ..structured_requirement_capability_energy_v2.knowledge import CapabilityAdapter, MODEL_DIM


@dataclass(frozen=True)
class GlobalCapabilityEncoding:
    """Only the globally pooled candidate representation needed for retrieval."""

    embedding: torch.Tensor
    token_mask: torch.Tensor
    modality_names: tuple[str, ...]


class GlobalCapabilityEncoder(nn.Module):
    """V2 modality adapters and masked global pooling, without K/V projections."""

    def __init__(self, modality_dims: Mapping[str, int], dropout: float = 0.10) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("at least one frozen capability modality is required")
        self.modality_names = tuple(modality_dims)
        self.adapters = nn.ModuleDict({
            name: CapabilityAdapter(int(dim), dropout) for name, dim in modality_dims.items()
        })
        self.modality_embedding = nn.Embedding(len(self.modality_names), MODEL_DIM)
        self.unknown_token = nn.Parameter(torch.empty(MODEL_DIM))
        self.global_projection = nn.Linear(MODEL_DIM, 256)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.unknown_token, std=0.02)

    def _modality_tokens(
        self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized, pre-pooling tokens from the frozen global path."""
        if set(descriptors) != set(self.modality_names) or set(masks) != set(self.modality_names):
            raise ValueError("descriptors and masks must equal this configuration's frozen modalities")
        count = len(descriptors[self.modality_names[0]])
        tokens, present = [], []
        for index, name in enumerate(self.modality_names):
            values, mask = descriptors[name], masks[name].bool().reshape(-1)
            if len(values) != count or len(mask) != count:
                raise ValueError("all candidate modalities must share the candidate axis")
            tokens.append(self.adapters[name](values) + self.modality_embedding.weight[index])
            present.append(mask)
        return F.normalize(torch.stack(tokens, dim=1), dim=-1, eps=1e-12), torch.stack(present, dim=1)

    def encode_modality_embeddings(
        self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Project each frozen pre-pooling modality token into the global retrieval space.

        This is an inference-only exposure of the existing M_SET encoder.  A candidate
        missing a restricted modality receives the already-trained shared UNKNOWN token,
        preserving the candidate pool and avoiding a learned per-candidate fallback.
        """
        token_tensor, mask_tensor = self._modality_tokens(descriptors, masks)
        unknown = F.normalize(self.unknown_token, dim=-1, eps=1e-12).expand(len(token_tensor), -1)
        return {
            name: F.normalize(
                self.global_projection(torch.where(mask_tensor[:, index, None], token_tensor[:, index], unknown)),
                dim=-1, eps=1e-12,
            )
            for index, name in enumerate(self.modality_names)
        }

    def forward(
        self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> GlobalCapabilityEncoding:
        token_tensor, mask_tensor = self._modality_tokens(descriptors, masks)
        count = len(token_tensor)
        neither = ~mask_tensor.any(dim=1)
        if neither.any():
            unknown = F.normalize(self.unknown_token, dim=-1, eps=1e-12)
            token_tensor = torch.cat((token_tensor, unknown[None, None].expand(count, 1, -1)), dim=1)
            mask_tensor = torch.cat((mask_tensor, neither[:, None]), dim=1)
            names = self.modality_names + ("UNKNOWN",)
        else:
            names = self.modality_names
        denominator = mask_tensor.sum(dim=1, keepdim=True).clamp_min(1).to(token_tensor.dtype)
        pooled = (token_tensor * mask_tensor[..., None]).sum(dim=1) / denominator
        return GlobalCapabilityEncoding(
            embedding=F.normalize(self.global_projection(pooled), dim=-1, eps=1e-12),
            token_mask=mask_tensor,
            modality_names=names,
        )


class GlobalObjectiveKnowledgeAttributionModel(nn.Module):
    """The V2 global path with no slots, cross attention, or residual branch."""

    def __init__(self, modality_dims: Mapping[str, int], *, dropout: float = 0.10) -> None:
        super().__init__()
        self.source_context_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.global_delta_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512),
        )
        self.global_context_modulation = nn.Sequential(nn.Linear(512, 1024))
        nn.init.zeros_(self.global_context_modulation[0].weight)
        with torch.no_grad():
            self.global_context_modulation[0].bias[:512].fill_(1.0)
            self.global_context_modulation[0].bias[512:].zero_()
        self.global_query_projection = nn.Linear(512, 256)
        self.capability_encoder = GlobalCapabilityEncoder(modality_dims, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def modality_names(self) -> tuple[str, ...]:
        return self.capability_encoder.modality_names

    def encode_query(self, source: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if source.ndim != 2 or source.shape != goal.shape or source.shape[-1] != STATE_DIM:
            raise ValueError("source and goal must be [B,8248]")
        source_context = self.source_context_encoder(source)
        delta_global = self.global_delta_encoder(goal - source)
        gamma, beta = self.global_context_modulation(source_context).chunk(2, dim=-1)
        return F.normalize(self.global_query_projection(gamma * delta_global + beta), dim=-1, eps=1e-12)

    def encode_candidates(
        self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> GlobalCapabilityEncoding:
        return self.capability_encoder(descriptors, masks)

    def encode_modality_candidates(
        self, descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference-only modality-restricted candidate banks from frozen M_SET weights."""
        return self.capability_encoder.encode_modality_embeddings(descriptors, masks)

    def score_embeddings(self, query: torch.Tensor, candidates: GlobalCapabilityEncoding | torch.Tensor) -> torch.Tensor:
        embedding = candidates.embedding if isinstance(candidates, GlobalCapabilityEncoding) else candidates
        return self.logit_scale.exp().clamp(max=100.0) * (query @ embedding.T)

    def forward(
        self, source: torch.Tensor, goal: torch.Tensor,
        descriptors: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.score_embeddings(self.encode_query(source, goal), self.encode_candidates(descriptors, masks))

    def load_v2_global_state(
        self, state: Mapping[str, torch.Tensor], *, source_modalities: tuple[str, ...] = ("STRING", "MAPKG", "TEXT"),
    ) -> None:
        """Copy the V2 global-path weights, including modality rows by name."""
        target = self.state_dict()
        copied: dict[str, torch.Tensor] = {}
        direct_prefixes = (
            "source_context_encoder.", "global_delta_encoder.", "global_context_modulation.",
            "global_query_projection.", "capability_encoder.adapters.",
            "capability_encoder.unknown_token", "capability_encoder.global_projection.", "logit_scale",
        )
        for name, value in target.items():
            if name.startswith(direct_prefixes) and name in state and state[name].shape == value.shape:
                copied[name] = state[name]
        embedding_name = "capability_encoder.modality_embedding.weight"
        if embedding_name in state:
            source_names = tuple(source_modalities)
            rows = []
            for name in self.modality_names:
                rows.append(state[embedding_name][source_names.index(name)])
            copied[embedding_name] = torch.stack(rows)
        missing = set(target) - set(copied)
        if missing:
            raise RuntimeError(f"V2 global-state transfer missed: {sorted(missing)}")
        self.load_state_dict(copied, strict=True)
