"""Semantically grounded candidate capability adapters and cached K/V bank."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_DIM = 128
HEADS = 4
HEAD_DIM = 32


class CapabilityAdapter(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, MODEL_DIM), nn.LayerNorm(MODEL_DIM),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass(frozen=True)
class CapabilityBank:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    global_embedding: torch.Tensor
    modality_names: tuple[str, ...]
    modality_index: torch.Tensor
    semantic_modality_count: int


class StructuredCapabilityEncoder(nn.Module):
    """Exactly one token per accepted modality, plus UNKNOWN when necessary."""

    def __init__(self, modality_dims: dict[str, int], dropout: float = 0.10) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("at least one public-knowledge modality is required")
        self.modality_names = tuple(modality_dims)
        self.adapters = nn.ModuleDict({
            name: CapabilityAdapter(dim, dropout) for name, dim in modality_dims.items()
        })
        self.modality_embedding = nn.Embedding(len(modality_dims), MODEL_DIM)
        self.unknown_token = nn.Parameter(torch.empty(MODEL_DIM))
        self.key_projection = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.value_projection = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.global_projection = nn.Linear(MODEL_DIM, 256)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.unknown_token, std=0.02)

    @staticmethod
    def _heads(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(*values.shape[:-1], HEADS, HEAD_DIM)

    def forward(
        self, descriptors: dict[str, torch.Tensor], masks: dict[str, torch.Tensor],
    ) -> CapabilityBank:
        if set(descriptors) != set(self.modality_names) or set(masks) != set(self.modality_names):
            raise ValueError("descriptor/mask modalities must equal the frozen capability manifest")
        count = len(descriptors[self.modality_names[0]])
        tokens, present = [], []
        for index, name in enumerate(self.modality_names):
            values = descriptors[name]
            mask = masks[name].bool().reshape(-1)
            if len(values) != count or len(mask) != count:
                raise ValueError("candidate modalities must share one candidate axis")
            token = self.adapters[name](values) + self.modality_embedding.weight[index]
            tokens.append(token)
            present.append(mask)
        token_tensor = torch.stack(tokens, dim=1)
        mask_tensor = torch.stack(present, dim=1)
        neither = ~mask_tensor.any(dim=1)
        if neither.any():
            token_tensor = torch.cat(
                (token_tensor, self.unknown_token[None, None].expand(count, 1, -1)), dim=1,
            )
            mask_tensor = torch.cat((mask_tensor, neither[:, None]), dim=1)
            names = self.modality_names + ("UNKNOWN",)
        else:
            names = self.modality_names
        token_tensor = F.normalize(token_tensor, dim=-1, eps=1e-12)
        denominator = mask_tensor.sum(dim=1, keepdim=True).clamp_min(1).to(token_tensor.dtype)
        pooled = (token_tensor * mask_tensor[..., None]).sum(dim=1) / denominator
        modality_index = torch.arange(len(names), device=token_tensor.device)[None].expand(count, -1).clone()
        return CapabilityBank(
            tokens=token_tensor,
            token_mask=mask_tensor,
            keys=self._heads(self.key_projection(token_tensor)),
            values=self._heads(self.value_projection(token_tensor)),
            global_embedding=F.normalize(self.global_projection(pooled), dim=-1, eps=1e-12),
            modality_names=names,
            modality_index=modality_index,
            semantic_modality_count=len(self.modality_names),
        )


class RequirementCapabilityCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_projection = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.output_projection = nn.Linear(MODEL_DIM, MODEL_DIM)

    def forward(
        self, requirements: torch.Tensor, bank: CapabilityBank,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.query_projection(requirements).reshape(
            len(requirements), requirements.shape[1], HEADS, HEAD_DIM,
        )
        logits = torch.einsum("brhd,njhd->bnrhj", queries, bank.keys) / math.sqrt(HEAD_DIM)
        logits = logits.masked_fill(
            ~bank.token_mask[None, :, None, None, :], torch.finfo(logits.dtype).min,
        )
        if bank.semantic_modality_count > 4:
            keep = min(3, bank.semantic_modality_count)
            threshold = logits.topk(keep, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        matched = torch.einsum("bnrhj,njhd->bnrhd", attention, bank.values).reshape(
            len(requirements), len(bank.tokens), requirements.shape[1], MODEL_DIM,
        )
        return self.output_projection(matched), attention
