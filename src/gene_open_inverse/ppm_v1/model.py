"""The PPM candidate-effect model: encoder, program tokenizer, sparse router, memory transport.

The forward pass takes standardized static features, raw presence flags and, for the memory
path, a memory of *other* training identities.  It never takes an identity, a query or a
held-context outcome, so there is no path through which a candidate's own label can reach
its own prediction except by being in the memory, which the caller controls and the run
asserts against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contract as pc


def topk_signed(values: torch.Tensor, k: int = pc.ACTIVE_PROGRAMS) -> torch.Tensor:
    """Keep the ``k`` entries of largest magnitude per row, with their sign; zero the rest."""
    index = values.abs().topk(k, dim=-1).indices
    mask = torch.zeros_like(values).scatter(-1, index, 1.0)
    return values * mask


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(pc.HIDDEN)
        self.inner = nn.Sequential(nn.Linear(pc.HIDDEN, pc.FFN_WIDTH), nn.GELU(),
                                   nn.Dropout(pc.DROPOUT), nn.Linear(pc.FFN_WIDTH, pc.HIDDEN),
                                   nn.Dropout(pc.DROPOUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner(self.norm(x))


class KnowledgeEncoder(nn.Module):
    """Input dropout, modality adapters, a presence-masked per-dimension gate, a residual MLP."""

    def __init__(self):
        super().__init__()
        self.widths = pc.MODALITY_WIDTHS
        self.input_dropout = nn.Dropout(pc.INPUT_DROPOUT)
        self.adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(w), nn.Linear(w, pc.HIDDEN), nn.GELU())
                                       for w in self.widths])
        self.mix = nn.Linear(len(self.widths) * pc.HIDDEN, len(self.widths) * pc.HIDDEN)
        self.blocks = nn.ModuleList([Block() for _ in range(pc.ENCODER_BLOCKS)])
        self.norm = nn.LayerNorm(pc.HIDDEN)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        if bool((presence.sum(dim=1) <= 0).any()):
            raise ValueError("a candidate without any legal modality cannot be encoded")
        features = self.input_dropout(features)
        parts, start = [], 0
        for index, (width, adapter) in enumerate(zip(self.widths, self.adapters)):
            parts.append(adapter(features[:, start:start + width]) * presence[:, index:index + 1])
            start += width
        stacked = torch.stack(parts, dim=1)                                   # [n, m, H]
        logits = self.mix(stacked.flatten(1)).view(stacked.shape)
        logits = logits.masked_fill(presence[:, :, None] <= 0, float("-inf"))
        mixed = (torch.softmax(logits, dim=1) * stacked).sum(dim=1)
        for block in self.blocks:
            mixed = block(mixed)
        return self.norm(mixed)


class ProgramTokenizer(nn.Module):
    """``K`` unit-norm effect programs, a response encoder, signed top-k codes."""

    def __init__(self, basis: np.ndarray):
        super().__init__()
        basis = torch.as_tensor(np.asarray(basis, dtype=np.float32))       # [G, K]
        if basis.shape[1] != pc.PROGRAMS:
            raise ValueError(f"program basis must have {pc.PROGRAMS} columns")
        self.programs = nn.Parameter(basis.T.contiguous().clone())
        self.encoder = nn.Linear(basis.shape[0], pc.PROGRAMS)
        with torch.no_grad():
            self.encoder.weight.copy_(basis.T)
            self.encoder.bias.zero_()

    def dictionary(self) -> torch.Tensor:
        return F.normalize(self.programs, dim=1)

    def code(self, directions: torch.Tensor) -> torch.Tensor:
        """The dense pre-sparsification code ``z(u)``."""
        return self.encoder(directions)

    def tokenize(self, directions: torch.Tensor) -> torch.Tensor:
        return topk_signed(self.code(directions))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.normalize(codes @ self.dictionary(), dim=1)


class SparseRouter(nn.Module):
    """Top-2 mixture of low-capacity experts from ``h_c`` to program codes."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(pc.HIDDEN, pc.EXPERTS)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(pc.HIDDEN), nn.Linear(pc.HIDDEN, pc.HIDDEN), nn.GELU(),
                          nn.Linear(pc.HIDDEN, pc.PROGRAMS)) for _ in range(pc.EXPERTS)])

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict]:
        probabilities = torch.softmax(self.router(hidden), dim=1)
        top = probabilities.topk(pc.ACTIVE_EXPERTS, dim=1)
        weights = top.values / top.values.sum(dim=1, keepdim=True)
        outputs = torch.stack([expert(hidden) for expert in self.experts], dim=1)  # [n, M, K]
        chosen = outputs.gather(1, top.indices[:, :, None].expand(-1, -1, pc.PROGRAMS))
        codes = (weights[:, :, None] * chosen).sum(dim=1)
        first = F.one_hot(top.indices[:, 0], pc.EXPERTS).float()
        balance = pc.EXPERTS * (first.mean(dim=0) * probabilities.mean(dim=0)).sum()
        return codes, {"probabilities": probabilities, "top": top.indices, "balance": balance}


class MemoryTransport(nn.Module):
    """Cross-attention from a candidate's knowledge to retrieved atlas memory.

    There is no residual from the query into the output: everything this path returns is a
    transformation of retrieved memory values, weighted by knowledge-key attention.
    """

    def __init__(self):
        super().__init__()
        self.value = nn.Linear(pc.PROGRAMS, pc.HIDDEN)
        self.attention = nn.MultiheadAttention(pc.HIDDEN, pc.HEADS, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(pc.HIDDEN), nn.Linear(pc.HIDDEN, pc.FFN_WIDTH), nn.GELU(),
                                 nn.Dropout(pc.DROPOUT), nn.Linear(pc.FFN_WIDTH, pc.HIDDEN))
        self.norm = nn.LayerNorm(pc.HIDDEN)
        self.out = nn.Linear(pc.HIDDEN, pc.PROGRAMS)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple:
        transported, weights = self.attention(query[:, None, :], keys, self.value(values),
                                              need_weights=True, average_attn_weights=True)
        transported = transported[:, 0]
        transported = transported + self.ffn(transported)
        return self.out(self.norm(transported)), weights[:, 0]


@dataclass
class Memory:
    """Training identities available for retrieval: standardized keys and measured directions."""

    features: torch.Tensor        # [M, F]
    presence: torch.Tensor        # [M, m]
    directions: torch.Tensor      # [M, G] u_bar
    identities: np.ndarray        # [M] str, for the firewall record only; never enters the model


class PPM(nn.Module):
    def __init__(self, arm: str, basis: np.ndarray):
        super().__init__()
        if arm not in pc.PPM_ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.encoder = KnowledgeEncoder()
        self.tokenizer = ProgramTokenizer(basis)
        self.router = SparseRouter() if pc.USES_ROUTER[arm] else None
        self.transport = MemoryTransport() if pc.USES_MEMORY[arm] else None
        self.gate = (nn.Sequential(nn.Linear(pc.HIDDEN, pc.GATE_WIDTH), nn.GELU(), nn.Linear(pc.GATE_WIDTH, 1))
                     if self.router is not None and self.transport is not None else None)
        self.anchored = arm in pc.ANCHORED
        if self.anchored:
            # ReZero: the correction enters at zero, so an untrained anchored model is A1.
            self.correction_scale = nn.Parameter(torch.zeros(()))

    def encode_memory(self, memory: Memory, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        """All memory keys and values without gradient, with dropout off, for retrieval."""
        was = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            keys = torch.cat([self.encoder(memory.features[s:s + chunk], memory.presence[s:s + chunk])
                              for s in range(0, memory.features.shape[0], chunk)])
            values = torch.cat([self.tokenizer.tokenize(memory.directions[s:s + chunk])
                                for s in range(0, memory.directions.shape[0], chunk)])
        self.encoder.train(was)
        return keys, values

    def retrieve(self, hidden: torch.Tensor, memory: Memory, exclude: torch.Tensor | None,
                 encoded: tuple | None = None) -> dict:
        keys_all, values_all = encoded if encoded is not None else self.encode_memory(memory)
        similarity = F.normalize(hidden.detach(), dim=1) @ F.normalize(keys_all, dim=1).T
        if exclude is not None:
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            valid = exclude >= 0
            similarity[rows[valid], exclude[valid]] = float("-inf")
        top = similarity.topk(pc.NEIGHBOURS, dim=1)
        index = top.indices
        flat = index.reshape(-1)
        keys = self.encoder(memory.features[flat], memory.presence[flat]).view(*index.shape, pc.HIDDEN)
        return {"keys": keys, "values": values_all[index], "index": index, "similarity": top.values}

    def forward(self, features: torch.Tensor, presence: torch.Tensor, memory: Memory | None = None,
                exclude: torch.Tensor | None = None, encoded: tuple | None = None,
                anchor: torch.Tensor | None = None) -> dict:
        hidden = self.encoder(features, presence)
        output = {"hidden": hidden}
        parametric = memory_codes = None
        if self.router is not None:
            parametric, routing = self.router(hidden)
            output.update(routing=routing, balance=routing["balance"])
        if self.transport is not None:
            if memory is None:
                raise ValueError(f"{self.arm} needs a memory")
            retrieved = self.retrieve(hidden, memory, exclude, encoded)
            memory_codes, attention = self.transport(hidden, retrieved["keys"], retrieved["values"])
            output.update(attention=attention, neighbours=retrieved["index"],
                          neighbour_similarity=retrieved["similarity"])
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(hidden))
            mixed = gate * parametric + (1.0 - gate) * memory_codes
            output["gate"] = gate[:, 0]
        else:
            mixed = parametric if parametric is not None else memory_codes
        codes = topk_signed(mixed)
        if self.anchored:
            if anchor is None:
                raise ValueError(f"{self.arm} needs the A1 anchor of every candidate")
            direction = F.normalize(anchor + self.correction_scale * (codes @ self.tokenizer.dictionary()), dim=1)
        else:
            direction = self.tokenizer.decode(codes)
        output.update(codes=codes, direction=direction)
        return output
