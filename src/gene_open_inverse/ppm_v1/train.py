"""Fitting one PPM arm on one pool: a selection run, then a refit at the selected epoch.

Selection holds out the inner-validation identities from the training rows *and* from the
memory, and predicts them from the remaining memory after every epoch, so the criterion is an
open-vocabulary one.  The refit uses every identity of the pool for exactly the selected
number of epochs with the same seed.  During every training step a row's own identity is
removed from the memory it can retrieve from.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import anchor as an
from . import contract as pc
from . import data as dt
from .model import PPM, Memory, topk_signed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass
class Fitted:
    model: PPM
    standardizer: dt.Standardizer
    memory: Memory | None
    pool: dt.Pool
    record: dict
    held: dt.Pool | None = None
    anchor_model: object = None


def _tensor(values, device=None) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device or DEVICE)


def build_memory(pool: dt.Pool, standardizer: dt.Standardizer, device=None) -> Memory:
    return Memory(features=_tensor(standardizer.transform(pool.features), device),
                  presence=_tensor(standardizer.presence(pool.features), device),
                  directions=_tensor(dt.identity_directions(pool), device),
                  identities=pool.identities.copy())


def _parameter_groups(model: PPM) -> list:
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and name != "tokenizer.programs":
            decay.append(parameter)
        else:
            plain.append(parameter)
    return [{"params": decay, "weight_decay": pc.WEIGHT_DECAY}, {"params": plain, "weight_decay": 0.0}]


def losses(model: PPM, output: dict, unit: torch.Tensor, view_a: torch.Tensor, view_b: torch.Tensor,
           identity: torch.Tensor, directions: torch.Tensor) -> dict:
    """Every term of the frozen objective for one batch of rows."""
    tokenizer = model.tokenizer
    terms = {"dir": (1.0 - (output["direction"] * unit).sum(dim=1)).mean(),
             "tok": (1.0 - (tokenizer.decode(tokenizer.tokenize(unit)) * unit).sum(dim=1)).mean(),
             "view": (1.0 - F.cosine_similarity(tokenizer.code(view_a), tokenizer.code(view_b), dim=1)).mean()}
    hidden = F.normalize(output["hidden"], dim=1)
    target = directions[identity] @ directions[identity].T
    distinct = identity[:, None] != identity[None, :]
    terms["metric"] = ((hidden @ hidden.T - target) ** 2)[distinct].mean()
    dictionary = tokenizer.dictionary()
    gram = dictionary @ dictionary.T
    terms["div"] = (gram[~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)] ** 2).mean()
    terms["balance"] = output.get("balance", torch.zeros((), device=unit.device))
    terms["total"] = sum(pc.LOSS_WEIGHTS[name] * value for name, value in terms.items())
    return terms


def _gradient_norms(model: PPM) -> dict:
    groups = {"encoder": model.encoder, "tokenizer": model.tokenizer, "router": model.router,
              "transport": model.transport, "gate": model.gate}
    norms = {}
    for name, module in groups.items():
        if module is None:
            continue
        squares = [float((p.grad.detach() ** 2).sum()) for p in module.parameters() if p.grad is not None]
        norms[name] = math.sqrt(sum(squares)) if squares else 0.0
    return norms


@torch.no_grad()
def predict(model: PPM, standardizer: dt.Standardizer, memory: Memory | None, features: np.ndarray,
            present: np.ndarray, anchor: np.ndarray | None = None, chunk: int = 1024) -> dict:
    """Unit effect directions for arbitrary candidates; a candidate with no modality gets zero.

    ``anchor`` is the A1 effect of every candidate, required by the anchored arms only.
    """
    model.eval()
    features = np.asarray(features, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    rows = np.flatnonzero(present)
    width = model.tokenizer.programs.shape[1]
    effect = np.zeros((features.shape[0], width), dtype=np.float64)
    extra = {"codes": np.zeros((features.shape[0], pc.PROGRAMS), dtype=np.float32)}
    encoded = model.encode_memory(memory) if memory is not None and model.transport is not None else None
    x = _tensor(standardizer.transform(features[rows]))
    flags = _tensor(standardizer.presence(features[rows]))
    anchors = _tensor(np.asarray(anchor)[rows]) if model.anchored else None
    collected: dict = {}
    for start in range(0, rows.size, chunk):
        output = model(x[start:start + chunk], flags[start:start + chunk], memory, None, encoded,
                       anchors[start:start + chunk] if anchors is not None else None)
        for key in ("direction", "codes", "gate", "attention", "neighbours", "neighbour_similarity", "hidden"):
            if key in output:
                collected.setdefault(key, []).append(output[key].detach().cpu().numpy())
        if "routing" in output:
            collected.setdefault("router_probabilities", []).append(
                output["routing"]["probabilities"].detach().cpu().numpy())
    merged = {key: np.concatenate(values) for key, values in collected.items()}
    effect[rows] = merged.pop("direction").astype(np.float64)
    extra["codes"][rows] = merged.pop("codes")
    extra.update({key: value for key, value in merged.items()})
    extra["rows"] = rows
    return {"effect": effect, **extra}


def _validation_cosine(model: PPM, standardizer: dt.Standardizer, memory: Memory | None,
                       held: dt.Pool, anchor: np.ndarray | None) -> float:
    predicted = predict(model, standardizer, memory, held.features, np.ones(held.identities.size, bool), anchor)
    rows = predicted["effect"][held.row_identity]
    return float(np.einsum("ij,ij->i", rows, held.unit.astype(np.float64)).mean())


def train(pool: dt.Pool, arm: str, seed: int, epochs: int | None = None,
          validation: np.ndarray | None = None, max_epochs: int = pc.MAX_EPOCHS,
          log=None) -> Fitted:
    """Selection run when ``validation`` is given, otherwise a fixed-length refit."""
    if (epochs is None) == (validation is None):
        raise ValueError("give exactly one of epochs (refit) or validation (selection)")
    deterministic(seed)
    held = None
    if validation is not None:
        held = dt.restrict(pool, validation)
        pool = dt.restrict(pool, ~validation)
    standardizer = dt.Standardizer(pool)
    basis = dt.program_basis(pool)
    model = PPM(arm, basis.basis).to(DEVICE)
    memory = build_memory(pool, standardizer) if model.transport is not None else None
    x = _tensor(standardizer.transform(pool.features))
    flags = _tensor(standardizer.presence(pool.features))
    unit, view_a, view_b = _tensor(pool.unit), _tensor(pool.view_a), _tensor(pool.view_b)
    identity = torch.as_tensor(pool.row_identity, device=DEVICE)
    directions = _tensor(dt.identity_directions(pool))
    optimizer = torch.optim.AdamW(_parameter_groups(model), lr=pc.LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    anchor_model, train_anchor, held_anchor = None, None, None
    if model.anchored:
        anchor_model = an.fit_a1(pool)
        train_anchor = _tensor(an.training_anchors(pool, anchor_model.penalty))
        if held is not None:
            held_anchor = anchor_model.effect(held.features, np.ones(held.identities.size, dtype=bool))

    history, best, best_epoch, stale = [], -np.inf, 0, 0
    if model.anchored and held is not None:
        # Epoch 0 is A1 itself; early stopping may return it.
        best = _validation_cosine(model, standardizer, memory, held, held_anchor)
        history.append({"epoch": 0, "validation_cosine": best})
    total = epochs if epochs is not None else max_epochs
    started = time.time()
    for epoch in range(1, total + 1):
        model.train()
        order = torch.randperm(pool.rows, generator=generator).to(DEVICE)
        sums: dict = {}
        steps = 0
        for start in range(0, pool.rows, pc.BATCH_ROWS):
            rows = order[start:start + pc.BATCH_ROWS]
            ids = identity[rows]
            output = model(x[ids], flags[ids], memory, ids if memory is not None else None, None,
                           train_anchor[ids] if train_anchor is not None else None)
            terms = losses(model, output, unit[rows], view_a[rows], view_b[rows], ids, directions)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: "
                                         f"{ {k: float(v) for k, v in terms.items()} }")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            if start + pc.BATCH_ROWS >= pool.rows:
                gradients = _gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pc.GRADIENT_CLIP)
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            steps += 1
        entry = {"epoch": epoch, **{f"loss_{k}": v / steps for k, v in sums.items()},
                 "gradient_norms_last_step": gradients, "seconds": time.time() - started}
        if "gate" in output:
            entry["gate_mean_last_batch"] = float(output["gate"].mean())
        if "routing" in output:
            entry["expert_top1_share_last_batch"] = np.bincount(
                output["routing"]["top"][:, 0].cpu().numpy(), minlength=pc.EXPERTS).tolist()
        if held is not None:
            entry["validation_cosine"] = _validation_cosine(model, standardizer, memory, held, held_anchor)
            if entry["validation_cosine"] > best:
                best, best_epoch, stale = entry["validation_cosine"], epoch, 0
            else:
                stale += 1
        history.append(entry)
        if log is not None:
            log(entry)
        if held is not None and stale >= pc.PATIENCE:
            break
    record = {"arm": arm, "seed": seed, "rows": pool.rows, "identities": int(pool.identities.size),
              "epochs_run": len(history), "history": history,
              "program_basis_explained": float(basis.explained)}
    if anchor_model is not None:
        record.update(anchor_penalty=float(anchor_model.penalty),
                      anchor_held_out_cosine=float(anchor_model.held_out_cosine),
                      correction_scale=float(model.correction_scale.detach()))
    if held is not None:
        record.update(selected_epoch=best_epoch, best_validation_cosine=best,
                      validation_identities=int(held.identities.size), validation_rows=held.rows)
    return Fitted(model=model, standardizer=standardizer, memory=memory, pool=pool, record=record, held=held,
                  anchor_model=anchor_model)


def select_and_refit(pool: dt.Pool, arm: str, seed: int, log=None) -> tuple[Fitted, Fitted]:
    """The frozen two-run procedure for one pool."""
    selection = train(pool, arm, seed, validation=dt.validation_mask(pool.identities), log=log)
    final = train(pool, arm, seed, epochs=selection.record["selected_epoch"], log=log)
    return selection, final


def program_usage(fitted: Fitted) -> dict:
    """D1 on the tokenizer's codes of every training row."""
    model = fitted.model
    model.eval()
    with torch.no_grad():
        codes = torch.cat([model.tokenizer.tokenize(_tensor(fitted.pool.unit[s:s + 2048]))
                           for s in range(0, fitted.pool.rows, 2048)]).cpu().numpy()
    return code_statistics(codes)


def code_statistics(codes: np.ndarray) -> dict:
    active = np.abs(codes) > 0
    frequency = active.mean(axis=0)
    share = np.abs(codes) / np.maximum(np.abs(codes).sum(axis=1, keepdims=True), 1e-30)
    entropy = -(share * np.log(np.maximum(share, 1e-30))).sum(axis=1)
    usage = frequency / max(frequency.sum(), 1e-30)
    dominant = int(np.argmax(frequency))
    return {"active_per_row_mean": float(active.sum(axis=1).mean()),
            "effective_programs_per_row_median": float(np.median(np.exp(entropy))),
            "usage_entropy_normalized": float(-(usage[usage > 0] * np.log(usage[usage > 0])).sum()
                                              / np.log(usage.size)),
            "dead_program_fraction": float((frequency == 0).mean()),
            "dominant_program": dominant,
            "dominant_program_frequency": float(frequency[dominant]),
            "dominant_program_mean_share": float(share[:, dominant].mean())}
