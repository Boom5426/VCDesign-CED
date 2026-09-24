#!/usr/bin/env python3
"""Phase D: train the clean base under the incumbent's own recipe.

The recipe is not reimplemented.  ``_epoch``, ``_role_tensors``, ``_seed``,
``capture_branch_state`` and every constant are imported from the incumbent's
training module, so the optimizer, scheduler, loss, sampler, batch size, seed,
horizon and objective schedule are identical by construction rather than by
inspection.  Exactly two things differ from the incumbent: the modality set is the
strict legal one without TEXT, and the capability encoder has no learnable unknown
token.

One network is trained.  ``CLEAN_BASE`` is its score; ``CLEAN_EFFECT`` is that score
fused with the frozen effect branch at ``beta = 1``.  They therefore share every
weight, which is what makes ``CLEAN_EFFECT - CLEAN_BASE`` a clean causal contrast
for the effect branch and nothing else.  Each arm selects its own checkpoint from
the same curve under its own scoring rule, which is Phase E.

``G_check`` is not read here and is not reachable through the asset loader.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.global_objective_knowledge_attribution_v1.train_factorial import (
    BASE_EPOCHS, BATCH_SIZE, CLIP_NORM, FINAL_EPOCH, LR, SEED, WEIGHT_DECAY,
    _epoch, _role_tensors, _seed, capture_branch_state,
)
from ..model.open_vocab_dual_encoder_v1.data import sha256
from . import contract as fc
from .model import CleanAttributionModel
from .scoring import strict_legal_dims, strict_legal_role


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = AuditConfig.load(Path(args.config))
    # The frozen loader admits only STRING or STRING+MAPKG+TEXT, so it is used exactly
    # as frozen and TEXT is dropped immediately afterwards, before any tensor exists.
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    dims = strict_legal_dims(assets)
    if set(dims) != set(fc.PRIMARY_MODALITIES) or "TEXT" in dims:
        raise RuntimeError("the clean track must carry exactly the strict legal modalities")

    _seed(SEED)
    role = strict_legal_role(assets.role_arrays("G_fit"))
    tensors = _role_tensors(role, device)
    if "TEXT" in tensors["descriptors"] or "TEXT" in tensors["masks"]:
        raise RuntimeError("TEXT reached a training tensor")
    model = CleanAttributionModel(dims).to(device)
    if any(name.endswith("unknown_token") for name, _ in model.named_parameters()):
        raise RuntimeError("the clean model must not carry a learnable unknown token")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    amp = device.type == "cuda"

    history = []
    checkpoints = {}
    for epoch in range(1, FINAL_EPOCH + 1):
        objective = "PAIR" if epoch <= BASE_EPOCHS else "SET"
        record = _epoch(model, optimizer, role, tensors, generator, epoch=epoch,
                        objective=objective, device=device, amp=amp)
        record["phase"] = "base" if epoch <= BASE_EPOCHS else "branch"
        history.append(record)
        print(f"  epoch {epoch:2d} [{objective}] loss={record['loss']:.6f} "
              f"pair={record['pair_loss']:.6f} set={record['set_loss']:.6f} "
              f"grad={record['gradient_norm']:.4f} {record['epoch_seconds']:.1f}s", flush=True)
        if epoch > BASE_EPOCHS:
            path = output / f"epoch_{epoch:03d}.pt"
            payload = capture_branch_state(model, optimizer, generator)
            payload.update({"schema": "VCDESIGN_FINAL_CLEAN_MODEL_V1", "epoch": epoch,
                            "objective": objective, "modalities": list(fc.PRIMARY_MODALITIES),
                            "all_missing_rule": fc.ALL_MISSING_RULE,
                            "check_response_accessed": False})
            torch.save(payload, path)
            checkpoints[str(epoch)] = {"path": str(path), "sha256": sha256(path)}

    manifest = {
        "schema": "VCDESIGN_FINAL_CLEAN_MODEL_V1_TRAIN",
        "method_of_record": fc.FINAL_METHOD,
        "read_G_check": False,
        "modalities": list(fc.PRIMARY_MODALITIES), "text_status": fc.TEXT_STATUS,
        "all_missing_rule": fc.ALL_MISSING_RULE,
        "recipe": {"source": "gene_open_inverse.model.global_objective_knowledge_attribution_v1."
                             "train_factorial, imported not reimplemented",
                   "seed": SEED, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "gradient_clip_norm": CLIP_NORM, "batch_size": BATCH_SIZE,
                   "base_epochs": [1, BASE_EPOCHS], "branch_epochs": [BASE_EPOCHS + 1, FINAL_EPOCH],
                   "objective_schedule": "PAIR then SET", "amp_bfloat16": amp},
        "modality_dimensions": dims,
        "text_reached_a_tensor": False,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history, "checkpoints": checkpoints,
        "runtime": {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "torch": torch.__version__},
        "provenance": config.provenance(),
    }
    (output / "train_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase D: clean base training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    manifest = run(parser.parse_args())
    print(f"\nparameters={manifest['parameters']:,}  checkpoints={sorted(manifest['checkpoints'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
