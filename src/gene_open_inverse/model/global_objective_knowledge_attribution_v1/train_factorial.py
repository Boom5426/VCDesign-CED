#!/usr/bin/env python3
"""FIT-only, checkpoint-branched 2×2 global objective/knowledge attribution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..open_vocab_dual_encoder_v1.data import FrozenTransitionAssets, sha256
from ..open_vocab_dual_encoder_v1.loss import top_focused_pairwise_loss
from ..structured_requirement_capability_energy_v2.loss import combined_v2_loss, gt_top20_positive_set_loss
from .assets import AttributionRoleArrays, GlobalAttributionAssets
from .model import GlobalObjectiveKnowledgeAttributionModel


SEED, LR, WEIGHT_DECAY, CLIP_NORM, BATCH_SIZE = 20260916, 1e-4, 1e-4, 1.0, 8
BASE_EPOCHS, FINAL_EPOCH = 5, 10


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def capture_branch_state(
    model: GlobalObjectiveKnowledgeAttributionModel, optimizer: torch.optim.Optimizer, generator: torch.Generator,
) -> dict:
    return {
        "model_state": copy.deepcopy(model.state_dict()), "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None, "amp_scaler_state": None, "generator_state": generator.get_state(),
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_branch_state(
    payload: Mapping, model: GlobalObjectiveKnowledgeAttributionModel, optimizer: torch.optim.Optimizer, generator: torch.Generator,
) -> None:
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    # map_location=device may move serialized RNG byte tensors to CUDA;
    # Generator.set_state and torch.set_rng_state require CPU byte tensors.
    generator.set_state(payload["generator_state"].cpu())
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])


def objective_loss(pair_loss: torch.Tensor, set_loss: torch.Tensor, *, objective: str) -> torch.Tensor:
    if objective == "PAIR":
        return combined_v2_loss(pair_loss, set_loss, stage="global")
    if objective == "SET":
        return combined_v2_loss(pair_loss, set_loss, stage="conditional")
    raise ValueError("objective must be PAIR or SET")


def _role_tensors(role: AttributionRoleArrays, device: torch.device) -> dict:
    return {
        "source": torch.from_numpy(role.source).to(device),
        "goal": torch.from_numpy(role.goal).to(device),
        "descriptors": {name: torch.from_numpy(value).to(device) for name, value in role.capabilities.items()},
        "masks": {name: torch.from_numpy(value).to(device) for name, value in role.capability_present.items()},
        "utility": tuple(torch.from_numpy(FrozenTransitionAssets.cross_view_utility(role.base, view)).to(device) for view in (0, 1)),
        "margins": torch.from_numpy(FrozenTransitionAssets.measurement_margin(role.base)).to(device),
    }


def _epoch(
    model: GlobalObjectiveKnowledgeAttributionModel, optimizer: torch.optim.Optimizer, role: AttributionRoleArrays,
    tensors: dict, generator: torch.Generator, *, epoch: int, objective: str, device: torch.device, amp: bool,
) -> dict:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    permutation = torch.randperm(len(role.ids), generator=generator, device=device)
    totals = {name: 0.0 for name in ("loss", "pair_loss", "set_loss", "pair_count", "hard_negative_fraction", "gradient_norm")}
    steps = 0
    for start in range(0, len(role.ids), BATCH_SIZE):
        rows = permutation[start:start + BATCH_SIZE]
        source = torch.cat((tensors["source"][0, rows], tensors["source"][1, rows]), dim=0)
        goal = torch.cat((tensors["goal"][0, rows], tensors["goal"][1, rows]), dim=0)
        utility = torch.cat((tensors["utility"][0][rows], tensors["utility"][1][rows]), dim=0)
        self_indices = torch.cat((rows, rows), dim=0)
        margins = torch.cat((tensors["margins"][rows], tensors["margins"][rows]), dim=0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            scores = model(source, goal, tensors["descriptors"], tensors["masks"])
            pair_loss, pair_diagnostic = top_focused_pairwise_loss(
                scores.float(), utility, self_indices, margins, generator=generator,
                gt_top_k=128, predicted_top_k=128, random_k=256, max_pairs_per_query=4096,
            )
            if objective == "SET":
                set_loss, _ = gt_top20_positive_set_loss(scores.float(), utility, self_indices, positive_k=20)
            else:
                set_loss = pair_loss.detach() * 0.0
            loss = objective_loss(pair_loss, set_loss, objective=objective)
        if not all(torch.isfinite(value).all() for value in (pair_loss, set_loss, loss)):
            raise FloatingPointError("non-finite factorial global training loss")
        loss.backward()
        squared = sum((parameter.grad.detach().float().norm().square() for parameter in model.parameters() if parameter.grad is not None), torch.zeros((), device=device))
        gradient = float(squared.sqrt().item())
        if not np.isfinite(gradient) or gradient == 0.0:
            raise FloatingPointError("factorial global gradient is non-finite or zero")
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        totals["loss"] += float(loss.detach().item())
        totals["pair_loss"] += float(pair_loss.detach().item())
        totals["set_loss"] += float(set_loss.detach().item())
        totals["pair_count"] += float(pair_diagnostic.pair_count)
        totals["hard_negative_fraction"] += float(pair_diagnostic.hard_negative_fraction)
        totals["gradient_norm"] += gradient
        steps += 1
    model.eval()
    return {
        "epoch": epoch, "objective": objective, "steps": steps,
        **{name: value / steps for name, value in totals.items()},
        "logit_scale": float(model.logit_scale.exp().clamp(max=100.0).detach().item()),
        "epoch_seconds": time.perf_counter() - started,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else 0.0,
        "check_response_accessed": False,
    }


def _write_json_line(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _train_base(
    name: str, modalities: tuple[str, ...], assets: GlobalAttributionAssets, out: Path, device: torch.device,
) -> Path:
    _seed(SEED)
    role = assets.role_arrays("G_fit")
    tensors = _role_tensors(role, device)
    model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device); generator.manual_seed(SEED)
    amp = device.type == "cuda"
    log = out / f"{name}_BASE_train_metrics.jsonl"
    for epoch in range(1, BASE_EPOCHS + 1):
        record = _epoch(model, optimizer, role, tensors, generator, epoch=epoch, objective="PAIR", device=device, amp=amp)
        record.update({"phase": "base", "base": name, "modalities": list(modalities)})
        _write_json_line(log, record)
    path = out / f"{name}_BASE_E5.pt"
    payload = capture_branch_state(model, optimizer, generator)
    payload.update({"schema": "VCDESIGN_GOKA_V1_BRANCH_BASE", "base": name, "modalities": list(modalities), "epoch": BASE_EPOCHS, "objective": "PAIR"})
    torch.save(payload, path)
    return path


def _train_branch(
    config: str, base_checkpoint: Path, assets: GlobalAttributionAssets, out: Path, *, objective: str, device: torch.device,
) -> Path:
    role = assets.role_arrays("G_fit")
    tensors = _role_tensors(role, device)
    payload = torch.load(base_checkpoint, map_location=device, weights_only=False)
    model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device)
    restore_branch_state(payload, model, optimizer, generator)
    branch = out / config
    branch.mkdir()
    log = branch / "train_metrics.jsonl"
    amp = device.type == "cuda"
    for epoch in range(BASE_EPOCHS + 1, FINAL_EPOCH + 1):
        record = _epoch(model, optimizer, role, tensors, generator, epoch=epoch, objective=objective, device=device, amp=amp)
        record.update({"phase": "branch", "config": config, "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": sha256(base_checkpoint)})
        _write_json_line(log, record)
    final = branch / "epoch_010.pt"
    final_payload = capture_branch_state(model, optimizer, generator)
    final_payload.update({
        "schema": "VCDESIGN_GOKA_V1_FINAL_CONFIG", "config": config, "modalities": list(assets.modalities),
        "epoch": FINAL_EPOCH, "stage_b_objective": objective, "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256(base_checkpoint), "check_response_accessed": False,
    })
    torch.save(final_payload, final)
    return final


def _make_assets(args: argparse.Namespace, modalities: tuple[str, ...]) -> GlobalAttributionAssets:
    values = {key: getattr(args, key) for key in ("freeze", "cache", "cache_ids", "cache_manifest", "string", "source_goal_manifest", "source_states", "goal_states")}
    if len(modalities) > 1:
        values.update({key: getattr(args, key) for key in ("mapkg", "mapkg_manifest", "candidate_knowledge", "candidate_knowledge_freeze")})
    return GlobalAttributionAssets(modalities=modalities, **values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "cache", "cache_ids", "cache_manifest", "string", "source_goal_manifest", "source_states", "goal_states", "mapkg", "mapkg_manifest", "candidate_knowledge", "candidate_knowledge_freeze", "capability_manifest", "out"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-after-string-base", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists() and not args.resume_after_string_base:
        raise FileExistsError(f"refusing to overwrite factorial FIT output: {out}")
    capability_manifest = json.loads(Path(args.capability_manifest).read_text())
    if capability_manifest.get("accepted_modalities") != ["STRING", "MAPKG", "TEXT"]:
        raise RuntimeError("factorial V1 requires the frozen V2 STRING/MAPKG/TEXT manifest")
    device = torch.device(args.device)
    if args.resume_after_string_base:
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest.get("status") != "GOKA_V1_G_FIT_ONLY_RUNNING":
            raise RuntimeError("only a failed, still-running-state factorial artifact may resume")
        if not (out / "STRING_BASE_E5.pt").is_file() or any((out / name).exists() for name in ("S_PAIR", "S_SET", "MULTI_BASE_E5.pt", "M_PAIR", "M_SET")):
            raise RuntimeError("resume requires exactly one completed STRING_BASE_E5 and no downstream branch artifact")
        manifest["resume"] = {
            "from": "STRING_BASE_E5", "reason": "CUDA map_location moved serialized RNG state tensors; restore now canonicalizes byte states to CPU",
            "resume_code_sha256": _code_hash(),
        }
    else:
        out.mkdir(parents=True)
        manifest = {
            "schema": "VCDESIGN_GLOBAL_OBJECTIVE_KNOWLEDGE_ATTRIBUTION_V1_TRAIN",
            "status": "GOKA_V1_G_FIT_ONLY_RUNNING", "seed": SEED, "base_epochs": [1, 5], "branch_epochs": [6, 10],
            "optimizer": {"lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": CLIP_NORM, "batch_size": BATCH_SIZE},
            "capability_manifest": {"path": str(Path(args.capability_manifest).resolve()), "sha256": sha256(args.capability_manifest)},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "amp_bfloat16": device.type == "cuda"},
            "code_sha256": _code_hash(), "check_response_accessed": False,
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    string_assets = _make_assets(args, ("STRING",))
    string_base = out / "STRING_BASE_E5.pt" if args.resume_after_string_base else _train_base("STRING", ("STRING",), string_assets, out, device)
    finals = {
        "S_PAIR": _train_branch("S_PAIR", string_base, string_assets, out, objective="PAIR", device=device),
        "S_SET": _train_branch("S_SET", string_base, string_assets, out, objective="SET", device=device),
    }
    multi_assets = _make_assets(args, ("STRING", "MAPKG", "TEXT"))
    multi_base = _train_base("MULTI", ("STRING", "MAPKG", "TEXT"), multi_assets, out, device)
    finals.update({
        "M_PAIR": _train_branch("M_PAIR", multi_base, multi_assets, out, objective="PAIR", device=device),
        "M_SET": _train_branch("M_SET", multi_base, multi_assets, out, objective="SET", device=device),
    })
    manifest.update({
        "status": "GOKA_V1_G_FIT_ONLY_COMPLETE", "final_checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in finals.items()},
        "string_base": {"path": str(string_base), "sha256": sha256(string_base)}, "multi_base": {"path": str(multi_base), "sha256": sha256(multi_base)},
        "assets": {"STRING": string_assets.asset_record(), "MULTI": multi_assets.asset_record()},
        "check_response_accessed": False,
    })
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
