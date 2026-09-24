#!/usr/bin/env python3
"""Post hoc companion: PPM arms under VCDesign evaluation contract V1 (BU@20 primary).

PPM's decision is its pre-registered MBRU decision (protocol Section 8); the contract was
frozen during the mission, so this readout is reported beside it and decides nothing.  It
uses the same saved effects, the same C-016 cross-fit and the same frozen fusion as ``grade``,
and computes every metric with ``evaluation_contract_v1.metrics``.  PHR@20 is not available
for these contexts (two-view packs) and is reported as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..decision_consistent_effect_v1 import arms as dca
from ..decision_consistent_effect_v1 import stats as st
from ..evaluation_contract_v1 import adoption as ad
from ..evaluation_contract_v1 import contract as ec
from ..evaluation_contract_v1 import metrics as mt
from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..open_vocab_generalization_v1 import masking as mk
from . import contract as pc
from .run import _load_arm


def _rows(pack, fused_by_view: list, columns: np.ndarray) -> dict:
    remap = {int(v): i for i, v in enumerate(columns.tolist())}
    inside = np.asarray([remap.get(int(p), -1) for p in np.asarray(pack.self_positions).tolist()])
    ids = np.asarray(pack.ids, dtype=str)[columns]
    per_view = [mt.row_metrics(np.asarray(fused_by_view[v])[:, columns], np.asarray(pack.utility[v])[:, columns],
                               ids, inside) for v in (0, 1)]
    return {k: np.concatenate([per_view[0][k], per_view[1][k]]) for k in per_view[0]}


def context_rows(pack, run: Path, held: str, arms: tuple) -> dict:
    ridge = np.load(run / f"ridge_{held}.npz", allow_pickle=False)
    eligible, folds = ridge["eligible"], ridge["folds"]
    columns = np.flatnonzero(eligible)
    effects = {pc.RIDGE: ridge["cross"]}
    for arm in arms:
        loaded = _load_arm(run / "fits", arm, held)
        effects[arm] = mk._crossfit(loaded["seen"]["effect"].astype(np.float64),
                                    [f["effect"].astype(np.float64) for f in loaded["folds"]], eligible, folds)
    rows = {pc.BASE: _rows(pack, pack.base, columns)}
    for name, effect in effects.items():
        cosine = dca.cosine_scores(pack, effect)
        rows[name] = _rows(pack, [ev.fuse(pack.base[v], cosine[v]) for v in (0, 1)], columns)
    identities = np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)]
    rows["labels"] = np.concatenate([identities, identities])
    return rows


def readout(run: Path, packs_dir: str, arms: tuple) -> dict:
    packs = {name: hn.load_pack(name, packs_dir) for name in fc.CONTEXTS}
    per = {held: context_rows(packs[held], run, held, arms) for held in ec.PRIMARY_CONTEXTS}
    labels = np.concatenate([per[h]["labels"] for h in ec.PRIMARY_CONTEXTS])
    result = {"contract": ec.VERSION, "status": "POST_HOC_COMPANION_NOT_DECISIVE", "regime": "MASKED fused",
              "PHR@20": "not available: the four-context packs keep two views, realized J needs four quarters"}
    for arm in arms:
        block = {}
        for metric in ("BU@10", "BU@20", "BU@50", "MU@20", "HvHit@20"):
            difference = np.concatenate([per[h][arm][metric] - per[h][pc.RIDGE][metric] for h in ec.PRIMARY_CONTEXTS])
            block[f"delta_{metric}_vs_{pc.RIDGE}"] = st.cluster_bootstrap_labels(difference, labels)
        block["BU@20_by_context"] = {}
        for h in ec.PRIMARY_CONTEXTS:
            ids = per[h]["labels"]
            entry = st.cluster_bootstrap_labels(per[h][arm]["BU@20"] - per[h][pc.RIDGE]["BU@20"], ids)
            block["BU@20_by_context"][h] = {"median": entry["median"], "mean": entry["mean"], "ci95": entry["median_ci95"]}
        gate = mt.qwr(np.concatenate([per[h][arm]["BU@20"] for h in ec.PRIMARY_CONTEXTS]),
                      np.concatenate([per[h][pc.RIDGE]["BU@20"] for h in ec.PRIMARY_CONTEXTS]))
        block["QWR@20_gate_vs_incumbent"] = gate
        block["QWR@20_paper_vs_BASE_by_context"] = {
            h: mt.qwr(per[h][arm]["BU@20"], per[h][pc.BASE]["BU@20"]) for h in ec.PRIMARY_CONTEXTS}
        block["adoption_gate"] = ad.decide({
            "bu20": {k: block[f"delta_BU@20_vs_{pc.RIDGE}"][k] for k in ("median", "state")},
            "bu20_by_context": block["BU@20_by_context"],
            "mu20_state": block[f"delta_MU@20_vs_{pc.RIDGE}"]["state"],
            "hvhit20_state": block[f"delta_HvHit@20_vs_{pc.RIDGE}"]["state"],
            "phr20_state": None, "qwr_gate20": gate["gate"]})
        result[arm] = block
    result["RIDGE_UNIT_paper"] = {h: {"QWR@20_vs_BASE": mt.qwr(per[h][pc.RIDGE]["BU@20"], per[h][pc.BASE]["BU@20"]),
                                      "delta_BU@20_vs_BASE": st.cluster_bootstrap_labels(
                                          per[h][pc.RIDGE]["BU@20"] - per[h][pc.BASE]["BU@20"], per[h]["labels"])}
                                  for h in ec.PRIMARY_CONTEXTS}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PPM under evaluation contract V1 (post hoc companion)")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--packs", required=True)
    parser.add_argument("--arms", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    target = run / f"contract_v1_readout__{args.arms.replace(',', '_')}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    record = readout(run, args.packs, tuple(args.arms.split(",")))
    target.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
