#!/usr/bin/env python3
"""Evidence experiments for RIDGE_UNIT (A1): atlas scale, transfer matrix, modality, rank, retention.

``fit``: every spec on one pool (SEEN or a C-016 fold) of one held context.  The stratum, the
masks, the queries, BASE, utility and fusion are the frozen ones for every spec; a spec changes
only what A1 trains on.  The natural full atlas at rank 256 must reproduce C-018's A1 to 1e-10
before anything is kept.  ``evaluate``: contract V1 grading of every spec against BASE and
against A1, MASKED and SEEN, and the prediction-versus-design relationship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..decision_consistent_effect_v1 import arms as dca
from ..decision_consistent_effect_v1 import stats as st
from ..evaluation_contract_v1 import metrics as mt
from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..four_context_v1 import predictors as pd
from ..open_vocab_generalization_v1 import contract as ov
from ..open_vocab_generalization_v1 import masking as mk
from ..ppm_v1 import run as pr
from ..ppm_v1.contract_readout import _rows
from ..rwed_v1 import estimator as es
from . import contract as xc

STRING, MAPKG = slice(0, 512), slice(512, 1536)
FLAGS = {"STRING": 1536, "MAPKG": 1537}


def _refuse(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists")


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-300)


def restrict_modality(features: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """Zero the other modality's block and flag; presence becomes this modality's flag."""
    x = np.array(features, dtype=np.float64, copy=True)
    if modality == "both":
        return x, (x[:, FLAGS["STRING"]] > 0) | (x[:, FLAGS["MAPKG"]] > 0)
    other = "MAPKG" if modality == "STRING" else "STRING"
    x[:, MAPKG if other == "MAPKG" else STRING] = 0.0
    x[:, FLAGS[other]] = 0.0
    return x, x[:, FLAGS[modality]] > 0


def row_subsample(keys: np.ndarray, present: np.ndarray, size: int) -> np.ndarray:
    """Keep the ``size`` present rows with the smallest SHA-256 of their key: outcome-free, deterministic."""
    rows = np.flatnonzero(present)
    digest = np.asarray([int.from_bytes(hashlib.sha256(f"{xc.ROW_NAMESPACE}:{k}".encode()).digest()[:8], "big")
                         for k in keys[rows].tolist()], dtype=np.uint64)
    keep = np.zeros(present.size, dtype=bool)
    keep[rows[np.argsort(digest, kind="stable")[:size]]] = True
    return keep


def fit_spec(packs: dict, spec: dict, excluded, matched_rows: int, pack) -> tuple[np.ndarray, dict]:
    pool = es.views_pool(packs, spec["contexts"], excluded)
    unit = dca.unit_target_pool(pool)
    features, present = restrict_modality(unit["features"], spec["modality"])
    present = present & np.asarray(unit["present"], dtype=bool)
    if spec["matched"]:
        present = row_subsample(unit["keys"], present, matched_rows)
    model = pd.fit(features, unit["responses"], present, unit["keys"], spec["contexts"], rank=spec["rank"])
    held_x, held_present = restrict_modality(pack.features, spec["modality"])
    held_present = held_present & np.asarray(pack.present, dtype=bool)
    return model.effect(held_x, held_present), {"rows": model.rows, "identities": model.identities,
                                                "penalty": model.penalty, "held_out_cosine": model.held_out_cosine}


def fit_stage(packs: dict, held: str, pool_name: str, output: Path) -> dict:
    setup = pr.held_setup(packs, held)
    pack = setup["pack"]
    excluded = pr.pool_exclusion(setup, pool_name)
    singles = [int(np.asarray(es.views_pool(packs, (c,), excluded)["present"], dtype=bool).sum()) for c in setup["training"]]
    matched_rows = min(singles)
    keep = (np.ones(pack.ids.size, dtype=bool) if pool_name == "SEEN"
            else setup["eligible"] & (setup["folds"] == int(pool_name)))
    effects, records = {}, {}
    for spec in xc.specs(held):
        effect, record = fit_spec(packs, spec, excluded, matched_rows, pack)
        effects[spec["name"]], records[spec["name"]] = effect[keep], record
    c018 = np.load(Path(xc.C018_RUN) / f"effects_{held}.npz", allow_pickle=False)
    reference = (c018["unit_seen"] if pool_name == "SEEN" else c018["unit_cross"])[keep]
    gate = {"full_rank256_vs_C018_A1_max_abs": float(np.abs(effects[xc.FULL] - reference).max()),
            "eligible_identical": bool(np.array_equal(c018["eligible"], setup["eligible"])),
            "folds_identical": bool(np.array_equal(c018["folds"], setup["folds"]))}
    gate["passes"] = bool(gate["full_rank256_vs_C018_A1_max_abs"] <= 1e-10 and gate["eligible_identical"] and gate["folds_identical"])
    if not gate["passes"]:
        raise RuntimeError(f"provenance gate failed: {gate}")
    np.savez_compressed(output / "effects.npz", rows=np.flatnonzero(keep), **effects)
    return {"held": held, "pool": pool_name, "matched_rows": matched_rows, "single_context_rows": singles,
            "provenance_gate": gate, "fits": records}


def _pooled(values: list, labels: list) -> dict:
    return st.cluster_bootstrap_labels(np.concatenate(values), np.concatenate(labels))


def evaluate(packs: dict, run: Path, sped_run: Path) -> dict:
    per, cand, meta = {}, {}, {}
    for held in xc.PRIMARY_CONTEXTS:
        pack = packs[held]
        c018 = np.load(Path(xc.C018_RUN) / f"effects_{held}.npz", allow_pickle=False)
        eligible, folds = c018["eligible"], c018["folds"]
        columns = np.flatnonzero(eligible)
        seen = np.load(run / "fits" / held / "SEEN" / "effects.npz", allow_pickle=False)
        parts = [np.load(run / "fits" / held / str(f) / "effects.npz", allow_pickle=False) for f in range(ov.MASK_FOLDS)]
        sped_seen = np.load(sped_run / "fits" / held / "SEEN" / "effects.npz", allow_pickle=False)
        common = {"SEEN": np.repeat(sped_seen["common"][None], pack.ids.size, 0)}
        common["MASKED"] = common["SEEN"].copy()
        for f in range(ov.MASK_FOLDS):
            common["MASKED"][eligible & (folds == f)] = np.load(sped_run / "fits" / held / str(f) / "effects.npz",
                                                                allow_pickle=False)["common"]
        truth = ev.consensus(np.asarray(pack.responses)[:, columns])
        per[held], cand[held] = {"SEEN": {}, "MASKED": {}}, {"SEEN": {}, "MASKED": {}}
        per[held]["SEEN"]["BASE"] = per[held]["MASKED"]["BASE"] = _rows(pack, pack.base, columns)
        for spec in xc.specs(held):
            name = spec["name"]
            seen_effect = seen[name]
            masked_effect = seen_effect.copy()
            for f, part in enumerate(parts):
                rows = part["rows"]
                masked_effect[rows] = part[name]
            for regime, effect in (("SEEN", seen_effect), ("MASKED", masked_effect)):
                cosine = dca.cosine_scores(pack, effect)
                per[held][regime][name] = {"fused": _rows(pack, [ev.fuse(pack.base[v], cosine[v]) for v in (0, 1)], columns),
                                           "effect_only": _rows(pack, cosine, columns)}
                e, c = effect[columns], common[regime][columns]
                spec_e = e - np.einsum("ij,ij->i", e, c)[:, None] * c
                spec_t = truth - np.einsum("ij,ij->i", truth, c)[:, None] * c
                cand[held][regime][name] = {"full": np.einsum("ij,ij->i", _unit(e), _unit(truth)),
                                            "specific": np.einsum("ij,ij->i", _unit(spec_e), _unit(spec_t))}
        identities = np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)]
        per[held]["labels"] = np.concatenate([identities, identities])
        cand[held]["labels"] = np.asarray(pack.ids, dtype=str)[columns]
        meta[held] = json.loads((run / "fits" / held / "SEEN" / "record.json").read_text())

    result = {"mission": xc.MISSION, "read_anchor_G_check": False, "per_context": {}, "pooled_classes": {}}
    points = []
    for held in xc.PRIMARY_CONTEXTS:
        block = {}
        for spec in xc.specs(held):
            name = spec["name"]
            entry = {"experiment": spec["experiment"], "contexts": list(spec["contexts"]),
                     "rows_SEEN": meta[held]["fits"][name]["rows"], "penalty_SEEN": meta[held]["fits"][name]["penalty"]}
            for regime in ("SEEN", "MASKED"):
                rows = per[held][regime]
                entry[f"{regime}|fused|BU@20_minus_BASE"] = st.cluster_bootstrap_labels(
                    rows[name]["fused"]["BU@20"] - rows["BASE"]["BU@20"], per[held]["labels"])
                entry[f"{regime}|QWR@20_vs_BASE"] = mt.qwr(rows[name]["fused"]["BU@20"], rows["BASE"]["BU@20"])["paper"]
                if name != xc.FULL:
                    for mode in ("fused", "effect_only"):
                        entry[f"{regime}|{mode}|BU@20_minus_A1"] = st.cluster_bootstrap_labels(
                            rows[name][mode]["BU@20"] - rows[xc.FULL][mode]["BU@20"], per[held]["labels"])
                entry[f"{regime}|candidate_full_median"] = float(np.median(cand[held][regime][name]["full"]))
                entry[f"{regime}|candidate_specific_median"] = float(np.median(cand[held][regime][name]["specific"]))
            seen_gain = entry["SEEN|fused|BU@20_minus_BASE"]["median"]
            entry["MASKED_over_SEEN_retained"] = entry["MASKED|fused|BU@20_minus_BASE"]["median"] / seen_gain if seen_gain else None
            points.append((held, name, entry["MASKED|candidate_specific_median"], entry["MASKED|fused|BU@20_minus_BASE"]["median"]))
            block[name] = entry
        result["per_context"][held] = block

    classes = {"1_context": lambda s: s["experiment"] == "atlas" and len(s["contexts"]) == 1,
               "2_context": lambda s: s["experiment"] == "atlas" and len(s["contexts"]) == 2,
               "3_context": lambda s: s["experiment"] == "atlas" and len(s["contexts"]) == 3,
               "1_context@matched": lambda s: s["experiment"] == "atlas_matched" and len(s["contexts"]) == 1,
               "2_context@matched": lambda s: s["experiment"] == "atlas_matched" and len(s["contexts"]) == 2,
               "3_context@matched": lambda s: s["experiment"] == "atlas_matched" and len(s["contexts"]) == 3,
               "K562_in_atlas": lambda s: s["experiment"] == "atlas" and "K562" in s["contexts"] and len(s["contexts"]) < 3,
               "K562_not_in_atlas": lambda s: s["experiment"] == "atlas" and "K562" not in s["contexts"],
               **{s: (lambda s2, n=s: s2["name"] == n) for s in ("STRING_only", "MAPKG_only", "rank64", "rank128", "rank512")}}
    for label, chosen in classes.items():
        for regime in ("SEEN", "MASKED"):
            values, labels, cvalues, clabels = [], [], [], []
            for held in xc.PRIMARY_CONTEXTS:
                names = [s["name"] for s in xc.specs(held) if chosen(s)]
                if not names:
                    continue
                rows = per[held][regime]
                values.append(np.mean([rows[n]["fused"]["BU@20"] for n in names], axis=0) - rows["BASE"]["BU@20"])
                labels.append(per[held]["labels"])
                cvalues.append(np.mean([cand[held][regime][n]["specific"] for n in names], axis=0))
                clabels.append(cand[held]["labels"])
            result["pooled_classes"][f"{label}|{regime}|fused|BU@20_minus_BASE"] = _pooled(values, labels)
            result["pooled_classes"][f"{label}|{regime}|candidate_specific_cosine_mean"] = float(np.mean(np.concatenate(cvalues)))
    for name in ("STRING_only", "MAPKG_only", "rank64", "rank128", "rank512"):
        for regime in ("SEEN", "MASKED"):
            result["pooled_classes"][f"{name}|{regime}|fused|BU@20_minus_A1"] = _pooled(
                [per[h][regime][name]["fused"]["BU@20"] - per[h][regime][xc.FULL]["fused"]["BU@20"] for h in xc.PRIMARY_CONTEXTS],
                [per[h]["labels"] for h in xc.PRIMARY_CONTEXTS])
    rho = spearmanr([p[2] for p in points], [p[3] for p in points])
    result["prediction_vs_design"] = {"points": [{"held": p[0], "spec": p[1], "masked_specific_cosine_median": p[2],
                                                  "masked_fused_BU20_minus_BASE_median": p[3]} for p in points],
                                      "spearman": float(rho.correlation), "p_value": float(rho.pvalue), "n": len(points)}
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=xc.MISSION)
    parser.add_argument("--stage", required=True, choices=("fit", "evaluate"))
    parser.add_argument("--held", default=None)
    parser.add_argument("--pool", default="SEEN")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--sped-run", dest="sped_run", default=None)
    args = parser.parse_args()
    run = Path(args.run_dir)
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    if args.stage == "fit":
        output = run / "fits" / args.held / args.pool
        output.mkdir(parents=True, exist_ok=True)
        target = output / "record.json"
        _refuse(target)
        record = fit_stage(packs, args.held, args.pool, output)
    else:
        target = run / "evidence_v1.json"
        _refuse(target)
        record = evaluate(packs, run, Path(args.sped_run))
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
