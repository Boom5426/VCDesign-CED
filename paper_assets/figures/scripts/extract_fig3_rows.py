#!/usr/bin/env python3
"""Extract canonical BU@20 source data for Figure 3.

The published closure stores canonical SEEN and MASKED score matrices, but it does
not store the unit-target size-control arm needed by the controlled-masking panel.
This script therefore:

1. regrades the frozen closure score matrices for the solver-regime panel;
2. rebuilds the five already-specified unit-target masking fits and evaluates the
   size-control arm directly, never by subtracting summary medians; and
3. recomputes the nested mixed-vocabulary series from the same unit-target effects.

No hyperparameter, split, mask, candidate pool, query set, or evaluation rule is
changed.  Run on the compute host and stream the deterministic JSON to stdout::

    python3 - --packs PACKS --dced-run DCED_RUN --closure-run CLOSURE_RUN \
        --output - < figures/scripts/extract_fig3_rows.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from gene_open_inverse.decision_consistent_effect_v1 import arms as dca
from gene_open_inverse.decision_consistent_effect_v1 import stats as st
from gene_open_inverse.external_baselines_v1 import contract as cc
from gene_open_inverse.external_baselines_v1 import scores as closure_scores
from gene_open_inverse.four_context_v1 import contract as fc
from gene_open_inverse.four_context_v1 import evaluate as ev
from gene_open_inverse.four_context_v1 import harness as hn
from gene_open_inverse.four_context_v1 import predictors as pd
from gene_open_inverse.open_vocab_generalization_v1 import masking as mk
from gene_open_inverse.open_vocab_generalization_v1.mixture import share_value
from gene_open_inverse.ppm_v1.contract_readout import _rows


HELD = ("RPE1", "HepG2", "Jurkat")
STATES = ("full_measured", "size_controlled", "response_unseen")
SHARES = (0.0, 0.25, 0.50, 0.75, 1.0)
METHODS = (
    cc.INTERNAL_FTM,
    cc.BASE,
    cc.VCDESIGN,
    cc.CELLNAVI_NATIVE,
    cc.PROFILED,
)
METHOD_LABELS = {
    cc.INTERNAL_FTM: "Forward-match",
    cc.BASE: "Base",
    cc.VCDESIGN: "VCDesign-CED",
    cc.CELLNAVI_NATIVE: "CellNavi",
    cc.PROFILED: "Profile retrieval",
}
ATOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(label: str, observed: float, expected: float, atol: float = ATOL) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise RuntimeError(
            f"{label}: reproduced {observed:.17g}, expected {expected:.17g}"
        )


def source(path: Path, kind: str) -> dict:
    return {"kind": kind, "sha256": sha256(path)}


def labels(pack) -> np.ndarray:
    queries = np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)]
    return np.concatenate([queries, queries])


def state_summary(values: np.ndarray, cluster_labels: np.ndarray) -> dict:
    block = st.cluster_bootstrap_labels(values, cluster_labels)
    return {
        "median": float(block["median"]),
        "ci95": [float(v) for v in block["median_ci95"]],
        "clusters": int(block["clusters"]),
        "query_view_rows": int(block["entries"]),
    }


def fit_unit_masking_effects(packs: dict, held: str, pack, eligible: np.ndarray,
                             folds: np.ndarray, reference_cross: np.ndarray) -> tuple[list, dict]:
    """Rebuild the five frozen unit-target fold fits and verify their cross-fit."""
    training = tuple(name for name in fc.CONTEXTS if name != held)
    effects, fits = [], {}
    for fold in range(cc.MASK_FOLDS):
        inside = eligible & (folds == fold)
        pool = mk.masked_training_pool(packs, training, np.asarray(pack.ids)[inside])
        unit_pool = dca.unit_target_pool(pool)
        model = pd.fit(
            unit_pool["features"], unit_pool["responses"], unit_pool["present"],
            unit_pool["keys"], training,
        )
        effect = model.effect(pack.features, pack.present)
        effects.append(effect)
        error = float(np.abs(effect[inside] - reference_cross[inside]).max())
        if error > 1e-10:
            raise RuntimeError(f"{held} fold {fold}: unit-target reproduction error {error}")
        fits[str(fold)] = {
            "masked_candidates": int(inside.sum()),
            "training_rows": int(model.rows),
            "training_identities": int(model.identities),
            "penalty": float(model.penalty),
            "held_out_gene_space_cosine": float(model.held_out_cosine),
            "reference_max_abs_error": error,
        }
        print(f"{held}: rebuilt unit-target fold {fold}", file=sys.stderr, flush=True)
    rebuilt_cross = mk._crossfit(reference_cross, effects, eligible, folds, offset=0)
    # The initial array is overwritten on every eligible row, so only those rows matter.
    error = float(np.abs(rebuilt_cross[eligible] - reference_cross[eligible]).max())
    if error > 1e-10:
        raise RuntimeError(f"{held}: rebuilt unit-target cross-fit error {error}")
    return effects, {"folds": fits, "crossfit_max_abs_error": error}


def evaluate(args: argparse.Namespace) -> dict:
    packs_dir = Path(args.packs)
    dced_run = Path(args.dced_run)
    closure_run = Path(args.closure_run)
    closure_result_path = closure_run / "external_baselines_v1.json"
    closure_result = json.loads(closure_result_path.read_text())
    packs = {name: hn.load_pack(name, packs_dir) for name in fc.CONTEXTS}

    panel_a = {
        "metric": "BU@20",
        "level_statistic": "median over query-view rows pooled across held contexts",
        "methods": {},
        "headline_contrasts": {},
    }
    panel_b_contexts, panel_c_contexts = {}, {}
    pooled_gains = {state: [] for state in STATES}
    pooled_labels = []
    row_source = {}
    fit_audit = {}
    source_files = {
        "closure_result": source(closure_result_path, "external_baselines_v1.json"),
    }

    for held in HELD:
        pack = packs[held]
        score_path = closure_run / "scores" / f"{held}.npz"
        effect_path = dced_run / f"effects_{held}.npz"
        source_files[f"closure_scores_{held}"] = source(score_path, "closure score matrix")
        source_files[f"dced_effects_{held}"] = source(effect_path, "RIDGE_UNIT effects")
        score_store = closure_scores.load(score_path)
        saved = np.load(effect_path, allow_pickle=False)
        eligible = np.asarray(saved["eligible"], dtype=bool)
        folds = np.asarray(saved["folds"], dtype=np.int64)
        columns = np.flatnonzero(eligible)
        query_labels = labels(pack)
        pooled_labels.append(query_labels)

        rows = {"SEEN": {}, "MASKED": {}}
        for regime in ("SEEN", "MASKED"):
            for method in METHODS:
                if method not in score_store["scores"].get(regime, {}):
                    continue
                rows[regime][method] = _rows(
                    pack, score_store["scores"][regime][method], columns
                )["BU@20"]
                expected = closure_result["main_table"][regime][method][held]["BU@20"]["median"]
                check_close(
                    f"{held} {regime} {method} BU@20",
                    float(np.median(rows[regime][method])), float(expected),
                )

        seen_gain = rows["SEEN"][cc.VCDESIGN] - rows["SEEN"][cc.BASE]
        masked_gain = rows["MASKED"][cc.VCDESIGN] - rows["MASKED"][cc.BASE]
        expected_seen = closure_result["contrasts"]["SEEN"][
            f"{cc.VCDESIGN}_minus_{cc.BASE}"
        ]["by_context"][held]["median"]
        expected_masked = closure_result["contrasts"]["MASKED"][
            f"{cc.VCDESIGN}_minus_{cc.BASE}"
        ]["by_context"][held]["median"]
        check_close(f"{held} SEEN VCDesign-BASE", float(np.median(seen_gain)), expected_seen)
        check_close(f"{held} MASKED VCDesign-BASE", float(np.median(masked_gain)), expected_masked)

        fold_effects, held_fit_audit = fit_unit_masking_effects(
            packs, held, pack, eligible, folds, saved["unit_cross"].astype(np.float64)
        )
        size_effect = mk._crossfit(
            saved["unit_seen"].astype(np.float64), fold_effects, eligible, folds, offset=1
        )
        size_cosine = dca.cosine_scores(pack, size_effect)
        size_fused = [ev.fuse(pack.base[v], size_cosine[v]) for v in (0, 1)]
        size_rows = _rows(pack, size_fused, columns)["BU@20"]
        base_rows = _rows(pack, pack.base, columns)["BU@20"]
        size_gain = size_rows - base_rows

        gains = {
            "full_measured": seen_gain,
            "size_controlled": size_gain,
            "response_unseen": masked_gain,
        }
        panel_b_contexts[held] = {
            "candidates": int(columns.size),
            "query_identities": int(np.asarray(pack.query_rows).size),
            "states": {
                state: state_summary(values, query_labels)
                for state, values in gains.items()
            },
        }
        for state, values in gains.items():
            pooled_gains[state].append(values)

        # Same unit-target estimator, same fixed pool, same queries, and the original nested
        # identity hash.  Only the candidate columns taking the cross-fitted effect change.
        ids = np.asarray(pack.ids, dtype=str)
        hash_values = np.zeros(ids.size, dtype=np.float64)
        hash_values[columns] = share_value(ids[columns])
        seen_cosine = dca.cosine_scores(pack, saved["unit_seen"].astype(np.float64))
        masked_cosine = dca.cosine_scores(pack, saved["unit_cross"].astype(np.float64))
        points, previous_unseen = [], np.zeros(ids.size, dtype=bool)
        for share in SHARES:
            unseen = eligible & (hash_values < share)
            if np.any(previous_unseen & ~unseen):
                raise RuntimeError(f"{held}: mixed-candidate assignment is not nested at {share}")
            mixed = [
                np.where(unseen[None, :], masked_cosine[v], seen_cosine[v])
                for v in (0, 1)
            ]
            fused = [ev.fuse(pack.base[v], mixed[v]) for v in (0, 1)]
            mixed_rows = _rows(pack, fused, columns)["BU@20"]
            gain = mixed_rows - base_rows
            points.append({
                "share": float(share),
                "realised_share": float(unseen[columns].mean()),
                "median": float(np.median(gain)),
            })
            previous_unseen = unseen
        check_close(f"{held} mixed share 0", points[0]["median"], float(np.median(seen_gain)))
        check_close(f"{held} mixed share 1", points[-1]["median"], float(np.median(masked_gain)))
        panel_c_contexts[held] = {"points": points}

        query_ids = np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)]
        row_source[held] = {
            "query": np.concatenate([query_ids, query_ids]).astype(str).tolist(),
            "view": ([0] * len(query_ids)) + ([1] * len(query_ids)),
            "gain_BU20_over_BASE": {
                state: [float(value) for value in gains[state]] for state in STATES
            },
        }
        fit_audit[held] = held_fit_audit

    all_labels = np.concatenate(pooled_labels)
    pooled = {
        state: state_summary(np.concatenate(pooled_gains[state]), all_labels)
        for state in STATES
    }
    expected = closure_result["contrasts"]["MASKED"][f"{cc.VCDESIGN}_minus_{cc.BASE}"]
    check_close("pooled MASKED contrast", pooled["response_unseen"]["median"], expected["median"])
    for got, want in zip(pooled["response_unseen"]["ci95"], expected["median_ci95"]):
        check_close("pooled MASKED contrast CI", got, want)

    for method in METHODS:
        block = {}
        for regime in ("MASKED", "SEEN"):
            if method not in closure_result["main_table"][regime]:
                continue
            entry = closure_result["main_table"][regime][method]
            if isinstance(entry, str):
                continue
            block[regime] = float(entry["pooled"]["BU@20"]["median"])
        panel_a["methods"][method] = {"label": METHOD_LABELS[method], "levels": block}

    masked_contrast = closure_result["contrasts"]["MASKED"][
        f"{cc.VCDESIGN}_minus_{cc.BASE}"
    ]
    retrieval_contrast = closure_result["contrasts"]["SEEN"][
        f"{cc.VCDESIGN}_minus_{cc.PROFILED}"
    ]
    panel_a["headline_contrasts"] = {
        "MASKED|VCDesign-CED_minus_Base": {
            "median": float(masked_contrast["median"]),
            "ci95": [float(v) for v in masked_contrast["median_ci95"]],
        },
        "SEEN|Profile_retrieval_minus_VCDesign-CED": {
            "median": float(-retrieval_contrast["median"]),
            "ci95": [float(-retrieval_contrast["median_ci95"][1]),
                     float(-retrieval_contrast["median_ci95"][0])],
        },
    }

    if panel_a["headline_contrasts"]["MASKED|VCDesign-CED_minus_Base"]["median"] <= 0:
        raise RuntimeError("mandatory assertion failed: MASKED VCDesign-CED <= Base")
    if panel_a["headline_contrasts"]["SEEN|Profile_retrieval_minus_VCDesign-CED"]["median"] <= 0:
        raise RuntimeError("mandatory assertion failed: SEEN profile retrieval <= VCDesign-CED")

    return {
        "schema": "VCDESIGN_FIG3_ROWS_V2",
        "generated_by": "figures/scripts/extract_fig3_rows.py",
        "estimator": {
            "id": "RIDGE_UNIT",
            "training_target": "unit consensus response direction",
            "score": "cosine to predicted unit-target effect",
            "fusion": "row_z(BASE) + row_z(effect), weight 1.0",
            "metric": "BU@20",
            "evaluation_contract": "ICLR_EXTERNAL_BASELINE_AND_EVALUATION_CLOSURE_V1",
        },
        "sources": source_files,
        "panel_a": panel_a,
        "panel_b": {
            "metric": "paired per-row BU@20 gain over identical BASE",
            "arm_definition": {
                "full_measured": "full three-context RIDGE_UNIT atlas",
                "size_controlled": (
                    "candidate fold f scored by the predictor masking fold (f + 1) mod 5"
                ),
                "response_unseen": "candidate fold f scored by the predictor masking fold f",
            },
            "contexts": panel_b_contexts,
            "pooled": pooled,
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260916,
                "unit": "query gene across views and contexts",
            },
        },
        "panel_c": {
            "metric": "median paired per-row BU@20 gain over identical BASE",
            "pool_definition": (
                "fixed eligible pool and queries; nested candidate assignment changes only "
                "whether each candidate uses the SEEN or MASKED RIDGE_UNIT effect"
            ),
            "shares": [float(v) for v in SHARES],
            "contexts": panel_c_contexts,
        },
        "row_source": row_source,
        "fit_reproduction_audit": fit_audit,
        "mandatory_assertions": {
            "masked_vcdesign_above_base": True,
            "seen_profile_retrieval_above_vcdesign": True,
            "size_controlled_from_direct_scores": True,
            "mixed_assignment_nested": True,
            "base_scorer_identical": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packs", required=True)
    parser.add_argument("--dced-run", required=True)
    parser.add_argument("--closure-run", required=True)
    parser.add_argument("--output", required=True, help="explicit output path, or '-' for stdout")
    args = parser.parse_args()
    data = evaluate(args)
    payload = json.dumps(data, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        output = Path(args.output)
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
