#!/usr/bin/env python3
"""Phase 10: is there a context residual worth modelling, before any V2 exists?

Decomposition, on the candidates every context measured:

    r_{c,x} = r_bar_c + delta r_{c,x}

Two choices are fixed here, before the first number.

Scale.  The primary decomposition is on per-context unit-normalized responses.  The
deployment score is a cosine, so a candidate's magnitude carries no decision value,
and a raw decomposition would book a context that simply responds harder as
"context-specific" when its direction is identical.  The raw version is reported
alongside as a sensitivity, not as the headline.

Reliability.  A residual computed from one measurement is partly noise, and noise
always looks context-specific because each context was measured separately.  The
residual is therefore recomputed independently in each of the two batch-disjoint
views and the two are compared.  The comparison is calibrated twice: against the
reliability of the full response in the same candidates, which is the ceiling any
residual could reach, and against a control in which context labels are permuted
within a candidate, which is what a pure-noise residual scores.

The oracle decomposition is the number that actually bounds V2.  It asks what a
model that knew each candidate's true context-specific response could buy over one
that knew only the true context-blind consensus.  If that gap is near zero, no
context conditioning can help, however it is parameterised.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contract as fc
from . import evaluate as ev
from . import harness as hn


PERMUTATION_SEED = 20260918
PERMUTATIONS = 20


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def stack_common(packs: dict, identities: np.ndarray) -> dict:
    """``[context, candidate, gene]`` responses for the candidates measured everywhere."""
    order = np.asarray(sorted(identities.tolist()), dtype=str)
    views, consensus = [], []
    for view in (0, 1):
        block = []
        for name in fc.CONTEXTS:
            pack = packs[name]
            position = {value: index for index, value in enumerate(pack.ids.tolist())}
            rows = np.asarray([position[value] for value in order.tolist()], dtype=np.int64)
            block.append(np.asarray(pack.responses[view][rows], dtype=np.float64))
        views.append(np.stack(block))
    for name_index in range(len(fc.CONTEXTS)):
        consensus.append(0.5 * (views[0][name_index] + views[1][name_index]))
    return {"identities": order, "by_view": views, "consensus": np.stack(consensus)}


def decompose(responses: np.ndarray, normalize: bool) -> dict:
    """Split ``[context, candidate, gene]`` into a shared part and a context residual."""
    values = _unit(responses) if normalize else np.asarray(responses, dtype=np.float64)
    shared = values.mean(axis=0)
    residual = values - shared[None]
    total = float((values ** 2).sum())
    return {"values": values, "shared": shared, "residual": residual,
            "total_energy": total,
            "shared_energy_share": float((shared ** 2).sum() * values.shape[0] / max(total, 1e-30)),
            "residual_variance_share": float((residual ** 2).sum() / max(total, 1e-30))}


def across_context_cosine(values: np.ndarray) -> dict:
    """Pairwise cosine between the same candidate's response in two contexts."""
    unit = _unit(values)
    record = {}
    for left in range(len(fc.CONTEXTS)):
        for right in range(left + 1, len(fc.CONTEXTS)):
            cosine = np.einsum("ij,ij->i", unit[left], unit[right])
            record[f"{fc.CONTEXTS[left]}|{fc.CONTEXTS[right]}"] = {
                "median": float(np.median(cosine)), "mean": float(cosine.mean()),
                "p05": float(np.quantile(cosine, 0.05)), "p95": float(np.quantile(cosine, 0.95))}
    return record


def residual_reliability(by_view: list, normalize: bool) -> dict:
    """Is the residual the same object in two independent measurements of it?"""
    left = decompose(by_view[0], normalize)
    right = decompose(by_view[1], normalize)
    residual_cosine = np.einsum("cij,cij->ci", _unit(left["residual"]), _unit(right["residual"]))
    response_cosine = np.einsum("cij,cij->ci", _unit(left["values"]), _unit(right["values"]))

    generator = np.random.default_rng(PERMUTATION_SEED)
    contexts, candidates = residual_cosine.shape
    control = []
    for _ in range(PERMUTATIONS):
        shuffled = np.empty_like(right["residual"])
        for candidate in range(candidates):
            shuffled[:, candidate] = right["residual"][generator.permutation(contexts), candidate]
        control.append(np.einsum("cij,cij->ci", _unit(left["residual"]), _unit(shuffled)))
    control = np.stack(control)

    return {
        "residual_cross_view_cosine": {
            "median": float(np.median(residual_cosine)), "mean": float(residual_cosine.mean()),
            "fraction_positive": float((residual_cosine > 0).mean()),
            "per_context_median": {name: float(np.median(residual_cosine[index]))
                                   for index, name in enumerate(fc.CONTEXTS)}},
        "response_cross_view_cosine_ceiling": {
            "median": float(np.median(response_cosine)), "mean": float(response_cosine.mean())},
        "context_permuted_control": {
            "median": float(np.median(control)), "mean": float(control.mean()),
            "p95_over_permutations": float(np.quantile(control.mean(axis=(1, 2)), 0.95))},
        "resolved_above_control": bool(float(np.median(residual_cosine))
                                       > float(np.quantile(control.mean(axis=(1, 2)), 0.95))),
    }


def oracle_decomposition(packs: dict, stack: dict) -> dict:
    """What perfect context knowledge buys over perfect context-blind knowledge.

    Both oracles are cross-fitted: the candidate response that builds the score comes
    from the view opposite the one grading it, so neither oracle is scored against the
    measurement that produced it.  This is the ceiling on any context-conditioned
    model, whatever its architecture.
    """
    identities = stack["identities"]
    record = {}
    for index, name in enumerate(fc.CONTEXTS):
        pack = packs[name]
        position = {value: row for row, value in enumerate(pack.ids.tolist())}
        columns = np.asarray([position[value] for value in identities.tolist()], dtype=np.int64)
        mask = np.zeros(pack.ids.size, dtype=bool)
        mask[columns] = True
        # Four oracles, in two pairs.
        #
        # The plain pair takes the candidate vector from view ``1-v``, which is the view
        # the graded utility is also built from.  That makes it a true ceiling, but a
        # candidate whose noise in that view happens to point at the query's own noisy
        # direction is rewarded twice, and the blind arm dilutes that noise by averaging
        # four contexts while the matched arm carries it in full.  Part of the plain gap
        # is therefore shared measurement noise rather than context biology.
        #
        # The cross-fitted pair takes the candidate vector from view ``v`` instead, which
        # is independent of the view that grades it.  It has no shared noise on either
        # side and is the honest headroom; the plain pair is an upper bound.
        arms: dict = {"BASE": pack.base}
        specification = {
            "ORACLE_CONTEXT_BLIND": (1, False),
            "ORACLE_CONTEXT_MATCHED": (1, True),
            "ORACLE_CONTEXT_BLIND_CROSSFIT": (0, False),
            "ORACLE_CONTEXT_MATCHED_CROSSFIT": (0, True),
        }
        for label, (opposite, matched) in specification.items():
            per_view = []
            for view in (0, 1):
                effect = np.zeros((pack.ids.size, pack.responses.shape[2]), dtype=np.float64)
                which = (1 - view) if opposite else view
                source = (stack["by_view"][which][index] if matched
                          else _unit(stack["by_view"][which]).mean(axis=0))
                effect[columns] = source
                per_view.append(ev.fuse(pack.base[view],
                                        ev.effect_cosine(pack.query_responses, effect, view)))
            arms[label] = per_view
        block = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions, mask)
        if "skipped" in block:
            record[name] = block
            continue
        vectors = block.pop("_vectors")
        record[name] = {
            "candidates": block["candidates"], "queries": block["queries"],
            "arms": block["arms"],
            "context_matched_minus_blind": ev.paired(vectors["ORACLE_CONTEXT_MATCHED"]["MBRU"],
                                                     vectors["ORACLE_CONTEXT_BLIND"]["MBRU"]),
            "context_matched_minus_blind_crossfit": ev.paired(
                vectors["ORACLE_CONTEXT_MATCHED_CROSSFIT"]["MBRU"],
                vectors["ORACLE_CONTEXT_BLIND_CROSSFIT"]["MBRU"]),
            "blind_minus_base": ev.paired(vectors["ORACLE_CONTEXT_BLIND"]["MBRU"],
                                          vectors["BASE"]["MBRU"]),
            "blind_crossfit_minus_base": ev.paired(
                vectors["ORACLE_CONTEXT_BLIND_CROSSFIT"]["MBRU"], vectors["BASE"]["MBRU"]),
            "headroom_note": ("the cross-fitted pair is the honest headroom; the plain pair "
                              "shares candidate-side measurement noise with the grader and is "
                              "an upper bound")}
    return record


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    identities = np.asarray(json.loads(Path(args.common4).read_text())["identities"], dtype=str)
    stack = stack_common(packs, identities)

    record: dict = {"schema": "VCDESIGN_FOUR_CONTEXT_V1_RESIDUAL",
                    "common4_candidates": int(identities.size),
                    "primary_scale": "per-context unit-normalized, because the deployment score is a cosine",
                    "gate": fc.RESIDUAL_GATE_NOTE, "read_anchor_G_check": False}
    for label, normalize in (("unit_normalized_primary", True), ("raw_sensitivity", False)):
        split = decompose(stack["consensus"], normalize)
        record[label] = {
            "shared_energy_share": split["shared_energy_share"],
            "residual_variance_share": split["residual_variance_share"],
            "across_context_cosine": across_context_cosine(stack["consensus"]) if normalize else None,
            "reliability": residual_reliability(stack["by_view"], normalize)}
    record["oracle_decomposition"] = oracle_decomposition(packs, stack)

    primary = record["unit_normalized_primary"]
    record["verdict"] = {
        "residual_variance_share": primary["residual_variance_share"],
        "residual_is_measurement_resolved": primary["reliability"]["resolved_above_control"],
        "build_v2": bool(primary["reliability"]["resolved_above_control"]
                         and primary["residual_variance_share"] > 0.0),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 10: context residual headroom")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--common4", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    primary = record["unit_normalized_primary"]
    print(f"COMMON4 candidates {record['common4_candidates']}")
    print(f"residual variance share {primary['residual_variance_share']:.4f}, "
          f"shared {primary['shared_energy_share']:.4f}")
    reliability = primary["reliability"]
    print(f"residual cross-view cosine {reliability['residual_cross_view_cosine']['median']:+.4f} "
          f"vs response ceiling {reliability['response_cross_view_cosine_ceiling']['median']:+.4f} "
          f"vs permuted control {reliability['context_permuted_control']['median']:+.4f}")
    for name, entry in record["oracle_decomposition"].items():
        if "context_matched_minus_blind" in entry:
            item = entry["context_matched_minus_blind"]
            cross = entry["context_matched_minus_blind_crossfit"]
            low, high = cross["bootstrap"]["median"]["ci95"]
            print(f"  {name:8s} matched minus blind: upper bound {item['median']:+.4f}, "
                  f"cross-fitted {cross['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {cross['fraction_query_improved']:.3f}")
    print(f"BUILD_V2 = {record['verdict']['build_v2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
