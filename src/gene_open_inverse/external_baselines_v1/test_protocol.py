#!/usr/bin/env python3
"""Data-free tests.  Each guards a failure that would otherwise look like a valid external comparison."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_open_inverse.action_conditioned_utility_v1 import contract as act   # noqa: E402
from gene_open_inverse.external_baselines_v1 import cells                     # noqa: E402
from gene_open_inverse.external_baselines_v1 import contract as cc            # noqa: E402
from gene_open_inverse.external_baselines_v1 import evaluate as evl           # noqa: E402
from gene_open_inverse.external_baselines_v1 import gears_fit as gf           # noqa: E402
from gene_open_inverse.external_baselines_v1 import phr                       # noqa: E402
from gene_open_inverse.external_baselines_v1 import scores as sc              # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def _four_context(tmp: Path, held: str, n: int = 12, g: int = 7, seed: int = 0):
    """A synthetic build directory whose views are exact count-weighted quarter means."""
    rng = np.random.default_rng(seed)
    genes = np.asarray([f"G{i}" for i in range(g)])
    ids = np.asarray([f"I{i}" for i in range(n)])
    batches = 8
    quarter_of_batch = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8)
    incidence = rng.integers(25, 60, size=(n, batches))
    quarters = rng.standard_normal((4, n, g)).astype(np.float32)
    counts = np.stack([incidence[:, quarter_of_batch == q].sum(1) for q in range(4)], 1).astype(np.float64)
    side = np.stack([(counts[:, 0, None] * quarters[0] + counts[:, 1, None] * quarters[1]) / (counts[:, 0] + counts[:, 1])[:, None],
                     (counts[:, 2, None] * quarters[2] + counts[:, 3, None] * quarters[3]) / (counts[:, 2] + counts[:, 3])[:, None]]).astype(np.float32)
    build = tmp / held
    build.mkdir(parents=True)
    (tmp / "genes").mkdir(exist_ok=True)
    np.save(tmp / "genes" / "G_4.npy", genes[::-1])            # a reordered axis, as in the real packs
    np.save(build / "identity_axis.npy", ids)
    np.save(build / "gene_axis.npy", genes)
    np.save(build / "four_way_eligible.npy", np.arange(n) % 4 != 3)
    np.save(build / "incidence.npy", incidence)
    np.save(build / "batch_quarter.npy", quarter_of_batch)
    np.save(build / "batch_side.npy", quarter_of_batch // 2)
    np.save(build / "quarter_responses.npy", quarters)
    np.save(build / "side_responses.npy", side)
    pool_rows = np.arange(n)
    pack = SimpleNamespace(ids=ids, pool_rows=pool_rows, self_positions=np.asarray([0, 1, 3, 5]),
                           responses=side[:, pool_rows][:, :, ::-1])
    return pack, quarters


def test_01_quarter_gate_passes_exact_and_fires_on_a_corrupted_quarter():
    with tempfile.TemporaryDirectory() as tmp:
        pack, quarters = _four_context(Path(tmp), "RPE1")
        gate = phr.reconstruction_gate(tmp, "RPE1", pack)
        check("01_gate_passes", gate["passes"], str(gate))
        corrupt = quarters.copy()
        corrupt[1, 0, 2] += 0.01
        np.save(Path(tmp) / "RPE1" / "quarter_responses.npy", corrupt)
        check("01_gate_fires", not phr.reconstruction_gate(tmp, "RPE1", pack)["passes"])


def test_02_realized_j_reads_only_the_other_side_and_uses_the_proposer_action():
    with tempfile.TemporaryDirectory() as tmp:
        pack, quarters = _four_context(Path(tmp), "RPE1")
        stratum = np.ones(pack.ids.size, bool)
        stratum[2] = False
        out = phr.realized_j(tmp, "RPE1", pack, stratum)
        fw = np.arange(pack.ids.size) % 4 != 3
        check("02_queries_are_four_way", np.array_equal(out["query_rows"], np.flatnonzero(fw[pack.self_positions])))
        check("02_candidates_are_four_way_stratum", np.array_equal(out["candidates"], np.flatnonzero(fw & stratum)))
        g4 = np.arange(quarters.shape[2])[::-1]
        members = np.flatnonzero(fw)
        q = [quarters[k][members][:, g4].astype(np.float64) for k in range(4)]
        local = {int(p): i for i, p in enumerate(members)}
        qi = [local[int(pack.self_positions[r])] for r in out["query_rows"]]
        ci = [local[int(c)] for c in out["candidates"]]
        for view in (0, 1):
            action, _ = act.optimal_attenuation(act.view_statistics(q[2 * view], q[2 * view + 1]))
            manual = act.view_statistics(q[2 - 2 * view], q[3 - 2 * view]).value_at(action)[np.ix_(qi, ci)]
            check(f"02_view{view}_matches_manual", np.allclose(out["J"][view], manual, atol=1e-12))
        # perturbing the grading side of view 0 (Q2, Q3) must not change view 0's action
        changed = quarters.copy()
        changed[2] *= 3.0
        np.save(Path(tmp) / "RPE1" / "quarter_responses.npy", changed)
        again = phr.realized_j(tmp, "RPE1", pack, stratum)
        q2 = [changed[k][members][:, g4].astype(np.float64) for k in range(4)]
        a0, _ = act.optimal_attenuation(act.view_statistics(q[0], q[1]))
        check("02_grader_does_not_move_action",
              np.allclose(again["J"][0], act.view_statistics(q2[2], q2[3]).value_at(a0)[np.ix_(qi, ci)], atol=1e-12))


def test_03_gears_split_never_contains_a_masked_identity():
    kept = [f"G{i}" for i in range(400)]
    excluded = kept[::9]
    split, train, val = gf.make_split(kept, excluded)
    names = {c.split("+")[0] for part in split.values() for c in part}
    check("03_no_masked", not (names & set(excluded)))
    check("03_ctrl_trains", "ctrl" in split["train"])
    check("03_val_is_hash_bucket", all(gf._bucket(g, cc.GEARS_VAL_NAMESPACE, cc.GEARS_VAL_MODULUS) == 0 for g in val))
    check("03_partition", sorted(train + val) == sorted(set(kept) - set(excluded)))
    split2, _, val2 = gf.make_split(kept[::-1], excluded)
    check("03_order_invariant", sorted(val) == sorted(val2))


def _pack(ids, genes=5, q=3, seed=1):
    rng = np.random.default_rng(seed)
    n = len(ids)
    return SimpleNamespace(ids=np.asarray(ids), self_positions=np.arange(q),
                           responses=rng.standard_normal((2, n, genes)),
                           query_responses=rng.standard_normal((2, q, genes)),
                           base=[rng.standard_normal((q, n)) for _ in (0, 1)], fit_mask=np.ones(n, bool))


def test_04_cellnavi_columns_map_by_name_and_uncovered_is_never_scored():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "cellnavi" / "RPE1").mkdir(parents=True)
        pack = _pack(["A", "B", "C", "D"])
        classes = np.asarray(["D", "B", "A", "Z"])            # C is not a class
        queries = np.asarray(["A", "B", "C", "A", "B", "C"])
        views = np.asarray([0, 0, 0, 1, 1, 1])
        logp = np.log(np.random.default_rng(3).dirichlet(np.ones(4), size=6))
        np.savez(run / "cellnavi" / "RPE1" / "scores.npz", queries=queries, views=views, classes=classes,
                 mean_log_prob=logp, cells=np.full(6, 16), cell_top1=np.zeros(3, int),
                 cell_query=np.asarray(["A", "B", "C"]), cell_view=np.zeros(3, int))
        out = sc.cellnavi_scores(run, "RPE1", pack)
        check("04_A_is_class_2", np.allclose(out["by_view"][1][:, 0], logp[3:, 2]))
        check("04_D_is_class_0", np.allclose(out["by_view"][0][:, 3], logp[:3, 0]))
        check("04_C_uncovered", not out["covered"][2] and np.all(out["by_view"][0][:, 2] == sc.NOT_SCORED))
        rank = out["identification"][0]["rank_of_own_class"]
        check("04_identification_rank", rank[0] == 1 + int((logp[0] > logp[0, 2]).sum()) and rank[2] == -1)


def test_05_profiles_exclude_the_held_context_and_respect_fit_masks():
    k562 = _pack(["A", "B", "C"], seed=5)
    k562.fit_mask = np.asarray([True, False, True])          # B is not G_fit
    other = _pack(["B", "C"], seed=6)
    held = _pack(["A", "B", "C", "E"], seed=7)
    sig, covered = sc.profiles({"K562": k562, "HepG2": other, "RPE1": held}, "RPE1", held)
    unit = lambda x: x / np.linalg.norm(x)
    cons = lambda p, i: unit(0.5 * (p.responses[0, i] + p.responses[1, i]))
    check("05_A_from_K562_only", np.allclose(sig[0], cons(k562, 0)))
    check("05_B_not_from_K562", np.allclose(sig[1], cons(other, 0)))
    check("05_C_mean_of_two", np.allclose(sig[2], unit(cons(k562, 2) + cons(other, 1))))
    check("05_E_uncovered_and_held_ignored", not covered[3] and np.all(sig[3] == 0))


def test_06_gears_crossfit_takes_the_fold_fit_and_uncovered_scores_zero():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        genes = np.asarray(["g0", "g1", "g2"])
        ids = np.asarray(["A", "B", "C", "D"])
        eligible = np.asarray([True, True, True, False])
        folds = np.asarray([0, 1, 0, 2])
        for name, value in [("SEEN", 1.0)] + [(f"FOLD_{f}", 10.0 + f) for f in range(5)]:
            (run / "gears" / name).mkdir(parents=True)
            extra = np.asarray(["OTHER_POOL_GENE"])                  # a fold union is a superset
            excluded = (np.concatenate([ids[eligible & (folds == int(name[-1]))], extra]) if name != "SEEN"
                        else np.asarray([], str))
            np.savez(run / "gears" / name / "effects.npz", identities=np.asarray(["A", "B", "D"]),
                     delta=np.full((3, 3), value), genes=genes, excluded=excluded,
                     training_identities=np.asarray([i for i in ["A", "B", "D"] if i not in set(excluded)]))
        out = sc.gears_effects(run, "RPE1", SimpleNamespace(ids=ids), eligible, folds, genes)
        check("06_masked_A_from_fold0", np.allclose(out["MASKED"][0], 10.0))
        check("06_masked_B_from_fold1", np.allclose(out["MASKED"][1], 11.0))
        check("06_ineligible_keeps_seen", np.allclose(out["MASKED"][3], 1.0))
        check("06_uncovered_zero", not out["covered"][2] and not out["MASKED"][2].any())
        # a fit that failed to exclude a masked identity must be refused
        bad = np.load(run / "gears" / "FOLD_0" / "effects.npz")
        np.savez(run / "gears" / "FOLD_0" / "effects.npz", **{k: bad[k] for k in bad.files if k != "excluded"},
                 excluded=np.asarray(["C"]))
        try:
            sc.gears_effects(run, "RPE1", SimpleNamespace(ids=ids), eligible, folds, genes)
            check("06_firewall_refuses", False)
        except RuntimeError:
            check("06_firewall_refuses", True)


def test_10_fold_union_contains_every_held_mask():
    with tempfile.TemporaryDirectory() as tmp:
        c018, fc_run = Path(tmp) / "c018", Path(tmp) / "fc"
        (c018).mkdir()
        (fc_run / "packs").mkdir(parents=True)
        rng = np.random.default_rng(4)
        for h in cc.PRIMARY_CONTEXTS:
            ids = np.asarray([f"G{i}" for i in rng.choice(60, 30, replace=False)])
            np.savez(c018 / f"effects_{h}.npz", eligible=rng.random(30) < 0.8, folds=rng.integers(0, 5, 30))
            np.savez(fc_run / "packs" / f"{h}.npz", ids=ids)
        for fold in range(5):
            union = set(gf.fold_union(str(c018), str(fc_run), fold))
            ok = all(set(gf.masked_set(str(c018), str(fc_run), h, fold)) <= union for h in cc.PRIMARY_CONTEXTS)
            check(f"10_fold{fold}_union_superset", ok)


def test_07_specific_cosine_removes_the_common_axis():
    rng = np.random.default_rng(8)
    common = np.tile(np.eye(6)[0], (4, 1))
    truth = rng.standard_normal((4, 6))
    effect = truth + 50.0 * common
    check("07_common_invisible", np.allclose(evl.specific_cosine(effect, truth, common), 1.0))


def test_08_cell_selection_is_deterministic_and_order_free():
    keys = [f"K|cell{i}" for i in range(300)]
    a = cells.cell_key_rank(keys, "NS")[:32]
    b = cells.cell_key_rank(keys[::-1], "NS")[:32]
    check("08_same_set", sorted(np.asarray(keys)[a].tolist()) == sorted(np.asarray(keys[::-1])[b].tolist()))
    check("08_namespace_matters", set(a.tolist()) != set(cells.cell_key_rank(keys, "OTHER")[:32].tolist()))


def test_09_contract_rows_are_labelled_and_no_g_check():
    check("09_taxonomy", all(a in cc.TAXONOMY and a in cc.LEGALITY for a in cc.MAIN_ROWS + (cc.RIDGE_K562, cc.PROFILED)))
    check("09_no_g_check", cc.READ_ANCHOR_G_CHECK is False)
    check("09_cellnavi_measured_candidate", "MEASURED-CANDIDATE" in cc.LEGALITY[cc.CELLNAVI])
    check("09_oracle_flagged", "NOT_DEPLOYABLE" in cc.LEGALITY[cc.ORACLE])


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except Exception as error:
                check(name, False, f"raised {type(error).__name__}: {error}")
    failed = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
