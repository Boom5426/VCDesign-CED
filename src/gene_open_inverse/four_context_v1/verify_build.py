#!/usr/bin/env python3
"""Independent verifier for a Phase 2 context build.

A separate implementation, run after the fact, that recomputes values from the
source file rather than trusting the build.  It does not import the builder's
accumulation code: the split bits are rederived from the written rule, and the
sampled responses are recomputed by reading exactly the cells of one identity and
of the control population, in a different order and with different arithmetic
grouping from the streaming pass.

Every numeric check carries an explicit tolerance and the verdict is gated on it.
The script refuses to overwrite its own output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np

from . import contract as fc
from . import dualguide as dg
from . import storage as st


SAMPLE_IDENTITIES = 12
SAMPLE_SEED = 20260918


def sampled_targets(batch_count: int, usable: np.ndarray) -> dict:
    """Exactly which batches and identities the verifier inspects.

    Shared with the corruption witness so that the witness corrupts something the
    verifier is guaranteed to look at.  Both draw from one generator in one order, so
    this function is the single definition of "what gets checked".
    """
    generator = np.random.default_rng(SAMPLE_SEED)
    batches = generator.choice(batch_count, size=min(4, batch_count), replace=False)
    pool = np.flatnonzero(usable)
    identities = generator.choice(pool, size=min(SAMPLE_IDENTITIES, pool.size), replace=False)
    return {"batches": batches, "identities": identities}


def _bit(namespace: str, seed: int, name: str) -> int:
    digest = hashlib.sha256(("%s|%d|%s" % (namespace, seed, name)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 1


def verify(source: str, built: Path, perturbation_column: str, batch_column: str,
           control_labels: tuple[str, ...], identity_mode: str = "column",
           guide_column: str | None = None, gene_symbol_column: str | None = None) -> dict:
    manifest = json.loads((built / "FOUR_CONTEXT_BUILD_MANIFEST.json").read_text())
    batches = np.asarray([str(v) for v in np.load(built / "batch_axis.npy", allow_pickle=True)])
    sides = np.load(built / "batch_side.npy")
    quarters = np.load(built / "batch_quarter.npy")
    identities = np.asarray([str(v) for v in np.load(built / "identity_axis.npy", allow_pickle=True)])
    responses = np.load(built / "side_responses.npy")
    sources = np.load(built / "side_sources.npy")
    control_mean = np.load(built / "control_mean.npy")
    incidence = np.load(built / "incidence.npy")
    usable = np.load(built / "usable.npy")

    checks: dict = {}

    # 1. The split rule, rederived.
    expected_side = np.asarray([_bit(fc.VIEW_NAMESPACE, fc.VIEW_SEED, name)
                                for name in batches.tolist()], dtype=np.int8)
    expected_quarter = np.asarray([2 * _bit(fc.VIEW_NAMESPACE, fc.VIEW_SEED, name)
                                   + _bit(fc.QUAD_NAMESPACE, fc.VIEW_SEED, name)
                                   for name in batches.tolist()], dtype=np.int8)
    checks["batch_side_rule"] = {"passed": bool(np.array_equal(expected_side, sides)),
                                 "tolerance": "exact"}
    checks["batch_quarter_rule"] = {"passed": bool(np.array_equal(expected_quarter, quarters)),
                                    "tolerance": "exact",
                                    "is_refinement": bool(np.array_equal(quarters // 2, sides))}

    # 2. Recompute a sample of responses straight from the source.
    handle = anndata.read_h5ad(source, backed="r")
    perturbation = (handle.obs[perturbation_column].astype(str).to_numpy()
                    if perturbation_column and perturbation_column in handle.obs
                    else np.full(int(handle.n_obs), "__UNDECLARED__", dtype=object).astype(str))
    batch = handle.obs[batch_column].astype(str).to_numpy()
    if identity_mode == "dual_guide":
        # Resolved independently of the build's own run, from the same declared rule.
        guides = handle.obs[guide_column].astype(str).to_numpy()
        universe = set(np.asarray(handle.var_names, dtype=str).tolist()) \
            | set(np.unique(perturbation).tolist())
        resolution = dg.resolve(guides, universe)
        control_mask = resolution["kind"] == dg.NON_TARGETING
        perturbation = np.where(control_mask, "__CONTROL__", resolution["identity"])
        keep_cell = (resolution["kind"] == dg.SINGLE_GENE) | control_mask
        perturbation = np.where(keep_cell, perturbation, "__EXCLUDED__")
        control_mask = perturbation == "__CONTROL__"
        checks["dual_guide_identity_agrees_with_the_build"] = {
            "identities_in_build_absent_from_this_resolution": int(
                np.setdiff1d(identities, np.unique(perturbation)).size),
            "tolerance": "exact",
            "passed": bool(np.setdiff1d(identities, np.unique(perturbation)).size == 0)}
    else:
        control_mask = np.isin(perturbation, np.asarray(control_labels, dtype=str))
    batch_position = {name: index for index, name in enumerate(batches.tolist())}
    column = np.asarray([batch_position[name] for name in batch.tolist()], dtype=np.int64)

    # Rederive which gene columns the build kept, from the written axis rather than
    # from the build's own variable, so a wrong restriction shows up as a mismatch.
    if gene_symbol_column:
        raw_genes = handle.var[gene_symbol_column].astype(str).to_numpy()
    else:
        raw_genes = np.asarray(handle.var_names, dtype=str)
    written = np.asarray([str(v) for v in np.load(built / "gene_axis.npy", allow_pickle=True)])
    symbols, counts = np.unique(raw_genes, return_counts=True)
    duplicated = set(symbols[counts > 1].tolist())
    gene_keep = np.asarray([name not in duplicated for name in raw_genes.tolist()], dtype=bool)
    checks["gene_axis_matches_the_written_axis"] = {
        "tolerance": "exact",
        "passed": bool(np.array_equal(raw_genes[gene_keep], written)),
        "columns_dropped": int((~gene_keep).sum())}

    def normalized(rows: np.ndarray) -> np.ndarray:
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        block = handle.X[rows]
        values = np.asarray(block.toarray() if hasattr(block, "toarray") else block, dtype=np.float32)
        library = values.sum(axis=1)
        good = np.isfinite(library) & (library > 0.0)
        values[good] = np.log1p(values[good] / library[good, None] * fc.LIBRARY_TARGET)
        values[~good] = 0.0
        return np.ascontiguousarray(values[good][:, gene_keep])

    def mean_of(rows: np.ndarray) -> np.ndarray:
        """Float64 accumulation, matching the build's precision but not its order."""
        return normalized(rows).mean(axis=0, dtype=np.float64)

    # Control means, batch by batch, recomputed independently.
    targets = sampled_targets(batches.size, usable)
    sampled_batches = targets["batches"]
    stored_block, expected_block = [], []
    for index in sampled_batches.tolist():
        rows = np.flatnonzero(control_mask & (column == index))
        stored_block.append(control_mean[index])
        expected_block.append(mean_of(rows))
    checks["control_mean"] = {
        "batches_checked": [str(batches[i]) for i in sampled_batches.tolist()],
        **st.compare(np.stack(stored_block), np.stack(expected_block), "control_mean")}

    # Identity responses on both sides.
    picked = targets["identities"]
    stored_response, expected_response = [], []
    stored_source, expected_source = [], []
    for index in picked.tolist():
        name = identities[index]
        for side in (0, 1):
            rows = np.flatnonzero((perturbation == name) & np.isin(column, np.flatnonzero(sides == side)))
            mean = mean_of(rows)
            weights = incidence[index].astype(np.float64).copy()
            weights[sides != side] = 0.0
            weights /= weights.sum()
            reference = weights @ control_mean.astype(np.float64)
            stored_source.append(sources[side][index])
            expected_source.append(reference)
            stored_response.append(responses[side][index])
            expected_response.append(mean - reference)
    checks["identity_response"] = {
        "identities_checked": int(picked.size),
        **st.compare(np.stack(stored_response), np.stack(expected_response), "identity_response")}
    checks["control_reference"] = st.compare(np.stack(stored_source), np.stack(expected_source),
                                             "control_reference")

    # 3. Structural claims in the manifest.
    control_per_batch = np.load(built / "control_per_batch.npy")
    checks["every_batch_has_controls"] = {
        "batches_without_controls": int((control_per_batch == 0).sum()),
        "tolerance": "exact zero",
        "passed": bool((control_per_batch == 0).sum() == 0)}
    checks["no_control_label_became_an_identity"] = {
        "passed": bool(not set(identities.tolist()) & set(control_labels)),
        "tolerance": "exact"}
    checks["unusable_rows_are_zero"] = {
        "passed": bool(np.abs(responses[:, ~usable]).max(initial=0.0) == 0.0),
        "tolerance": "exact"}
    checks["manifest_counts"] = {
        "passed": bool(manifest["identities"]["usable"] == int(usable.sum())
                       and manifest["batches"]["total"] == int(batches.size)),
        "tolerance": "exact"}
    handle.file.close()

    return {"schema": "VCDESIGN_FOUR_CONTEXT_V1_VERIFY", "context": manifest["context"],
            "numeric_contract": st.CONTRACT, "tolerance": st.TOLERANCE_EXPRESSION,
            "source": str(source), "built": str(built),
            "checks": checks,
            "verdict": "PASS" if all(item.get("passed", False) for item in checks.values()) else "FAIL",
            "runtime": {"timestamp": datetime.now(timezone.utc).isoformat(),
                        "python": platform.python_version(), "numpy": np.__version__}}


def corruption_witness(source: str, built: Path, perturbation_column: str, batch_column: str,
                       control_labels: tuple[str, ...], workspace: Path,
                       identity_mode: str = "column", guide_column: str | None = None,
                       gene_symbol_column: str | None = None) -> dict:
    """Prove the gate still fires on a real defect, rather than asserting that it does.

    A copy of the asset is made, one stored response is moved by 1e-03, and the same
    verifier is run against the same source data.  A tolerance loose enough to pass a
    corrupted asset is not a verification, so this record is kept next to the real one
    and a verification whose witness does not FAIL is itself a failure.
    """
    import shutil

    corrupted = workspace / f"{built.name}__CORRUPTED_WITNESS"
    if corrupted.exists():
        shutil.rmtree(corrupted)
    shutil.copytree(built, corrupted)
    responses = np.load(corrupted / "side_responses.npy")
    usable = np.load(corrupted / "usable.npy")
    batch_count = int(np.load(corrupted / "batch_axis.npy", allow_pickle=True).size)
    inspected = sampled_targets(batch_count, usable)["identities"]
    block = responses[:, inspected]
    responses[:, inspected] = st.corrupt_for_witness(block)
    np.save(corrupted / "side_responses.npy", responses)
    record = verify(source, corrupted, perturbation_column, batch_column, control_labels,
                    identity_mode, guide_column, gene_symbol_column)
    shutil.rmtree(corrupted)
    return {"perturbation": "one stored response element moved by 1e-03, inside the "
                            "identities the verifier samples",
            "identities_inspected": int(inspected.size),
            "verdict_on_corrupted_asset": record["verdict"],
            "identity_response_check": record["checks"]["identity_response"],
            "witness_passed": record["verdict"] == "FAIL"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently verify a context build")
    parser.add_argument("--source", required=True)
    parser.add_argument("--built", required=True)
    parser.add_argument("--perturbation-column", dest="perturbation_column", default=None,
                        help="optional under dual_guide, where identity comes from the pair")
    parser.add_argument("--batch-column", dest="batch_column", required=True)
    parser.add_argument("--control-labels", dest="control_labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity-mode", dest="identity_mode", default="column",
                        choices=("column", "dual_guide"))
    parser.add_argument("--guide-column", dest="guide_column", default=None)
    parser.add_argument("--gene-symbol-column", dest="gene_symbol_column", default=None)
    parser.add_argument("--witness-workspace", dest="witness_workspace", default=None,
                        help="directory for the deliberately corrupted asset witness")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite an existing verification record: {destination}")
    record = verify(args.source, Path(args.built), args.perturbation_column,
                    args.batch_column, tuple(args.control_labels),
                    args.identity_mode, args.guide_column, args.gene_symbol_column)
    if args.witness_workspace:
        record["corruption_witness"] = corruption_witness(
            args.source, Path(args.built), args.perturbation_column, args.batch_column,
            tuple(args.control_labels), Path(args.witness_workspace),
            args.identity_mode, args.guide_column, args.gene_symbol_column)
        if not record["corruption_witness"]["witness_passed"]:
            record["verdict"] = "FAIL"
            record["verdict_reason"] = ("the verifier accepted a deliberately corrupted asset, "
                                        "so its tolerance does not verify anything")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
