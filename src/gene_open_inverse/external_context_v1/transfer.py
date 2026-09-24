"""Shared machinery for moving the frozen anchor model into an external context.

Two things have to cross the boundary and both are done once, declared, and never
adjusted on a result: the gene axis and the candidate features.

The gene axis.  ``G_common`` is the intersection of the two measured axes.  An RPE1
state vector is embedded into the anchor's 8248-gene axis by placing the measured
value at each shared gene and zero elsewhere, which lets the frozen anchor model run
unchanged.  The effect cosine is computed on ``G_common`` only.

The candidate features.  STRING and MAPKG are keyed by gene symbol, so they transfer
directly, but their standardisation was fitted on the anchor's ``G_fit`` rows and must
be reused rather than refitted.  ``external_capabilities`` reproduces the anchor's own
statistics and then verifies, on the identities the two contexts share, that it
reproduces the anchor's feature matrix to floating point.
"""
from __future__ import annotations

import numpy as np


def gene_bridge(external_genes: np.ndarray, anchor_genes: np.ndarray) -> dict:
    """Positions of the shared genes in both axes, taken once."""
    common = np.intersect1d(np.asarray(external_genes, dtype=str), np.asarray(anchor_genes, dtype=str))
    external_index = {name: index for index, name in enumerate(np.asarray(external_genes, dtype=str).tolist())}
    anchor_index = {name: index for index, name in enumerate(np.asarray(anchor_genes, dtype=str).tolist())}
    return {
        "common": common,
        "external_positions": np.asarray([external_index[name] for name in common.tolist()], dtype=np.int64),
        "anchor_positions": np.asarray([anchor_index[name] for name in common.tolist()], dtype=np.int64),
        "size": int(common.size),
        "fraction_of_anchor_axis": float(common.size / len(anchor_genes)),
        "fraction_of_external_axis": float(common.size / len(external_genes)),
    }


def embed_into_anchor(values: np.ndarray, bridge: dict, anchor_width: int) -> np.ndarray:
    """Place shared-gene values on the anchor axis, zero elsewhere."""
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros(values.shape[:-1] + (anchor_width,), dtype=np.float32)
    output[..., bridge["anchor_positions"]] = values[..., bridge["external_positions"]]
    return output


def external_capabilities(assets, ids: np.ndarray) -> tuple[dict, dict, dict]:
    """STRING and MAPKG for arbitrary gene symbols, standardised exactly as the anchor does."""
    ids = np.asarray(ids, dtype=str)
    raw, present = assets.base._raw_string(ids)
    string = (raw - assets.base.string_mean) / assets.base.string_std
    string[~present] = 0.0

    with np.load(assets.paths["mapkg"], allow_pickle=False) as archive:
        universe = np.asarray(archive["universe"]).astype(str)
        values = archive["MAPKG"].astype(np.float32)
        flag = np.asarray(archive["MAPKG_present"])
    legal = assets._finite_present(values, flag)
    lookup = {name: index for index, name in enumerate(universe.tolist())}
    mapkg_raw = np.zeros((ids.size, values.shape[1]), dtype=np.float32)
    mapkg_present = np.zeros(ids.size, dtype=bool)
    for row, name in enumerate(ids.tolist()):
        index = lookup.get(name)
        if index is not None and legal[index]:
            mapkg_raw[row] = values[index]
            mapkg_present[row] = True

    # The anchor's own fit statistics, recomputed from its cache axis rather than refitted here.
    cache_raw, cache_present = assets._map_to_cache(universe, values, flag, values.shape[1])
    fit = assets.base.role_indices["G_fit"]
    rows = cache_raw[fit][cache_present[fit]]
    mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(rows.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    mapkg = (mapkg_raw - mean) / std
    mapkg[~mapkg_present] = 0.0

    # Verify on the identities the two contexts share.
    anchor_ids = np.asarray([str(v) for v in assets.base.cache_ids])
    anchor_lookup = {name: index for index, name in enumerate(anchor_ids.tolist())}
    shared = [(row, anchor_lookup[name]) for row, name in enumerate(ids.tolist()) if name in anchor_lookup]
    errors = {"shared_identities": len(shared)}
    if shared:
        here = np.asarray([row for row, _ in shared])
        there = np.asarray([index for _, index in shared])
        errors["STRING_max_abs_error"] = float(np.abs(string[here] - assets.values["STRING"][there]).max())
        errors["MAPKG_max_abs_error"] = float(np.abs(mapkg[here] - assets.values["MAPKG"][there]).max())
        errors["STRING_present_matches"] = bool(np.array_equal(present[here], assets.present["STRING"][there]))
        errors["MAPKG_present_matches"] = bool(np.array_equal(mapkg_present[here], assets.present["MAPKG"][there]))
    return ({"STRING": string.astype(np.float32), "MAPKG": mapkg.astype(np.float32)},
            {"STRING": present, "MAPKG": mapkg_present}, errors)


def cross_view_cosine(responses: np.ndarray, view: int) -> np.ndarray:
    """The anchor's evaluator convention: a view-``v`` query is graded by view ``1-v`` responses."""
    other = np.asarray(responses[1 - view], dtype=np.float64)
    unit = other / np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-30)
    return unit @ unit.T
