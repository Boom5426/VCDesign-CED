"""Deployment-time support diagnostics for a candidate, computed from static knowledge alone.

Every quantity here is a function of the candidate's static features and of which
identities the training atlas measured.  None of them reads a response, a score or a
result, so all of them are available at deployment time for a gene nobody has ever
perturbed.  That is the whole point: a quantity that needed the answer could not be
used to decide whether to trust the answer.

None of these is a model input.  They are read only to describe a candidate and to
split results by, and this mission implements no gate that consumes them.

Leverage is the primary one because it is the ridge's own statement about
extrapolation rather than a metric invented beside it.  For a ridge fitted on X with
penalty lambda, ``L_c = k_c^T (X^T X + lambda I)^{-1} k_c`` is small when k_c lies in a
direction the training rows span strongly and grows toward ``||k_c||^2 / lambda`` in a
direction they do not span at all.  The null-space term is kept explicitly, because
with 1538 features and fewer training rows than that the unspanned part is most of the
story and dropping it would make every candidate look equally well supported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contract as ov


@dataclass(frozen=True)
class SupportTable:
    """One row per candidate, in the candidate axis' own order."""

    ids: np.ndarray
    string_present: np.ndarray
    mapkg_present: np.ndarray
    coverage: np.ndarray            # one of ov.COVERAGE_PATTERNS
    string_degree: np.ndarray       # STRING graph degree, 0 when absent from the graph
    string_in_graph: np.ndarray
    training_neighbours: np.ndarray        # STRING neighbours inside the training atlas
    training_neighbour_weight: np.ndarray  # summed edge weight to those neighbours
    leverage: np.ndarray
    leverage_percentile: np.ndarray
    d1: np.ndarray
    d10: np.ndarray
    d50: np.ndarray
    d10_percentile: np.ndarray
    mapkg_d10: np.ndarray           # the same neighbourhood inside the MAPKG block alone

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def modality_flags(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the two present flags from the frozen feature layout, without re-deriving them."""
    features = np.asarray(features)
    if features.shape[1] != ov.FEATURE_WIDTH:
        raise ValueError(f"expected the frozen {ov.FEATURE_WIDTH}-wide feature vector, "
                         f"received width {features.shape[1]}")
    string = features[:, ov.STRING_FLAG_COLUMN] > 0.5
    mapkg = features[:, ov.MAPKG_FLAG_COLUMN] > 0.5
    return string, mapkg


def coverage_pattern(string_present: np.ndarray, mapkg_present: np.ndarray) -> np.ndarray:
    pattern = np.where(string_present & mapkg_present, "BOTH",
              np.where(string_present & ~mapkg_present, "STRING_ONLY",
              np.where(~string_present & mapkg_present, "MAPKG_ONLY", "NEITHER")))
    return pattern.astype("<U12")


def string_neighbourhood(ids: np.ndarray, adjacency: dict,
                         training_ids: np.ndarray) -> dict:
    """Degree, and how much of that degree lands inside the training atlas.

    The adjacency is undirected and stored once per edge, so both endpoints are
    counted.  A candidate absent from the STRING universe gets degree zero and no
    neighbours, which is the same state the feature contract already treats as absent.
    """
    ids = np.asarray(ids, dtype=str)
    genes = np.asarray(adjacency["genes"], dtype=str)
    degree = np.asarray(adjacency["degree"], dtype=np.int64)
    row = np.asarray(adjacency["edge_row"], dtype=np.int64)
    column = np.asarray(adjacency["edge_col"], dtype=np.int64)
    weight = np.asarray(adjacency["edge_w"], dtype=np.float64)

    index = {name: position for position, name in enumerate(genes.tolist())}
    training = np.zeros(genes.size, dtype=bool)
    for name in np.asarray(training_ids, dtype=str).tolist():
        position = index.get(name)
        if position is not None:
            training[position] = True

    counts = np.zeros(genes.size, dtype=np.int64)
    weights = np.zeros(genes.size, dtype=np.float64)
    for left, right in ((row, column), (column, row)):
        np.add.at(counts, left, training[right].astype(np.int64))
        np.add.at(weights, left, weight * training[right])

    positions = np.asarray([index.get(name, -1) for name in ids.tolist()], dtype=np.int64)
    inside = positions >= 0
    safe = np.where(inside, positions, 0)
    return {"string_in_graph": inside,
            "string_degree": np.where(inside, degree[safe], 0),
            "training_neighbours": np.where(inside, counts[safe], 0),
            "training_neighbour_weight": np.where(inside, weights[safe], 0.0)}


def _sorted_distances(query: np.ndarray, reference: np.ndarray, wanted: int,
                      exclude: np.ndarray | None = None, chunk: int = 512) -> np.ndarray:
    """The ``wanted`` smallest Euclidean distances from each query row to the reference.

    ``exclude[i]`` is one reference row to remove for query ``i``, or a negative value
    for none.  It exists because the static feature vector is a pure function of the gene
    symbol, so a candidate whose identity is in the training atlas has a bitwise identical
    row there and finds itself at distance exactly zero.  Leaving that in would give every
    atlas-measured candidate a free zero in its neighbourhood, which is precisely the
    seen-against-unseen comparison this diagnostic is used to make, and precisely the axis
    the support matching matches on.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape[0] < wanted + 1:
        raise ValueError(f"cannot take {wanted} neighbours from {reference.shape[0]} rows")
    reference_square = (reference ** 2).sum(axis=1)
    output = np.empty((query.shape[0], wanted), dtype=np.float64)
    for start in range(0, query.shape[0], chunk):
        stop = start + chunk
        block = query[start:stop]
        squared = (block ** 2).sum(axis=1)[:, None] + reference_square[None, :] \
            - 2.0 * (block @ reference.T)
        np.maximum(squared, 0.0, out=squared)
        if exclude is not None:
            drop = np.asarray(exclude[start:stop], dtype=np.int64)
            rows = np.flatnonzero(drop >= 0)
            squared[rows, drop[rows]] = np.inf
        part = np.partition(squared, wanted - 1, axis=1)[:, :wanted]
        output[start:start + block.shape[0]] = np.sqrt(np.sort(part, axis=1))
    return output


def self_columns(ids: np.ndarray, training_identities: np.ndarray) -> np.ndarray:
    """Where each candidate sits inside the training identity axis, or -1 if absent."""
    index = {name: position for position, name
             in enumerate(np.asarray(training_identities, dtype=str).tolist())}
    return np.asarray([index.get(name, -1) for name in np.asarray(ids, dtype=str).tolist()],
                      dtype=np.int64)


def neighbourhood(features: np.ndarray, training_features: np.ndarray,
                  k: int = ov.NEIGHBOUR_K,
                  sensitivity: tuple[int, ...] = ov.NEIGHBOUR_SENSITIVITY,
                  exclude: np.ndarray | None = None) -> dict:
    """Mean distance to the k nearest training identities, plus the fixed sensitivities.

    ``training_features`` must already be reduced to unique identities.  A candidate
    measured in three contexts has three training rows but one static feature vector,
    and counting it three times would make every candidate near it look better
    supported for a reason that has nothing to do with knowledge.
    """
    wanted = max((k,) + tuple(sensitivity))
    distances = _sorted_distances(features, training_features, wanted, exclude)
    record = {f"d{k}": distances[:, :k].mean(axis=1)}
    for value in sensitivity:
        record[f"d{value}"] = distances[:, :value].mean(axis=1)
    return record


def ridge_leverage(path, features: np.ndarray, penalty: float) -> np.ndarray:
    """``k_c^T (X^T X + lambda I)^{-1} k_c`` for the fit that ``path`` actually is.

    The SVD stored on the path is of the standardized, row-weighted training matrix,
    so the spanned part is read straight off it and the unspanned remainder is damped
    by the penalty alone.  With fewer training rows than features the remainder is not
    a correction, it is the dominant term.
    """
    centred = (np.asarray(features, dtype=np.float64) - path.feature_mean) / path.feature_scale
    projected = centred @ path.right.T                 # [n, rank]
    singular = np.asarray(path.singular, dtype=np.float64)
    spanned = (projected ** 2) / (singular ** 2 + float(penalty))
    remainder = (centred ** 2).sum(axis=1) - (projected ** 2).sum(axis=1)
    np.maximum(remainder, 0.0, out=remainder)
    return spanned.sum(axis=1) + remainder / float(penalty)


def percentile_of(values: np.ndarray) -> np.ndarray:
    """Rank in [0, 1], ties averaged, computed on whatever population is passed in."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    totals = np.zeros(unique.size, dtype=np.float64)
    np.add.at(totals, inverse, ranks)
    return (totals / counts)[inverse] / max(values.size - 1, 1)


def build(ids: np.ndarray, features: np.ndarray, path, penalty: float,
          training_features: np.ndarray, training_ids: np.ndarray,
          adjacency: dict) -> SupportTable:
    """Every diagnostic for one candidate axis against one training atlas."""
    string_present, mapkg_present = modality_flags(features)
    graph = string_neighbourhood(ids, adjacency, training_ids)
    exclude = self_columns(ids, training_ids)
    distances = neighbourhood(features, training_features, exclude=exclude)
    mapkg_block = slice(ov.STRING_WIDTH, ov.STRING_WIDTH + ov.MAPKG_WIDTH)
    mapkg_distances = neighbourhood(features[:, mapkg_block], training_features[:, mapkg_block],
                                    exclude=exclude)
    leverage = ridge_leverage(path, features, penalty)
    return SupportTable(
        ids=np.asarray(ids, dtype=str),
        string_present=string_present, mapkg_present=mapkg_present,
        coverage=coverage_pattern(string_present, mapkg_present),
        string_degree=graph["string_degree"], string_in_graph=graph["string_in_graph"],
        training_neighbours=graph["training_neighbours"],
        training_neighbour_weight=graph["training_neighbour_weight"],
        leverage=leverage, leverage_percentile=percentile_of(leverage),
        d1=distances["d1"], d10=distances["d10"], d50=distances["d50"],
        d10_percentile=percentile_of(distances["d10"]),
        mapkg_d10=mapkg_distances["d10"])


def unique_training(features: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a pooled ``context|identity`` training pool to one row per identity."""
    identities = np.asarray([key.split("|", 1)[1] for key in np.asarray(keys, dtype=str).tolist()])
    values, first = np.unique(identities, return_index=True)
    return np.asarray(features, dtype=np.float64)[first], values
