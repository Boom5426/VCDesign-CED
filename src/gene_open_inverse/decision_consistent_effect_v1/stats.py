"""Inference and the diagnostics' arithmetic, nothing that scores a candidate.

The evaluator emits one vector of length ``2Q`` per arm and metric: every query's view 0
value, then every query's view 1 value.  Entries ``i`` and ``i + Q`` are one query seen
twice.  Every interval here resamples query identities and carries both views of a drawn
identity together, which is the cluster unit the programme's frozen bootstrap names but
does not implement.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..four_context_v1 import evaluate as ev
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from . import contract as dc


def view_pairs(vector: np.ndarray) -> np.ndarray:
    """``[2Q] -> [Q, 2]``, one row per query identity."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size % 2:
        raise ValueError("the evaluator emits two views per query, so the vector is even")
    half = vector.size // 2
    return np.stack([vector[:half], vector[half:]], axis=1)


def state_of(interval: list) -> str:
    low, high = float(interval[0]), float(interval[1])
    if low > 0.0:
        return dc.POSITIVE
    if high < 0.0:
        return dc.NEGATIVE
    return dc.NULL


def cluster_bootstrap_rows(blocks: list, replicates: int = dc.BOOTSTRAP_REPLICATES,
                           seed: int = dc.BOOTSTRAP_SEED, chunk: int = 500) -> dict:
    """Median and mean of all entries, resampling rows (clusters) with replacement.

    ``blocks`` holds one ``[clusters, entries]`` array per context and every row is one
    cluster.  Pooling contexts concatenates their rows, so a gene queried in two contexts
    is two clusters.  The draws depend only on the seed and the cluster count, so every
    contrast of one regime is resampled with the same draws.
    """
    rows = np.concatenate([np.asarray(block, dtype=np.float64) for block in blocks])
    count = rows.shape[0]
    # One call for every draw, so the stream never depends on how the arithmetic is
    # chunked; only the gather and the statistic are done in chunks.
    draws = np.random.default_rng(seed).integers(0, count, size=(replicates, count))
    medians, means = [], []
    for start in range(0, replicates, chunk):
        sample = rows[draws[start:start + chunk]].reshape(min(chunk, replicates - start), -1)
        medians.append(np.median(sample, axis=1))
        means.append(sample.mean(axis=1))
    medians, means = np.concatenate(medians), np.concatenate(means)
    flat = rows.reshape(-1)
    median_ci = [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]
    return {"clusters": int(count), "entries": int(flat.size),
            "replicates": int(replicates), "seed": int(seed),
            "median": float(np.median(flat)), "mean": float(flat.mean()),
            "median_ci95": median_ci,
            "mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "fraction_query_views_improved": float((flat > 0).mean()),
            "fraction_replicates_median_above_zero": float((medians > 0).mean()),
            "state": state_of(median_ci)}


def cluster_bootstrap_labels(values: np.ndarray, labels: np.ndarray,
                             replicates: int = dc.BOOTSTRAP_REPLICATES,
                             seed: int = dc.BOOTSTRAP_SEED, chunk: int = 200) -> dict:
    """The same statistic when clusters hold different numbers of entries.

    Used for the per-fold design, where one query identity is graded in every fold whose
    pool it ranks, so its cluster holds two entries per fold.  Clusters are padded to a
    common width with NaN and the statistic ignores the padding.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    unique, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    width = int(sizes.max())
    padded = np.full((unique.size, width), np.nan)
    fill = np.zeros(unique.size, dtype=np.int64)
    for index, cluster in enumerate(inverse.tolist()):
        padded[cluster, fill[cluster]] = values[index]
        fill[cluster] += 1
    draws = np.random.default_rng(seed).integers(0, unique.size, size=(replicates, unique.size))
    medians, means = [], []
    for start in range(0, replicates, chunk):
        sample = padded[draws[start:start + chunk]].reshape(min(chunk, replicates - start), -1)
        medians.append(np.nanmedian(sample, axis=1))
        means.append(np.nanmean(sample, axis=1))
    medians, means = np.concatenate(medians), np.concatenate(means)
    median_ci = [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]
    return {"clusters": int(unique.size), "entries": int(values.size),
            "replicates": int(replicates), "seed": int(seed),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "median_ci95": median_ci,
            "mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "fraction_query_views_improved": float((values > 0).mean()),
            "fraction_replicates_median_above_zero": float((medians > 0).mean()),
            "state": state_of(median_ci)}


def _legal_orders(scores_by_view: list, ids: np.ndarray, self_positions: np.ndarray,
                  mask: np.ndarray):
    """Yield ``(view, row, legal order within the stratum, stratum columns)``.

    The same construction as ``four_context_v1.evaluate.evaluate_arms``: restrict to the
    stratum's columns, remap the query's own position, exclude it through the frozen
    ``stable_order`` and drop it, or pass no exclusion when the query is not in the stratum.
    """
    columns = np.flatnonzero(np.asarray(mask, dtype=bool))
    stratum_ids = np.asarray(ids, dtype=str)[columns]
    remap = {int(value): index for index, value in enumerate(columns.tolist())}
    inside = np.asarray([remap.get(int(p), -1) for p in np.asarray(self_positions).tolist()],
                        dtype=np.int64)
    for view in (0, 1):
        block = np.asarray(scores_by_view[view], dtype=np.float64)[:, columns]
        for row in range(block.shape[0]):
            position = int(inside[row])
            excluded = position if position >= 0 else None
            order = stable_order(block[row], stratum_ids, excluded)
            if excluded is not None:
                order = order[order != position]
            yield view, row, order, columns


def best_at(scores_by_view: list, utility_by_view: list, ids: np.ndarray,
            self_positions: np.ndarray, mask: np.ndarray) -> dict:
    """``Best@B``: the largest graded utility among the top B, a companion metric only."""
    output = {f"Best@{budget}": [] for budget in (10, 20, 50)}
    for view, row, order, columns in _legal_orders(scores_by_view, ids, self_positions, mask):
        utility = np.asarray(utility_by_view[view], dtype=np.float64)[row, columns]
        for budget in (10, 20, 50):
            output[f"Best@{budget}"].append(float(utility[order[:budget]].max()))
    return {key: np.asarray(values) for key, values in output.items()}


def per_query_overlap(first: list, second: list, ids: np.ndarray, self_positions: np.ndarray,
                      mask: np.ndarray, budget: int = dc.OVERLAP_BUDGET) -> np.ndarray:
    """Top-``budget`` overlap of two arms' legal orderings, one value per query view."""
    left = [set(order[:budget].tolist()) for _, _, order, _ in
            _legal_orders(first, ids, self_positions, mask)]
    right = [set(order[:budget].tolist()) for _, _, order, _ in
             _legal_orders(second, ids, self_positions, mask)]
    return np.asarray([len(a & b) / budget for a, b in zip(left, right)], dtype=np.float64)


def goal_blind_overlap(first: np.ndarray, second: np.ndarray, ids: np.ndarray,
                       budget: int = dc.OVERLAP_BUDGET) -> float:
    """Top-``budget`` overlap of two goal-blind orderings of one candidate set."""
    ids = np.asarray(ids, dtype=str)
    a = stable_order(np.asarray(first, dtype=np.float64), ids)[:budget]
    b = stable_order(np.asarray(second, dtype=np.float64), ids)[:budget]
    return float(len(set(a.tolist()) & set(b.tolist())) / budget)


def _ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=np.float64), method="average")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def partial_spearman(x: np.ndarray, y: np.ndarray, *covariates: np.ndarray) -> float:
    """Rank correlation of ``x`` and ``y`` after the rank-linear part of the covariates.

    Each variable is rank transformed, the ranks of ``x`` and ``y`` are each regressed on
    an intercept and the covariates' ranks, and the residuals are correlated.  With one
    covariate this equals the textbook first-order partial correlation of the ranks.
    """
    rx, ry = _ranks(x), _ranks(y)
    design = np.column_stack([np.ones(rx.size)] + [_ranks(z) for z in covariates])
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return _pearson(residual_x, residual_y)


def spearman_interval(x: np.ndarray, y: np.ndarray, covariates: tuple = (),
                      replicates: int = dc.DIAGNOSTIC_BOOTSTRAP_REPLICATES,
                      seed: int = dc.DIAGNOSTIC_BOOTSTRAP_SEED) -> dict:
    """A candidate-level bootstrap interval for a (partial) rank correlation."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    covariates = tuple(np.asarray(z, dtype=np.float64) for z in covariates)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        draw = generator.integers(0, x.size, size=x.size)
        values.append(partial_spearman(x[draw], y[draw], *(z[draw] for z in covariates))
                      if covariates else spearman(x[draw], y[draw]))
    point = partial_spearman(x, y, *covariates) if covariates else spearman(x, y)
    return {"value": float(point), "ci95": [float(np.nanquantile(values, 0.025)),
                                            float(np.nanquantile(values, 0.975))],
            "candidates": int(x.size), "replicates": int(replicates), "seed": int(seed)}


def _cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.einsum("ij,ij->i", first, second) / np.maximum(norm, 1e-30)


def direction_diagnostics(effect: np.ndarray, responses: np.ndarray, rows: np.ndarray,
                          intercept: np.ndarray, interval: bool = True) -> dict:
    """D1 and D2 for one predictor on one candidate set.

    ``responses`` are the held context's two measured views and ``intercept`` is the
    predictor's own intercept reconstruction, whose direction ``a`` is the common response
    direction the predictor learned from its training rows.  The held responses enter
    nothing but these numbers.  ``interval=False`` returns point values only, for the
    per-fold diagnostics whose summary is a median over folds.
    """
    truth = ev.consensus(np.asarray(responses)[:, rows])
    predicted = np.asarray(effect, dtype=np.float64)[rows]
    common = np.asarray(intercept, dtype=np.float64)
    common = common / max(float(np.linalg.norm(common)), 1e-30)

    def remove_common(values):
        return values - np.outer(values @ common, common)

    norm = np.linalg.norm(predicted, axis=1)
    amplitude = np.linalg.norm(truth, axis=1)
    accuracy = _cosine_rows(predicted, truth)
    specific = _cosine_rows(remove_common(predicted), remove_common(truth))
    alignment = truth @ common / np.maximum(amplitude, 1e-30)
    predicted_common = predicted @ common

    def correlate(first, second, covariates=()):
        if interval:
            return spearman_interval(first, second, covariates)
        return {"value": float(partial_spearman(first, second, *covariates) if covariates
                               else spearman(first, second)), "candidates": int(first.size)}

    return {
        "candidates": int(rows.size),
        "norm": {"median": float(np.median(norm)), "p05": float(np.quantile(norm, 0.05)),
                 "p95": float(np.quantile(norm, 0.95)),
                 "p95_over_p05": float(np.quantile(norm, 0.95) / max(np.quantile(norm, 0.05), 1e-30))},
        "direction_accuracy": {"median": float(np.median(accuracy)),
                               "fraction_non_positive": float((accuracy <= 0).mean())},
        "candidate_specific_accuracy": {"median": float(np.median(specific)),
                                        "fraction_non_positive": float((specific <= 0).mean())},
        "D1_spearman_norm_accuracy": correlate(norm, accuracy),
        "D1_spearman_norm_candidate_specific_accuracy": correlate(norm, specific),
        "D2_spearman_norm_amplitude": correlate(norm, amplitude),
        "D2_spearman_norm_common_alignment": correlate(norm, alignment),
        "D2_partial_norm_specific_accuracy_given_amplitude_and_alignment":
            correlate(norm, specific, (amplitude, alignment)),
        "D2_partial_norm_accuracy_given_amplitude": correlate(norm, accuracy, (amplitude,)),
        "spearman_norm_predicted_common_component": float(spearman(norm, predicted_common)),
        "_norm": norm, "_amplitude": amplitude,
    }
