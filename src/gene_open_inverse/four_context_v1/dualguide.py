"""Dual-sgRNA identity semantics, resolved explicitly and never guessed.

GSE264667 transduces a dual-sgRNA vector, so a cell carries two guides and "which
intervention is this cell" is a question about the pair, not about one label.  Three
pair kinds have three different meanings and only two of them are single-gene
interventions:

    gene X + gene X      one intervention on X, stronger knockdown than one guide
    gene X + NTC         one intervention on X
    gene A + gene B      a combination, which this programme does not model

The third kind cannot be folded into the first two.  Calling ``A + B`` an
intervention on ``A`` would put a second perturbation's response into ``A``'s
measured effect and then into the distillation target, and every downstream number
would inherit it.  Those rows are excluded from the primary programme and counted,
never silently relabelled, and if excluding them leaves too few candidates the
context is downgraded rather than the task redefined.

Parsing strategy
----------------
Separators are tried from a declared list, but the decisive step is not the
separator: each part is resolved against a *known symbol universe* and against the
declared non-targeting tokens.  A part that matches neither is unparsed and is
reported as such.  That way a format this list does not anticipate produces a loud
unparsed fraction instead of a confident wrong answer.
"""
from __future__ import annotations

import re

import numpy as np


# Separators, most specific pair delimiter first.  Order matters less than it looks,
# because a separator is only accepted when *every* part it produces resolves, but it
# still matters: the first attempt at GSE264667 put "_+_" ahead of "|" and "_+_"
# occurs *inside* each guide name, so the split produced fragments and the whole
# dataset came back unparsed.  The data were fine; the parser was not.
SEPARATORS = ("|", "_|_", "_+_", "__", "_and_", "+")
STRAND_SUFFIX = re.compile(r"[_.\-](?:[+-]|[0-9]+|[ABab]|sg[0-9]+)$")
NTC_TOKENS = ("non-targeting", "non_targeting", "nontargeting", "ntc", "negctrl",
              "negative_control", "neg_ctrl", "safe-targeting", "safe_targeting",
              "control", "scrambled", "*")

SINGLE_GENE = "SINGLE_GENE"
NON_TARGETING = "NON_TARGETING"
CROSS_GENE = "CROSS_GENE"
UNPARSED = "UNPARSED"
KINDS = (SINGLE_GENE, NON_TARGETING, CROSS_GENE, UNPARSED)

COLLAPSE_RULES = {
    SINGLE_GENE: "both guides resolve to one gene, or one guide is non-targeting; "
                 "the cell is that gene's intervention",
    NON_TARGETING: "both guides are non-targeting; the cell is a control",
    CROSS_GENE: "the two guides target different genes; excluded from the primary "
                "single-gene programme and counted",
    UNPARSED: "the label did not resolve against the declared separators, the symbol "
              "universe or the non-targeting tokens; excluded and counted",
}


def _clean(part: str) -> str:
    value = part.strip()
    previous = None
    while value and value != previous:
        previous = value
        value = STRAND_SUFFIX.sub("", value)
    return value


def _is_ntc(part: str) -> bool:
    lowered = part.lower()
    return any(token in lowered for token in NTC_TOKENS)


def _known_prefix(part: str, universe: frozenset[str]) -> str | None:
    """The longest underscore-delimited prefix of a guide name that is a known symbol.

    A guide name in GSE264667 reads ``AAAS_-_53715438.23-P1P2``: the target symbol,
    the strand, and the cut coordinate.  Stripping suffixes by pattern cannot recover
    the symbol from that, because the tail is not a fixed shape.  Reading the symbol
    as a prefix against a known universe can, and it fails loudly on a name whose
    prefix is not a gene instead of inventing one.

    Longest first, so a symbol that itself contains an underscore is preferred over
    its own first token.
    """
    tokens = part.strip().split("_")
    for length in range(len(tokens), 0, -1):
        candidate = "_".join(tokens[:length])
        if candidate in universe:
            return candidate
    cleaned = _clean(part)
    return cleaned if cleaned in universe else None


def _resolve_part(part: str, universe: frozenset[str]) -> tuple[str, str]:
    if _is_ntc(part) or _is_ntc(_clean(part)):
        return NON_TARGETING, ""
    symbol = _known_prefix(part, universe)
    return (SINGLE_GENE, symbol) if symbol else (UNPARSED, "")


def split_pair(label: str, universe: frozenset[str] | None = None) -> tuple[str, ...]:
    """Split one dual-guide label into its parts.

    When a universe is supplied, a separator is accepted only if every part it
    produces resolves.  That is what stops a delimiter which also occurs inside a
    guide name from silently winning and shredding the label.
    """
    fallback: tuple[str, ...] | None = None
    for separator in SEPARATORS:
        if separator not in label:
            continue
        parts = tuple(piece for piece in label.split(separator) if piece != "")
        if len(parts) < 2:
            continue
        if universe is None:
            return parts
        if all(_resolve_part(piece, universe)[0] != UNPARSED for piece in parts):
            return parts
        fallback = fallback or parts
    return fallback or (label,)


def resolve_label(label: str, universe: frozenset[str]) -> tuple[str, str]:
    """Return ``(kind, identity)`` for one dual-guide label.

    ``identity`` is the gene symbol for a single-gene intervention, the literal
    label for a control, and the empty string otherwise.
    """
    parts = split_pair(label, universe)
    resolved = [_resolve_part(part, universe) for part in parts]
    if any(kind == UNPARSED for kind, _ in resolved):
        return UNPARSED, ""
    genes = sorted({name for kind, name in resolved if kind == SINGLE_GENE})
    if not genes:
        return NON_TARGETING, label
    if len(genes) == 1:
        return SINGLE_GENE, genes[0]
    return CROSS_GENE, ""


def resolve(labels: np.ndarray, universe) -> dict:
    """Resolve every cell's dual-guide label.  One pass over the distinct values."""
    labels = np.asarray(labels, dtype=str)
    known = frozenset(str(value) for value in np.asarray(list(universe), dtype=str).tolist())
    distinct, inverse = np.unique(labels, return_inverse=True)
    kinds = np.empty(distinct.size, dtype=object)
    identities = np.empty(distinct.size, dtype=object)
    for index, value in enumerate(distinct.tolist()):
        kinds[index], identities[index] = resolve_label(value, known)
    return {"distinct": distinct, "kind_per_label": kinds.astype(str),
            "identity_per_label": identities.astype(str), "universe": known,
            "kind": kinds.astype(str)[inverse], "identity": identities.astype(str)[inverse]}


def summarise(resolution: dict, labels: np.ndarray) -> dict:
    """The numbers the hard gate is decided on, at cell and at label level."""
    kind = resolution["kind"]
    identity = resolution["identity"]
    cells = kind.size
    record: dict = {
        "cells": int(cells),
        "distinct_labels": int(resolution["distinct"].size),
        "cell_fraction": {name: float((kind == name).mean()) for name in KINDS},
        "cell_count": {name: int((kind == name).sum()) for name in KINDS},
        "label_count": {name: int((resolution["kind_per_label"] == name).sum()) for name in KINDS},
        "single_gene_identities": int(np.unique(identity[kind == SINGLE_GENE]).size),
        "cross_gene_cells": int((kind == CROSS_GENE).sum()),
        "unparsed_cells": int((kind == UNPARSED).sum()),
        "collapse_rules": COLLAPSE_RULES,
        "example_labels": {},
    }
    for name in KINDS:
        picked = resolution["distinct"][resolution["kind_per_label"] == name]
        record["example_labels"][name] = picked[:6].tolist()
    return record


def pair_shape(resolution: dict, universe=None) -> dict:
    """How the two guides relate, at label level, for the audit table."""
    known = frozenset(str(v) for v in np.asarray(list(universe or []), dtype=str).tolist())
    kinds = resolution["kind_per_label"]
    shape = {"same_gene_pair": 0, "gene_plus_ntc": 0, "ntc_plus_ntc": 0,
             "geneA_plus_geneB": 0, "unparsed": 0, "single_part_label": 0}
    for label, kind in zip(resolution["distinct"].tolist(), kinds.tolist()):
        parts = split_pair(label, known or None)
        if kind == UNPARSED:
            shape["unparsed"] += 1
            continue
        if len(parts) < 2:
            shape["single_part_label"] += 1
            continue
        ntc = [_is_ntc(part) or _is_ntc(_clean(part)) for part in parts]
        if all(ntc):
            shape["ntc_plus_ntc"] += 1
        elif any(ntc):
            shape["gene_plus_ntc"] += 1
        elif kind == SINGLE_GENE:
            shape["same_gene_pair"] += 1
        else:
            shape["geneA_plus_geneB"] += 1
    total = max(sum(shape.values()), 1)
    return {"label_counts": shape,
            "label_fractions": {key: value / total for key, value in shape.items()}}


def unparsed_pair_shape(resolution: dict) -> dict:
    """Are the labels we could not resolve nevertheless same-gene pairs?

    "No cross-gene pairs" would otherwise be a claim about the labels that parsed,
    and an unresolved symbol is exactly where a combination could hide.  A guide name
    leads with its target, so comparing the leading token of the two halves answers
    the question without needing to know what the symbol is.  Aliases that the symbol
    universe does not carry, ASUN for INTS13 for instance, land here.
    """
    known = resolution.get("universe") or frozenset()
    same, differing, examples = 0, 0, []
    for label, kind in zip(resolution["distinct"].tolist(),
                           resolution["kind_per_label"].tolist()):
        if kind != UNPARSED:
            continue
        parts = split_pair(label, known or None)
        if len(parts) < 2:
            differing += 1
            continue
        leading = {part.strip().split("_")[0] for part in parts}
        if len(leading) == 1:
            same += 1
        else:
            differing += 1
            if len(examples) < 6:
                examples.append(label)
    return {"unparsed_labels": same + differing,
            "identical_leading_token": same,
            "differing_leading_token": differing,
            "differing_examples": examples,
            "reading": "identical leading tokens mean the label is a same-gene pair whose "
                       "symbol the universe does not carry, usually an alias; a differing "
                       "leading token is a possible cross-gene pair and is not collapsible"}


def case_verdict(summary: dict, shape: dict, minimum_identities: int,
                 unparsed: dict | None = None) -> dict:
    """Which of the protocol's three cases this context is, and whether it qualifies."""
    fractions = shape["label_fractions"]
    dominant = max(("same_gene_pair", "gene_plus_ntc", "geneA_plus_geneB"),
                   key=lambda key: fractions.get(key, 0.0))
    case = {"same_gene_pair": "A_SAME_GENE_DUAL_GUIDES",
            "gene_plus_ntc": "B_GENE_PLUS_NTC",
            "geneA_plus_geneB": "C_CROSS_GENE_PAIRS"}[dominant]
    enough = summary["single_gene_identities"] >= minimum_identities
    possible_cross = shape["label_counts"]["geneA_plus_geneB"]
    if unparsed is not None:
        possible_cross += unparsed["differing_leading_token"]
    return {
        "case": case,
        "dominant_label_shape": dominant,
        "unparsed_pair_shape": unparsed,
        "labels_that_could_be_cross_gene": possible_cross,
        "no_cross_gene_pair_anywhere": bool(possible_cross == 0),
        "dominant_label_fraction": float(fractions.get(dominant, 0.0)),
        "single_gene_identities_after_exclusion": summary["single_gene_identities"],
        "minimum_required": minimum_identities,
        "cross_gene_labels_excluded": shape["label_counts"]["geneA_plus_geneB"],
        "unparsed_labels_excluded": shape["label_counts"]["unparsed"],
        "single_gene_unit_constructible": bool(enough),
        "note": "cross-gene pairs are excluded rather than relabelled; the task "
                "definition is not changed to preserve a candidate count",
    }
