#!/usr/bin/env python3
"""Render Figure 2: held-out K562 performance, paired gains and budget payoff.

The original three panels read ``fig2_hero_data.json``; optional benchmark
panels read the published external-baseline levels CSV. The script validates
the frozen headline values and exports editable PDF/SVG plus a PNG, at 140 mm
width (56 mm high for three panels, 118 mm for five).

    python3 figures/scripts/plot_fig2_hero.py \
        --data figures/data/fig2_hero_data.json \
        --external-levels supplementary/data/external_baseline_levels.csv \
        --out figures/fig2_hero
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from figstyle import GUIDE, INK, apply_figure_style, fig_text, place


apply_figure_style()

FIG_W, FIG_H = 5.50, 2.20  # 139.7 x 55.9 mm, three aligned peer panels
A_BOX = (0.95, 0.40, 1.08, 1.52)
B_BOX = (2.52, 0.40, 1.18, 1.52)
C_BOX = (4.20, 0.40, 1.18, 1.52)
D_BOX = (1.45, 0.43, 1.55, 1.60)
E_BOX = (3.73, 0.43, 1.63, 1.60)

BENCHMARK_METHODS = (
    ("cellnavi_official", "CellNavi (official)"),
    ("cellnavi_native", "CellNavi (native)"),
    ("internal_forward_then_match", "Forward-then-match"),
    ("base", "Base scorer"),
    ("vcdesign_ced", "VCDesign-CED"),
    ("profiled_signature_retrieval", "Profile retrieval"),
)

PRIMARY = "#0f777b"
PRIMARY_TEXT = "#0b6266"
BASE = "#42484a"
TEXT = "#78838a"
FORWARD = "#a4aaad"
PRIOR = "#c2c6c8"

METHOD_ORDER = (
    "Goal-blind prior",
    "Forward-then-match",
    "Base scorer",
    "Text-augmented ranker",
    "VCDesign-CED",
)
METHOD_STYLE = {
    "Goal-blind prior": (PRIOR, 0.14),
    "Forward-then-match": (FORWARD, 0.16),
    "Base scorer": (BASE, 0.16),
    "Text-augmented ranker": (TEXT, 0.18),
    "VCDesign-CED": (PRIMARY, 0.32),
}

# Validation targets only. Plotting values always come from the data artifact.
EXPECTED_LEVELS = {
    "Base scorer": 0.3551,
    "Text-augmented ranker": 0.3850,
    "VCDesign-CED": 0.3905,
}
EXPECTED_PAIRED = {
    "Base scorer": (0.0319, 0.751),
    "Text-augmented ranker": (0.0268, 0.582),
}
EXPECTED_HVHIT = {
    "Base scorer": (4.0, 6.0, 9.0),
    "Text-augmented ranker": (4.0, 7.0, 10.0),
    "VCDesign-CED": (4.0, 7.0, 12.0),
}


def _assert_close(label: str, value: float, expected: float, tolerance: float) -> None:
    if not np.isclose(value, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{label}: {value:.8f} != expected {expected:.8f}")


def validate(data: dict) -> None:
    if data.get("schema") != "VCDESIGN_FIG2_DATA_V3":
        raise AssertionError(f"unexpected Figure 2 schema: {data.get('schema')}")
    if int(data["population"]["query_view_rows"]) != 684:
        raise AssertionError("Figure 2 requires 684 paired query-view rows")
    for method, expected in EXPECTED_LEVELS.items():
        values = np.asarray(data["panel_a"]["methods"][method]["values"], dtype=float)
        if values.size != 684:
            raise AssertionError(f"{method}: expected 684 values, found {values.size}")
        _assert_close(f"{method} median", float(np.median(values)), expected, 5e-5)
    for comparator, (median, win_rate) in EXPECTED_PAIRED.items():
        entry = data["panel_b"]["comparisons"][comparator]
        values = np.asarray(entry["values"], dtype=float)
        _assert_close(f"CED - {comparator} median", float(np.median(values)), median, 5e-5)
        _assert_close(f"CED - {comparator} win rate", float(np.mean(values > 0)), win_rate, 5e-4)
    for method, expected in EXPECTED_HVHIT.items():
        actual = tuple(float(value) for value in data["panel_c"]["methods"][method])
        if actual != expected:
            raise AssertionError(f"{method} HvHit: {actual} != expected {expected}")


def _panel_letter(fig, box, letter: str, x_in: float | None = None) -> None:
    fig_text(
        fig,
        max(0.04, box[0] - 0.43) if x_in is None else x_in,
        box[1] + box[3] + 0.10,
        letter,
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def draw_a(fig, data: dict):
    """Method comparison: per-row distributions with restrained summary glyphs."""
    ax = place(fig, *A_BOX)
    entries = data["panel_a"]["methods"]
    all_values = np.concatenate(
        [np.asarray(entries[method]["values"], dtype=float) for method in METHOD_ORDER]
    )
    span = float(all_values.max() - all_values.min())
    xlim = (float(all_values.min() - 0.035 * span), float(all_values.max() + 0.035 * span))
    ax.set_xlim(*xlim)
    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    ax.set_ylim(-0.34, len(METHOD_ORDER) - 0.58)
    ax.axvline(0.0, color=GUIDE, lw=0.65, ls=(0, (2, 2)), zorder=0)

    for method, y in zip(METHOD_ORDER, y_positions):
        values = np.asarray(entries[method]["values"], dtype=float)
        summary = entries[method]["summary"]
        color, alpha = METHOD_STYLE[method]
        grid = np.linspace(float(values.min()), float(values.max()), 300)
        density = gaussian_kde(values)(grid)
        density = density / density.max() * 0.29
        ax.fill_between(grid, y, y + density, color=color, alpha=alpha, lw=0, zorder=1)
        ax.plot(grid, y + density, color=color, alpha=0.72, lw=0.55, zorder=2)
        ax.plot(
            [summary["p05"], summary["p95"]],
            [y, y],
            color=color,
            lw=0.75,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            [summary["q25"], summary["q75"]],
            [y, y],
            color=color,
            lw=2.5,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.plot(
            summary["median"],
            y,
            "o",
            ms=4.0 if method == "VCDesign-CED" else 3.4,
            mfc=color,
            mec="white",
            mew=0.45,
            zorder=5,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Goal-blind\nprior", "Forward-then-\nmatch", "Base scorer", "Text-augmented\nranker", "VCDesign-CED"], fontsize=7.0)
    for label, method in zip(ax.get_yticklabels(), METHOD_ORDER):
        label.set_color(PRIMARY_TEXT if method == "VCDesign-CED" else INK)
        label.set_fontweight("bold" if method == "VCDesign-CED" else "normal")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel(r"Mean budgeted utility", labelpad=2)
    ax.spines["left"].set_visible(False)
    _panel_letter(fig, A_BOX, "a", x_in=0.08)
    return ax


def draw_b(fig, data: dict):
    """ECDFs expose the complete paired row-level gain structure."""
    ax = place(fig, *B_BOX)
    comparisons = data["panel_b"]["comparisons"]
    styles = {
        "Base scorer": (PRIMARY, 1.25, "vs. Base"),
        "Text-augmented ranker": (TEXT, 0.95, "vs. Text"),
    }
    all_values = []
    for comparator in ("Base scorer", "Text-augmented ranker"):
        values = np.sort(np.asarray(comparisons[comparator]["values"], dtype=float))
        all_values.append(values)
        y = np.arange(1, values.size + 1) / values.size
        color, linewidth, label = styles[comparator]
        ax.step(values, y, where="post", color=color, lw=linewidth, label=label, zorder=2)
    combined = np.concatenate(all_values)
    span = float(combined.max() - combined.min())
    ax.set_xlim(float(combined.min() - 0.025 * span), float(combined.max() + 0.025 * span))
    ax.set_ylim(0.0, 1.0)
    ax.axvline(0.0, color=GUIDE, lw=0.65, zorder=0)
    ax.set_xlabel(r"CED gain, $\Delta\overline{\mathrm{BU}}$", labelpad=2)
    ax.set_ylabel("Cumulative fraction", labelpad=2)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.15, labelspacing=0.25)

    for comparator, dx, dy in (
        ("Base scorer", 0.035, -0.070),
        ("Text-augmented ranker", 0.100, -0.035),
    ):
        entry = comparisons[comparator]
        color = styles[comparator][0]
        cumulative_at_zero = 1.0 - float(entry["win_rate"])
        ax.text(
            dx,
            cumulative_at_zero + dy,
            f"{entry['win_rate']:.1%}",
            color=color,
            fontsize=6.5,
            fontweight="bold" if comparator == "Base scorer" else "normal",
            ha="left",
            va="center",
        )
    _panel_letter(fig, B_BOX, "b")
    return ax


def draw_c(fig, data: dict):
    """Discrete experimental budgets translated into high-value selections."""
    ax = place(fig, *C_BOX)
    x = np.arange(3, dtype=float)
    budgets = data["panel_c"]["budgets"]
    styles = {
        "Base scorer": (BASE, 0.70, "o", 2.5),
        "Text-augmented ranker": (TEXT, 0.78, "s", 2.5),
        "VCDesign-CED": (PRIMARY, 1.15, "o", 3.2),
    }
    for method in ("Base scorer", "Text-augmented ranker", "VCDesign-CED"):
        color, linewidth, marker, size = styles[method]
        y = np.asarray(data["panel_c"]["methods"][method], dtype=float)
        ax.plot(
            x,
            y,
            color=color,
            lw=linewidth,
            marker=marker,
            ms=size,
            mfc=color,
            mec=color,
            zorder=3 if method == "VCDesign-CED" else 2,
        )

    handles = ax.get_lines()
    ax.legend(
        [handles[2], handles[1], handles[0]],
        ["VCDesign-CED", "Text", "Base"],
        loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3,
    )
    ax.set_xlim(-0.12, 2.15)
    ax.set_ylim(0.0, 13.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in budgets])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_xlabel(r"Experimental budget, $B$", labelpad=2)
    ax.set_ylabel(r"High-value hits in top $B$", labelpad=2)
    _panel_letter(fig, C_BOX, "c")
    return ax


def draw_external(fig, rows):
    """Context-specific and pooled comparisons, from the published source table.

    Failed candidate-invariant GEARS fits are not performance estimates and
    are deliberately excluded. Missing MASKED arms are never represented as zero.
    """
    lookup = {(r["regime"], r["method_id"]): r for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError("duplicate method/regime in external source table")
    for method, _ in BENCHMARK_METHODS:
        for context in ("rpe1", "hepg2", "jurkat", "pooled"):
            if not np.isfinite(float(lookup[("SEEN", method)][context])):
                raise AssertionError(f"non-finite external level: {method}/{context}")
    d = place(fig, *D_BOX)
    matrix = np.array([[float(lookup[("SEEN", m)][c])
                        for c in ("rpe1", "hepg2", "jurkat")]
                       for m, _ in BENCHMARK_METHODS])
    cmap = LinearSegmentedColormap.from_list("design_utility", ["#d4c4e9", "#ffffff", "#0f777b"])
    norm = TwoSlopeNorm(vmin=-0.20, vcenter=0, vmax=0.70)
    d.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rgba = cmap(norm(matrix[i, j]))
            rgb = np.array(rgba[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            lum = float(linear @ [.2126, .7152, .0722])
            d.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                   fontsize=6.5, color="white" if lum < .179 else INK)
    d.set_yticks(range(len(BENCHMARK_METHODS)), [label for _, label in BENCHMARK_METHODS])
    for label, (method, _) in zip(d.get_yticklabels(), BENCHMARK_METHODS):
        label.set_fontsize(7)
        if method == "vcdesign_ced":
            label.set_color(PRIMARY_TEXT)
            label.set_fontweight("bold")
    d.set_xticks(range(3), ["RPE1", "HepG2", "Jurkat"])
    d.tick_params(length=0)
    d.set_xlabel("Response available: BU@20", labelpad=5)
    for spine in d.spines.values():
        spine.set_visible(False)
    _panel_letter(fig, D_BOX, "d", x_in=.08)

    e = place(fig, *E_BOX)
    for i, (method, _) in enumerate(BENCHMARK_METHODS):
        for regime, color, marker, dy in [("MASKED", PRIMARY, "o", -.10),
                                           ("SEEN", "#7a58c1", "s", .10)]:
            entry = lookup.get((regime, method))
            if entry is not None:
                e.plot(float(entry["pooled"]), i + dy, marker=marker, color=color,
                       ms=3.5, mfc="white" if regime == "MASKED" else color,
                       mew=.8, linestyle="none")
    e.axvline(0, color=GUIDE, lw=.65, ls=(0, (2, 2)))
    e.set_ylim(5.5, -.5)
    e.set_xlim(-.07, .75)
    e.set_xticks([0, .3, .6])
    e.set_yticks(range(6), [])
    e.tick_params(axis="y", length=0)
    e.spines["left"].set_visible(False)
    e.set_xlabel("Pooled BU@20", labelpad=5)
    for regime, label, color, marker in [
        ("MASKED", "Response unseen", PRIMARY, "o"),
        ("SEEN", "Response available", "#7a58c1", "s")]:
        e.plot([], [], marker=marker, color=color, linestyle="none", ms=3.5,
               mfc="white" if regime == "MASKED" else color, label=label)
    e.legend(loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
             handlelength=1, labelspacing=.25)
    _panel_letter(fig, E_BOX, "e", x_in=3.29)
    return d, e


def build(data: dict, external_rows=None):
    validate(data)
    # Keep the three original panels aligned and unchanged in physical size.
    # The optional benchmark row adds distinct contexts and information regimes.
    fig = plt.figure(figsize=(FIG_W, FIG_H + (2.45 if external_rows else 0)))
    axes = {
        "a": draw_a(fig, data),
        "b": draw_b(fig, data),
        "c": draw_c(fig, data),
    }
    if external_rows:
        for ax in axes.values():
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + 2.45 / 4.65, pos.width, pos.height])
        for text in fig.texts:
            x, y = text.get_position()
            text.set_position((x, y + 2.45 / 4.65))
        axes["d"], axes["e"] = draw_external(fig, external_rows)
    return fig, axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output path without suffix")
    parser.add_argument("--audit-scripts", type=Path, default=None)
    parser.add_argument("--external-levels", type=Path, default=None,
                        help="published external_baseline_levels.csv for benchmark panels d/e")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    external_rows = None
    if args.external_levels:
        with args.external_levels.open(newline="") as stream:
            external_rows = list(csv.DictReader(stream))
    fig, axes = build(data, external_rows)
    if args.audit_scripts is not None:
        import sys

        sys.path.insert(0, str(args.audit_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        report = require_matplotlib_panel_alignment(
            fig,
            axes=list(axes.values()),
            panel_ids=list(axes.keys()),
            row_groups=[["a", "b", "c"]] + ([["d", "e"]] if external_rows else []),
            exemptions=[
                {
                    "panels": ["a", "b", "c"],
                    "checks": ["panel-label"],
                    "reason": "lowercase letters are figure text placed at fixed physical offsets",
                }
            ],
        )
        print("panel alignment:", report.get("verdict"), report.get("summary"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    fig.savefig(args.out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
