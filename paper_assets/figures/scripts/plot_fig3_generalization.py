#!/usr/bin/env python3
"""Render Figure 3: solver regime, controlled masking, and mixed deployment.

Every plotted value comes from ``fig3_generalization_data.json`` and uses the same
RIDGE_UNIT estimator, BU@20 metric, eligible pools, queries, and base scorer.
"""
from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from figstyle import GUIDE, INK, apply_figure_style, fig_text, place


apply_figure_style()

FIG_WIDTH_MM = 139.7
FIG_W, FIG_H = 5.50, 3.55  # 139.7 x 90.2 mm
A_BOX = (0.48, 0.43, 2.65, 2.76)
B_BOX = (3.72, 1.88, 1.53, 1.25)
C_BOX = (3.72, 0.38, 1.53, 0.92)

PRIMARY = "#0f777b"
PRIMARY_TEXT = "#0b6266"
BASE = "#42484a"
FORWARD = "#9aa4a9"
RETRIEVAL = "#c47635"
CELLNAVI = "#a69bbb"
CI = "#73afb1"
MUTED = "#697174"

INTERNAL_FTM = "INTERNAL_FORWARD_THEN_MATCH"
BASE_ID = "BASE"
VCDESIGN = "VCDESIGN_BASE_PLUS_RIDGE_UNIT"
CELLNAVI_ID = "CELLNAVI_NATIVE_SCALE"
PROFILED = "PROFILED_SIGNATURE_RETRIEVAL"
HELD = ("RPE1", "HepG2", "Jurkat")
STATES = ("full_measured", "size_controlled", "response_unseen")

EXPECTED_LEVELS = {
    ("MASKED", INTERNAL_FTM): 0.2935704126698596,
    ("SEEN", INTERNAL_FTM): 0.3958669537321793,
    ("MASKED", BASE_ID): 0.43959549177214163,
    ("SEEN", BASE_ID): 0.43959549177214163,
    ("MASKED", VCDESIGN): 0.4989576535408368,
    ("SEEN", VCDESIGN): 0.5075526284382246,
    ("SEEN", CELLNAVI_ID): 0.015011397473915482,
    ("SEEN", PROFILED): 0.6746711115615611,
}
EXPECTED_CONTRASTS = {
    "MASKED|VCDesign-CED_minus_Base": 0.049201783576071106,
    "SEEN|Profile_retrieval_minus_VCDesign-CED": 0.15193164983577273,
}


def assert_close(label: str, observed: float, expected: float, atol: float = 1e-12) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {observed:.17g} != {expected:.17g}")


def validate(data: dict) -> None:
    if data.get("schema") != "VCDESIGN_FIG3_DATA_V2":
        raise AssertionError(f"unexpected Figure 3 schema: {data.get('schema')}")
    estimator = data["contract"]["estimator"]
    if estimator["id"] != "RIDGE_UNIT" or estimator["metric"] != "BU@20":
        raise AssertionError("Figure 3 mixes estimator or metric variants")
    for (regime, method), expected in EXPECTED_LEVELS.items():
        value = data["panel_a"]["methods"][method]["levels"][regime]
        assert_close(f"{regime} {method}", value, expected)
    for key, expected in EXPECTED_CONTRASTS.items():
        assert_close(key, data["panel_a"]["headline_contrasts"][key]["median"], expected)
    if not all(data["mandatory_assertions"].values()):
        raise AssertionError(f"mandatory assertions failed: {data['mandatory_assertions']}")
    for held in HELD:
        points = data["panel_c"]["contexts"][held]["points"]
        if [point["share"] for point in points] != [0.0, 0.25, 0.5, 0.75, 1.0]:
            raise AssertionError(f"{held}: mixed-deployment shares changed")


def panel_letter(fig, box, letter: str, x: float | None = None, y: float | None = None) -> None:
    fig_text(
        fig,
        box[0] - 0.39 if x is None else x,
        box[1] + box[3] + 0.08 if y is None else y,
        letter,
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def vertical_delta(ax, x: float, low: float, high: float, label: str, color: str) -> None:
    tick = 0.025
    ax.plot([x, x], [low, high], color=color, lw=0.65, zorder=1)
    ax.plot([x - tick, x + tick], [low, low], color=color, lw=0.65, zorder=1)
    ax.plot([x - tick, x + tick], [high, high], color=color, lw=0.65, zorder=1)
    ax.text(
        x + 0.04,
        (low + high) / 2,
        label,
        color=color,
        fontsize=6.5,
        ha="left",
        va="center",
    )


def draw_a(fig, data: dict):
    """Hero panel: a two-regime solver landscape."""
    ax = place(fig, *A_BOX)
    methods = data["panel_a"]["methods"]
    common = (
        (INTERNAL_FTM, FORWARD, 0.75, 3.2),
        (BASE_ID, BASE, 0.85, 3.4),
        (VCDESIGN, PRIMARY, 1.35, 4.0),
    )
    for method, color, linewidth, marker_size in common:
        values = [methods[method]["levels"][regime] for regime in ("MASKED", "SEEN")]
        ax.plot(
            [0, 1], values, color=color, lw=linewidth, marker="o", ms=marker_size,
            mfc=color, mec="white", mew=0.45, zorder=4 if method == VCDESIGN else 3,
        )

    seen_only = (
        (PROFILED, RETRIEVAL, 3.8),
        (CELLNAVI_ID, CELLNAVI, 3.0),
    )
    for method, color, marker_size in seen_only:
        ax.plot(
            [1], [methods[method]["levels"]["SEEN"]], "o", color=color,
            ms=marker_size, mec="white", mew=0.45, zorder=4,
        )

    labels = {
        PROFILED: ("Profile retrieval", RETRIEVAL, 0.681),
        VCDESIGN: ("VCDesign-CED", PRIMARY_TEXT, 0.514),
        BASE_ID: ("Base", BASE, 0.448),
        INTERNAL_FTM: ("Forward-match", MUTED, 0.385),
        CELLNAVI_ID: ("CellNavi", CELLNAVI, 0.015),
    }
    for method, (label, color, target_y) in labels.items():
        value = methods[method]["levels"]["SEEN"]
        if abs(target_y - value) > 0.008:
            ax.plot([1.02, 1.08], [value, target_y], color=color, lw=0.5, clip_on=False)
        ax.text(
            1.11, target_y, label, color=color, fontsize=6.5,
            fontweight="bold" if method in (VCDESIGN, PROFILED) else "normal",
            ha="left", va="center", clip_on=False,
        )

    masked_delta = data["panel_a"]["headline_contrasts"][
        "MASKED|VCDesign-CED_minus_Base"
    ]["median"]
    retrieval_delta = data["panel_a"]["headline_contrasts"][
        "SEEN|Profile_retrieval_minus_VCDesign-CED"
    ]["median"]
    vertical_delta(
        ax, -0.08, methods[BASE_ID]["levels"]["MASKED"],
        methods[VCDESIGN]["levels"]["MASKED"], f"+{masked_delta:.3f}", PRIMARY_TEXT,
    )
    vertical_delta(
        ax, 0.80, methods[VCDESIGN]["levels"]["SEEN"],
        methods[PROFILED]["levels"]["SEEN"], f"+{retrieval_delta:.3f}", RETRIEVAL,
    )

    ax.axhline(0.0, color=GUIDE, lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-0.16, 1.46)
    ax.set_ylim(-0.035, 0.735)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MASKED", "SEEN"], fontsize=7.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("BU@20", labelpad=2)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.spines["bottom"].set_visible(False)
    panel_letter(fig, A_BOX, "a", x=0.08)
    return ax


def draw_b(fig, data: dict):
    """Three directly scored information states, with query-gene bootstrap intervals."""
    ax = place(fig, *B_BOX)
    rows = ("RPE1", "HepG2", "Jurkat", "Pooled")
    y_positions = dict(zip(rows, (3.0, 2.0, 1.0, 0.0)))
    markers = {
        "full_measured": ("o", "white", PRIMARY),
        "size_controlled": ("D", PRIMARY, PRIMARY),
        "response_unseen": ("o", PRIMARY, "white"),
    }
    for context in rows:
        entry = data["panel_b"]["pooled"] if context == "Pooled" else data["panel_b"]["contexts"][context]["states"]
        y = y_positions[context]
        values = [entry[state]["median"] for state in STATES]
        ax.plot(
            values, [y] * len(values), color=CI if context != "Pooled" else PRIMARY,
            lw=0.65 if context != "Pooled" else 0.95, zorder=1,
        )
        for state in STATES:
            summary = entry[state]
            ax.plot(
                summary["ci95"], [y, y], color=CI, lw=0.55,
                solid_capstyle="butt", zorder=2,
            )
        # Size control and full atlas can be nearly identical. Draw the diamond
        # first and the smaller filled circle above it so both states remain visible.
        for state in ("size_controlled", "full_measured", "response_unseen"):
            summary = entry[state]
            marker, edge, face = markers[state]
            ax.plot(
                [summary["median"]], [y], marker=marker,
                ms=(4.0 if state == "size_controlled" else 3.5) if context == "Pooled"
                else (3.3 if state == "size_controlled" else 2.9),
                mec=edge, mfc=face, mew=0.6 if state == "full_measured" else 0.8,
                color=PRIMARY, zorder=4 if state == "full_measured" else 3,
            )

    ax.set_xlim(0.028, 0.086)
    ax.set_ylim(-0.45, 3.45)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(rows)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks([0.04, 0.06, 0.08])
    ax.set_xlabel(r"$\Delta$BU@20 over Base", labelpad=2)
    ax.spines["left"].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", ms=3.1, mec="white", mfc=PRIMARY, mew=0.6,
               label="full"),
        Line2D([], [], marker="D", ls="", ms=3.1, mec=PRIMARY, mfc=PRIMARY, mew=0.8,
               label="size-control"),
        Line2D([], [], marker="o", ls="", ms=3.3, mec=PRIMARY, mfc="white", mew=0.8,
               label="unseen"),
    ]
    fig.legend(
        handles=handles, ncol=3, loc="upper left",
        bbox_to_anchor=((B_BOX[0] - 0.03) / FIG_W, (B_BOX[1] + B_BOX[3] + 0.11) / FIG_H),
        fontsize=6.0, handletextpad=0.15, columnspacing=0.35,
        borderaxespad=0.0, handlelength=0.65,
    )
    panel_letter(fig, B_BOX, "b", x=3.32, y=3.24)
    return ax


def draw_c(fig, data: dict):
    """Nested mixed-information candidate pools at five observed compositions."""
    ax = place(fig, *C_BOX)
    styles = {
        "RPE1": ("o", 3.1),
        "HepG2": ("s", 3.0),
        "Jurkat": ("^", 3.2),
    }
    ends = {}
    for context in HELD:
        points = data["panel_c"]["contexts"][context]["points"]
        x = np.asarray([point["share"] for point in points])
        y = np.asarray([point["median"] for point in points])
        marker, marker_size = styles[context]
        ax.plot(
            x, y, color=PRIMARY, lw=0.75, marker=marker, ms=marker_size,
            mfc="white", mec=PRIMARY, mew=0.75, zorder=2,
        )
        ends[context] = float(y[-1])

    targets = {"HepG2": 0.0705, "RPE1": 0.0530, "Jurkat": 0.0390}
    for context in HELD:
        ax.plot([1.02, 1.08], [ends[context], targets[context]], color=PRIMARY, lw=0.5,
                clip_on=False)
        ax.text(1.10, targets[context], context, color=PRIMARY_TEXT, fontsize=6.0,
                ha="left", va="center", clip_on=False)

    ax.set_xlim(-0.04, 1.31)
    ax.set_ylim(0.034, 0.076)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks([0.04, 0.06])
    ax.set_xlabel("Response-unseen candidates (%)", labelpad=2)
    ax.set_ylabel(r"$\Delta$BU@20 over Base", labelpad=2)
    panel_letter(fig, C_BOX, "c", x=3.32)
    return ax


def build(data: dict):
    validate(data)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = {"a": draw_a(fig, data), "b": draw_b(fig, data), "c": draw_c(fig, data)}
    return fig, axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output path without suffix")
    parser.add_argument("--audit-scripts", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    fig, axes = build(data)
    if args.audit_scripts is not None:
        import sys

        sys.path.insert(0, str(args.audit_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        report = require_matplotlib_panel_alignment(
            fig,
            axes=list(axes.values()),
            panel_ids=list(axes.keys()),
            column_groups=[["b", "c"]],
            exemptions=[
                {
                    "panels": ["a", "b", "c"],
                    "checks": ["panel-label"],
                    "reason": "lowercase panel letters are figure text outside the axes boxes",
                },
                {
                    "panels": ["a"],
                    "checks": ["row", "column"],
                    "reason": "the hero panel deliberately spans the two stacked support panels",
                },
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
