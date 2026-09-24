#!/usr/bin/env python3
"""Render the four-panel Figure 4 insight figure from its frozen V2 data artifact.

The panels deliberately retain their own metrics and references. This script never compares
effect sizes across panels; it only renders the controlled contrasts supplied by
build_fig4_data.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import Bbox

from figstyle import (
    CONTEXT,
    GUIDE,
    INK,
    MEASURED,
    MEASURED_TEXT,
    MUTED,
    PRIMARY,
    S_BASE,
    S_LETTER,
    S_NOTE,
    data_y_to_inches,
    fig_text,
    place,
    set_ticks,
)


FIG_W, FIG_H = 5.50, 4.05  # 139.7 x 102.9 mm
FIG_WIDTH_MM = 139.7
GRAPHITE = "#4b4b4b"
GRID_POINT = "#c9cdd1"
CAPACITY = "#81798a"
CAPACITY_TEXT = "#625b69"
INFORMATION = "#568f91"
INFORMATION_TEXT = "#356f72"

TOP_Y, TOP_H = 2.34, 1.30
BOTTOM_Y, BOTTOM_H = 0.43, 1.28
A_BOX = (0.58, TOP_Y, 1.46, TOP_H)
B_BOX = (2.55, TOP_Y, 2.60, TOP_H)
C_BOX = (1.01, BOTTOM_Y, 1.06, BOTTOM_H)
D_GLOBAL_BOX = (2.53, BOTTOM_Y, 1.36, BOTTOM_H)
D_INSET_BOX = (4.20, BOTTOM_Y, 1.16, BOTTOM_H)


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def panel_letter(fig, x_in: float, y_in: float, letter: str) -> None:
    fig_text(fig, x_in, y_in, letter, fontsize=S_LETTER, fontweight="bold", color=INK,
             ha="left", va="center")


def draw_a(fig, data: dict):
    """Two-track substitution of candidate information within one factorial probe."""
    panel = data["panel_a"]
    ax = place(fig, *A_BOX)
    ax.set_xlim(0.09, 0.69)
    ax.set_ylim(-0.38, 1.43)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Learned\nquery", "Raw\ntransition"], linespacing=1.25)
    set_ticks(ax.xaxis, (0.2, 0.4, 0.6))
    ax.set_xlabel(r"Probe $\overline{\mathrm{BU}}$")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)

    rows = (("Q_CURRENT", 1), ("Q_RAW", 0))
    for query, y in rows:
        static = panel["cells"][f"{query}__C_LEGAL"]
        measured = panel["cells"][f"{query}__C_ORACLE"]
        ax.plot([static, measured], [y, y], color="#9c9c9c", lw=1.0, zorder=1)
        ax.plot(static, y, "o", ms=5.1, mfc=GRAPHITE, mec=GRAPHITE, mew=0.8, zorder=3)
        ax.plot(measured, y, "o", ms=5.5, mfc="white", mec=MEASURED, mew=1.25, zorder=3)
        ax.text((static + measured) / 2, y + 0.14, signed(measured - static, 2),
                fontsize=S_NOTE, color=MUTED, ha="center", va="bottom")

    ax.text(panel["cells"]["Q_CURRENT__C_LEGAL"], 1.28, "Static\nknowledge",
            fontsize=S_NOTE, color=GRAPHITE, ha="center", va="bottom", linespacing=1.25)
    ax.text(panel["cells"]["Q_CURRENT__C_ORACLE"], 1.28, "Measured\nresponse†",
            fontsize=S_NOTE, color=MEASURED_TEXT, ha="center", va="bottom", linespacing=1.25)
    return ax


def draw_b(fig, data: dict):
    """Same identities, fixed-context measurement intervention."""
    panel = data["panel_b"]
    ax = place(fig, *B_BOX)
    ax.set_xlim(-0.12, 1.33)
    ax.set_ylim(0.0, 0.061)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$M$ spread", r"$3M$"])
    set_ticks(ax.yaxis, (0.0, 0.02, 0.04, 0.06))
    ax.set_xlabel("Atlas measurements")
    ax.set_ylabel(r"Gain over base, $\Delta\overline{\mathrm{BU}}$")
    ax.spines["bottom"].set_bounds(0, 1)

    label_offsets = {"RPE1": -0.0016, "HepG2": 0.0016, "Jurkat": 0.0}
    for context in ("RPE1", "HepG2", "Jurkat"):
        entry = panel["contexts"][context]
        ys = [entry["M_spread"], entry["three_M"]]
        color = CONTEXT[context]
        ax.plot([0, 1], ys, color=color, lw=1.15, zorder=2)
        ax.plot([0, 1], ys, "o", ms=4.5, color=color, mec="white", mew=0.55, zorder=3)
        label_y = ys[1] + label_offsets[context]
        ax.plot([1.0, 1.055], [ys[1], label_y], color=color, lw=0.6, clip_on=False)
        ax.text(1.07, label_y, context, fontsize=S_BASE, color=color, ha="left", va="center")
    return ax


def draw_c(fig, data: dict):
    """Zero-centred capacity and candidate-information interventions."""
    panel = data["panel_c"]
    ax = place(fig, *C_BOX)
    ax.set_xlim(-0.030, 0.007)
    ax.set_ylim(-0.55, 6.05)
    set_ticks(ax.xaxis, (-0.03, -0.02, -0.01, 0.0))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color="#aeb2b5", lw=0.7, zorder=0)
    ax.set_xlabel(r"$\Delta$BU@20 vs unit-direction ridge", labelpad=3)

    labels = {
        "Predictive response basis": "Predictive basis",
        "Context-invariant adapter": "Invariant adapter",
        "Nonlinear memory, anchored": "Anchored memory",
        "Nonlinear effect memory": "Nonlinear memory",
        "Drop MAP-KG (STRING only)": "STRING only",
        "Drop STRING (MAP-KG only)": "MAP-KG only",
    }
    y_positions = [5.45, 4.55, 3.65, 2.75, 1.35, 0.45]
    to_in = data_y_to_inches(ax, C_BOX)
    for item, y in zip(panel["variants"], y_positions):
        is_information = item["group"] == "information"
        color = INFORMATION if is_information else CAPACITY
        text_color = INFORMATION_TEXT if is_information else CAPACITY_TEXT
        ax.plot(item["ci95"], [y, y], color=color, lw=1.15, solid_capstyle="butt", zorder=2)
        ax.plot(item["median"], y, "o", ms=4.4, mfc=color, mec="white", mew=0.5, zorder=3)
        fig_text(fig, 0.94, to_in(y), labels[item["label"]], fontsize=S_NOTE,
                 color=text_color, ha="right", va="center")
    return ax


def _controlled_global_points(panel: dict) -> list[dict]:
    wanted = ("[full]", "[rank64]", "[rank128]", "[rank512]", "[STRING_only]",
              "[MAPKG_only]")
    return [point for point in panel["grid_points"]
            if any(token in point["estimator"] for token in wanted)]


def draw_d(fig, data: dict):
    """Global prediction-design association and within-atlas centred changes."""
    panel = data["panel_d"]
    global_ax = place(fig, *D_GLOBAL_BOX)
    grid = panel["grid_points"]
    global_ax.scatter([p["prediction"] for p in grid],
                      [p["design_effect_only_BU@20"] for p in grid],
                      s=9, c=GRID_POINT, edgecolors="none", zorder=1)
    global_ax.set_xlim(-0.065, 0.175)
    global_ax.set_ylim(0.0, 0.425)
    set_ticks(global_ax.xaxis, (-0.05, 0.0, 0.05, 0.10, 0.15))
    set_ticks(global_ax.yaxis, (0.0, 0.2, 0.4))
    global_ax.set_xlabel("Held-context prediction\n(candidate-specific cosine)", labelpad=2)
    global_ax.set_ylabel("Effect-only BU@20", labelpad=2)
    fig_text(fig, 2.58, 1.78, rf"$\rho={panel['spearman']:.2f}$",
             fontsize=S_NOTE, color=MUTED, ha="left", va="bottom")

    controlled = _controlled_global_points(panel)
    xs = [p["prediction"] for p in controlled]
    ys = [p["design_effect_only_BU@20"] for p in controlled]
    margin_x, margin_y = 0.006, 0.018
    x0, x1 = min(xs) - margin_x, max(xs) + margin_x
    y0, y1 = min(ys) - margin_y, max(ys) + margin_y
    global_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  edgecolor=GRAPHITE, lw=0.65, zorder=3))
    full = [p for p in controlled if p["estimator"].endswith("[full]")]
    global_ax.scatter([p["prediction"] for p in full],
                      [p["design_effect_only_BU@20"] for p in full],
                      marker="x", s=17, linewidths=0.9, color=GRAPHITE, zorder=4)

    inset_ax = place(fig, *D_INSET_BOX)
    changes = panel["controlled_changes"]
    inset_ax.axhline(0, color=GUIDE, lw=0.65, zorder=0)
    inset_ax.axvline(0, color=GUIDE, lw=0.65, zorder=0)
    inset_ax.scatter([p["d_prediction"] for p in changes], [p["d_design"] for p in changes],
                     s=13, c=PRIMARY, edgecolors="white", linewidths=0.35, zorder=2)
    inset_ax.scatter([0], [0], marker="x", s=22, linewidths=1.0, color=GRAPHITE, zorder=3)
    inset_ax.set_xlim(-0.043, 0.012)
    inset_ax.set_ylim(-0.255, 0.052)
    set_ticks(inset_ax.xaxis, (-0.04, -0.02, 0.0))
    set_ticks(inset_ax.yaxis, (-0.2, -0.1, 0.0))
    inset_ax.set_xlabel("Δ prediction", labelpad=2)
    inset_ax.set_ylabel("Δ design", labelpad=4)
    inset_ax.text(0.04, 1.05,
                  f"{panel['same_sign']['agree']}/{panel['same_sign']['pairs']} same sign",
                  transform=inset_ax.transAxes, fontsize=S_NOTE, color=INK,
                  fontweight="bold", ha="left", va="top")

    for y_global, y_inset in ((y1, 0.95), (y0, 0.05)):
        fig.add_artist(ConnectionPatch(xyA=(x1, y_global), coordsA=global_ax.transData,
                                       xyB=(0.0, y_inset), coordsB=inset_ax.transAxes,
                                       color="#8d9194", lw=0.55, zorder=0, clip_on=False))
    return global_ax, inset_ax


def build(data: dict):
    if data.get("schema") != "VCDESIGN_FIG4_DATA_V2":
        raise RuntimeError(f"unexpected Figure 4 schema: {data.get('schema')}")
    if not all(data["mandatory_assertions"].values()):
        raise RuntimeError("Figure 4 audit assertions did not all pass")
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = {
        "a": draw_a(fig, data),
        "b": draw_b(fig, data),
        "c": draw_c(fig, data),
    }
    axes["d"], axes["d-inset"] = draw_d(fig, data)
    panel_letter(fig, 0.05, 3.86, "a")
    panel_letter(fig, 2.24, 3.86, "b")
    panel_letter(fig, 0.05, 1.90, "c")
    panel_letter(fig, 2.24, 1.90, "d")
    return fig, axes


def export_panels(fig, output_dir: Path) -> None:
    """Export editable panel crops from the composed figure for downstream layout work."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = {
        "a": (0.00, 2.17, 2.17, 4.02),
        "b": (2.18, 2.17, 5.48, 4.02),
        "c": (0.00, 0.08, 2.17, 2.06),
        "d": (2.18, 0.08, 5.48, 2.06),
    }
    for panel, bounds in crops.items():
        bbox = Bbox.from_extents(*bounds)
        fig.savefig(output_dir / f"fig4{panel}.pdf", bbox_inches=bbox, pad_inches=0)
        fig.savefig(output_dir / f"fig4{panel}.svg", bbox_inches=bbox, pad_inches=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output path without suffix")
    parser.add_argument("--panels-dir", type=Path)
    parser.add_argument("--audit-scripts", type=Path)
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
            row_groups=[["a", "b"], ["c", "d", "d-inset"]],
            exemptions=[{
                "panels": list(axes),
                "checks": ["panel-label"],
                "reason": "lowercase panel letters are figure text outside the axes",
            }, {
                "panels": ["c", "d", "d-inset"],
                "checks": ["panel-width", "horizontal-gutter"],
                "reason": "the specified asymmetric layout makes d a global plot plus inset",
            }],
        )
        print("panel alignment:", report.get("verdict"), report.get("summary"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    fig.savefig(args.out.with_suffix(".png"), dpi=600)
    if args.panels_dir is not None:
        export_panels(fig, args.panels_dir)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
