#!/usr/bin/env python3
"""Figure 2 (display v2): intervention-selection performance.

Top row, held-out cellular contexts (unit-direction CED, BU@20):
  a  per-context levels of every evaluated solver in both information regimes
  b  paired gain of VCDesign-CED over the identical base scorer, with query-gene intervals
Bottom row, held-out K562 identities (raw-target CED, the locked mean-BU endpoint):
  c  row-level levels and paired effects (estimation plot)
  d  paired gain in high-value candidates selected, by budget

Reads only figures/data/display_v2_data.json (built by build_display_v2_data.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

import figstyle_v2 as st

H = 3.86
CONTEXTS = st.CONTEXTS

# Vertical budget (inches from the bottom of the canvas).
HEAD_1 = H - 0.15            # row-1 panel headings (baseline)
KEY_Y = H - 0.36             # regime key
TOP_A = H - 0.62             # top of the panel-a plotting area (facet titles sit above it)
PITCH, GROUP_GAP = 0.146, 0.12
ROW2_TOP = 1.36              # top of the row-2 plotting areas
ROW2_Y0, ROW2_H = 0.42, 0.94
HEAD_2 = ROW2_TOP + 0.14     # row-2 panel headings (baseline)

# Panel a rows, top to bottom: (method id, label, colour, emphasis); None starts a group.
GROUPS = [
    ("No response needed", [
        ("goal_blind_prior", "Goal-blind prior", st.GOAL_BLIND, False),
        ("internal_forward_then_match", "CED effect only", st.EFFECT_ONLY, False),
        ("base", "Base scorer (no atlas)", st.BASE, False),
        ("vcdesign_ced", "VCDesign-CED", st.CED, True)]),
    ("Needs measured response", [
        ("cellnavi_official", "CellNavi, official", st.CELLNAVI, False),
        ("cellnavi_native", "CellNavi, native", st.CELLNAVI, False),
        ("profiled_signature_retrieval", "Profile retrieval", st.RETRIEVAL, False)]),
]
DODGE = 0.046  # inches between the masked (upper) and measured (lower) marker of one row
LABEL_X = 1.06
FACET_L, FACET_W, FACET_GAP = 1.16, 0.79, 0.16
ZERO_A = "#dfe3e8"  # a quiet zero line: facets, not the axis origin, should separate contexts
XLIM_A = (-0.16, 0.76)


def rows_a():
    """Row centres (inches) for panel a, the group-header positions, and the bottom edge."""
    y = TOP_A
    rows, heads = [], []
    for gi, (gname, members) in enumerate(GROUPS):
        if gi:
            y -= GROUP_GAP
        heads.append((gname, y - 0.01))
        for m in members:
            y -= PITCH
            rows.append((m, y + PITCH / 2))
    return rows, heads, y


def panel_a(fig, d):
    levels = d["fig2_a"]["methods"]
    rows, heads, bottom = rows_a()
    facet_r = FACET_L + 3 * FACET_W + 2 * FACET_GAP

    # Row bands are drawn per block (label column and each facet) so white gutters separate
    # the three held contexts.
    blocks = [(0.02, FACET_L - 0.07)] + [
        (FACET_L + i * (FACET_W + FACET_GAP), FACET_W) for i in range(len(CONTEXTS))]
    for bx, bw in blocks:
        band = st.place(fig, bx, bottom, bw, TOP_A - bottom)
        band.set_axis_off()
        band.set_xlim(0, 1)
        band.set_ylim(bottom, TOP_A)
        for gname, members in GROUPS:
            for k, m in enumerate(members):
                y = dict((mm[0], yy) for mm, yy in rows)[m[0]]
                if k % 2 == 0:
                    band.axhspan(y - PITCH / 2, y + PITCH / 2, color=st.BAND, lw=0, zorder=0)

    for (mid, label, color, bold), y in rows:
        st.text_in(fig, LABEL_X, y, label, fontsize=st.S_LABEL, ha="right", va="center",
                   color=st.CED if bold else st.INK,
                   fontweight="semibold" if bold else "normal")
    for gname, y in heads:
        st.text_in(fig, 0.04, y + 0.012, gname, fontsize=st.S_TICK, color=st.INK_2,
                   va="bottom", style="italic")

    for i, ctx in enumerate(CONTEXTS):
        x0 = FACET_L + i * (FACET_W + FACET_GAP)
        ax = st.place(fig, x0, bottom, FACET_W, TOP_A - bottom)
        ax.set_xlim(*XLIM_A)
        ax.set_ylim(bottom, TOP_A)
        st.clean(ax, left=False)
        ax.set_yticks([])
        st.set_ticks(ax.xaxis, [0, 0.3, 0.6], nd=1)
        for gx in (0.3, 0.6):
            ax.axvline(gx, color=st.GRID, lw=0.5, zorder=0.4)
        st.zero_line(ax, color=ZERO_A, lw=0.6)
        for (mid, label, color, bold), y in rows:
            reg = levels[mid]["levels"]
            size = 5.2 if bold else 4.5
            if mid == "base":
                ax.plot([reg["MASKED"][ctx]] * 2, [y + DODGE, y - DODGE], color=color, lw=0.8,
                        alpha=0.55, zorder=3)
                st.point(ax, reg["MASKED"][ctx], y + DODGE, color, True, size=size)
                st.point(ax, reg["MASKED"][ctx], y - DODGE, color, False, size=size)
            elif "MASKED" in reg:
                ax.plot([reg["MASKED"][ctx], reg["SEEN"][ctx]], [y + DODGE, y - DODGE], color=color,
                        lw=0.8, alpha=0.55, zorder=3, solid_capstyle="butt")
                st.point(ax, reg["MASKED"][ctx], y + DODGE, color, True, size=size)
                st.point(ax, reg["SEEN"][ctx], y - DODGE, color, False, size=size)
            else:
                st.point(ax, reg["SEEN"][ctx], y, color, False, size=size)
        for ln in ax.lines:  # top-row markers overhang the facet top by 0.6 pt
            ln.set_clip_on(False)
        st.text_in(fig, x0 + FACET_W / 2, TOP_A + 0.06, ctx, fontsize=st.S_LABEL, ha="center",
                   va="bottom", fontweight="semibold")
    st.text_in(fig, FACET_L + (facet_r - FACET_L) / 2, bottom - 0.30, "BU@20 (0 = random ranking)",
               fontsize=st.S_LABEL, ha="center", va="center")
    return bottom


def panel_b(fig, d, bottom):
    b = d["fig2_b"]["regimes"]
    x0, w = 4.58, 0.84
    ax = st.place(fig, x0, bottom, w, TOP_A - bottom)
    ax.set_xlim(-0.004, 0.09)
    ax.set_ylim(bottom, TOP_A)
    st.clean(ax, left=False)
    ax.set_yticks([])
    st.set_ticks(ax.xaxis, [0, 0.04, 0.08])
    ax.axvline(0.04, color=st.GRID, lw=0.6, zorder=0.4)
    ax.axvline(0.08, color=st.GRID, lw=0.6, zorder=0.4)
    st.zero_line(ax)
    span = TOP_A - bottom
    names = list(CONTEXTS) + ["pooled"]
    ys = [TOP_A - span * f for f in (0.13, 0.35, 0.57, 0.84)]
    ax.axhline(TOP_A - span * 0.705, color=st.RULE, lw=0.5, zorder=0.4)
    for name, y in zip(names, ys):
        pooled = name == "pooled"
        st.text_in(fig, x0 - 0.05, y, "Pooled" if pooled else name, fontsize=st.S_LABEL,
                   ha="right", va="center", fontweight="semibold" if pooled else "normal")
        for regime, dy, open_ in (("MASKED", 1.2 * DODGE, True), ("SEEN", -1.2 * DODGE, False)):
            s = b[regime][name]
            st.interval(ax, s["ci95"][0], s["ci95"][1], y + dy, st.CED, lw=1.3)
            st.point(ax, s["median"], y + dy, st.CED, open_, size=5.2 if pooled else 4.5)
    st.text_in(fig, x0 + w / 2, bottom - 0.30, "Δ BU@20 over Base", fontsize=st.S_LABEL,
               ha="center", va="center")


def half_eye(ax, values, y, color, soft, height, bold=False, xmin=None):
    values = np.asarray(values)
    lo = values.min() if xmin is None else max(values.min(), xmin)
    grid = np.linspace(lo, values.max(), 400)
    dens = gaussian_kde(values, bw_method=0.28)(grid)
    dens = dens / dens.max() * height
    ax.fill_between(grid, y, y + dens, color=soft, lw=0, zorder=1)
    ax.plot(grid, y + dens, color=color, lw=0.6, alpha=0.6, zorder=1.5)
    p5, q1, med, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    if xmin is not None:
        p5 = max(p5, xmin)
    ax.plot([p5, p95], [y, y], color=color, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.plot([q1, q3], [y, y], color=color, lw=2.6, solid_capstyle="butt", zorder=2)
    ax.plot([med], [y], marker="o", ms=5.0 if bold else 4.3, mfc="white", mec=color,
            mew=1.25, zorder=3)


def panel_c(fig, d):
    c = d["fig2_c"]
    rows = [("Goal-blind prior", "Goal-blind prior", st.GOAL_BLIND, "#eceef1"),
            ("Forward-then-match", "CED effect only", st.EFFECT_ONLY, "#e9ebee"),
            ("Base scorer", "Base scorer", st.BASE, "#e0e3e7"),
            ("Text-augmented ranker", "Text-augmented", st.TEXT_RANKER, "#e6e8eb"),
            ("VCDesign-CED", "VCDesign-CED", st.CED, st.CED_SOFT)]
    x0, w = 1.06, 1.66
    ax = st.place(fig, x0, ROW2_Y0, w, ROW2_H)
    n = len(rows)
    lo, hi = -0.28, n - 0.12
    ax.set_ylim(lo, hi)
    xmin = -0.25  # the comparison lives to the right; the share of rows cut off is printed
    ax.set_xlim(xmin - 0.15, 0.80)
    st.clean(ax, left=False)
    ax.set_yticks([])
    ax.spines["bottom"].set_bounds(xmin, 0.80)
    st.set_ticks(ax.xaxis, [-0.2, 0, 0.2, 0.4, 0.6], nd=1)
    st.zero_line(ax)
    for i, (key, label, color, soft) in enumerate(rows):
        y = n - 1 - i
        bold = key == "VCDesign-CED"
        vals = np.asarray(c["levels"][key]["values"])
        half_eye(ax, vals, y, color, soft, height=0.74, bold=bold, xmin=xmin)
        cut = float((vals < xmin).mean())
        if cut >= 0.005:
            ax.plot([xmin - 0.012], [y], marker="<", ms=3.2, color=color, ls="none", zorder=3,
                    clip_on=False)
            ax.text(xmin - 0.035, y, f"{100 * cut:.0f}%", fontsize=st.S_TICK - 0.5,
                    color=st.INK_2, ha="right", va="center")
        yy = ROW2_Y0 + (y - lo) / (hi - lo) * ROW2_H
        st.text_in(fig, x0 - 0.04, yy + 0.02, label, fontsize=st.S_LABEL, ha="right",
                   va="bottom", color=st.CED if bold else st.INK,
                   fontweight="semibold" if bold else "normal")
    ax.set_xlabel(st.MBU + " per query view", labelpad=3)

    # Paired effects of VCDesign-CED, locked-record medians and intervals.
    px, pw = 3.36, 0.62
    py0, ph = ROW2_Y0, ROW2_H * 0.74
    ax2 = st.place(fig, px, py0, pw, ph)
    ax2.set_xlim(-0.003, 0.047)
    ax2.set_ylim(-0.7, 1.75)
    st.clean(ax2, left=False)
    ax2.set_yticks([])
    st.set_ticks(ax2.xaxis, [0, 0.02, 0.04])
    st.zero_line(ax2)
    for y, key, label in ((1.0, "Base scorer", "vs Base"), (0.0, "Text-augmented ranker", "vs Text")):
        p = c["paired"][key]
        cl = c["k562_contrasts_identity_cluster"][f"VCDesign-CED minus {key}"]
        st.interval(ax2, cl["median_ci95"][0], cl["median_ci95"][1], y, st.CED, lw=1.6)
        st.point(ax2, p["median"], y, st.CED, True, size=5.2)
        yy = py0 + (y + 0.7) / 2.45 * ph
        st.text_in(fig, px - 0.05, yy, label, fontsize=st.S_LABEL, ha="right", va="center")
        ax2.text(p["median"], y + 0.42, f"{100 * p['win_rate']:.0f}% improved",
                 fontsize=st.S_TICK, color=st.INK_2, ha="center", va="center")
    ax2.set_xlabel("Paired Δ " + st.MBU, labelpad=3)


def panel_d(fig, d):
    dd = d["fig2_d"]
    x0, w = 4.64, 0.78
    ax = st.place(fig, x0, ROW2_Y0, w, ROW2_H)
    budgets = dd["budgets"]
    xs = np.log10(np.asarray(budgets, dtype=float))
    ax.set_xlim(0.86, 1.86)
    ax.set_ylim(-0.4, 2.65)
    st.zero_line(ax, vertical=False)
    for gy in (1, 2):
        ax.axhline(gy, color=st.GRID, lw=0.6, zorder=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    st.set_ticks(ax.yaxis, [0, 1, 2], nd=0)
    series = (("Base scorer", -0.05, "-", "vs Base"),
              ("Text-augmented ranker", 0.05, (0, (2.0, 1.4)), "vs Text"))
    for comp, dx, ls, lab in series:
        p = dd["paired"][comp]
        means = [p[str(b)]["mean"] for b in budgets]
        ax.plot(xs + dx, means, color=st.CED, lw=1.0, ls=ls, zorder=2)
        for x, b in zip(xs + dx, budgets):
            lo, hi = p[str(b)]["mean_ci95"]
            st.interval(ax, lo, hi, x, st.CED, lw=1.2, horizontal=False)
            st.point(ax, x, p[str(b)]["mean"], st.CED, True, size=4.3)
    ax.text(0.90, 0.74, "vs Base", fontsize=st.S_TICK, color=st.CED, ha="left", va="center")
    ax.text(1.40, -0.22, "vs Text", fontsize=st.S_TICK, color=st.CED, ha="left", va="center")
    ax.set_xlabel("Budget B", labelpad=3)
    ax.set_ylabel("Δ high-value hits", labelpad=2)


def build(d):
    fig = st.figure(H)
    st.panel_head(fig, 0.02, HEAD_1, "a", "Held-out contexts: CED leads the solvers that need no response")
    st.panel_head(fig, 4.12, HEAD_1, "b", "Gain in every context")
    st.regime_key(fig, 1.16, KEY_Y)
    bottom = panel_a(fig, d)
    panel_b(fig, d, bottom)

    st.panel_head(fig, 0.02, HEAD_2, "c", "Unseen K562 identities: CED improves on both rankers")
    st.panel_head(fig, 4.12, HEAD_2, "d", "Extra hits at B = 50")
    panel_c(fig, d)
    panel_d(fig, d)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("figures/data/display_v2_data.json"))
    p.add_argument("--out", type=Path, default=Path("figures/fig2_performance"))
    args = p.parse_args()
    d = json.loads(args.data.read_text())
    fig = build(d)
    st.save(fig, args.out, sources={"data": args.data})
    print(f"wrote {args.out}.pdf/.svg/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
