#!/usr/bin/env python3
"""Figure 4 (display v2): what limits response-unseen design, and how to select estimators.

  a  response-unseen gain over Base as the effect atlas grows: all measured rows (natural atlases)
     against a fixed 840-row budget spread over the same contexts (row-matched atlases)
  b  capacity additions and candidate-information removals, against the unit-direction ridge
  c  prediction accuracy versus effect-only design utility: the 57-point atlas grid (left) and the
     within-atlas estimator changes (right, broken y-axis)

All panels: pooled or per-context response-unseen (MASKED) candidates of RPE1, HepG2 and Jurkat,
unit-direction CED, BU@20. Reads only figures/data/display_v2_data.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import figstyle_v2 as st

H = 3.38
ROW1_Y0, ROW1_H = 2.04, 0.98
ROW2_Y0, ROW2_H = 0.40, 0.94


def panel_a(fig, d):
    arms = d["fig4_a"]["arms"]
    x0, w = 0.52, 1.66
    ax = st.place(fig, x0, ROW1_Y0, w, ROW1_H)
    ax.set_xlim(0.7, 3.3)
    ax.set_ylim(-0.004, 0.061)
    ax.set_xticks([1, 2, 3])
    st.set_ticks(ax.yaxis, [0, 0.02, 0.04])
    for gy in (0.02, 0.04):
        ax.axhline(gy, color=st.GRID, lw=0.6, zorder=0.4)
    st.zero_line(ax, vertical=False)
    style = {"natural": ("-", "all measured rows"), "row_matched": ((0, (2.2, 1.5)), "840 rows")}
    for arm, (ls, lab) in style.items():
        pts = arms[arm]
        xs = np.array([p["contexts"] for p in pts], dtype=float)
        ys = np.array([p["median"] for p in pts])
        ax.plot(xs, ys, color=st.CED, lw=1.1, ls=ls, zorder=2)
        for x, p in zip(xs, pts):
            st.interval(ax, p["ci95"][0], p["ci95"][1], x, st.CED, lw=1.2, horizontal=False)
            st.point(ax, x, p["median"], st.CED, True, size=4.6)
    ax.text(0.78, 0.0545, "all measured rows\n(2.7k, 5.4k, 8.0k)", fontsize=st.S_TICK, color=st.CED,
            va="center", ha="left", linespacing=1.05)
    ax.text(2.12, 0.0135, "fixed 840 rows", fontsize=st.S_TICK, color=st.CED, va="center",
            ha="left")
    ax.set_xlabel("Training contexts in the atlas", labelpad=3)
    ax.set_ylabel("Δ BU@20 over Base", labelpad=2)


INFO_ROWS = [
    ("Model changes", [
        ("Predictive basis", "Predictive response basis"),
        ("Invariant adapter", "Context-invariant adapter"),
        ("Anchored memory", "Nonlinear memory, anchored"),
        ("Unanchored memory", "Nonlinear effect memory")]),
    ("Removed information", [
        ("Drop MAP-KG", "Drop MAP-KG (STRING only)"),
        ("Drop STRING", "Drop STRING (MAP-KG only)")]),
]


def panel_b(fig, d):
    variants = {v["label"]: v for v in d["fig4_b"]["variants"]}
    x0, w = 3.92, 1.50
    top, pitch, gap = ROW1_Y0 + ROW1_H + 0.10, 0.135, 0.09
    y = top
    rows, heads = [], []
    for gi, (gname, members) in enumerate(INFO_ROWS):
        if gi:
            y -= gap
        heads.append((gname, y))
        y -= 0.09
        for short, key in members:
            y -= pitch
            rows.append((short, key, y + pitch / 2))
    bottom = y
    ax = st.place(fig, x0, bottom, w, top - bottom)
    ax.set_xlim(-0.029, 0.007)
    ax.set_ylim(bottom, top)
    st.clean(ax, left=False)
    ax.set_yticks([])
    st.set_ticks(ax.xaxis, [-0.02, -0.01, 0])
    for gx in (-0.02, -0.01):
        ax.axvline(gx, color=st.GRID, lw=0.6, zorder=0.4)
    st.zero_line(ax)
    for short, key, yy in rows:
        v = variants[key]
        st.interval(ax, v["ci95"][0], v["ci95"][1], yy, st.CED, lw=1.3)
        st.point(ax, v["median"], yy, st.CED, True, size=4.6)
        st.text_in(fig, x0 - 0.05, yy, short, fontsize=st.S_LABEL, ha="right", va="center")
    for gname, yy in heads:
        st.text_in(fig, x0 - 1.18, yy + 0.01, gname, fontsize=st.S_TICK, color=st.INK_2,
                   style="italic", va="top")
    ax.set_xlabel("Δ BU@20 vs linear ridge", labelpad=3)


def panel_c_left(fig, d):
    c = d["fig4_c"]
    x0, w = 0.52, 1.94
    ax = st.place(fig, x0, ROW2_Y0, w, ROW2_H)
    ax.set_xlim(-0.062, 0.17)
    ax.set_ylim(-0.01, 0.46)
    st.set_ticks(ax.xaxis, [-0.05, 0, 0.05, 0.1, 0.15])
    st.set_ticks(ax.yaxis, [0, 0.2, 0.4], nd=1)
    dup = {(x["held"], x["estimator"]) for x in c["duplicates_removed"]}
    variants = ("rank64", "rank128", "rank512", "STRING_only", "MAPKG_only")
    for g in c["grid_points"]:
        if (g["held"], g["estimator"]) in dup:
            continue
        is_variant = any(f"[{v}]" in g["estimator"] for v in variants)
        is_ref = g["estimator"] == "RIDGE_UNIT[full]"
        color = st.CED if (is_ref or is_variant) else st.CED_MID
        ax.plot([g["prediction"]], [g["design_effect_only_BU@20"]],
                marker=st.CONTEXT_MARKER[g["held"]], ms=5.0 if is_ref else 3.8, ls="none",
                mfc="white", mec=color, mew=1.8 if is_ref else 0.9, zorder=3 if is_ref else 2)
    pc = c["per_context"]
    ax.text(-0.057, 0.442, "Spearman ρ within context", fontsize=st.S_TICK, color=st.INK_2,
            ha="left", va="center", style="italic")
    for i, k in enumerate(st.CONTEXTS):
        yy = 0.398 - 0.044 * i
        ax.plot([-0.053], [yy], marker=st.CONTEXT_MARKER[k], ms=3.8, mfc="white", mec=st.CED,
                mew=0.9, ls="none")
        ax.text(-0.046, yy, k, fontsize=st.S_TICK, color=st.INK, ha="left", va="center")
        ax.text(0.018, yy, f"{pc[k]['rho']:.2f}", fontsize=st.S_TICK, color=st.INK, ha="right",
                va="center")
    ax.plot([0, 0], [-0.01, 0.262], color=st.GRID, lw=0.6, zorder=0.4)
    ax.set_xlabel("Held-context prediction (candidate-specific cosine)", labelpad=3)
    ax.text(0.5, 1.03, "across atlases", transform=ax.transAxes, fontsize=st.S_TICK,
            color=st.INK_2, ha="center", va="bottom", style="italic")
    ax.set_ylabel("Effect-only BU@20", labelpad=2)


LABEL_POS = {  # label anchor in data units, and alignment
    "rank 64": (0.0085, -0.0235, "left"),
    "rank 128": (0.0085, -0.0080, "left"),
    "rank 512": (-0.0080, 0.0065, "right"),
    "STRING only": (-0.0125, 0.0300, "right"),
    "MAP-KG only": (-0.017, -0.12, "left"),
}


def panel_c_right(fig, d):
    ch = d["fig4_c"]["controlled_changes"]
    x0, w = 3.40, 2.02
    split = ROW2_Y0 + ROW2_H * 0.30
    up = st.place(fig, x0, split + 0.05, w, ROW2_Y0 + ROW2_H - split - 0.05)
    lo = st.place(fig, x0, ROW2_Y0, w, split - ROW2_Y0 - 0.02)
    xlim = (-0.043, 0.021)
    for ax in (up, lo):
        ax.set_xlim(*xlim)
        ax.axvline(0, color=st.ZERO, lw=0.7, zorder=0.5)
    up.set_ylim(-0.033, 0.036)
    lo.set_ylim(-0.272, -0.045)
    up.axhline(0, color=st.ZERO, lw=0.7, zorder=0.5)
    # Quadrants where prediction and design agree in sign are tinted.
    up.fill_between([0, xlim[1]], 0, 0.036, color=st.BAND, lw=0, zorder=0)
    up.fill_between([xlim[0], 0], -0.033, 0, color=st.BAND, lw=0, zorder=0)
    lo.fill_between([xlim[0], 0], -0.272, -0.045, color=st.BAND, lw=0, zorder=0)
    up.text(xlim[1] - 0.0008, 0.032, "agree", fontsize=st.S_TICK, color=st.INK_2,
            style="italic", ha="right", va="top")
    up.text(xlim[0] + 0.0008, -0.030, "agree", fontsize=st.S_TICK, color=st.INK_2,
            style="italic", ha="left", va="bottom")
    up.spines["bottom"].set_visible(False)
    up.tick_params(axis="x", length=0, labelbottom=False)
    st.set_ticks(up.yaxis, [-0.02, 0, 0.02])
    st.set_ticks(lo.yaxis, [-0.2, -0.1], nd=1)
    st.set_ticks(lo.xaxis, [-0.04, -0.02, 0])
    # Break marks on the y axis.
    for ax in (up, lo):
        yv = ax.get_ylim()[0] if ax is up else ax.get_ylim()[1]
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.plot([xlim[0] - 0.0012, xlim[0] + 0.0012], [yv - 0.03 * span, yv + 0.03 * span],
                color=st.INK_3, lw=0.7, clip_on=False)
    for var, per in ch.items():
        xs, ys = [], []
        for ctx, v in per.items():
            ax = lo if v["d_design"] < -0.04 else up
            ax.plot([v["d_prediction"]], [v["d_design"]], marker=st.CONTEXT_MARKER[ctx], ms=4.2,
                    mfc="white", mec=st.CED, mew=0.95, ls="none",
                    zorder=3, clip_on=False)
            xs.append(v["d_prediction"])
            ys.append(v["d_design"])
        lx, ly, ha = LABEL_POS[var]
        if var == "MAP-KG only":
            lo.text(-0.0265, -0.215, "Drop STRING", fontsize=st.S_TICK,
                    color=st.INK, ha="left", va="center")
        else:
            name = {"STRING only": "Drop MAP-KG", "rank 64": "Rank 64", "rank 128": "Rank 128",
                    "rank 512": "Rank 512"}[var]
            up.text(lx, ly, name, fontsize=st.S_TICK, color=st.INK, ha=ha, va="center")
        # thin hull line binds the three contexts of one variant
        order = np.argsort(xs)
        (lo if var == "MAP-KG only" else up).plot(np.array(xs)[order], np.array(ys)[order],
                                                  color=st.CED_MID, lw=0.6, zorder=1.5)
    lo.set_xlabel("Δ held-context prediction", labelpad=3)
    up.text(0.5, 1.03, "within one atlas", transform=up.transAxes, fontsize=st.S_TICK,
            color=st.INK_2, ha="center", va="bottom", style="italic")
    st.text_in(fig, x0 - 0.40, ROW2_Y0 + ROW2_H / 2, "Δ effect-only BU@20",
               fontsize=st.S_LABEL, rotation=90, ha="center", va="center")


def build(d):
    fig = st.figure(H)
    st.panel_head(fig, 0.02, H - 0.15, "a", "More rows help; contexts alone do not")
    st.panel_head(fig, 2.52, H - 0.15, "b", "Model changes do not help; STRING does")
    st.panel_head(fig, 0.02, ROW2_Y0 + ROW2_H + 0.19, "c",
                  "Accuracy tracks atlas changes, not nearby estimators")
    panel_a(fig, d)
    panel_b(fig, d)
    panel_c_left(fig, d)
    panel_c_right(fig, d)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("figures/data/display_v2_data.json"))
    p.add_argument("--out", type=Path, default=Path("figures/fig4_limits"))
    args = p.parse_args()
    d = json.loads(args.data.read_text())
    fig = build(d)
    st.save(fig, args.out, sources={"data": args.data})
    print(f"wrote {args.out}.pdf/.svg/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
