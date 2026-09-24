#!/usr/bin/env python3
"""Figure 3 (display v2): when to retrieve, when to infer, and why inference works.

  a  solver regime: pooled BU@20 of the base scorer, VCDesign-CED and profile retrieval with the
     candidate's own response masked from the effect atlas or measured in training contexts;
     labels are the registered paired median gains
  b  controlled masking: the full, size-controlled and own-response-masked atlases differ by one
     factor at a time; forests give the paired BU@20 lost at each step, query-gene intervals
  c  mixed candidate libraries: gain over Base as the response-unseen share of a fixed pool grows

Unit-direction CED, BU@20, RPE1/HepG2/Jurkat held out in turn. Reads only
figures/data/display_v2_data.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, Rectangle

import figstyle_v2 as st

H = 2.32
ATLAS_ROW = "#d3d8df"
REMOVED = "#aab1ba"
Y0 = 0.40          # bottom of every plotting area (x-axis line)


def panel_a(fig, d):
    """Solver regime: the measured response decides whether to retrieve or to infer."""
    r = d["fig3_regime"]
    lv, ct = r["levels"], r["contrasts"]
    x0, w, h = 0.44, 1.18, 1.46
    ax = st.place(fig, x0, Y0, w, h)
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(0.405, 0.715)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["masked", "measured"])
    st.set_ticks(ax.yaxis, [0.45, 0.55, 0.65])
    for gy in (0.45, 0.55, 0.65):
        ax.axhline(gy, color=st.GRID, lw=0.5, zorder=0.4)
    base, ced, ret = lv["Base"], lv["VCDesign-CED"], lv["Profile retrieval"]
    ax.plot([0, 1], [base["MASKED"], base["SEEN"]], color=st.BASE, lw=0.8, alpha=0.5, zorder=2)
    ax.plot([0, 1], [ced["MASKED"], ced["SEEN"]], color=st.CED, lw=1.0, alpha=0.7, zorder=2)
    st.point(ax, 0, base["MASKED"], st.BASE, True, size=4.6)
    st.point(ax, 1, base["SEEN"], st.BASE, False, size=4.6)
    st.point(ax, 0, ced["MASKED"], st.CED, True, size=5.2)
    st.point(ax, 1, ced["SEEN"], st.CED, False, size=5.2)
    st.point(ax, 1, ret["SEEN"], st.RETRIEVAL, False, size=5.2)
    # Direct labels (left of the masked column; retrieval has no masked arm).
    ax.text(-0.12, base["MASKED"] - 0.004, "Base", fontsize=st.S_TICK, color=st.BASE, ha="right",
            va="center")
    ax.text(-0.12, ced["MASKED"] + 0.004, "CED", fontsize=st.S_TICK, color=st.CED, ha="right",
            va="center", fontweight="semibold")
    ax.text(0.88, ret["SEEN"], "Retrieval", fontsize=st.S_TICK, color=st.RETRIEVAL, ha="right",
            va="center")
    # Paired median gains (not level differences).
    g1 = ct["masked: VCDesign-CED minus Base"]["median"]
    g2 = ct["measured: Profile retrieval minus VCDesign-CED"]["median"]
    ax.text(0.12, (base["MASKED"] + ced["MASKED"]) / 2, st.minus(f"+{g1:.3f}"), fontsize=st.S_TICK,
            color=st.CED, ha="left", va="center")
    ax.text(1.12, (ced["SEEN"] + ret["SEEN"]) / 2, st.minus(f"+{g2:.3f}"), fontsize=st.S_TICK,
            color=st.RETRIEVAL, ha="left", va="center")
    ax.set_ylabel("Pooled BU@20", labelpad=2)
    st.text_in(fig, x0 + w / 2, Y0 - 0.30, "Candidate's own response", fontsize=st.S_LABEL,
               ha="center", va="center")


def atlas_icon(fig, cx, y0, removed=(), candidate=2, candidate_present=True, w=0.28, h=0.30):
    """A stack of atlas rows; the evaluated candidate's row is orange (a measured response)."""
    ax = st.place(fig, cx - w / 2, y0, w, h)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    for i in range(8):
        y = 7 - i + 0.14
        if i == candidate:
            if candidate_present:
                ax.add_patch(Rectangle((0, y), 1, 0.72, color=st.RETRIEVAL, lw=0))
            else:
                ax.add_patch(Rectangle((0.01, y + 0.02), 0.98, 0.68, fill=False, ec=st.RETRIEVAL,
                                       lw=0.7, ls=(0, (2, 1.5))))
        elif i in removed:
            ax.add_patch(Rectangle((0.01, y + 0.02), 0.98, 0.68, fill=False, ec=REMOVED, lw=0.7,
                                   ls=(0, (2, 1.5))))
        else:
            ax.add_patch(Rectangle((0, y), 1, 0.72, color=ATLAS_ROW, lw=0))


def arrow(fig, x0, x1, y, label, label_y):
    fw, fh = fig.get_size_inches()
    fig.patches.append(FancyArrowPatch((x0 / fw, y / fh), (x1 / fw, y / fh), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=6, lw=0.8, color=st.INK_2))
    st.text_in(fig, (x0 + x1) / 2, label_y, label, fontsize=st.S_TICK, ha="center", va="bottom",
               color=st.INK_2, style="italic", linespacing=1.0)


def panel_b(fig, d):
    v = d["fig3_a"]["values"]
    label_r = 2.36
    col_w, col_gap = 0.56, 0.10
    col1 = label_r + 0.07
    cols = (("atlas_size", col1), ("own_response", col1 + col_w + col_gap))
    centers_col = [c + col_w / 2 for _, c in cols]
    c2 = sum(centers_col) / 2
    c1, c3 = 2 * centers_col[0] - c2, 2 * centers_col[1] - c2
    icon_y0 = 1.54
    atlas_icon(fig, c1, icon_y0)
    atlas_icon(fig, c2, icon_y0, removed=(5,))
    atlas_icon(fig, c3, icon_y0, candidate_present=False)
    for cx, lab in zip((c1, c2, c3), ("Full", "Size-\ncontrolled", "Own response\nmasked")):
        st.text_in(fig, cx, icon_y0 - 0.04, lab, fontsize=st.S_TICK, ha="center", va="top",
                   linespacing=1.0)
    mid = icon_y0 + 0.15
    arrow(fig, c1 + 0.17, c2 - 0.17, mid, "6% fewer\nrows", icon_y0 + 0.30)
    arrow(fig, c2 + 0.17, c3 - 0.17, mid, "own response\nremoved", icon_y0 + 0.30)

    names = list(st.CONTEXTS) + ["pooled"]
    h = 0.80
    ys = [Y0 + h * f for f in (0.88, 0.64, 0.40, 0.10)]
    for key, x0 in cols:
        ax = st.place(fig, x0, Y0, col_w, h)
        ax.set_xlim(-0.008, 0.019)
        ax.set_ylim(Y0, Y0 + h)
        st.clean(ax, left=False)
        ax.set_yticks([])
        st.set_ticks(ax.xaxis, [0, 0.01], nd=2)
        ax.axvline(0.01, color=st.GRID, lw=0.6, zorder=0.4)
        st.zero_line(ax)
        ax.axhline(Y0 + h * 0.25, color=st.RULE, lw=0.5, zorder=0.4)
        for name, y in zip(names, ys):
            s = v[key][name]
            pooled = name == "pooled"
            st.interval(ax, s["ci95"][0], s["ci95"][1], y, st.CED, lw=1.4)
            st.point(ax, s["median"], y, st.CED, False, size=5.0 if pooled else 4.4)
    for name, y in zip(names, ys):
        pooled = name == "pooled"
        st.text_in(fig, label_r, y, "Pooled" if pooled else name, fontsize=st.S_LABEL, ha="right",
                   va="center", fontweight="semibold" if pooled else "normal")
    st.text_in(fig, c2, Y0 - 0.30, "BU@20 lost at each step", fontsize=st.S_LABEL, ha="center",
               va="center")


def panel_c(fig, d):
    ctx = d["fig3_b"]["contexts"]
    x0, w, h = 4.42, 0.66, 1.46
    ax = st.place(fig, x0, Y0, w, h)
    ax.set_xlim(-12, 112)
    ax.set_ylim(0.032, 0.076)
    st.set_ticks(ax.yaxis, [0.04, 0.05, 0.06, 0.07])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"])
    for gy in (0.04, 0.05, 0.06, 0.07):
        ax.axhline(gy, color=st.GRID, lw=0.5, zorder=0.4)
    dodge = {"RPE1": -4.0, "HepG2": 4.0, "Jurkat": 0.0}
    for name in st.CONTEXTS:
        pts = ctx[name]["points"]
        shares = [round(100 * p["share"]) for p in pts]
        xs = [s_ + dodge[name] for s_ in shares]
        ys = [p["median"] for p in pts]
        ax.plot(xs, ys, color=st.CED, lw=1.0, zorder=2)
        for x, y, share in zip(xs, ys, shares):
            fill = "full" if share == 0 else ("none" if share == 100 else "bottom")
            ax.plot([x], [y], marker=st.CONTEXT_MARKER[name], ms=4.0, color=st.CED, fillstyle=fill,
                    mfc=st.CED, mfcalt="white", mec=st.CED, mew=0.9, ls="none", zorder=3,
                    clip_on=False)
        ax.text(116, ys[-1] + (0.0030 if name == "HepG2" else -0.0030 if name == "RPE1" else 0),
                name, fontsize=st.S_TICK, color=st.CED, va="center", ha="left")
    ax.set_ylabel("Gain over Base", labelpad=2)
    st.text_in(fig, x0 + w / 2, Y0 - 0.30, "Unseen share (%)", fontsize=st.S_LABEL, ha="center",
               va="center")


def build(d):
    fig = st.figure(H)
    st.panel_head(fig, 0.02, H - 0.16, "a", "Retrieve measured, infer unseen")
    st.panel_head(fig, 2.02, H - 0.16, "b", "Gain survives masking")
    st.panel_head(fig, 4.08, H - 0.16, "c", "Holds in mixed pools")
    panel_a(fig, d)
    panel_b(fig, d)
    panel_c(fig, d)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("figures/data/display_v2_data.json"))
    p.add_argument("--out", type=Path, default=Path("figures/fig3_masking"))
    args = p.parse_args()
    d = json.loads(args.data.read_text())
    fig = build(d)
    st.save(fig, args.out, sources={"data": args.data})
    print(f"wrote {args.out}.pdf/.svg/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
