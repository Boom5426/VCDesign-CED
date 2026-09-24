#!/usr/bin/env python3
"""Render the supplementary natural-versus-row-matched atlas-context diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from figstyle import GUIDE, MEASURED, MEASURED_TEXT, MUTED, S_BASE, apply_figure_style, place


apply_figure_style()
FIG_WIDTH_MM = 139.7
FIG_W, FIG_H = 5.50, 2.05
BOX = (0.68, 0.42, 4.27, 1.37)
MATCHED = "#72777b"


def validate(data: dict) -> None:
    if data.get("schema") != "VCDESIGN_SUPP_ATLAS_CONTEXT_DATA_V1":
        raise RuntimeError(f"unexpected atlas-context schema: {data.get('schema')}")
    panel = data["panel"]
    for arm in ("natural", "row_matched"):
        points = panel["arms"][arm]
        if [item["contexts"] for item in points] != [1, 2, 3]:
            raise RuntimeError(f"{arm} context levels changed")
        if any(item["clusters"] != 976 for item in points):
            raise RuntimeError(f"{arm} query-gene bootstrap population changed")


def build(data: dict):
    validate(data)
    panel = data["panel"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = place(fig, *BOX)
    ax.set_xlim(0.78, 3.38)
    ax.set_ylim(-0.007, 0.058)
    ax.set_xticks([1, 2, 3])
    ax.set_yticks([0.0, 0.02, 0.04])
    ax.set_yticklabels(["0", "0.02", "0.04"])
    ax.axhline(0, color=GUIDE, lw=0.65, zorder=0)
    ax.set_xlabel("Training contexts in the effect atlas")
    ax.set_ylabel(r"Gain over base, $\Delta$BU@20")

    styles = {
        "natural": (MEASURED, "Natural atlas", "-", 4.2),
        "row_matched": (MATCHED, "Row-matched atlas", (0, (3, 1.7)), 3.8),
    }
    for arm in ("natural", "row_matched"):
        color, label, linestyle, marker_size = styles[arm]
        points = panel["arms"][arm]
        xs = [item["contexts"] for item in points]
        ys = [item["median"] for item in points]
        ax.plot(xs, ys, color=color, lw=1.15, ls=linestyle, zorder=2)
        for item in points:
            ax.plot([item["contexts"], item["contexts"]], item["ci95"], color=color,
                    lw=0.8, solid_capstyle="butt", zorder=2)
        ax.plot(xs, ys, "o", ms=marker_size, mfc=color, mec="white", mew=0.5, zorder=3)
        offset = 0.0005 if arm == "natural" else 0.0035
        ax.text(3.10, ys[-1] + offset, label, fontsize=S_BASE,
                color=MEASURED_TEXT if arm == "natural" else MUTED,
                ha="left", va="center")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fig = build(json.loads(args.data.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    fig.savefig(args.out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
