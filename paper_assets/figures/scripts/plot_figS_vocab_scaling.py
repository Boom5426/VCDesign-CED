#!/usr/bin/env python3
"""Render the supplementary K562 vocabulary-scaling stress test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from figstyle import BAND, GUIDE, MUTED, PRIMARY_LIGHT, PRIMARY_TEXT, apply_figure_style, place


apply_figure_style()
FIG_WIDTH_MM = 139.7
FIG_W, FIG_H = 5.50, 2.05
BOX = (0.66, 0.42, 4.35, 1.38)


def validate(data: dict) -> None:
    if data.get("schema") != "VCDESIGN_SUPP_VOCAB_SCALING_DATA_V1":
        raise AssertionError(f"unexpected SI scaling schema: {data.get('schema')}")
    panel = data["panel"]
    sizes = [entry["size"] for key in ("response_unseen", "measured") for entry in panel[key]]
    if np.any(np.asarray(sizes) <= 0):
        raise AssertionError("log-scale candidate vocabulary sizes must be strictly positive")
    unseen = {entry["size"]: entry for entry in panel["response_unseen"]}
    if not np.isclose(unseen[6255]["median"], -0.010281249762853273, atol=1e-12, rtol=0.0):
        raise AssertionError("the 6,255-candidate stress-test endpoint changed")


def build(data: dict):
    validate(data)
    panel = data["panel"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = place(fig, *BOX)
    ax.set_xscale("log")
    ax.set_xlim(210, 9000)
    ax.xaxis.set_minor_locator(NullLocator())
    ticks = [250, 500, 1000, 1570, 3000, 6255]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["250", "500", "1k", "1.6k", "3k", "6.3k"])
    ax.set_ylim(-0.025, 0.072)
    ax.set_yticks([-0.02, 0.00, 0.02, 0.04, 0.06])
    ax.axhline(0.0, color=GUIDE, lw=0.65, zorder=0)
    low, high = panel["held_context_pool_band"]
    ax.axvspan(low, high, color=BAND, lw=0, zorder=0)
    ax.text((low * high) ** 0.5, 0.069, "main held-context pools", color=MUTED,
            fontsize=6.0, ha="center", va="top")

    styles = {
        "response_unseen": (PRIMARY_TEXT, "white", 1.1, "response-unseen"),
        "measured": (PRIMARY_LIGHT, PRIMARY_LIGHT, 0.85, "measured"),
    }
    for key in ("response_unseen", "measured"):
        points = panel[key]
        x = np.asarray([point["size"] for point in points])
        y = np.asarray([point["median"] for point in points])
        edge, face, linewidth, label = styles[key]
        ax.plot(x, y, color=edge, lw=linewidth, zorder=2)
        for point in points:
            if point["full_class"]:
                ax.plot([point["size"], point["size"]], point["ci95"], color=edge,
                        lw=0.7, zorder=2)
            else:
                ax.plot([point["size"], point["size"]],
                        [min(point["draws"]), max(point["draws"])], color=edge,
                        lw=0.5, alpha=0.6, zorder=1)
            ax.plot([point["size"]], [point["median"]], "o", ms=3.4,
                    mec=edge, mfc=face, mew=0.75, zorder=3)
        end = points[-1]
        # Keep the measured label above the descending response-unseen trajectory.
        label_y = end["median"] + (0.020 if key == "measured" else -0.001)
        ax.text(end["size"] * 1.08, label_y, label, color=edge, fontsize=6.5,
                ha="left", va="center", clip_on=False)

    ax.text(6255 / 1.10, -0.0135, "−0.010", color=PRIMARY_TEXT, fontsize=6.5,
            fontweight="bold", ha="right", va="top")
    ax.set_xlabel("Candidate vocabulary size", labelpad=2)
    ax.set_ylabel(r"$\Delta\overline{\mathrm{BU}}$ over Base", labelpad=2)
    return fig, {"scaling": ax}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fig, _ = build(json.loads(args.data.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".svg"))
    fig.savefig(args.out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
