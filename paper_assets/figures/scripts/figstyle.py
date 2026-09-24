"""Shared style for the manuscript's data figures (Figures 2 to 4).

One text-size ladder, one palette and a few layout helpers, so every figure renders with the
same visual language. Import this module before pyplot draws anything.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

# Four text sizes, mapped to role, so the ladder stays readable after a page is scaled down:
#   S_LETTER  panel letters A/B/C, bold
#   S_TITLE   the panel claim, semibold (new role: it no longer shares a size with axis labels)
#   S_BASE    axis labels, series and method names, key numbers (bold where they are the point)
#   S_NOTE    tick labels, group headers, small annotations and keys
# S_TICK is kept as a separate name because the other figure scripts import it, but it holds the
# same value as S_NOTE: four distinct sizes in total.
S_LETTER, S_TITLE, S_BASE, S_NOTE = 9.0, 7.5, 7.0, 6.5
S_TICK = S_NOTE

# Arial ships no semibold face, so matplotlib resolves "semibold" to weight 700 (Arial Bold).
# Asking for "bold" directly keeps the render quiet and gives exactly the same glyphs.
W_SEMIBOLD = "bold"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": S_BASE,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": S_TICK,
    "ytick.labelsize": S_TICK,
    "axes.labelsize": S_BASE,
    "legend.frameon": False,
})

# Paper-wide colour meaning, taken from Figure 1: teal is VCDesign and its effect branch (CED),
# purple is candidate knowledge, orange is measured perturbation responses (the atlas and the
# measured-response oracle), greys are comparators and model machinery. Data marks use darker
# shades of Figure 1's pastel fills so that thin lines and text stay legible on white.
PRIMARY = "#16858a"          # VCDesign, CED (Figure 1 teal #29b4b8, darkened)
PRIMARY_LIGHT = "#94cfd1"    # VCDesign w/o CED
PRIMARY_TEXT = "#0f6468"
KNOWLEDGE = "#7a58c1"        # candidate knowledge (Figure 1 purple)
KNOWLEDGE_TEXT = "#5b3f9c"
MEASURED = "#dd7433"         # measured responses (Figure 1 orange)
MEASURED_TEXT = "#a9531f"
STRONG = "#4a4a4a"
SIMPLE = "#a3a3a3"
ORACLE = MEASURED            # the oracle ranks by measured responses
ORACLE_TEXT = MEASURED_TEXT
CHANCE = "#bdbdbd"
GUIDE = "#d6d6d6"
BAND = "#eef1f5"
MUTED = "#6b6b6b"
INK = "#262626"

# Held contexts: every Figure 3 mark is a CED gain, so contexts are shades of the teal method
# hue (dark to light), always paired with a marker shape so that shade is never the only cue.
CONTEXT = {"RPE1": "#0b5357", "HepG2": "#16858a", "Jurkat": "#5fb8bb"}
CONTEXT_TEXT = {"RPE1": "#0b5357", "HepG2": "#12747a", "Jurkat": "#3a9699"}
CONTEXT_MARKER = {"RPE1": "o", "HepG2": "s", "Jurkat": "^"}

MBU = r"$\overline{\mathrm{BU}}$"
DMBU = r"$\Delta\overline{\mathrm{BU}}$"


def apply_figure_style():
    """Apply the three-role print-scale ladder used by final Figure 2.

    Figure 2 has no internal panel titles, so 7 pt labels, 6.5 pt annotations and
    6 pt ticks are sufficient. Panel letters are the one intentional exception.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "savefig.bbox": None,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def place(fig, left, bottom, width, height):
    """Add axes at a position given in inches, without a background patch."""
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes((left / fig_w, bottom / fig_h, width / fig_w, height / fig_h))
    # No axes background: the page is white, and annotations may extend past an axes edge.
    ax.patch.set_visible(False)
    return ax


def fig_text(fig, x_in, y_in, text, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    return fig.text(x_in / fig_w, y_in / fig_h, text, **kwargs)


def panel_title(fig, x_in, y_in, letter, title, letter_gap=0.19):
    """Panel letter (bold, S_LETTER) followed by the panel's claim (semibold, S_TITLE)."""
    fig_text(fig, x_in, y_in, letter, fontsize=S_LETTER, fontweight="bold", va="center")
    fig_text(fig, x_in + letter_gap, y_in, title, fontsize=S_TITLE, fontweight=W_SEMIBOLD,
             va="center", color=INK)


def set_ticks(axis, ticks):
    """Ticks with plain labels: 0 for zero, two decimals otherwise, a true minus sign."""
    axis.set_ticks(ticks)
    axis.set_ticklabels([("0" if t == 0 else f"{t:.2f}").replace("-", "−") for t in ticks])


def data_y_to_inches(ax, box):
    """Map a data y-value of an axes placed at box (inches) to a figure y in inches."""
    low, high = ax.get_ylim()
    scale = box[3] / (high - low)
    return lambda y: box[1] + (y - low) * scale
