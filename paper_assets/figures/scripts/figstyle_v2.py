"""Shared visual system for the main-text data figures (Figures 2 to 4, display v2).

One typeface (Lato, embedded as TrueType), one role-based size ladder, and one semantic palette:

* colour encodes the method (VCDesign-CED is the only saturated hue among solvers that do not read
  measured responses; orange marks solvers that read them, as in Figure 1),
* marker fill encodes information availability: open = the candidate's own response is masked
  from the effect atlas, filled = the candidate's responses were measured in training contexts,
* marker shape encodes the held cellular context wherever contexts share one axis.

Figures are drawn at their final print size (5.5 in, the ICLR text width), so every size here is
the size on the page. Import this module before pyplot draws anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG_W = 5.5  # ICLR single-column text width, inches

# Size ladder (points, at print size).
S_LETTER = 10.0   # panel letters
S_HEAD = 8.0      # panel headings (the panel's claim)
S_LABEL = 7.5     # axis labels, method names, key numbers
S_TICK = 7.0      # tick labels, annotations, small keys
FONT = "Lato"

# Ink and structure.
INK = "#1d2127"
INK_2 = "#4b525d"        # secondary text
INK_3 = "#7a828e"        # tertiary text, notes
RULE = "#c4c9d1"         # axes and hairlines
ZERO = "#b9bec6"         # zero / random reference
BAND = "#f4f5f7"         # alternating row band
GRID = "#eceef1"

# Methods. Teal is VCDesign-CED (Figure 1 teal, deepened for thin marks on white).
CED = "#0e7c7f"
CED_SOFT = "#cfe7e7"
CED_MID = "#5aa8aa"
BASE = "#39404b"
TEXT_RANKER = "#7f8792"
EFFECT_ONLY = "#98a0aa"
GOAL_BLIND = "#b3b9c1"
CELLNAVI = "#6d82a8"      # muted slate: fixed-output identification
RETRIEVAL = "#d0782f"     # measured-response retrieval (Figure 1 orange = measured responses)
RETRIEVAL_SOFT = "#f3dcc6"
LOSS = "#b0584f"          # a variant that lowers design value

CONTEXTS = ("RPE1", "HepG2", "Jurkat")
CONTEXT_MARKER = {"RPE1": "o", "HepG2": "s", "Jurkat": "^", "Pooled": "D"}

MBU = r"$\overline{\mathrm{BU}}$"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.default": "regular",
    "font.size": S_LABEL,
    "axes.labelsize": S_LABEL,
    "axes.titlesize": S_HEAD,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.unicode_minus": True,
    "xtick.labelsize": S_TICK,
    "ytick.labelsize": S_TICK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": S_TICK,
    "lines.solid_capstyle": "round",
    "savefig.bbox": None,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def figure(height: float):
    return plt.figure(figsize=(FIG_W, height))


def place(fig, left: float, bottom: float, width: float, height: float):
    """Axes at a position given in inches from the lower-left corner, transparent background."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes((left / fw, bottom / fh, width / fw, height / fh))
    ax.patch.set_visible(False)
    return ax


def text_in(fig, x: float, y: float, s: str, **kw):
    """Figure text at a position in inches."""
    fw, fh = fig.get_size_inches()
    kw.setdefault("color", INK)
    return fig.text(x / fw, y / fh, s, **kw)


def panel_head(fig, x: float, y: float, letter: str, heading: str, gap: float = 0.17):
    """Bold panel letter followed by the panel's one-line claim."""
    text_in(fig, x, y, letter, fontsize=S_LETTER, fontweight="bold", va="baseline")
    if heading:
        text_in(fig, x + gap, y, heading, fontsize=S_HEAD, fontweight="semibold",
                va="baseline", color=INK)


def minus(s: str) -> str:
    return s.replace("-", "−")


def fmt(v: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    if not sign and abs(v) < 0.5 * 10 ** (-nd):
        s = f"{0:.{nd}f}"
    return minus(s)


def set_ticks(axis, ticks, nd: int = 2, zero: str = "0"):
    axis.set_ticks(ticks)
    axis.set_ticklabels([zero if abs(t) < 1e-12 else fmt(t, nd) for t in ticks])


def clean(ax, left: bool = True, bottom: bool = True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)


def zero_line(ax, x: float = 0.0, vertical: bool = True, **kw):
    style = dict(color=ZERO, lw=0.7, zorder=0.5)
    style.update(kw)
    if vertical:
        ax.axvline(x, **style)
    else:
        ax.axhline(x, **style)


def point(ax, x, y, color, open_: bool, marker: str = "o", size: float = 4.6, zorder=4,
          edge_w: float = 1.0):
    """One dot. Open markers keep a white face so lines behind them do not show through."""
    ax.plot([x], [y], marker=marker, ms=size + edge_w + 0.6, mfc="white", mec="none",
            ls="none", zorder=zorder - 0.05)  # halo separates the dot from lines behind it
    ax.plot([x], [y], marker=marker, ms=size, mfc="white" if open_ else color,
            mec=color, mew=edge_w, ls="none", zorder=zorder)


def interval(ax, lo, hi, y, color, lw: float = 1.2, horizontal: bool = True, zorder=3,
             alpha: float = 1.0):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder,
                alpha=alpha)
    else:
        ax.plot([y, y], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=zorder,
                alpha=alpha)


def regime_key(fig, x: float, y: float, color: str = INK_2, gap: float = 0.12):
    """Inline key for marker fill: open = own response masked, filled = measured elsewhere."""
    fw, fh = fig.get_size_inches()
    ax = place(fig, x, y - 0.06, 0.1, 0.12)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    point(ax, 0.5, 0.5, color, True, size=4.4)
    t1 = text_in(fig, x + gap, y, "own response absent from the effect atlas",
                 fontsize=S_TICK, color=INK_2, va="center")
    fig.canvas.draw()
    bb = t1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x2 = bb.x1 + 0.2
    ax2 = place(fig, x2, y - 0.06, 0.1, 0.12)
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    point(ax2, 0.5, 0.5, color, False, size=4.4)
    text_in(fig, x2 + gap, y, "responses measured in training contexts", fontsize=S_TICK,
            color=INK_2, va="center")


def save(fig, out: Path, sources: dict | None = None):
    """Write PDF (vector, editable text), SVG, and a 600-dpi PNG review raster; no bbox cropping."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "VCDesign figure scripts (display v2)"}
    fig.savefig(out.with_suffix(".pdf"), metadata=meta)
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    if sources is not None:
        record = {"figure": out.name, "inputs": {k: sha256(Path(v)) for k, v in sources.items()},
                  "outputs": {s: sha256(out.with_suffix(s)) for s in (".pdf",)}}
        out.with_suffix(".provenance.json").write_text(json.dumps(record, indent=1) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legend_handle(color, open_: bool, marker="o", size=4.4):
    return Line2D([], [], marker=marker, ms=size, ls="none", mfc="white" if open_ else color,
                  mec=color, mew=1.0)
