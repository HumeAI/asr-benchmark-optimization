"""Hume chart-style module for RW-Voice-EQ Bench (and any other Hume figures).

Reference: hume_style.txt (Jeff's brand guidance — Fellix font, Accent palette,
ink #353535, transparent background, vector-safe export).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════

Minimal:

    from hume_style import apply_style, HUME, ROLE_COLORS
    apply_style()

    fig, ax = plt.subplots()
    ax.barh(y, values, color=HUME["primary"])

Full pattern for a horizontal bar chart (the dominant Hume pattern):

    from hume_style import (
        apply_style, HUME, ROLE_COLORS, ERR_KW,
        style_h_bar_ax, save_figure, value_label,
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(y, values, xerr=stdevs,
                   color=HUME["primary"], edgecolor=HUME["ink"],
                   error_kw=ERR_KW)
    for i, v in enumerate(values):
        value_label(ax, x=v, y=i, text=f"{v:.2f}")

    style_h_bar_ax(ax, xlabel="Spearman ρ", title="…")
    save_figure(fig, "my_plot")            # → my_plot.pdf / .svg / .png

═══════════════════════════════════════════════════════════════════════════════
What this module covers (vs. what stays per-plot)
═══════════════════════════════════════════════════════════════════════════════

Covered here (call once via apply_style):
    • Fellix font registration (auto-discovered from common locations,
      silent fallback to Helvetica / Arial / DejaVu Sans if absent)
    • Type sizes, weights, and ink colour (#353535) for all text elements
    • Clean frame: top + right spines off, gridlines below data
    • Gridline colour (#E6E6E1) and width (0.8)
    • Transparent figure/axes/savefig background (correct for white-page papers
      and flexible for slide decks)
    • Vector-safe export: 300 DPI PNG, embedded TrueType in PDF/SVG (fonttype 42)
    • Palette (HUME) + role assignments (ROLE_COLORS)
    • Shared error-bar keyword args (ERR_KW)
    • Helpers: style_h_bar_ax / style_v_bar_ax / value_label / save_figure

Stays per-plot (author's responsibility):
    • Category / model naming (apply the corrected labels per figure)
    • Sort order (best-first within panel)
    • Which colour role each series maps to (top-5 vs specialist vs SV)
    • Axis ranges (don't force a shared scale across different metrics)
    • Legend contents

═══════════════════════════════════════════════════════════════════════════════
Fellix
═══════════════════════════════════════════════════════════════════════════════

If Fellix isn't found, matplotlib will fall back to Helvetica / Arial / DejaVu
Sans without error. To install: drop the Fellix .ttf/.otf files into any of:

    ~/Fellix/
    ~/fonts/Fellix/
    ~/Library/Fonts/                       (system-wide)
    $HUME_FELLIX_DIR (env var)

then re-run the plot script.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Palette (confirm against Primary Design Resource Figma "Accent" colours)
# ─────────────────────────────────────────────────────────────────────────────

HUME: dict[str, str] = {
    # Hero accents (from investor deck; confirm hex against Figma)
    "primary":  "#F5836F",   # coral   — "In overall top-5"
    "special":  "#A78BDB",   # purple  — "Category / scenario specialist"
    "sv":       "#6E9EE8",   # blue    — Dedicated speaker-verification models
    # Optional deck accents if a panel needs more series
    "mint":     "#6FC59E",
    "amber":    "#F2B15C",
    # Non-data colours
    "ink":      "#353535",   # Brand dark: text, spines, ticks
    "grid":     "#E6E6E1",   # Gridlines
    "err":      "#6B6B6B",   # Error bars
}

# The default role assignment. Extend or remap for your own figure if needed.
ROLE_COLORS: dict[str, str] = {
    "top5":       HUME["primary"],
    "specialist": HUME["special"],
    "sv":         HUME["sv"],
}

# CLEAR-concept accents (optional; from the investor deck).
# Only use when you deliberately want the paper to feel unified with the deck.
CLEAR_ACCENTS: dict[str, str] = {
    "conversation": HUME["amber"],
    "listening":    HUME["primary"],
    "expression":   HUME["special"],
    "accuracy":     HUME["sv"],
    "reliability":  HUME["mint"],
}

# Error-bar keyword args — pass into ax.bar / ax.errorbar / ax.barh
ERR_KW: dict = dict(ecolor=HUME["err"], elinewidth=1, capsize=3, capthick=1)


# ─────────────────────────────────────────────────────────────────────────────
# Fellix font registration
# ─────────────────────────────────────────────────────────────────────────────

_FELLIX_SEARCH_PATHS: tuple[Path, ...] = (
    Path.home() / "Fellix",
    Path.home() / "fonts" / "Fellix",
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    # Fellix .otf shipped with the Hume study-runner on the cluster
    Path(os.environ.get("BENCHMARK_OPT_FONT_DIR", "")) if os.environ.get("BENCHMARK_OPT_FONT_DIR") else Path("/nonexistent"),
    Path(os.environ.get("HUME_FELLIX_DIR", "")) if os.environ.get("HUME_FELLIX_DIR") else Path(""),
)


def _register_fellix() -> bool:
    """Register any Fellix .ttf/.otf files found on this machine.
    Returns True if at least one Fellix file was registered."""
    registered = False
    for path in _FELLIX_SEARCH_PATHS:
        if not path or not path.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for f in path.glob(ext):
                if "fellix" in f.name.lower():
                    try:
                        font_manager.fontManager.addfont(str(f))
                        registered = True
                    except Exception:
                        pass
    return registered


# ─────────────────────────────────────────────────────────────────────────────
# apply_style — the main entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_style(verbose: bool = False) -> None:
    """Apply the full Hume chart style to matplotlib rcParams.

    Idempotent — safe to call multiple times. If Fellix is not installed,
    matplotlib falls back through the sans-serif family (Helvetica → Arial →
    DejaVu Sans) without error.
    """
    fellix_ok = _register_fellix()
    if verbose:
        print(f"[hume_style] Fellix registered: {fellix_ok}")

    mpl.rcParams.update({
        # ── Typography ──
        # Family="sans-serif" with Fellix at the front of the fallback list
        # lets matplotlib pick Fellix when available and fall back cleanly
        # otherwise. (More robust than setting font.family="Fellix" directly.)
        "font.family": "sans-serif",
        "font.sans-serif": ["Fellix", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,

        # ── Ink colour (#353535) for all text elements ──
        "text.color":       HUME["ink"],
        "axes.labelcolor":  HUME["ink"],
        "axes.edgecolor":   HUME["ink"],
        "xtick.color":      HUME["ink"],
        "ytick.color":      HUME["ink"],
        "axes.titlecolor":  HUME["ink"],

        # ── Clean frame ──
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": HUME["grid"],
        "grid.linewidth": 0.8,

        # ── Transparent background ──
        # Correct for white-page papers; also lets figures drop cleanly onto
        # dark slide backgrounds if reused later.
        "figure.facecolor":  "none",
        "axes.facecolor":    "none",
        "savefig.facecolor": "none",

        # ── Vector-safe export ──
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # Embed TrueType — no Type-3 (arXiv-compatible)
        "ps.fonttype": 42,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Per-axes helpers
# ─────────────────────────────────────────────────────────────────────────────

def style_h_bar_ax(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    invert_y: bool = True,
) -> Axes:
    """Apply the standard treatment for a horizontal bar chart.

    Grid on the value axis (x) only, category labels on y, tick marks
    suppressed, best-first ordering via inverted y-axis.
    """
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=8)
    if invert_y:
        ax.invert_yaxis()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def style_v_bar_ax(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Apply the standard treatment for a vertical bar chart.
    Grid on the value axis (y) only, category labels on x.
    """
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def value_label(
    ax: Axes,
    *,
    x: float,
    y: float,
    text: str,
    dx: float = 0.012,
    fontsize: float = 9.5,
    fontweight: str = "normal",
) -> None:
    """Place a numeric value label just past the end of a bar."""
    ax.text(x + dx, y, text, va="center", ha="left",
            fontsize=fontsize, color=HUME["ink"], fontweight=fontweight)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-format vector-safe export
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    fig: Figure,
    path_stem: str | Path,
    *,
    formats: Iterable[str] = ("png",),
) -> list[Path]:
    """Save a figure. Defaults to PNG only.

    ``path_stem`` is a path *without* extension. Files are written as
    ``<stem>.<ext>`` for each ``ext`` in ``formats``. To also emit vector
    formats (e.g. for arXiv or a slide deck) pass them explicitly, e.g.
    ``formats=("pdf", "png")`` or ``("pdf", "svg", "png")``.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", transparent=True)
        out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# __all__ for `from hume_style import *`
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "HUME",
    "ROLE_COLORS",
    "CLEAR_ACCENTS",
    "ERR_KW",
    "apply_style",
    "style_h_bar_ax",
    "style_v_bar_ax",
    "value_label",
    "save_figure",
]
