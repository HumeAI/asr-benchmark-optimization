"""Orthographic switch rate, one panel per convention scope.

Rebuilds the paper's switch-rate panels from ``repro/data/ortho_switch.json``,
which ``repro/precompute_cells.py`` derives with :mod:`benchmaxx.ortho`. Unlike
``paired_grids.py`` — the original generator, which re-reads every model's raw
output — this runs offline in a clean clone.

Chance is 0.5 for a two-arm family, so the reference line is at 0.5 rather than
at 0. Bars are colored by whether the interval clears chance, not by a
hand-picked model set.

  python repro/figures/ortho_switch_fig.py [spacing|hon_mister|both]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

CHANCE = 0.5
TITLES = {
    "spacing": "archaic spacing, within LibriSpeech",
    "hon_mister": "honorific, VoxPopuli vs LibriSpeech",
}


def panel(ax, cell, title):
    res = cell["results"]
    order = sorted(res, key=lambda m: res[m]["switch"], reverse=True)
    for i, m in enumerate(order):
        r = res[m]
        clears = r["lo"] > CHANCE
        ax.barh(i, r["switch"], color=HUME["primary"] if clears else HUME["grid"], height=0.68, zorder=2)
        ax.plot([r["lo"], r["hi"]], [i, i], color=HUME["err"], lw=1.2, zorder=3, solid_capstyle="round")
    ax.axvline(CHANCE, color=HUME["ink"], lw=1.0, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1)
    style_h_bar_ax(ax, xlabel=f"switch rate ({title})", invert_y=True)
    return order


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    apply_style()
    cells = json.loads(P.cell("ortho_switch.json").read_text())
    keys = list(TITLES) if which == "both" else [which]

    for key in keys:
        cell = cells[key]
        n = len(cell["results"])
        fig, ax = plt.subplots(figsize=(6.6, 0.32 * n + 1.2))
        order = panel(ax, cell, TITLES[key])
        fig.tight_layout()
        out = save_figure(fig, str(P.FIGURES / f"ortho_switch_{key}"))
        above = sum(1 for m in order if cell["results"][m]["lo"] > CHANCE)
        print(f"wrote {', '.join(p.name for p in out)}  ({n} models, {above} above chance)")
        plt.close(fig)


if __name__ == "__main__":
    main()
