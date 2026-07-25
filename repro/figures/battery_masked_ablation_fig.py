"""paper/figures/battery_masked_ablation.png — masked-number recovery as the trigger is
removed (Fig 3d): same geometry/palette as battery_ablation_fig.py, masked-probe readout
from nummask_ablation_cells.json (shared format-robust matcher across WS2 and steering
decodes; run nummask_ablation_cells.py first).

  (ulimit -v 16000000; .venv-data/bin/python scripts/vmt/battery_masked_ablation_fig.py)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from battery_ablation_fig import COND, LABELS, ORDER, TRIM, wilson  # noqa: E402
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

SRC = Path(str(P.CELLS / "vmt" / "nummask_ablation_cells.json"))
OUT = str(P.FIGURES / "battery_masked_ablation.png")


def render(cells, out_stem, row_order, legend=False):
    order = [m for m in reversed(row_order) if cells.get(m)]
    nb = len(COND)
    h = 0.8 / nb
    y = np.arange(len(order))
    ypos = {m: y[i] for i, m in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(order) + 0.9))
    labeled = set()
    for bi, (ckey, clab, col) in enumerate(COND):
        offs = (bi - (nb - 1) / 2) * h
        for m in order:
            kn = cells[m].get(ckey)
            if not kn or not kn[1]:
                continue
            k, n = kn
            lo, hi = wilson(k, n)
            yy = ypos[m] + offs
            lab = clab if ckey not in labeled else None
            labeled.add(ckey)
            ax.barh(yy, k / n, height=h, color=col, linewidth=0, zorder=2, label=lab)
            ax.plot([lo, hi], [yy, yy], color=HUME["err"], lw=1.0, zorder=3)

    ax.set_yticks([ypos[m] for m in order])
    ax.set_yticklabels([LABELS.get(m, m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, None)
    style_h_bar_ax(ax, xlabel="masked-number accept-ref", invert_y=False)
    if legend:
        ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    print("wrote", *[p.name for p in paths], f"({len(order)} models)")


def main():
    apply_style()
    cells = json.load(open(SRC))
    stem = OUT[:-4]  # strip .png; save_figure adds extensions
    # right column of the battery grid: legend lives on the consensus (left) panels
    render(cells, stem, [m for m in ORDER if m not in TRIM])
    render(cells, stem + "_full", ORDER)


if __name__ == "__main__":
    main()
