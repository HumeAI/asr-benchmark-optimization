"""paper/figures/patch_dissociation.png — insertion vs omission edit-locus dissociation.

Per model (Cohere / Granite / Canary-Qwen), four quantities on consensus edits the
model reproduces in full context:
  insertion, patch removes it     : re-encoding the edit frames context-free drops the
                                    inserted (not-in-audio) token  -> ENCODER-side.
  insertion, faithful in isolation: the isolated window transcribes without the token.
  omission,  patch revives it     : re-encoding context-free brings the dropped (audible)
                                    token back  -> ~0 => DECODER-side.
  omission,  faithful in isolation: the isolated window keeps the audible token.

The story: both edit types are faithful in isolation (models perceive the span), but the
causal patch fixes only insertions (encoder-side), not omissions (decoder-side).

NOTE ON PROVENANCE: the original plotting script for this figure was lost (never committed).
The per-clip dumps (analysis/voxmode/repaint/editlocus_{cohere,granite,canaryqwen}_clean.json
+ the paired isolation/attn readouts) no longer share a single schema and the
faithful-in-isolation arms were sourced outside the *_clean.json rows, so these are the
published, reviewed figure values reproduced here for a presentation-only restyle — not a
recompute. If the metric is re-derived from source, replace VALUES below.

  python scripts/vmt/patch_dissociation_fig.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

OUT = str(P.FIGURES / "patch_dissociation")
MODELS = ["Cohere-Transcribe", "Granite-Speech-4.1-2B", "Canary-Qwen-2.5B"]
# (rate, n) per model, in series order below. Published figure values.
SERIES = [
    ("insertion: patch removes it",      HUME["primary"], [(0.40, 121), (0.52, 164), (0.46, 196)]),
    ("insertion: faithful in isolation", HUME["grid"],    [(0.90, 156), (0.94, 156), (0.90, 155)]),
    ("omission: patch revives it",       HUME["sv"],      [(0.005, 51), (0.01, 67), (0.01, 38)]),
    ("omission: faithful in isolation",  HUME["grid"],    [(0.67, 51), (0.84, 67), (0.79, 38)]),
]


def main():
    apply_style()
    ng, ns = len(MODELS), len(SERIES)
    x = np.arange(ng)
    w = 0.8 / ns
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for si, (label, color, vals) in enumerate(SERIES):
        off = (si - (ns - 1) / 2) * w
        rates = [v[0] for v in vals]
        # hatch the second "faithful in isolation" grey so the two grey series stay distinct
        hatch = "///" if si == 3 else None
        ax.bar(x + off, rates, w, color=color, edgecolor=HUME["ink"], lw=0.4, label=label, hatch=hatch, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of edits")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    main()
