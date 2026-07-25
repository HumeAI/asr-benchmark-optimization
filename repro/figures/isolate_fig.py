"""paper/figures/isolate_gating.png — general context-gating (all-model truncation).

For each flagged model, on consensus edits it reproduces in full context: the rate at
which the edit disappears when the audio is cut to a tight window around it (delete runs,
reference-extra words) and at which the suppressed audible word surfaces (insert runs).
Honest models are the calibration: they almost never reproduce these edits (nothing to
gate), and their surface rate on the same windows bounds what the short-window instrument
can detect. Reads analysis/voxmode/vmt/isolate_edits.json.

  python scripts/vmt/isolate_fig.py
"""

import json
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

SRC = str(P.CELLS / "vmt" / "isolate_edits.json")
OUT = str(P.FIGURES / "isolate_gating")
SUSPECTS = [
    ("cohere-transcribe", "Cohere-Transcribe"),
    ("canary-qwen-2.5b", "Canary-Qwen-2.5B"),
    ("granite-speech-4.1-2b", "Granite-Speech-4.1-2B"),
    ("phi4-multimodal", "Phi-4-Multimodal"),
    ("parakeet-tdt-0.6b-v2", "Parakeet-TDT-0.6B-v2"),
    ("higgs-audio-v3-8b-stt-v2", "Higgs-Audio-v3-8B"),
]
HONEST = ["moonshine-streaming-medium", "whisper-large-v3", "kimi-audio-7b", "qwen3-asr-0.6b", "voxtral-mini-3b"]
COND = [
    ("reproduced_delete", "reference-extra word vanishes in isolation", HUME["primary"]),
    ("reproduced_insert", "suppressed audible word surfaces in isolation", HUME["special"]),
]


import os as _os  # noqa: E402


_sw = _os.environ.get("MODEL_SWAP")
if _sw:
    _a, _b = _sw.split(":")
    SUSPECTS = [((_b, "Higgs-Audio-v3-8B") if m == _a else (m, lb)) for m, lb in SUSPECTS]


def main():
    apply_style()
    d = json.load(open(SRC))
    order = [m for m, _ in reversed(SUSPECTS) if m in d]
    labels = dict(SUSPECTS)
    nb = len(COND)
    h = 0.8 / nb
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.8, 0.62 * len(order) + 1.1))
    for bi, (ck, cl, col) in enumerate(COND):
        offs = (bi - (nb - 1) / 2) * h
        for i, m in enumerate(order):
            c = d[m].get(ck)
            if not c or c.get("rate") is None:
                continue
            ax.barh(y[i] + offs, c["rate"], height=h, color=col, zorder=2, label=cl if i == 0 else None)
            lo, hi = c["ci"]
            ax.plot([lo, hi], [y[i] + offs, y[i] + offs], color=HUME["err"], lw=0.8, zorder=3)
            ax.text(0.012, y[i] + offs, f"n={c['n']}", va="center", fontsize=7, color="white", zorder=4)
    # honest-model detection ceiling for the surface readout (their 'all' insert rate)
    ceil = np.mean([d[m]["all_insert"]["rate"] for m in HONEST if m in d])
    ax.axvline(ceil, color=HUME["err"], lw=1.0, ls=":", zorder=1)
    ax.text(ceil + 0.008, len(order) - 0.42, "honest-model\nsurface ceiling", fontsize=7.5, color=HUME["err"])

    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in order], fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of context-reproduced edits")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=HUME["grid"], linewidth=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    # legend reading order (here left-to-right) must match the bars' top-to-bottom order;
    # offs puts COND[0] at the bottom of each group, so reverse the insertion-order handles.
    _h, _l = ax.get_legend_handles_labels()
    ax.legend(_h[::-1], _l[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths], f"(honest surface ceiling {ceil:.2f})")


if __name__ == "__main__":
    main()
