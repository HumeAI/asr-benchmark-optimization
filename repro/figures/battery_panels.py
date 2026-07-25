"""Trigger-battery ladder panels for Figure 4 (consensus + masked), from battery_ladder.json.

Grouped horizontal bars per model. Voice conditions on identical transcripts (real /
vox-clone / ep-fresh clone / generic) as solid bars; ep-fresh REAL audio (own content,
noisier references) as a hatched bar, visually set apart because its denominator differs.
Same palette/geometry as consfull_fig.py / nummask_lift_fig.py.

  python scripts/vmt/battery_panels.py            # both panels
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

SRC = str(P.CELLS / "vmt" / "battery_ladder.json")
FIGS = str(P.FIGURES)
# Shared trigger-battery colour scheme: an intact->removed dial. Position 0 (the intact
# "trigger present / elevated" condition) is the coral focal series; position 3 (the
# trigger-removed control / honest floor) is de-emphasised grey. The SAME ordinal->colour
# map is used in battery_ablation_fig.py so both rows of the battery read as one dial.
COND = [
    # epfresh_real (own content, noisier refs) deliberately EXCLUDED: its denominator is
    # not comparable to the same-transcript voice conditions; it lives in tab:realfresh.
    ("real", "real", HUME["primary"]),
    ("voxclone", "vox-clone", HUME["special"]),
    ("epfresh_clone", "ep-fresh clone", HUME["sv"]),
    ("generic", "generic", HUME["grid"]),
]
ORDER = [
    "cohere-transcribe",
    "canary-qwen-2.5b",
    "granite-speech-4.1-2b",
    "phi4-multimodal",
    "parakeet-tdt-0.6b-v2",
    "higgs-audio-v3-8b-stt-v2",
    "moonshine-streaming-medium",
    "whisper-large-v3",
    "kimi-audio-7b",
    "qwen3-asr-0.6b",
    "voxtral-mini-3b",
]
LABELS = {
    "cohere-transcribe": "Cohere-Transcribe",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "moonshine-streaming-medium": "Moonshine-Streaming",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
}


import os as _os  # noqa: E402


_sw = _os.environ.get("MODEL_SWAP")  # e.g. "omni-3b-llm:higgs-audio-v3-8b-stt-v2"
if _sw:
    _a, _b = _sw.split(":")
    ORDER = [(_b if _m == _a else _m) for _m in ORDER]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# dropped from the MAIN figure (at the honest floor throughout); kept in the _full
# appendix variants so no data leaves the paper.
TRIM = ("kimi-audio-7b", "moonshine-streaming-medium", "qwen3-asr-0.6b")


def panel(block, out_stem, xlabel, order=None, legend=True):
    order = [m for m in reversed(order or ORDER) if m in block]
    nb = len(COND)
    h = 0.85 / nb
    y = np.arange(len(order))
    ypos = {m: y[i] for i, m in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.2, 0.6 * len(order) + 0.9))
    labeled = set()
    for bi, (ckey, clab, col) in enumerate(COND):
        offs = (bi - (nb - 1) / 2) * h
        for m in order:
            c = block[m].get(ckey)
            if not c or c.get("rate") is None:
                continue
            k = c.get("ref", c.get("leak", 0))
            lo, hi = wilson(k, c["n"])
            yy = ypos[m] + offs
            lab = clab if ckey not in labeled else None
            labeled.add(ckey)
            ax.barh(yy, c["rate"], height=h, color=col, linewidth=0, zorder=2, label=lab)
            ax.plot([lo, hi], [yy, yy], color=HUME["err"], lw=1.0, zorder=3)

    ax.set_yticks([ypos[m] for m in order])
    ax.set_yticklabels([LABELS.get(m, m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, None)
    style_h_bar_ax(ax, xlabel=xlabel, invert_y=False)
    if legend:
        # legend top-to-bottom must match the bars' top-to-bottom order within a group.
        # offs puts COND[0] (real) at the BOTTOM, so reverse the insertion-order handles.
        _h, _l = ax.get_legend_handles_labels()
        ax.legend(_h[::-1], _l[::-1], loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    print("wrote", *[p.name for p in paths])


def main():
    apply_style()
    d = json.load(open(SRC))
    trimmed = [m for m in ORDER if m not in TRIM]
    # left column = consensus (carries the shared voice-condition legend); right = masked (no legend)
    panel(d["consensus"], f"{FIGS}/battery_consensus", "reference-disagreement accept-ref", order=trimmed)
    panel(d["masked"], f"{FIGS}/battery_masked", "masked-number accept-ref", order=trimmed, legend=False)
    panel(d["consensus"], f"{FIGS}/battery_consensus_full", "reference-disagreement accept-ref")
    panel(d["masked"], f"{FIGS}/battery_masked_full", "masked-number accept-ref", legend=False)


if __name__ == "__main__":
    main()
