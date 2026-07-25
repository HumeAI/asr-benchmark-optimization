"""paper/figures/nummask_lift_voice.png — masked-number audio lift by voice condition.

Grouped horizontal bars, one group per model, conditions = real / vox-clone / ep-fresh /
generic (same palette and geometry as consfull_fig.py so the trigger-battery panels read
as one figure). Value = mean lambda(r)/char of the silenced number span vs the
zeroed-waveform prior, bootstrap 95% CI, 147 paired sentences (Eq. lift). Reads
analysis/voxmode/vmt/nummask_lift_conditions_<model>.json.

  python scripts/vmt/nummask_lift_fig.py
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

SRC = Path(str(P.CELLS / "vmt"))
STRICT = SRC / "nummask_lift_strict_table.json"
OUT = str(P.FIGURES / "nummask_lift_voice")
# Shared voice-condition palette across the three *_lift_voice figures, assigned by MEANING:
#   real = focal (coral), matched-speaker clone = purple, held-out fresh = blue, generic = grey floor.
COND = [
    ("real", "real", HUME["primary"]),
    ("voxclone", "vox-clone", HUME["special"]),
    ("epfresh", "ep-fresh", HUME["sv"]),
    ("generic", "generic", HUME["grid"]),
]
ORDER = [  # canonical Fig-battery order; all models except Parakeet (no teacher-forced readout)
    ("cohere-transcribe", "Cohere-Transcribe"),
    ("canary-qwen-2.5b", "Canary-Qwen-2.5B"),
    ("granite-speech-4.1-2b", "Granite-Speech-4.1-2B"),
    ("phi4-multimodal", "Phi-4-Multimodal"),
    ("higgs-audio-v3-8b-stt-v2", "Higgs-Audio-v3-8B"),
    ("moonshine-streaming-medium", "Moonshine-Streaming"),
    ("whisper-large-v3", "Whisper-Large-v3"),
    ("kimi-audio-7b", "Kimi-Audio-7B"),
    ("qwen3-asr-0.6b", "Qwen3-ASR-0.6B"),
    ("voxtral-mini-3b", "Voxtral-Mini-3B"),
]


def main():
    apply_style()
    strict = json.load(open(STRICT))
    data = {m: strict[m] for m, _ in ORDER if m in strict}
    order = [m for m, _ in reversed(ORDER) if m in data]
    labels = dict(ORDER)
    nb = len(COND)
    h = 0.8 / nb
    y = np.arange(len(order))
    ypos = {m: y[i] for i, m in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(order) + 0.9))
    labeled = set()
    for bi, (ckey, clab, col) in enumerate(COND):
        offs = (bi - (nb - 1) / 2) * h
        for m in order:
            s = data[m].get(ckey)
            if not s:
                continue
            yy = ypos[m] + offs
            lab = clab if ckey not in labeled else None
            labeled.add(ckey)
            ax.barh(yy, s["mean"], height=h, color=col, zorder=2, label=lab)
            ax.plot([s["lo"], s["hi"]], [yy, yy], color=HUME["err"], lw=0.8, zorder=3)

    ax.axvline(0, color=HUME["ink"], lw=0.9, zorder=1)
    ax.set_yticks([ypos[m] for m in order])
    ax.set_yticklabels([labels[m] for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("masked-number audio lift (nats/char)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    # legend top-to-bottom must match bars' top-to-bottom; offs puts COND[0] at the bottom.
    _h, _l = ax.get_legend_handles_labels()
    ax.legend(_h[::-1], _l[::-1], loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths], f"({len(order)} models)")


if __name__ == "__main__":
    main()
