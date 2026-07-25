"""paper/figures/libnum_voice_ladder.png — LibriSpeech masked-number voice ladder.

One row per model, four rungs on identical masked sentences: real recording,
clone of the same test narrator, clone of a 2026-debut LibriVox narrator
(same register + TTS pipeline, voice unseen in training), stock TTS voice.
Trained voices filled, untrained voices open. Reads libnum_voice_ladder_full2.json.

  python scripts/vmt/libnum_voice_ladder_fig.py
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

SRC = str(P.CELLS / "vmt" / "libnum_voice_ladder_full2.json")
OUT = str(P.FIGURES / "libnum_voice_ladder")
LABELS = {
    "cohere-transcribe": "Cohere-Transcribe",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "whisper-large-v3": "Whisper-large-v3",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
    "moonshine-streaming-medium": "Moonshine-medium",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
}
RUNGS = [
    ("real", "real recording", HUME["sv"], "o", True),
    ("libclone_pooled", "trained-narrator clone", HUME["primary"], "o", True),
    ("freshlv_pooled", "unseen-narrator clone", HUME["primary"], "o", False),
    ("generic_pooled", "stock TTS voice", HUME["err"], "s", False),
]


def main():
    apply_style()
    d = json.load(open(SRC))["models"]
    order = sorted(d, key=lambda m: -(d[m]["real"]["rate"] or 0))
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 0.44 * len(order) + 1.2))
    off = np.linspace(0.27, -0.27, len(RUNGS))
    for (rk, rl, col, mk, filled), o in zip(RUNGS, off):
        for i, m in enumerate(order):
            c = d[m].get(rk)
            if not c:
                continue
            lo, hi = c["ci"]
            ax.plot([lo, hi], [y[i] + o, y[i] + o], color=HUME["err"], lw=0.9, zorder=1)
            ax.plot(
                c["rate"], y[i] + o, mk, ms=4.6, mfc=col if filled else "white", mec=col, mew=1.1,
                zorder=3, label=rl if i == 0 else None,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in order], fontsize=8.5)
    ax.set_xlim(0, 0.68)
    ax.set_xlabel("masked-number recovery")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=HUME["grid"], linewidth=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    main()
