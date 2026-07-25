"""paper/figures/libnum_voice_gap.png — the voice-exposure gap, one dot per model.

Sentence-clustered mean difference in masked-number recovery between clones of
LibriSpeech test narrators (voices in training data) and clones of 2026-debut
LibriVox narrators (same register, same TTS pipeline, voices unseen), bootstrap
95% CIs over sentences. Reads libnum_voice_ladder_full2.json. Hume house style.

  python scripts/vmt/libnum_voice_gap_fig.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

SRC = str(P.CELLS / "vmt" / "libnum_voice_ladder_full2.json")
OUT = str(P.FIGURES / "libnum_voice_gap")
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


def main():
    apply_style()
    d = json.load(open(SRC))["models"]
    order = sorted(d, key=lambda m: d[m]["contrast_libclone_vs_freshlv"]["mean_diff"], reverse=True)
    y = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(6.4, 0.46 * len(order) + 1.0))
    for i, m in enumerate(order):
        c = d[m]["contrast_libclone_vs_freshlv"]
        lo, hi = c["ci"]
        mu = c["mean_diff"]
        sig = lo > 0
        col = HUME["primary"] if sig else HUME["grid"]
        ax.plot([lo, hi], [i, i], color=HUME["err"], lw=1.2, zorder=2, solid_capstyle="round")
        ax.plot(mu, i, "o", ms=7, color=col, mec=HUME["ink"], mew=0.8, zorder=3)
    ax.axvline(0, color=HUME["ink"], lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in order])
    style_h_bar_ax(ax, xlabel="recovery gap: trained $-$ unseen narrator clone", invert_y=True)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    main()
