"""paper/figures/orthohon_voice2x2.png — honorific convention follows the voice.

Two panels, one dumbbell per model, x = base-anchored "Mister" rate for the same
sentence rendered in a VoxPopuli-speaker clone vs a LibriSpeech-narrator clone.
Left: VoxPopuli Mr-sentences (transfer — libri voice writes Mister in). Right:
LibriSpeech Mister-sentences (erosion — vox voice strips Mister out). Open ticks
mark the real-recording rate. Reads orthohon_voice2x2.json.

  python scripts/vmt/orthohon_voice2x2_fig.py
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

SRC = str(P.CELLS / "vmt" / "orthohon_voice2x2.json")
OUT = str(P.FIGURES / "orthohon_voice2x2")
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
VOX_VOICE, LIB_VOICE = HUME["primary"], HUME["sv"]  # warm=vox-speaker clone, cool=libri-narrator clone


def main():
    apply_style()
    d = json.load(open(SRC))
    tr = d["contrasts"]["voice_transfer_vox_sentence_libri_vs_vox_voice"]["per_model"]
    er = d["contrasts"]["voice_erosion_libri_sentence_libri_vs_vox_voice"]["per_model"]
    base = d["real_baselines"]

    order = sorted(tr, key=lambda m: -tr[m]["orthovox2libri-hon_rate"])
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.4 * len(order) + 1.3), sharey=True)
    panels = [
        (axes[0], tr, "orthovox2vox-hon_rate", "orthovox2libri-hon_rate", "vox", "VoxPopuli sentences"),
        (axes[1], er, "ortholibri2vox-hon_rate", "ortholibri2libri-hon_rate", "libri", "LibriSpeech sentences"),
    ]
    for ax, cc, vk, lk, bk, title in panels:
        for i, m in enumerate(order):
            v, li = cc[m][vk], cc[m][lk]
            ax.plot([v, li], [y[i], y[i]], color=HUME["grid"], lw=1.6, zorder=1)
            ax.plot(v, y[i], "o", ms=5.5, color=VOX_VOICE, zorder=3, label="vox-speaker voice" if i == 0 else None)
            ax.plot(li, y[i], "o", ms=5.5, color=LIB_VOICE, zorder=3, label="libri-narrator voice" if i == 0 else None)
            rb = base.get(bk, {}).get("models", {}).get(m, {}).get("mister_rate")
            if rb is not None:
                ax.plot(rb, y[i], "|", ms=10, color=HUME["ink"], mew=1.6, zorder=2,
                        label="real recording" if i == 0 else None)
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.0)
        ax.set_xlabel('"Mister" rate')
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m in order])
    axes[0].legend(loc="lower right", frameon=False, handletextpad=0.3)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    main()
