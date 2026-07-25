"""Regenerate paper/figures/masking_voice.png with the FULL model panel.

Per-model paired difference in masked accept-ref, REAL VoxPopuli minus a
held-out ep-fresh clone (same transcript), on clips each model retrieves in its
OWN unmasked rendering of both conditions (intelligibility gate). Positive => the
model reads a silenced number from the benchmark voice but not a held-out voice.
Paired bootstrap 95% CI; red when the CI excludes 0.

Reuses the scoring primitives (hit / load_hyps / load_meta / TAGs) from
nummask_realvsgeneric_gated.py so the metric matches the black-box table exactly.

  python scripts/vmt/masking_voice_fig.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib  # noqa: E402
import nummask_realvsgeneric_gated as G  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hume_style import HUME, apply_style, save_figure  # noqa: E402
from pathlib import Path

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

OUT = str(P.FIGURES / "masking_voice")
MODELS = [
    "cohere-transcribe",
    "canary-qwen-2.5b",
    "granite-speech-4.1-2b",
    "phi4-multimodal",
    "parakeet-tdt-0.6b-v2",
    "higgs-audio-v3-8b-stt-v2",
    "qwen3-asr-0.6b",
    "voxtral-mini-3b",
    "whisper-large-v3",
    "kimi-audio-7b",
    "moonshine-streaming-medium",
]
LABELS = {
    "cohere-transcribe": "Cohere-Transcribe",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "moonshine-streaming-medium": "Moonshine-Streaming",
}


def paired_diffs(model, meta_real, meta_epf, keys):
    ru, rm = G.load_hyps(G.REAL_UNMASKED, model), G.load_hyps(G.REAL_MASKED, model)
    eu, em = G.load_hyps(G.EPF_UNMASKED, model), G.load_hyps(G.EPF_MASKED, model)
    if any(x is None for x in (ru, rm, eu, em)):
        return None
    ds = []
    for k in keys:
        hid_r, hid_e = meta_real.get(k), meta_epf.get(k)
        if not hid_r or not hid_e:
            continue
        # gate: model must retrieve the number UNMASKED in BOTH conditions
        if k not in ru or k not in eu or k not in rm or k not in em:
            continue
        if not (G.hit(hid_r, ru[k]) and G.hit(hid_e, eu[k])):
            continue
        ds.append(int(G.hit(hid_r, rm[k])) - int(G.hit(hid_e, em[k])))
    return np.array(ds) if ds else None


def boot_ci(d, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    apply_style()
    meta_real, meta_epf = G.load_meta(G.REAL_MASKED), G.load_meta(G.EPF_MASKED)
    keys = set(meta_real or {}) & set(meta_epf or {})
    rows = []
    for m in MODELS:
        d = paired_diffs(m, meta_real, meta_epf, keys)
        if d is None or len(d) < 5:
            print(f"  skip {m} (no data / n<5)")
            continue
        lo, hi = boot_ci(d)
        rows.append((m, float(d.mean()), lo, hi, len(d)))
        print(f"  {m:26s} diff={d.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}] n={len(d)}")
    rows.sort(key=lambda r: r[1])  # ascending; largest at top after invert
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 0.8))
    for i, (m, pt, lo, hi, n) in enumerate(rows):
        sig = lo > 0 or hi < 0
        col = HUME["primary"] if sig else HUME["grid"]
        ax.plot([lo, hi], [i, i], color=HUME["err"], lw=1.2, solid_capstyle="round", zorder=2)
        ax.plot(pt, i, "o", color=col, mec=HUME["ink"], mew=0.8, ms=7, zorder=3)
    ax.axvline(0, color=HUME["ink"], lw=1.0, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS.get(m, m) for m, *_ in rows], fontsize=9)
    ax.set_xlabel("masked accept-ref:  real $-$ held-out clone")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=HUME["grid"], linewidth=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths], f"({len(rows)} models)")


if __name__ == "__main__":
    main()
