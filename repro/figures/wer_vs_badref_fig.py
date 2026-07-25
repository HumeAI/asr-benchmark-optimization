"""Regenerate paper/figures/wer_vs_badref.png with corrected terminology.

Reproduces the reference-disagreement scatter (formerly "flagged-ref agreement"):
each model's reference-disagreement ACCEPT-REF (consensus benchmaxx_score) against its
vox-EN WER on error-containing utterances (vs the original, error-bearing references).

This is a faithful re-render of the previously published figure -- SAME data, only the
y-axis / title terminology updated to "reference-disagreement accept-ref". The x-axis is
the internal error-utterance WER, NOT the Open ASR Leaderboard WER cited in the caption
(see note in the paper edit summary).

  python scripts/vmt/wer_vs_badref_fig.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

OUT = str(P.FIGURES / "wer_vs_badref")
RUN = "vox_en_3wl_test"

KEEP = {
    "cohere-transcribe",
    "voxtral-mini-3b",
    "kimi-audio-7b",
    "phi4-multimodal",
    "whisper-large-v3",
    "qwen3-asr-0.6b",
    "moonshine-streaming-medium",
    "canary-qwen-2.5b",
    "granite-speech-4.1-2b",
    "parakeet-tdt-0.6b-v2",
    "higgs-audio-v3-8b-stt-v2",
}


def short(m):
    return (
        m.replace("-transcribe", "")
        .replace("-multimodal", "")
        .replace("granite-speech", "granite")
        .replace("parakeet-", "prkt-")
        .replace("qwen3-asr", "qwen3")
        .replace("whisper-large-v3", "whisper-lv3")
        .replace("voxtral-mini-3b", "voxtral")
        .replace("kimi-audio-7b", "kimi")
        .replace("moonshine-streaming-medium", "moonshine")
        .replace("higgs-audio-v3-8b-stt-v2", "higgs")
    )


def _lev(a, b):
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _is_gemini(m):
    return "gemini" in m.lower()


def main():
    apply_style()
    # y-axis = Table 1's accept-ref: newwl4-panel-unanimous verdict rate for non-members
    # (battery_ablation_cells.json "full" column); leave-one-out (unanimity of the other
    # three) for the four panel members.
    cells = json.loads(P.cell("vmt", "battery_ablation_cells.json").read_text())
    cons = {
        m: {"benchmaxx_score": kn["full"][0] / kn["full"][1]} for m, kn in cells.items() if m in KEEP and kn.get("full")
    }
    loo = {
        "moonshine-streaming-medium": "vox_en_3wl_test",  # 3wl IS the other-three panel for moonshine
        "kimi-audio-7b": "vox_en_loo_kimi",
        "qwen3-asr-0.6b": "vox_en_loo_qwen3",
        "voxtral-mini-3b": "vox_en_loo_voxtral",
    }
    for m, run in loo.items():
        p = P.CELLS / "consensus" / f"{run}_aggregate.json"
        if not p.exists():
            print(f"[warn] LOO aggregate missing for {m} ({run}); keeping panel-scored value")
            continue
        row = [r for r in json.loads(p.read_text())["benchmaxx_leaderboard"] if r["model"] == m]
        if row:
            cons[m] = {"benchmaxx_score": row[0]["benchmaxx_score"]}

    # x-axis = Table 1's VoxPopuli WER (Open ASR Leaderboard; ddagger models via our
    # leaderboard-faithful pipeline), so the scatter is the visualization of Table 1.
    wer = {
        "cohere-transcribe": 5.58,
        "canary-qwen-2.5b": 5.38,
        "granite-speech-4.1-2b": 5.40,
        "higgs-audio-v3-8b-stt-v2": 5.62,
        "phi4-multimodal": 5.77,
        "parakeet-tdt-0.6b-v2": 5.68,
        "moonshine-streaming-medium": 8.01,
        "whisper-large-v3": 8.70,
        "kimi-audio-7b": 7.72,
        "qwen3-asr-0.6b": 6.80,
        "voxtral-mini-3b": 6.77,
    }

    ms = [m for m in cons if m in wer]
    xs = [wer[m] for m in ms]
    ys = [cons[m]["benchmaxx_score"] for m in ms]

    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.scatter(xs, ys, s=120, color=HUME["primary"], edgecolor=HUME["ink"], lw=0.6, zorder=3)
    OFF = {
        "voxtral-mini-3b": (12, 20),
        "qwen3-asr-0.6b": (12, -22),
        "kimi-audio-7b": (8, 8),
        "moonshine-streaming-medium": (10, 16),
        "whisper-large-v3": (10, -16),
        "parakeet-tdt-0.6b-v2": (8, -13),
        "granite-speech-4.1-2b": (7, -14),
        "higgs-audio-v3-8b-stt-v2": (8, 8),
    }
    LEADER = {
        "voxtral-mini-3b",
        "qwen3-asr-0.6b",
        "moonshine-streaming-medium",
        "whisper-large-v3",
        "parakeet-tdt-0.6b-v2",
    }
    for m, x, y in zip(ms, xs, ys):
        dx, dy = OFF.get(m, (6, 4))
        kw = dict(fontsize=9, xytext=(dx, dy), textcoords="offset points", color=HUME["ink"])
        if m in LEADER:
            kw["arrowprops"] = dict(arrowstyle="-", color=HUME["err"], lw=0.6)
        ax.annotate(short(m), (x, y), **kw)
    ax.set_xlabel("VoxPopuli WER (%)")
    ax.set_ylabel("Reference-disagreement accept-ref")
    ax.grid(axis="both", color=HUME["grid"], linewidth=0.8)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    plt.close(fig)
    print("wrote", *[p.name for p in paths], f"({len(ms)} models)")
    for m in sorted(ms, key=lambda m: wer[m]):
        print(f"  {m:28s} WER={wer[m]:5.2f}%  accept-ref={cons[m]['benchmaxx_score']:.3f}")


if __name__ == "__main__":
    main()
