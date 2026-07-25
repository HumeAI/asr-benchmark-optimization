"""Render the consensus accept-ref tightness figure (clone vs generic vs real).

Grouped horizontal bars, one group per model, conditions = real / vox-clone /
[ep-fresh] / generic. Suspects show real >> vox-clone > generic; honest models
sit at the ~0 floor. Reads consfull_accept_ref.json (auto-uses whatever
conditions are present) + Wilson CIs. Minimal chart-text convention.

  python scripts/vmt/consfull_fig.py
"""

import json
import math
import os
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

SRC = str(P.CELLS / "vmt" / "consfull_accept_ref.json")

# two presets: voice-axis "tightness" (default) and channel-axis "robustness".
# Colours by meaning: "real" (the focal reference) = coral; every other condition = a
# distinct HUME accent, with the floor/least-interesting condition in grey.
MODE = sys.argv[1] if len(sys.argv) > 1 else "tightness"
if MODE == "robustness":
    # accept-ref under content-preserving perturbations vs the real baseline. One bar per family.
    # second arg picks the reverb source: "realrir" (paper default, real measured RIR RT60 0.52) or
    # "synthetic" (our low-DRR exp-decay reverb; harsher, for comparison).
    REV = sys.argv[2] if len(sys.argv) > 2 else "realrir"
    if REV == "synthetic":
        OUT = str(P.FIGURES / "consensus_robustness_synthetic.png")
        rev_cond = ("pert-reverb0.6", "+reverb (synthetic)", HUME["amber"])
    else:
        OUT = str(P.FIGURES / "consensus_robustness.png")
        rev_cond = ("realrir-mid", "+reverb", HUME["amber"])  # real measured RIR (RT60 0.52); noted in caption
    COND = [("real", "real", HUME["primary"]), ("pert-noise10db", "+noise 10dB", HUME["sv"]), rev_cond]
else:
    OUT = str(P.FIGURES / "consensus_tightness.png")
    # condition -> (display label, color). real (focal) -> clones -> generic floor (grey).
    COND = [
        ("real", "real", HUME["primary"]),
        ("vox", "vox-clone", HUME["sv"]),
        ("epfresh", "ep-fresh", HUME["special"]),
        ("generic", "generic", HUME["grid"]),
    ]
# labels for display
LABELS = {
    "cohere-transcribe": "Cohere-Transcribe",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "moonshine-streaming-medium": "Moonshine-Streaming",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    apply_style()
    d = json.load(open(SRC))
    conds = [c for c in COND if any(d[m].get(c[0]) for m in d)]
    # shared canonical order + geometry with the masked-number panel (masking_datasets_fig.py) so
    # the two subfigures of Fig.~3 have parallel model rows. Cohere at top; reversed -> y=0 at bottom.
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
    import os as _os

    _sw = _os.environ.get("MODEL_SWAP")
    if _sw:
        _a, _b = _sw.split(":")
        ORDER = [(_b if _m == _a else _m) for _m in ORDER]
    order = [m for m in reversed(ORDER) if m in d]
    rows = order
    nb = len(conds)
    h = 0.8 / nb
    y = np.arange(len(order))
    ypos = {m: y[i] for i, m in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(order) + 0.9))
    labeled = set()
    for bi, (ckey, clab, col) in enumerate(conds):
        offs = (bi - (nb - 1) / 2) * h
        for m in rows:
            c = d[m].get(ckey)
            if not c or not c.get("n"):
                continue
            rate = c["rate"]
            lo, hi = wilson(c["ref"], c["n"])
            yy = ypos[m] + offs
            lab = clab if ckey not in labeled else None
            labeled.add(ckey)
            ax.barh(yy, rate, height=h, color=col, zorder=2, label=lab)
            ax.plot([lo, hi], [yy, yy], color=HUME["err"], lw=0.9, zorder=3)

    ax.set_yticks([ypos[m] for m in rows])
    ax.set_yticklabels([LABELS.get(m, m) for m in rows], fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("accept-ref rate", fontsize=10)
    ax.set_xlim(0, None)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    # legend top-to-bottom must match bars' top-to-bottom; offs puts COND[0] at the bottom.
    _h, _l = ax.get_legend_handles_labels()
    ax.legend(_h[::-1], _l[::-1], loc="lower right", frameon=False, fontsize=9, ncol=1)
    fig.tight_layout()
    save_figure(fig, OUT[: -len(".png")])
    print(f"wrote {OUT}  (conditions: {[c[0] for c in conds]}, {len(rows)} models)")


if __name__ == "__main__":
    main()
