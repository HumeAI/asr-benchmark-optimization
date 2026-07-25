"""Causal-control figures for "The mode is steerable" (input level + activation level).

steer_input_level.png  — REAL VoxPopuli flagged clips: consensus accept-ref for baseline /
  +8s VoxPopuli donor / +8s conversational donor (splice_trigger_full_real8.json). The
  input-level OFF switch: a non-benchmark suffix collapses the behavior; a benchmark
  suffix leaves it intact; Granite is immune (decoder re-imposes).

steer_activation_level.png — two axes, all encoder-instrumented models:
  left  = real VoxPopuli edits, no-steer / ablate d / random  (ablate_acceptref_*.json)
  right = generic clones (no benchmark errors in audio), none / +8d / random
          (induce_generic_*.json). The activation-level OFF and ON switches.

  python scripts/vmt/steer_causal_fig.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

apply_style()

DR = Path(str(P.CELLS))
FIGS = str(P.FIGURES)
ORDER = [
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
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "moonshine-streaming-medium": "Moonshine-Streaming",
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


def barh_panel(ax, rows, conds, data, legend=True):
    """rows: model keys top-to-bottom; conds: (key, label, color); data[m][ck] = (rate, lo, hi)."""
    order = [m for m in reversed(rows) if m in data]
    nb = len(conds)
    h = 0.8 / nb
    y = np.arange(len(order))
    labeled = set()
    for bi, (ck, cl, col) in enumerate(conds):
        offs = (bi - (nb - 1) / 2) * h
        for i, m in enumerate(order):
            v = data[m].get(ck)
            if v is None:
                continue
            rate, lo, hi = v
            lab = cl if ck not in labeled else None
            labeled.add(ck)
            ax.barh(y[i] + offs, rate, height=h, color=col, zorder=2, label=lab)
            ax.plot([lo, hi], [y[i] + offs, y[i] + offs], color=HUME["err"], lw=0.9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(m, m) for m in order], fontsize=11)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, None)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="lower right", frameon=False, fontsize=11)


# dropped from the MAIN figure (honest floor throughout); kept in the _full appendix variant.
TRIM = ("kimi-audio-7b", "moonshine-streaming-medium", "qwen3-asr-0.6b")


def input_level(row_order=None, suffix=""):
    """Two axes mirroring the activation panel: input-level OFF (real clips + donors) and
    input-level ON (ep-fresh-clone bases + donors). The opposite-register donor is the
    intervention; the same-register donor is the specificity control."""
    row_order = row_order or ORDER
    # STRICT-gate cutover: spliced bases filtered by the strict TTS-intelligibility gate.
    dr8 = json.load(open(DR / "vmt/splice_trigger_full_real8_strict.json"))["consensus"]
    dep = json.load(open(DR / "vmt/splice_trigger_full_epfresh_strict.json"))["consensus"]
    # colour by meaning (shared with the activation figure): baseline = blue reference,
    # the causal steering intervention = coral, the matched-register control = grey.
    abl_conds = [
        ("baseline", "real clip", HUME["sv"]),
        ("ctlsuf8", "+8s conversational donor", HUME["primary"]),
        ("voxsuf8", "+8s VoxPopuli donor (control)", HUME["grid"]),
    ]
    ind_conds = [
        ("baseline", "ep-fresh clone", HUME["sv"]),
        ("voxsuf8", "+8s VoxPopuli donor", HUME["primary"]),
        ("ctlsuf8", "+8s conversational donor (control)", HUME["grid"]),
    ]

    def cells(src, conds):
        data = {}
        for m, r in src.items():
            if m not in ORDER:
                continue
            data[m] = {}
            for ck, _, _ in conds:
                c = r.get(ck)
                if c and c.get("rate") is not None:
                    data[m][ck] = (c["rate"], *wilson(c["ref"], c["n"]))
        return data

    abl = cells(dr8, abl_conds)
    ind = cells(dep, ind_conds)
    rows = [m for m in row_order if m in abl or m in ind]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.55 * len(rows) + 1.4))
    barh_panel(axes[0], rows, abl_conds, abl, legend=False)
    axes[0].set_xlabel("accept-ref, real VoxPopuli clips", fontsize=12)
    axes[0].set_title("ablation (OFF)", fontsize=12)
    barh_panel(axes[1], rows, ind_conds, ind, legend=False)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("accept-ref, ep-fresh clones", fontsize=12)
    axes[1].set_title("induction (ON)", fontsize=12)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    # legend top-to-bottom must match the bars' top-to-bottom order within a group; offs puts
    # conds[0] (baseline) at the BOTTOM, so reverse each panel's handles before concatenating.
    seen, H, L = set(), [], []
    for hd, lb in list(zip(h0[::-1], l0[::-1])) + list(zip(h1[::-1], l1[::-1])):
        if lb in seen:
            continue
        seen.add(lb)
        H.append(hd)
        L.append(lb)
    fig.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    save_figure(fig, f"{FIGS}/steer_input_level{suffix}")
    print(f"wrote {FIGS}/steer_input_level{suffix}.png ({len(rows)} models)")


def activation_level():
    steer_models = [
        ("cohere", "cohere-transcribe"),
        ("canary", "canary-qwen-2.5b"),
        # higgs excluded: no clean operating point (a4 degrades WER 3x + garbage, a<=2 inert; app:steer)
        ("granite", "granite-speech-4.1-2b"),
        ("phi4", "phi4-multimodal"),
        ("parakeet", "parakeet-tdt-0.6b-v2"),
    ]
    steer_models = [(s, f) for s, f in steer_models if (DR / f"steer/newwl4/induce_generic_{s}_strict.json").exists()]
    # same colour meaning as the input-level figure: baseline = blue, the steering
    # intervention (ablate / add direction) = coral, random-direction control = grey.
    abl_conds = [
        ("none", "no steer", HUME["sv"]),
        ("ablate", "ablate direction", HUME["primary"]),
        ("random", "random direction", HUME["grid"]),
    ]
    ind_conds = [
        ("none", "no steer", HUME["sv"]),
        ("induce_a8", "add direction", HUME["primary"]),
        ("random_a8", "random direction", HUME["grid"]),
    ]
    abl, ind = {}, {}
    for short, full in steer_models:
        # rescored against the newwl4-unanimous edit set (same panel as the input-level figs)
        o = json.load(open(DR / f"steer/newwl4/ablate_acceptref_{short}.json"))
        abl[full] = {ck: (o[ck]["accept_ref"], *o[ck]["wilson95"]) for ck, _, _ in abl_conds}
        # right panel (induction ON) cut over to the strict-gate ∩ newwl4 generic clones.
        g = json.load(open(DR / f"steer/newwl4/induce_generic_{short}_strict.json"))
        ind[full] = {ck: (g[ck]["accept_ref"], *g[ck]["wilson95"]) for ck, _, _ in ind_conds}
    rows = [full for _, full in steer_models]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.62 * len(rows) + 1.4))
    barh_panel(axes[0], rows, abl_conds, abl, legend=False)
    axes[0].set_xlabel("accept-ref, real VoxPopuli edits", fontsize=12)
    axes[0].set_title("ablation (OFF)", fontsize=12)
    barh_panel(axes[1], rows, ind_conds, ind, legend=False)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("accept-ref, generic clones", fontsize=12)
    axes[1].set_title("induction (ON)", fontsize=12)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    # legend top-to-bottom must match the bars' top-to-bottom order within a group; offs puts
    # conds[0] (baseline) at the BOTTOM, so reverse each panel's handles before concatenating.
    seen, H, L = set(), [], []
    for hd, lb in list(zip(h0[::-1], l0[::-1])) + list(zip(h1[::-1], l1[::-1])):
        if lb in seen:
            continue
        seen.add(lb)
        H.append(hd)
        L.append(lb)
    fig.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=11, ncol=4)
    fig.tight_layout()
    save_figure(fig, f"{FIGS}/steer_activation_level")
    print(f"wrote {FIGS}/steer_activation_level.png")


if __name__ == "__main__":
    input_level([m for m in ORDER if m not in TRIM])
    input_level(suffix="_full")
    activation_level()
