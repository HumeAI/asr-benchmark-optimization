"""paper/figures/masking_blackbox_datasets.png (fig:mask-bb).

Per-model masked accept-ref (silenced numeric span reproduced in the hypothesis) on four
datasets: two public benchmarks (VoxPopuli, LibriSpeech) and two recipe-matched held-out
sets (daikon, ep-fresh). Same ungated regex-match definition as Table~1's vox column.
Shows the public >> held-out gap across models.

  python scripts/vmt/masking_datasets_fig.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

DATA = P.require_data()
OUT = str(P.FIGURES / "masking_blackbox_datasets")
TAG = "mask-num-all-numexp-silence"
DATASETS = [  # (label, variant-prefix, color, public?)
    ("VoxPopuli", "voxpopuli", HUME["primary"], True),
    ("LibriSpeech", "librispeech-clean", HUME["special"], True),
    ("daikon", "daikon", HUME["err"], False),
    ("ep-fresh", "ep-fresh-vox", HUME["grid"], False),
]
MODELS = [
    "cohere-transcribe",
    "canary-qwen-2.5b",
    "granite-speech-4.1-2b",
    "phi4-multimodal",
    "parakeet-tdt-0.6b-v2",
    "qwen3-asr-0.6b",
    "voxtral-mini-3b",
    "whisper-large-v3",
    "kimi-audio-7b",
    "moonshine-streaming-medium",
    "higgs-audio-v3-8b-stt-v2",
]
LABELS = {
    "cohere-transcribe": "Cohere-Transcribe",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "moonshine-streaming-medium": "Moonshine-Streaming",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
}


def hyp(r):
    return r.get("hypothesis_raw") or r.get("hypothesis") or r.get("hyp") or ""


def load_hyps_any(variant, model):
    """{key: hyp} from either results layout: legacy results.jsonl or wsds shards."""
    p = DATA / "results" / variant / model / "test" / "results.jsonl"
    if p.exists():
        out = {}
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r.get("__key__")] = hyp(r)
        return out
    import glob
    import os

    from pyarrow import ipc

    dones = glob.glob(str(DATA / "results" / variant / "test" / model / "*" / "DONE"))
    if not dones:
        return None
    run_dir = sorted(dones, key=os.path.getmtime)[-1].rsplit("/", 1)[0]
    out = {}
    for shard in sorted(glob.glob(f"{run_dir}/*.wsds")):
        t = ipc.open_file(shard).read_all()
        hc = next((c for c in ("hyp_raw", "hypothesis_raw", "hyp", "hypothesis") if c in t.schema.names), None)
        if hc is None:
            continue
        for k, h in zip(t.column("__key__").to_pylist(), t.column(hc).to_pylist()):
            out[k] = h or ""
    return out


def rate(variant, model):
    mp = DATA / "datasets" / variant / "test" / "truncation_meta.parquet"
    if not mp.exists():
        return None
    hyps = load_hyps_any(variant, model)
    if hyps is None:
        return None
    meta = {r["__key__"]: (r["hidden_ref"] or "") for r in pq.read_table(mp).to_pylist()}
    k = n = 0
    for key, tgt in meta.items():
        tgt = (tgt or "").lower().strip(".,;:!?\"'")
        h = hyps.get(key)
        if not tgt or h is None:
            continue
        n += 1
        if re.search(r"(^|\W)" + re.escape(tgt) + r"($|\W)", h.lower()):
            k += 1
    return k / n if n else None


def main():
    apply_style()
    data = {}  # model -> {ds_label: rate}
    for m in MODELS:
        data[m] = {}
        for label, pref, _c, _pub in DATASETS:
            data[m][label] = rate(f"{pref}-{TAG}", m)
        print(
            f"  {m:26s} "
            + "  ".join(f"{lb}={data[m][lb]:.2f}" if data[m][lb] is not None else f"{lb}=NA" for lb, *_ in DATASETS)
        )
    # shared canonical order with the reference-disagreement panel (Table 1 / headline order,
    # Cohere at top) so rows align across both panels of Figure 4. reversed -> highest y = top.
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
    order = [m for m in reversed(ORDER) if m in data]
    y = np.arange(len(order))
    h = 0.8 / len(DATASETS)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(order) + 0.9))
    for j, (label, _pref, col, _pub) in enumerate(DATASETS):
        off = ((len(DATASETS) - 1) / 2 - j) * h
        vals = [data[m][label] if data[m][label] is not None else 0 for m in order]
        ax.barh([yy + off for yy in y], vals, h, color=col, edgecolor=HUME["ink"], linewidth=0.4, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(m, m) for m in order], fontsize=11)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("masked accept-ref")
    ax.set_xlim(0, None)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=HUME["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    paths = save_figure(fig, OUT)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    main()
