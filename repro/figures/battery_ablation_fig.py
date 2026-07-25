"""paper/figures/battery_ablation.png — accept-ref as the trigger is removed (row 2 of
the trigger figure: same models/geometry as consfull_fig.py, but the independent variable
is context/intervention rather than voice).

Conditions per model (all on the same consensus-flagged VoxPopuli-EN edits):
  full              : full clip, verdict accept-ref over unanimous runs (as tab:isolate).
  truncated         : tight window around the edit (isolate_edits.json; accept-ref =
                      1 - audio-true per class, pooled).
  donor ablated     : full clip + 8 s conversational donor
                      (splice_trigger_full_real8.json, ctlsuf8 arm).
  activation ablated: register direction projected out of one encoder layer
                      (ablate_acceptref_<m>.json, ablate arm; instrumented models only).

  (ulimit -v 16000000; .venv-data/bin/python scripts/vmt/battery_ablation_fig.py)
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
sys.path.insert(0, str(Path(__file__).parent.parent))
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

DR = Path(str(P.CELLS))
OUT = str(P.FIGURES / "battery_ablation.png")
CACHE = DR / "vmt/battery_ablation_cells.json"

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
STEER_SHORT = {
    "cohere-transcribe": "cohere",
    "parakeet-tdt-0.6b-v2": "parakeet",
    "canary-qwen-2.5b": "canary",
    "granite-speech-4.1-2b": "granite",
    "phi4-multimodal": "phi4",
    # higgs excluded: no clean operating point (a4 degrades WER 3x + garbage, a<=2 inert; app:steer)
}
# Shared trigger-battery colour scheme (see battery_panels.py): same intact->removed dial
# so the voice row and this ablation row read as one dial. Position 0 (full clip, trigger
# intact/elevated) is the coral focal series; position 3 (activation ablated, the strongest
# removal / honest floor) is de-emphasised grey. Also imported by battery_masked_ablation_fig.py.
COND = [
    ("full", "full clip", HUME["primary"]),
    ("truncated", "truncated to edit", HUME["special"]),
    ("donor", "donor ablated", HUME["sv"]),
    ("activation", "activation ablated", HUME["grid"]),
]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_cells():
    """Compute (k, n) per model x condition; cached because the full-clip verdict pass is slow."""
    import _voxablate_cons_run as C
    from isolate_edits_score import SAMPLES, load_wsds

    iso = json.load(open(DR / "vmt/isolate_edits.json"))
    r8 = json.load(open(DR / "vmt/splice_trigger_full_real8.json"))["consensus"]
    cells = {}
    for m in ORDER:
        cells[m] = {}
        hyps = load_wsds("voxpopuli", m)
        if hyps is not None:
            ref = tot = 0
            for key, s in SAMPLES.items():
                h = hyps.get(key)
                if h is None:
                    continue
                canon = C._canon(C._match_stored_format(h), "en")
                for r in s["runs"]:
                    if r["n_wl_agree"] != r["n_wl_total"]:
                        continue
                    v = C.cohere_verdict_for_run(r, s["ref_tokens"], canon)
                    if v in ("ref", "consensus"):
                        tot += 1
                        ref += v == "ref"
            cells[m]["full"] = (ref, tot)
        if m in iso:
            d = iso[m]
            nd, ni = d["all_delete"]["n"], d["all_insert"]["n"]
            kd = round(nd * (1 - d["all_delete"]["rate"]))
            ki = round(ni * (1 - d["all_insert"]["rate"]))
            cells[m]["truncated"] = (kd + ki, nd + ni)
        c = r8.get(m, {}).get("ctlsuf8")
        if c and c.get("rate") is not None:
            cells[m]["donor"] = (c["ref"], c["n"])
        s = STEER_SHORT.get(m)
        if s and (DR / f"steer/newwl4/ablate_acceptref_{s}.json").exists():
            # rescored against the newwl4-unanimous edit set (same panel as every other column)
            o = json.load(open(DR / f"steer/newwl4/ablate_acceptref_{s}.json"))["ablate"]
            cells[m]["activation"] = (o["n_ref"], o["n_eligible"])
        print(f"[cells] {m}: " + "  ".join(f"{ck}={k}/{n}={k / n:.3f}" for ck, (k, n) in cells[m].items()), flush=True)
    CACHE.write_text(json.dumps(cells, indent=1))
    return cells


# dropped from the MAIN figure (honest floor throughout); kept in the _full appendix variant.
TRIM = ("kimi-audio-7b", "moonshine-streaming-medium", "qwen3-asr-0.6b")


def render(cells, out_stem, row_order, xlabel="reference-disagreement accept-ref", legend=True):
    order = [m for m in reversed(row_order) if m in cells and cells[m]]
    nb = len(COND)
    h = 0.8 / nb
    y = np.arange(len(order))
    ypos = {m: y[i] for i, m in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(order) + 0.9))
    labeled = set()
    for bi, (ckey, clab, col) in enumerate(COND):
        offs = (bi - (nb - 1) / 2) * h
        for m in order:
            kn = cells[m].get(ckey)
            if not kn or not kn[1]:
                continue
            k, n = kn
            lo, hi = wilson(k, n)
            yy = ypos[m] + offs
            lab = clab if ckey not in labeled else None
            labeled.add(ckey)
            ax.barh(yy, k / n, height=h, color=col, linewidth=0, zorder=2, label=lab)
            ax.plot([lo, hi], [yy, yy], color=HUME["err"], lw=1.0, zorder=3)

    ax.set_yticks([ypos[m] for m in order])
    ax.set_yticklabels([LABELS.get(m, m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, None)
    style_h_bar_ax(ax, xlabel=xlabel, invert_y=False)
    if legend:
        # legend order must match the bars' top-to-bottom order; offs puts COND[0] at the
        # bottom of each group, so reverse the insertion-order handles.
        _h, _l = ax.get_legend_handles_labels()
        ax.legend(_h[::-1], _l[::-1], loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    print("wrote", *[p.name for p in paths], f"({len(order)} models)")


def main():
    apply_style()
    if "--rebuild" in sys.argv or not CACHE.exists():
        cells = build_cells()
    else:
        cells = json.load(open(CACHE))
    stem = OUT[:-4]  # strip .png; save_figure adds extensions
    render(cells, stem, [m for m in ORDER if m not in TRIM])
    render(cells, stem + "_full", ORDER)


if __name__ == "__main__":
    main()
