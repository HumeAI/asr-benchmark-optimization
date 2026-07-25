"""Consistent masked-number readout across the ablation conditions (Fig 3d cells).

Rescores every arm with the shared format-robust matcher (nummask_match.masked_hit):
  full        : nummask-real-splice8 __baseline WS2 hyps (real masked clips).
  donor       : nummask-real-splice8 __ctlsuf8 (8 s conversational donor appended).
  voxdonor    : __voxsuf8 (specificity control, not plotted in the main panel).
  truncated   : voxpopuli-mask-num-truncated WS2 hyps (window around the silenced span),
                when decoded.
  activation  : ablate_masked_<m>.json steering hyps (ablate arm; none/random kept for QA).

Out: analysis/voxmode/vmt/nummask_ablation_cells.json  {model: {cond: [k, n]}}

  (ulimit -v 16000000; .venv-data/bin/python scripts/vmt/nummask_ablation_cells.py)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nummask_match import masked_hit  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

DR = P.require_data()
OUT = DR / "analysis/voxmode/vmt/nummask_ablation_cells.json"
META = DR / "datasets/voxpopuli-mask-num-all-numexp-silence/test/truncation_meta.parquet"
SPLICE_DS = "nummask-real-splice8"
TRUNC_DS = "voxpopuli-mask-num-truncated"
MODELS = [
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
STEER_SHORT = {
    "cohere-transcribe": "cohere",
    "parakeet-tdt-0.6b-v2": "parakeet",
    "canary-qwen-2.5b": "canary",
    "granite-speech-4.1-2b": "granite",
    "phi4-multimodal": "phi4",
}


def load_wsds(ds, model):
    """Latest DONE run's {__key__: hyp_raw} for ds/model (same walk as splice_trigger_full_score)."""
    import glob
    import os

    from pyarrow import ipc

    base = DR / f"results/{ds}/test/{model}"
    dones = glob.glob(f"{base}/*/DONE")
    if not dones:
        return None
    run_dir = sorted(dones, key=os.path.getmtime)[-1].rsplit("/", 1)[0]
    hyps = {}
    for shard in sorted(glob.glob(f"{run_dir}/*.wsds")):
        rd = ipc.open_file(shard)
        names = rd.schema.names
        hc = next((c for c in ("hyp_raw", "hypothesis_raw", "hyp", "hypothesis") if c in names), None)
        if "__key__" not in names or hc is None:
            continue
        t = rd.read_all().select(["__key__", hc])
        for k, h in zip(t.column("__key__").to_pylist(), t.column(hc).to_pylist()):
            hyps[k] = h or ""
    return hyps


def main():
    import pyarrow.parquet as pq

    meta = {r["__key__"]: (r["hidden_ref"] or "") for r in pq.read_table(META).to_pylist()}
    targets = {k: t for k, t in meta.items() if t.strip()}
    cells = {}
    for m in MODELS:
        cells[m] = {}
        sp = load_wsds(SPLICE_DS, m)
        if sp:
            for cond, suf in (("full", "baseline"), ("donor", "ctlsuf8"), ("voxdonor", "voxsuf8")):
                k_ = n_ = 0
                for b, tgt in targets.items():
                    h = sp.get(f"{b}__{suf}")
                    if h is None:
                        continue
                    n_ += 1
                    k_ += masked_hit(tgt, h)
                cells[m][cond] = [k_, n_]
        tr = load_wsds(TRUNC_DS, m)
        if tr:
            k_ = n_ = 0
            for b, tgt in targets.items():
                h = tr.get(b)
                if h is None:
                    continue
                n_ += 1
                k_ += masked_hit(tgt, h)
            cells[m]["truncated"] = [k_, n_]
        s = STEER_SHORT.get(m)
        ab_p = DR / f"analysis/voxmode/steer/ablate_masked_{s}.json" if s else None
        if ab_p and ab_p.exists():
            rows = json.load(open(ab_p))["rows"]
            for cond, hyp_key in (
                ("activation", "hyp_ablate"),
                ("steer_none", "hyp_none"),
                ("steer_random", "hyp_random"),
            ):
                pairs = [(r["tgt"], r[hyp_key]) for r in rows if not r[f"garbage_{hyp_key.split('_')[1]}"]]
                cells[m][cond] = [sum(masked_hit(t, h) for t, h in pairs), len(pairs)]
        print(f"{m:28s} " + "  ".join(f"{c}={k}/{n}={k / n:.3f}" for c, (k, n) in cells[m].items() if n))
    OUT.write_text(json.dumps(cells, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
