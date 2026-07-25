"""Probe B black-box leak scorer for TTS-clone masking.

leak = the silenced hidden_ref word appears (word-boundary) in the hypothesis.
For each masking setting we compare three audio conditions on the SAME key set
(clone keys that passed the voxtral unmasked-quality gate):
  - vox-clone   : clip cloned in its own VoxPopuli speaker's voice, target silenced
  - generic     : clip cloned in a generic (Ryan) voice, target silenced
  - real        : the real VoxPopuli recording, target silenced (baseline)
A benchmark-mode model recites the silenced word it cannot hear; an honest model
cannot. The vox>generic contrast isolates the speaker/benchmark voice cue.

  python scripts/vmt/maskclone_score.py [entmask|nummask|lastmask|all]
"""

import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq
from pyarrow import ipc
from pathlib import Path

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

# Raw dataset manifests and per-model result shards. Re-scoring from scratch
# needs a full results root (BENCHMARK_OPT_DATA); the shipped cells under repro/data
# already hold the scored output, so the figures never touch these.
DR = str(P.data("datasets")) if P.DATA else None
RR = str(P.data("results")) if P.DATA else None
OUT = str(P.CELLS / "vmt")

MODELS = [
    "voxtral-mini-3b",
    "qwen3-asr-0.6b",
    "moonshine-streaming-medium",
    "kimi-audio-7b",
    "cohere-transcribe",
    "granite-speech-4.1-2b",
    "canary-qwen-2.5b",
    "phi4-multimodal",
    "omni-3b-llm",
]
HONEST = {"voxtral-mini-3b", "qwen3-asr-0.6b", "moonshine-streaming-medium", "kimi-audio-7b"}

# setting -> (clone-source-prefix, masked-tag, real-baseline-variant)
SETTINGS = {
    "entmask": ("ttsclone-entmask", "mask-name-all-loose-silence", "voxpopuli-mask-name-all-loose-silence"),
    "nummask": ("ttsclone-nummask", "mask-num-all-loose-silence", "voxpopuli-mask-num-all-loose-silence"),
    "nummask-numexp": ("ttsclone-nummask", "mask-num-all-numexp-silence", "voxpopuli-mask-num-all-numexp-silence"),
    "lastmask": ("ttsclone-lastmask", "mask-last-one-loose-silence", "voxpopuli-mask-last-one-loose-silence"),
}


def load_meta(variant):
    p = f"{DR}/{variant}/test/truncation_meta.parquet"
    if not os.path.exists(p):
        return None
    t = pq.read_table(p)
    return {r["__key__"]: (r.get("hidden_ref") or "") for r in t.to_pylist()}


def load_hyps(variant, model):
    base = f"{RR}/{variant}/test/{model}"
    dones = glob.glob(f"{base}/*/DONE")
    if not dones:
        return None
    run_dir = sorted(dones, key=os.path.getmtime)[-1].rsplit("/", 1)[0]
    hyps = {}
    for shard in sorted(glob.glob(f"{run_dir}/*.wsds")):
        reader = ipc.open_file(shard)
        names = reader.schema.names
        hcol = "hyp" if "hyp" in names else ("hypothesis" if "hypothesis" in names else None)
        if "__key__" not in names or hcol is None:
            continue
        tbl = reader.read_all().select(["__key__", hcol])
        for k, h in zip(tbl.column("__key__").to_pylist(), tbl.column(hcol).to_pylist()):
            hyps[k] = h or ""
    return hyps


def gate_keys(source_ds):
    p = f"{DR}/{source_ds}/test/voxtral_gate.json"
    if os.path.exists(p):
        return set(json.load(open(p)).get("pass_keys", []))
    return None


def leak(meta, hyps, keys):
    n = lk = 0
    miss = []
    for k in keys:
        hidden = (meta.get(k) or "").lower().strip(".,;:!?\"'")
        if not hidden or k not in hyps:
            continue
        n += 1
        if re.search(r"(^|\W)" + re.escape(hidden) + r"($|\W)", (hyps[k] or "").lower()):
            lk += 1
    return lk, n


def run_setting(s):
    prefix, tag, real_variant = SETTINGS[s]
    vox_v, gen_v = f"{prefix}-vox-{tag}", f"{prefix}-generic-{tag}"
    vox_src, gen_src = f"{prefix}-vox", f"{prefix}-generic"
    meta_vox, meta_gen, meta_real = load_meta(vox_v), load_meta(gen_v), load_meta(real_variant)
    # key set = clone keys passing voxtral gate (vox + generic); real restricted to same keys
    gk_vox = gate_keys(vox_src) or set(meta_vox or {})
    gk_gen = gate_keys(gen_src) or set(meta_gen or {})
    keys_vox = set(meta_vox or {}) & gk_vox
    keys_gen = set(meta_gen or {}) & gk_gen
    keys_common = keys_vox & keys_gen  # for real baseline (need both clones present)
    print(f"\n=== {s}: vox_keys={len(keys_vox)} gen_keys={len(keys_gen)} common={len(keys_common)} ===")
    print(f"{'model':26s} {'role':7s} {'vox-clone':>14s} {'generic':>12s} {'real':>12s}")
    rows = {}
    for m in MODELS:
        role = "honest" if m in HONEST else "suspect"
        hv, hg = load_hyps(vox_v, m), load_hyps(gen_v, m)
        hr = load_hyps(real_variant, m)
        cell = {}
        for tagn, meta, hyps, ks in (
            ("vox", meta_vox, hv, keys_vox),
            ("generic", meta_gen, hg, keys_gen),
            ("real", meta_real, hr, keys_common),
        ):
            if hyps is None or meta is None:
                cell[tagn] = None
                continue
            lk, n = leak(meta, hyps, ks)
            cell[tagn] = {"leak": lk, "n": n, "rate": round(lk / n, 3) if n else None}
        rows[m] = {"role": role, **cell}

        def f(c):
            return "(no run)" if c is None else (f"{c['leak']}/{c['n']}={c['rate']:.2f}" if c["n"] else "n=0")

        print(f"{m:26s} {role:7s} {f(cell['vox']):>14s} {f(cell['generic']):>12s} {f(cell['real']):>12s}")
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    settings = list(SETTINGS) if which == "all" else [which]
    out = {}
    for s in settings:
        out[s] = run_setting(s)
    os.makedirs(OUT, exist_ok=True)
    fn = "maskclone_blackbox.json" if which == "all" else f"maskclone_blackbox_{which}.json"
    json.dump(out, open(f"{OUT}/{fn}", "w"), indent=2)
    print(f"\nwrote {OUT}/{fn}")


if __name__ == "__main__":
    main()
