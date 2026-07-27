"""Export our result shards to the published prediction dataset.

Produces the layout of
[HumeAI/ASR-benchmark-optimization-predictions](https://huggingface.co/datasets/HumeAI/ASR-benchmark-optimization-predictions):

    predictions/{corpus}/{model}.jsonl    Open ASR Leaderboard manifest format
    masks/{variant}.parquet               truncation_meta, the mask recipes

Predictions carry raw text, since the orthographic switch rate needs it. Timing
comes from ``inference_time`` in the shard; duration from the source manifest.

Needs BENCHMARK_OPT_DATA. Nothing under ``daikon`` is exported — it is a private
held-out control.

    BENCHMARK_OPT_DATA=/path/to/results python repro/export_predictions.py --out staging/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import pyarrow.ipc as ipc

import paths as P

#: Corpora the paper's figures draw on. Deliberately an allowlist: the results
#: root holds ~300 experiment variants, most of which are not paper artifacts.
BASE_CORPORA = [
    "voxpopuli",
    "librispeech-clean",
    "librispeech-other",
    "ep-fresh-vox",
    "librivoxfresh",
]

#: Masked variants of the above, by tag. Missing combinations are skipped.
MASK_TAGS = [
    "mask-num-all-numexp-silence",
    "mask-num-all-loose-silence",
    "mask-name-all-loose-silence",
    "mask-last-one-loose-silence",
    "mask-num-truncated",
]

#: Never exported, at any level.
EXCLUDE = ("daikon",)

LANGUAGE = "en"


def corpora() -> list[str]:
    out = list(BASE_CORPORA)
    out += [f"{c}-{t}" for c in BASE_CORPORA for t in MASK_TAGS]
    return [c for c in out if not any(x in c for x in EXCLUDE)]


def load_manifest(corpus: str, split: str) -> dict[str, tuple[str, float]]:
    """``{key: (raw reference, duration)}`` for one corpus, filtered to LANGUAGE."""
    man = P.data("datasets", corpus, split, "manifest.parquet")
    if not man.exists():
        return {}
    have = set(pl.read_parquet_schema(man))
    # The truncation builder writes raw_text/normalized_text where the wav
    # converter writes text/text_normalized.
    text_col = next((c for c in ("text", "raw_text") if c in have), None)
    if text_col is None:
        print(f"  skip {corpus}: no reference column in manifest ({sorted(have)})")
        return {}
    cols = [c for c in ("__key__", text_col, "duration", "language") if c in have]
    df = pl.read_parquet(man, columns=cols)
    if "language" in have:
        df = df.filter(pl.col("language") == LANGUAGE)
    dur = df["duration"].to_list() if "duration" in have else [None] * df.height
    return {k: (t or "", d) for k, t, d in zip(df["__key__"].to_list(), df[text_col].to_list(), dur)}


def newest_run(model_dir: Path) -> Path | None:
    runs = [
        r for r in model_dir.iterdir()
        if r.is_dir() and (r / "DONE").exists() and not r.name.startswith("legacy_")
    ]
    return max(runs, key=lambda r: r.stat().st_mtime) if runs else None


def read_run(run: Path, keys: set[str]) -> dict[str, tuple[str, float | None]]:
    """``{key: (raw hypothesis, inference_time)}`` for one run."""
    out: dict[str, tuple[str, float | None]] = {}
    for shard in sorted(run.glob("*.wsds")):
        try:
            tbl = ipc.open_file(str(shard)).read_all()
        except Exception:
            continue
        names = tbl.column_names
        if "__key__" not in names:
            continue
        ks = tbl.column("__key__").to_pylist()
        raw = tbl.column("hyp_raw").to_pylist() if "hyp_raw" in names else [None] * len(ks)
        norm = tbl.column("hyp").to_pylist() if "hyp" in names else [None] * len(ks)
        secs = tbl.column("inference_time").to_pylist() if "inference_time" in names else [None] * len(ks)
        for k, r, n, s in zip(ks, raw, norm, secs):
            if k not in keys:
                continue
            h = r if (r and r.strip()) else n
            if h is not None:
                out[k] = (h, s)
    return out


def export_predictions(out_root: Path, split: str) -> list[tuple[str, str, int, int]]:
    rows = []
    for corpus in corpora():
        refs = load_manifest(corpus, split)
        if not refs:
            continue
        results = P.data("results", corpus, split)
        if not results.is_dir():
            continue
        dest = out_root / "predictions" / corpus
        for model_dir in sorted(p for p in results.iterdir() if p.is_dir()):
            run = newest_run(model_dir)
            if run is None:
                continue
            hyps = read_run(run, set(refs))
            if not hyps:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{model_dir.name}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for key, (hyp, secs) in sorted(hyps.items()):
                    ref, dur = refs[key]
                    rec = {"audio_filepath": key, "text": ref, "pred_text": hyp}
                    if dur is not None:
                        rec["duration"] = round(float(dur), 3)
                    if secs is not None:
                        rec["time"] = round(float(secs), 4)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append((corpus, model_dir.name, len(hyps), path.stat().st_size))
    return rows


def export_masks(out_root: Path, split: str) -> list[tuple[str, int, int]]:
    rows = []
    dest = out_root / "masks"
    for corpus in corpora():
        if "mask" not in corpus:
            continue
        src = P.data("datasets", corpus, split, "truncation_meta.parquet")
        if not src.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        df = pl.read_parquet(src)
        if "source_dataset" in df.columns:
            df = df.filter(~pl.col("source_dataset").str.contains("|".join(EXCLUDE)))
        if not df.height:
            continue
        path = dest / f"{corpus}.parquet"
        df.write_parquet(path)
        rows.append((corpus, df.height, path.stat().st_size))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging directory")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    P.require_data()

    out = Path(args.out)
    preds = export_predictions(out, args.split)
    masks = export_masks(out, args.split)

    by_corpus: dict[str, list[int]] = {}
    for corpus, _model, n, size in preds:
        c = by_corpus.setdefault(corpus, [0, 0, 0])
        c[0] += 1
        c[1] += n
        c[2] += size
    print(f"{'corpus':46} {'models':>6} {'clips':>9} {'MB':>7}")
    print("-" * 72)
    for corpus, (nm, nc, sz) in sorted(by_corpus.items()):
        print(f"{corpus:46} {nm:6} {nc:9} {sz/1e6:7.1f}")
    print(f"\npredictions: {len(preds)} files, {sum(r[3] for r in preds)/1e6:.1f} MB")
    print(f"masks:       {len(masks)} files, {sum(r[2] for r in masks)/1e6:.1f} MB")
    for corpus, n, _s in sorted(masks):
        print(f"   {corpus:56} {n:6} rows")


if __name__ == "__main__":
    main()
