"""Re-derive shipped figure data from a full results root.

The repository ships small derived "cells" under ``repro/data`` so that figures
rebuild in a clean clone. This script regenerates the cells that come from raw
per-model outputs, which are too large to ship. It needs ``BENCHMARK_OPT_DATA``
pointing at a results root laid out as:

    $BENCHMARK_OPT_DATA/
      datasets/{corpus}/{split}/manifest.parquet      # __key__, language, text
      results/{corpus}/{split}/{model}/{run}/*.wsds   # __key__, hyp[, hyp_raw]
                                                     # plus a DONE marker per run

Arrow IPC files are read for hypotheses; ``hyp_raw`` is preferred over ``hyp``
because the switch probe needs un-normalized text.

Currently regenerates:

``ortho_switch.json``
    Orthographic switch rates, computed with :mod:`benchmark_optimization.ortho`. Two cells:
    the within-LibriSpeech pooled spacing group, and the cross-corpus honorific
    comparison between VoxPopuli and LibriSpeech.

Usage::

    BENCHMARK_OPT_DATA=/path/to/results python repro/precompute_cells.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow.ipc as ipc

import paths as P
from benchmark_optimization import conventions, ortho

LIBRI = ["librispeech-clean", "librispeech-other"]
VOX = "voxpopuli"


def load_refs(corpus: str, split: str = "test", language: str = "en") -> dict[str, str]:
    man = P.data("datasets", corpus, split, "manifest.parquet")
    df = pl.read_parquet(man, columns=["__key__", "language", "text"])
    df = df.filter(pl.col("language") == language)
    return dict(zip(df["__key__"].to_list(), df["text"].to_list()))


def load_hyps(corpus: str, keys: set[str], split: str = "test") -> dict[str, dict[str, str]]:
    """``{key: {model: raw hypothesis}}`` for the newest completed run per model."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    root = P.data("results", corpus, split)
    if not root.is_dir():
        return out
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        runs = [
            r for r in model_dir.iterdir()
            if r.is_dir() and (r / "DONE").exists() and not r.name.startswith("legacy_")
        ]
        if not runs:
            continue
        run = max(runs, key=lambda r: r.stat().st_mtime)
        for shard in sorted(run.glob("*.wsds")):
            try:
                tbl = ipc.open_file(str(shard)).read_all()
            except Exception:
                continue
            names = tbl.column_names
            if "__key__" not in names:
                continue
            # Prefer raw text per row, falling back to the normalized column
            # where raw is blank. Choosing one column for the whole shard would
            # silently drop every row a model left raw-empty, which can push a
            # model under the per-arm floor and out of the figure.
            raw_col = next((c for c in ("hyp_raw", "hypothesis_raw") if c in names), None)
            norm_col = next((c for c in ("hyp", "hypothesis") if c in names), None)
            if raw_col is None and norm_col is None:
                continue
            n = tbl.num_rows
            ks = tbl.column("__key__").to_pylist()
            raws = tbl.column(raw_col).to_pylist() if raw_col else [None] * n
            norms = tbl.column(norm_col).to_pylist() if norm_col else [None] * n
            for k, raw, norm in zip(ks, raws, norms):
                if k not in keys:
                    continue
                h = raw if (raw and raw.strip()) else norm
                if h is not None:
                    out[k][model_dir.name] = h
    return out


def clips_for(corpora: list[str]) -> list[tuple[str, dict[str, str]]]:
    clips = []
    for corpus in corpora:
        refs = load_refs(corpus)
        hyps = load_hyps(corpus, set(refs))
        for key, ref in refs.items():
            got = hyps.get(key)
            if got:
                clips.append((ref or "", got))
    return clips


def _serialize(results: dict[str, ortho.SwitchResult]) -> dict:
    return {
        model: {
            "switch": r.switch,
            "lo": r.lo,
            "hi": r.hi,
            "limiting_arm": r.limiting_arm,
            "arms": {a: {"hits": t.hits, "n": t.n} for a, t in r.arms.items()},
        }
        for model, r in results.items()
    }


def main() -> None:
    P.require_data()

    libri = clips_for(LIBRI)
    print(f"librispeech clips with hypotheses: {len(libri)}")
    spacing = ortho.pooled_switch_rate(
        list(conventions.SPACING_PAIRS),
        libri,
        arm_names=conventions.SPACING_ARMS,
        name="spacing",
        min_per_arm=5,
    )
    print(f"  pooled spacing: {len(spacing)} models measurable")

    both = clips_for([VOX] + LIBRI)
    print(f"vox+librispeech clips with hypotheses: {len(both)}")
    # Cross-corpus: VoxPopuli references abbreviate, LibriSpeech spells out, so
    # the two arms come from different corpora. min_per_arm is lower here
    # because honorifics are rarer than the pooled spacing group.
    mister = ortho.switch_rate(conventions.family("hon_mister"), both, min_per_arm=3)
    print(f"  honorific mister: {len(mister)} models measurable")

    payload = {
        "spacing": {
            "scope": "within-librispeech, pooled over 4 spacing families",
            "arms": list(conventions.SPACING_ARMS),
            "chance": 0.5,
            "min_per_arm": 5,
            "results": _serialize(spacing),
        },
        "hon_mister": {
            "scope": "cross-corpus: voxpopuli (abbreviated) vs librispeech (spelled out)",
            "arms": list(conventions.family("hon_mister").arm_labels),
            "chance": 0.5,
            "min_per_arm": 3,
            "results": _serialize(mister),
        },
    }
    out = P.CELLS / "ortho_switch.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
