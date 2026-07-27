#!/usr/bin/env python
"""Assign each ep-fresh video to its best-matching CRE.

Plenary sittings cluster in session weeks, so the YouTube upload date only
hints at the date. We instead pick, per video, the cached CRE whose EN gold
maximizes the video's floor-English yield: for each candidate CRE we score
every staged clip's silver text (rapidfuzz partial_ratio) against the CRE's
concatenated EN gold and count clips >= --match-thresh. The CRE with the most
matches wins; we record the yield so low-yield videos can be flagged.

CPU-only; run on a SLURM cpu node (or login under ulimit -v).

    .venv-data/bin/python scripts/ep_fresh/assign_cre.py \
        --cache-dir <cre_cache> --video-dates <video_dates.json> \
        --staging <_staging> --out <assignments.json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cre_parse import gold_concat, parse_cre  # noqa: E402

_ALNUM_SP = re.compile(r"[^a-z0-9 ]")


def norm(s: str) -> str:
    return _ALNUM_SP.sub("", s.lower())


def window(yyyymmdd: str, before: int = 16, after: int = 16) -> list[str]:
    d = dt.datetime.strptime(yyyymmdd, "%Y%m%d").date()
    out = []
    for off in sorted(range(-before, after + 1), key=abs):
        c = d + dt.timedelta(days=off)
        if c.weekday() <= 4:
            out.append(c.strftime("%Y-%m-%d"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--video-dates", required=True)
    ap.add_argument("--staging", default="$BENCHMARK_OPT_DATA/datasets/ep-fresh-2026/_staging")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-thresh", type=float, default=85.0)
    ap.add_argument("--sample", type=int, default=120, help="clips per video to score for assignment (speed).")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    index = json.loads((cache / "cre_index.json").read_text())
    valid = {d: v for d, v in index.items() if v.get("status") == "ok"}
    # pre-parse + concat each valid CRE once
    gold_norm: dict[str, str] = {}
    for d, v in valid.items():
        turns = parse_cre(cache / v["file"], lang="EN")
        _, gn, _ = gold_concat(turns)
        gold_norm[d] = gn
    print(f"{len(valid)} valid CREs loaded", flush=True)

    video_dates = json.loads(Path(args.video_dates).read_text())
    staging = Path(args.staging)
    assignments = {}
    flagged = []
    for vid, ud in sorted(video_dates.items()):
        clips_path = staging / f"clips_{vid}.jsonl"
        if not clips_path.exists():
            flagged.append((vid, "no_clips"))
            continue
        clips = [json.loads(l) for l in clips_path.read_text().splitlines() if l.strip()]
        sample = clips[: args.sample]
        qs = [norm(c["text"]) for c in sample if c.get("text")]
        # candidate CREs = those in this video's date window that are valid
        cands = [d for d in window(ud) if d in gold_norm]
        if not cands:  # fall back to all valid (rare)
            cands = list(gold_norm)
        best_d, best_yield = None, -1
        scores = {}
        for d in cands:
            gn = gold_norm[d]
            hits = sum(1 for q in qs if q and fuzz.partial_ratio(q, gn) >= args.match_thresh)
            yld = hits / max(1, len(qs))
            scores[d] = round(yld, 3)
            if hits > best_yield:
                best_yield, best_d = hits, d
        yield_frac = scores.get(best_d, 0.0)
        assignments[vid] = {
            "upload_date": ud,
            "cre_date": best_d,
            "cre_file": valid[best_d]["file"] if best_d else None,
            "match_yield": yield_frac,
            "n_clips": len(clips),
            "candidate_scores": scores,
        }
        flag = "" if yield_frac >= 0.20 else "  <-- LOW YIELD"
        print(f"{vid} (up {ud}) -> CRE {best_d} yield={yield_frac:.2f} n_clips={len(clips)}{flag}", flush=True)
        if yield_frac < 0.20:
            flagged.append((vid, f"low_yield={yield_frac:.2f}"))

    Path(args.out).write_text(json.dumps(assignments, indent=2))
    print("=" * 60, flush=True)
    print(f"assigned {len(assignments)} videos -> {args.out}", flush=True)
    if flagged:
        print(f"FLAGGED ({len(flagged)}): {flagged}", flush=True)


if __name__ == "__main__":
    main()
