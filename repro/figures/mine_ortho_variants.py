"""Mine orthographic-variant choices across libri + vox + tedlium.

Tests dataset-specific formatting conventions: Mr. vs mister, U.S. vs US,
gonna vs going to, % vs percent, etc. A model that conditionally matches
the SAME orthography the ref uses, where audio gives no cue, is showing
contamination from the benchmark's ref corpus.

For each group, computes per-model `min(P(hyp=V | ref=V))` across variants
with ≥3 samples (switch score). Renders cards split by which variant the
ref uses so you can eyeball.
"""

from __future__ import annotations

import html
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
import pyarrow.ipc as ipc
import sys

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

try:  # removed from wsds_query in the manifest refactor; run ids sort lexicographically
    from asr_benchmarking.leaderboard.wsds_query import _run_sort_key
except ImportError:

    def _run_sort_key(run_dir):
        return run_dir.name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mine_ortho")

# Each group: list of (label, regex-pattern-string). All regexes match
# case-insensitively. Use raw pattern strings so the script controls boundary
# logic per-variant.
#   • Punctuation-bearing variants (Mr., Dr., U.S.) capture the period.
#   • Token-internal variants ("don't") capture the apostrophe in both ASCII
#     ' and curly ’ forms.
#   • "Mr" without period: use Mr\b(?!\.) — match "Mr " but NOT "Mr."
# All groups below have variants that are ACOUSTICALLY IDENTICAL in normal
# speech — differ only in surface orthography (capitalization, punctuation,
# hyphenation, spacing, abbreviation choice). High switch score on these
# = model is reading the ref, not the audio.
#
# Dropped from earlier version: yes/yeah, cannot/can't, do not/don't,
# going to/gonna, until/till, etc. — those are phonetically distinct.
ORTHO_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    # ── Abbreviation vs spelled-out (identical pronunciation) ──────────
    ("okay / ok", [
        ("okay", r"(?i)\bokay\b"),
        ("ok",   r"(?i)\bok\b(?!\.)"),
        ("O.K.", r"(?i)\bo\.k\.\b"),
    ]),
    ("mister / Mr.", [
        ("mister", r"(?i)\bmister\b"),
        ("Mr.",    r"(?i)\bmr\."),
        ("Mr",     r"(?i)\bmr\b(?!\.)"),
    ]),
    ("doctor / Dr.", [
        ("doctor", r"(?i)\bdoctor\b"),
        ("Dr.",    r"(?i)\bdr\."),
        ("Dr",     r"(?i)\bdr\b(?!\.)"),
    ]),
    ("professor / Prof.", [
        ("professor", r"(?i)\bprofessor\b"),
        ("Prof.",     r"(?i)\bprof\."),
        ("Prof",      r"(?i)\bprof\b(?!\.)"),
    ]),
    ("versus / vs.", [
        ("versus", r"(?i)\bversus\b"),
        ("vs.",    r"(?i)\bvs\."),
        ("vs",     r"(?i)\bvs\b(?!\.)"),
    ]),
    ("dollars / $", [
        ("dollars", r"(?i)\bdollars?\b"),
        ("$",       r"\$\s*\d"),
    ]),
    ("euros / €", [
        ("euros", r"(?i)\beuros?\b"),
        ("€",     r"€\s*\d|\d\s*€"),
    ]),
    ("percent / %", [
        ("percent",  r"(?i)\bpercent\b"),
        ("per cent", r"(?i)\bper cent\b"),
        ("%",        r"\d\s*%"),
    ]),
    ("all right / alright", [
        ("all right", r"(?i)\ball right\b"),
        ("alright",   r"(?i)\balright\b"),
    ]),

    # ── Compound noun spacing/hyphenation (identical pronunciation) ────
    ("email", [
        ("email",  r"(?i)\bemail\b"),
        ("e-mail", r"(?i)\be[- ]mail\b"),
    ]),
    ("online", [
        ("online",  r"(?i)\bonline\b"),
        ("on-line", r"(?i)\bon[- ]line\b"),
    ]),
    ("website", [
        ("website",  r"(?i)\bwebsite\b"),
        ("web site", r"(?i)\bweb site\b"),
    ]),
    ("database", [
        ("database",  r"(?i)\bdatabase\b"),
        ("data base", r"(?i)\bdata base\b"),
    ]),
    ("policymaker", [
        ("policymaker",  r"(?i)\bpolicymakers?\b"),
        ("policy maker", r"(?i)\bpolicy makers?\b"),
    ]),
    ("decision making", [
        ("decision-making", r"(?i)\bdecision-making\b"),
        ("decision making", r"(?i)\bdecision making\b"),
    ]),
    ("long term", [
        ("long-term", r"(?i)\blong-term\b"),
        ("long term", r"(?i)\blong term\b"),
    ]),
    ("state of the art", [
        ("state-of-the-art", r"(?i)\bstate-of-the-art\b"),
        ("state of the art", r"(?i)\bstate of the art\b"),
    ]),
    ("follow up", [
        ("follow-up", r"(?i)\bfollow-up\b"),
        ("follow up", r"(?i)\bfollow up\b"),
    ]),
    ("cooperate", [
        ("cooperate",  r"(?i)\bcoo?perate\b"),
        ("co-operate", r"(?i)\bco-operate\b"),
    ]),
    ("reenter", [
        ("reenter",  r"(?i)\breenter\b"),
        ("re-enter", r"(?i)\bre-enter\b"),
    ]),
    ("nonzero", [
        ("nonzero",  r"(?i)\bnonzero\b"),
        ("non-zero", r"(?i)\bnon-zero\b"),
    ]),
    ("twentieth century", [
        ("twentieth-century", r"(?i)\btwentieth-century\b"),
        ("twentieth century", r"(?i)\btwentieth century\b"),
    ]),

    # ── Year transcription (digits vs words — SAME pronunciation) ──────
    # "2020" is read as "twenty twenty"; written-form choice doesn't affect
    # audio. (We drop "two thousand twenty" because that IS a different
    # pronunciation.)
    ("year 2020", [
        ("twenty twenty",       r"(?i)\btwenty twenty\b(?!.{0,3}\d)"),
        ("2020",                r"\b2020\b"),
    ]),
    ("year 1999", [
        ("nineteen ninety nine", r"(?i)\bnineteen ninety[ -]?nine\b"),
        ("1999",                 r"\b1999\b"),
    ]),
]

# DROPPED (phonetically distinguishable — keeping these confused the test):
#   • U.S. / United States  (2 vs 4 syllables)
#   • U.K. / United Kingdom
#   • E.U. / European Union
#   • two thousand twenty / twenty twenty (different word counts)
# Also dropped from earlier version:
#   • yes/yeah, cannot/can't, do not/don't, going to/gonna, want to/wanna,
#     got to/gotta, because/'cause, until/till — all phonetically distinct.

WL_EN = {
    "gemini-3-flash-preview", "whisper-large-v3", "qwen3-asr-1.7b",
    "phi4-multimodal", "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite-preview-langhint"}

DATA = P.require_data()
DATASETS = ["voxpopuli", "librispeech-clean", "librispeech-other", "tedlium"]
MAX_PER_VARIANT_RENDER = 15


def compile_groups():
    out = []
    for label, variants in ORTHO_GROUPS:
        out.append((label, [(name, re.compile(pat)) for name, pat in variants]))
    return out


def find_first_variant(text: str, variants: list[tuple[str, re.Pattern]]) -> str | None:
    """Return the label of the first variant pattern that matches in text."""
    for name, pat in variants:
        if pat.search(text):
            return name
    return None


def load_hyps(ds: str, keys: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    root = DATA / "results" / ds / "test"
    if not root.is_dir():
        return out
    for md in root.iterdir():
        if not md.is_dir():
            continue
        runs = [r for r in md.iterdir() if r.is_dir() and (r / "DONE").exists() and not r.name.startswith("legacy_")]
        if not runs:
            continue
        rd = max(runs, key=_run_sort_key)
        for shard in sorted(rd.glob("*.wsds")):
            try:
                tbl = ipc.open_file(str(shard)).read_all()
            except Exception:
                continue
            if "__key__" not in tbl.column_names or "hyp" not in tbl.column_names:
                continue
            ks_ = tbl.column("__key__").to_pylist()
            hs_ = tbl.column("hyp").to_pylist()
            hrs_ = tbl.column("hyp_raw").to_pylist() if "hyp_raw" in tbl.column_names else [None]*len(ks_)
            for k, h, hr in zip(ks_, hs_, hrs_):
                if k in keys:
                    v = hr if (hr and hr.strip()) else h
                    if v is not None:
                        out[k][md.name] = v
    return out


SPANISH_ORTHO_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("señor / Sr.", [
        ("señor", r"(?i)\bse[ñn]or\b"),
        ("Sr.",   r"(?i)\bsr\."),
        ("Sr",    r"(?i)\bsr\b(?!\.)"),
    ]),
    ("señora / Sra.", [
        ("señora", r"(?i)\bse[ñn]ora\b"),
        ("Sra.",   r"(?i)\bsra\."),
        ("Sra",    r"(?i)\bsra\b(?!\.)"),
    ]),
    ("señorita / Srta.", [
        ("señorita", r"(?i)\bse[ñn]orita\b"),
        ("Srta.",    r"(?i)\bsrta\."),
        ("Srta",     r"(?i)\bsrta\b(?!\.)"),
    ]),
    ("don / D.", [
        ("Don", r"\bDon\b"),
        ("D.",  r"\bD\."),
    ]),
    ("doña / Dña.", [
        ("doña", r"(?i)\bdo[ñn]a\b"),
        ("Dña.", r"(?i)\bd[ñn]a\."),
    ]),
    ("etcétera / etc.", [
        ("etcétera", r"(?i)\betc[eé]tera\b"),
        ("etc.",     r"(?i)\betc\."),
    ]),
    # DROPPED — `\bn[ºo]\b` matched bare Spanish "no" (negation), not "Nº".
    # Vox ES has 0 actual "Nº" abbreviations in refs. Test isn't testable.
    ("euros / €", [
        ("euros", r"(?i)\beuros?\b"),
        ("€",     r"€\s*\d|\d\s*€"),
    ]),
    ("por ciento / %", [
        ("por ciento", r"(?i)\bpor ciento\b"),
        ("%",          r"\d\s*%"),
    ]),
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="Comma-separated dataset list (overrides default)")
    ap.add_argument("--language", default="en", help="Filter manifest by this language code")
    ap.add_argument("--out", default=str(P.FIGURES / "ortho_variants.html"))
    args = ap.parse_args()
    datasets = args.datasets.split(",") if args.datasets else DATASETS
    out_path_arg = Path(args.out)

    # Pick the variant set by language
    if args.language == "es":
        groups_def = SPANISH_ORTHO_GROUPS
    else:
        groups_def = ORTHO_GROUPS

    def compile_es_or_en():
        return [(label, [(name, re.compile(pat)) for name, pat in variants]) for label, variants in groups_def]
    groups = compile_es_or_en()

    # Pass 1: scan refs, record which variant fires per (sample, group)
    # samples_by_group_variant[gi][variant] = [(ds, key, ref), ...]
    samples_by_group_variant: dict[int, dict[str, list[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(list))
    per_ds_total_hits: Counter = Counter()
    for ds in datasets:
        p = DATA / f"datasets/{ds}/test/manifest.parquet"
        if not p.exists():
            continue
        m = pl.read_parquet(p, columns=["__key__", "text", "language"])
        if "language" in m.columns:
            m = m.filter(pl.col("language") == args.language)
        n_hit = 0
        for k, t in zip(m["__key__"].to_list(), m["text"].to_list()):
            t = t or ""
            for gi, (label, variants) in enumerate(groups):
                v = find_first_variant(t, variants)
                if v is not None:
                    samples_by_group_variant[gi][v].append((ds, k, t))
                    n_hit += 1
        per_ds_total_hits[ds] = n_hit
        log.info("  %-22s scanned %5d EN samples; %d ortho hits", ds, m.height, n_hit)

    # Pass 2: load hyps for all keys with hits
    keys_by_ds: dict[str, set[str]] = defaultdict(set)
    for gi, vmap in samples_by_group_variant.items():
        for v, samples in vmap.items():
            for ds, k, _ in samples:
                keys_by_ds[ds].add(k)
    hyps_by_ds_key: dict[str, dict[str, dict[str, str]]] = {}
    for ds, keys in keys_by_ds.items():
        log.info("loading hyps for %s (%d keys)", ds, len(keys))
        hyps_by_ds_key[ds] = load_hyps(ds, keys)

    # Pass 3: compute per-model switch scores per group
    # group_stats[gi][mdl][variant] = {match: ..., total: ...}
    group_stats: dict[int, dict[str, dict[str, dict[str, int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"match": 0, "total": 0})))
    for gi, vmap in samples_by_group_variant.items():
        variants = groups[gi][1]
        for ref_v, samples in vmap.items():
            for ds, k, _ in samples:
                hyps = hyps_by_ds_key.get(ds, {}).get(k, {})
                for mdl, hyp in hyps.items():
                    hyp_v = find_first_variant(hyp, variants)
                    group_stats[gi][mdl][ref_v]["total"] += 1
                    if hyp_v == ref_v:
                        group_stats[gi][mdl][ref_v]["match"] += 1

    # Print per-group balanced switch scores
    print()
    print("=== Per-group variant counts (ref) ===")
    balanced_groups: list[int] = []
    for gi, (label, variants) in enumerate(groups):
        vmap = samples_by_group_variant[gi]
        if not vmap:
            continue
        counts = {v: len(vmap[v]) for v in vmap}
        # Balanced = ≥2 variants with ≥3 samples each
        big = [v for v, n in counts.items() if n >= 3]
        if len(big) < 2:
            continue
        balanced_groups.append(gi)
        print(f"  {label:<22} {counts}")
    print()
    print(f"balanced groups: {len(balanced_groups)}")

    print()
    print("=== Per-group switch scores (top 15) ===")
    agg_switch: dict[str, list[float]] = defaultdict(list)
    for gi in balanced_groups:
        label, variants = groups[gi]
        rows = []
        for mdl, vstats in group_stats[gi].items():
            per_v = {}
            for v_name, _ in variants:
                st = vstats.get(v_name, {"match": 0, "total": 0})
                if st["total"] >= 2:
                    per_v[v_name] = (st["match"], st["total"])
            if len(per_v) < 2:
                continue
            rates = {v: m / t for v, (m, t) in per_v.items()}
            switch = min(rates.values())
            rows.append((mdl, switch, per_v))
            agg_switch[mdl].append(switch)
        rows.sort(key=lambda r: -r[1])
        print(f"\n--- {label} ---")
        print(f'  {"model":<46}{"WL":>4}{"switch":>8}   per-variant rates')
        for mdl, switch, per_v in rows[:15]:
            wl = "WL" if mdl in WL_EN else ""
            rates_str = " | ".join(f"{v}={m}/{t}({m/t:.0%})" for v, (m, t) in per_v.items())
            print(f"  {mdl:<46}{wl:>4}{switch:>8.0%}   {rates_str}")

    print()
    print("=== Aggregate switch across balanced groups ===")
    print(f'  {"model":<46}{"WL":>4}{"n_groups":>10}{"mean_switch":>13}')
    for mdl in sorted(agg_switch, key=lambda x: -sum(agg_switch[x]) / max(len(agg_switch[x]), 1)):
        s = agg_switch[mdl]
        if len(s) < 2:
            continue
        mean = sum(s) / len(s)
        wl = "WL" if mdl in WL_EN else ""
        print(f"  {mdl:<46}{wl:>4}{len(s):>10}{mean:>13.0%}")

    # ─── HTML render ───────────────────────────────────────────────────────
    parts: list[str] = ['''<!doctype html><html><head><meta charset="utf-8">
<title>Orthographic variant switch test</title>
<style>
  body { font: 13px -apple-system, sans-serif; margin: 16px; max-width: 1300px; }
  h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 28px; padding: 6px 10px; background: #e3f2fd; border-left: 4px solid #1976d2; border-radius: 0 4px 4px 0; }
  h3 { font-size: 14px; margin-top: 18px; }
  .card { border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; padding: 10px; }
  .ref { font-size: 13px; margin: 4px 0; }
  audio { width: 100%; margin: 4px 0; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 6px 0; }
  th, td { padding: 3px 8px; vertical-align: top; border-bottom: 1px solid #f0f0f0; text-align: left; }
  th { background: #f5f5f5; }
  td.m { color: #555; font-family: ui-monospace, monospace; font-size: 11px; white-space: nowrap; width: 28%; }
  td.h { font-family: ui-monospace, monospace; font-size: 11px; word-break: break-word; }
  tr.wl td.m { color: #1565c0; font-weight: 600; }
  tr.match { background: #e8f5e9; }
  tr.other { background: #fff3e0; }
  .legend { font-size: 12px; color: #555; background: #f0f0f0; padding: 8px; border-radius: 4px; margin: 12px 0; }
  .vA { background: #4caf50; color: #fff; padding: 0 3px; border-radius: 2px; font-weight: 600; }
  .vB { background: #ff5722; color: #fff; padding: 0 3px; border-radius: 2px; font-weight: 600; }
  .vC { background: #9c27b0; color: #fff; padding: 0 3px; border-radius: 2px; font-weight: 600; }
  .vD { background: #607d8b; color: #fff; padding: 0 3px; border-radius: 2px; font-weight: 600; }
  .toc { font-size: 12px; margin: 12px 0; }
  .toc a { margin-right: 14px; }
  details { margin: 4px 0; }
  summary { cursor: pointer; color: #1976d2; font-weight: 600; }
</style></head><body>''']
    parts.append("<h1>Orthographic variant switch test — libri + vox + tedlium</h1>")
    parts.append('<div class="legend">'
                 'For each orthographic group, the ref uses one specific variant (Mr. vs mister, % vs percent, etc.) — the audio carries no cue. '
                 'A row that\'s GREEN in both sub-sections (e.g., "Ref uses Mr." AND "Ref uses mister") means the model matched the ref formatting both times. '
                 'Since the audio is identical, that\'s strong contamination evidence.'
                 '</div>')
    parts.append('<h2>Aggregate switch leaderboard</h2>')
    parts.append('<table><tr><th>model</th><th>WL</th><th>n_groups</th><th>mean switch</th></tr>')
    for mdl in sorted(agg_switch, key=lambda x: -sum(agg_switch[x]) / max(len(agg_switch[x]), 1)):
        s = agg_switch[mdl]
        if len(s) < 2:
            continue
        mean = sum(s) / len(s)
        wl = "WL" if mdl in WL_EN else ""
        cls = "wl" if mdl in WL_EN else ""
        parts.append(f'<tr class="{cls}"><td class="m">{html.escape(mdl)}</td><td>{wl}</td><td>{len(s)}</td><td><b>{mean:.0%}</b></td></tr>')
    parts.append('</table>')

    parts.append('<div class="toc"><b>Groups:</b> ')
    for gi in balanced_groups:
        label, _ = groups[gi]
        parts.append(f'<a href="#g{gi}">{html.escape(label)}</a>')
    parts.append('</div>')

    for gi in balanced_groups:
        label, variants = groups[gi]
        vmap = samples_by_group_variant[gi]
        counts = " | ".join(f"<b>{v}</b>={len(vmap.get(v, []))}" for v, _ in variants if vmap.get(v))
        parts.append(f'<h2 id="g{gi}">{html.escape(label)} — {counts}</h2>')
        variant_class = {v: f"v{'ABCD'[i] if i < 4 else 'D'}" for i, (v, _) in enumerate(variants)}

        def hl(s: str) -> str:
            esc = html.escape(s)
            for v_name, pat in variants:
                cls = variant_class.get(v_name, "vD")
                esc = pat.sub(lambda m: f'<span class="{cls}">{m.group(0)}</span>', esc)
            return esc

        for v_name, _ in variants:
            samples = vmap.get(v_name, [])[:MAX_PER_VARIANT_RENDER]
            if not samples:
                continue
            parts.append(f'<h3>Ref uses <span class="{variant_class[v_name]}">{html.escape(v_name)}</span> ({len(vmap[v_name])} total, {len(samples)} shown)</h3>')
            parts.append(f'<details><summary>show {len(samples)} sample cards</summary>')
            for ds, k, ref in sorted(samples):
                hyps = hyps_by_ds_key.get(ds, {}).get(k, {})
                parts.append('<div class="card">')
                parts.append(f'<div style="color:#666;font-size:11px">{html.escape(ds)} · {html.escape(k)}</div>')
                parts.append(f'<div class="ref"><b>REF:</b> {hl(ref)}</div>')
                parts.append(f'<audio controls preload="none" src="/api/audio/{html.escape(ds)}/{html.escape(k)}"></audio>')
                if hyps:
                    parts.append('<table>')
                    for mdl in sorted(hyps, key=lambda x: (0 if x in WL_EN else 1, x)):
                        hyp = hyps[mdl]
                        hyp_v = find_first_variant(hyp, variants)
                        classes = []
                        if mdl in WL_EN: classes.append("wl")
                        if hyp_v == v_name: classes.append("match")
                        elif hyp_v is not None: classes.append("other")
                        parts.append(f'<tr class="{" ".join(classes)}"><td class="m">{html.escape(mdl)}</td><td class="h">{hl(hyp)}</td></tr>')
                    parts.append('</table>')
                parts.append('</div>')
            parts.append('</details>')

    parts.append("</body></html>")
    out_path_arg.write_text("".join(parts))
    log.info("wrote %s", out_path_arg)


if __name__ == "__main__":
    main()
