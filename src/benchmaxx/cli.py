"""Command line entry point: ``benchmaxx``.

Two subcommands, one per probe that runs on prediction files alone:

    benchmaxx ref-disagreement --preds DIR --panel a,b,c,d
    benchmaxx switch-rate      --preds DIR

The third probe family in the paper, masked-entity recovery, needs the audio
and model weights and so lives in ``repro/probes/`` rather than here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import conventions, ortho, predictions, refdis
from .normalize import has_whisper_normalizer


def _load(args) -> predictions.PredictionSet:
    src = Path(args.preds)
    if src.is_dir():
        ps = predictions.load_dir(src, pattern=args.pattern, language=args.language)
    else:
        ps = predictions.load_manifest(src, language=args.language)
    if not ps:
        sys.exit(f"no clips loaded from {src}")
    if ps.ref_conflicts:
        print(
            f"warning: {len(ps.ref_conflicts)} clips had disagreeing references "
            f"across files (dataset version mismatch?)",
            file=sys.stderr,
        )
    return ps


def _table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    cells = [list(headers)] + [[str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [
        "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells[0], widths))),
        "  ".join("-" * w for w in widths),
    ]
    for row in cells[1:]:
        lines.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))))
    return "\n".join(lines)


def cmd_ref_disagreement(args) -> None:
    ps = _load(args)
    if not has_whisper_normalizer():
        print(
            "warning: transformers not installed, falling back to basic normalization. "
            "Results will not match the Open ASR Leaderboard.",
            file=sys.stderr,
        )
    panel = [m.strip() for m in args.panel.split(",") if m.strip()]
    missing = [m for m in panel if m not in ps.models]
    if missing:
        sys.exit(f"panel models not found in predictions: {missing}\navailable: {ps.models}")
    if len(panel) < refdis.DEFAULTS["min_panel"]:
        sys.exit(f"need at least {refdis.DEFAULTS['min_panel']} panel models, got {len(panel)}")

    norm = ps.normalized()
    all_edits: list[refdis.RefEdit] = []
    for key, ref in norm.refs.items():
        hyps = norm.hyps.get(key, {})
        panel_hyps = {m: hyps[m] for m in panel if m in hyps}
        if len(panel_hyps) < refdis.DEFAULTS["min_panel"]:
            continue
        # Already normalized by ps.normalized(), so a whitespace split is the
        # tokenization the probe expects.
        all_edits += refdis.find_ref_edits(
            ref.split(),
            panel_hyps,
            hyps,
            majority=args.majority,
            include_middle=args.include_middle,
            min_run_len=args.min_run_len,
        )

    rates = refdis.accept_ref_rate(all_edits)
    print(f"clips: {len(norm)}   panel: {len(panel)}   reference errors found: {len(all_edits)}")
    print()
    rows = [
        (m, f"{s['rate']:.3f}", f"[{s['lo']:.3f}, {s['hi']:.3f}]", s["n_ref"], s["n_eligible"])
        for m, s in rates.items()
    ]
    print(_table(rows, ("model", "accept-ref", "95% CI", "n_ref", "n_eligible")))
    print()
    print("accept-ref = share of reference errors where the model reproduced the")
    print("reference instead of the panel consensus. Higher = more benchmark-optimized.")

    if args.out:
        payload = {
            "n_clips": len(norm),
            "panel": panel,
            "n_edits": len(all_edits),
            "params": {
                "majority": args.majority,
                "min_run_len": args.min_run_len,
                "include_middle": args.include_middle,
            },
            "accept_ref": rates,
            "edits": [
                {
                    "kind": e.kind,
                    "position": e.position,
                    "text": e.text,
                    "n_panel_agree": e.n_panel_agree,
                    "n_panel": e.n_panel,
                    "consensus_cer": e.consensus_cer,
                    "verdict": e.verdict,
                }
                for e in all_edits
            ],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


def cmd_switch_rate(args) -> None:
    ps = _load(args)
    if args.spacing:
        fams = list(conventions.SPACING_PAIRS)
    elif args.families:
        fams = [conventions.family(n.strip()) for n in args.families.split(",") if n.strip()]
    else:
        fams = conventions.families_for(args.language)
    clips = list(ps.clips())

    if args.pool or args.spacing:
        # Pooling is usually necessary, not optional: any single family is rare
        # enough that its interval spans chance. Arms must line up positionally,
        # which holds within the shipped spacing group.
        arm_names = conventions.SPACING_ARMS if args.spacing else fams[0].arm_labels
        try:
            pooled = ortho.pooled_switch_rate(
                fams, clips, arm_names=arm_names, name="pooled", min_per_arm=args.min_per_arm
            )
        except ValueError as e:
            sys.exit(f"cannot pool the requested families: {e}")
        results = {"pooled(" + ",".join(f.name for f in fams) + ")": pooled} if pooled else {}
    else:
        results = ortho.switch_rates(fams, clips, min_per_arm=args.min_per_arm)

    if not results:
        sys.exit(
            "no convention family had enough clips on both arms. Switch rate needs "
            "references that use BOTH spellings; a single corpus often uses only one. "
            "Pass predictions from two corpora with opposing conventions, or lower --min-per-arm."
        )

    for name, per_model in results.items():
        arms = next(iter(per_model.values())).arms
        print(f"\n=== {name}  ({' / '.join(arms)}) ===")
        rows = [
            (
                m,
                f"{r.switch:.3f}",
                f"[{r.lo:.3f}, {r.hi:.3f}]",
                r.limiting_arm,
                r.n_total,
                "yes" if r.follows_reference else "no",
            )
            for m, r in per_model.items()
        ]
        print(_table(rows, ("model", "switch", "95% CI", "limiting arm", "n", ">chance")))
    print()
    print("switch = min over arms of P(model emits the arm the reference used).")
    print("Chance is 0.500 for a two-arm family. A model with a fixed convention")
    print("sits near 0; only reference-following raises the minimum.")

    if args.out:
        payload = {
            fam: {
                m: {
                    "switch": r.switch, "lo": r.lo, "hi": r.hi,
                    "limiting_arm": r.limiting_arm,
                    "arms": {a: {"hits": t.hits, "n": t.n} for a, t in r.arms.items()},
                }
                for m, r in per_model.items()
            }
            for fam, per_model in results.items()
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmaxx",
        description="Quantify benchmark optimization in ASR models from prediction files.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--preds", required=True, help="prediction JSONL file, or a directory of them")
        sp.add_argument("--pattern", default="*.jsonl", help="glob used when --preds is a directory")
        sp.add_argument("--language", default="en", help="ISO language code of the references")
        sp.add_argument("--out", help="write full results to this JSON file")

    rd = sub.add_parser(
        "ref-disagreement",
        help="accept-ref rate: does the model reproduce erroneous reference text?",
    )
    common(rd)
    rd.add_argument(
        "--panel",
        required=True,
        help="comma-separated models forming the trusted panel. Choose them "
        "independently of the models you are testing.",
    )
    rd.add_argument("--majority", type=float, default=refdis.DEFAULTS["majority"])
    rd.add_argument("--min-run-len", type=int, default=refdis.DEFAULTS["min_run_len"])
    rd.add_argument(
        "--include-middle",
        action="store_true",
        help="also count edits interior to the reference (noisier; the paper reports boundaries only)",
    )
    rd.set_defaults(func=cmd_ref_disagreement)

    sr = sub.add_parser(
        "switch-rate",
        help="orthographic switch rate: does the model's spelling track the corpus?",
    )
    common(sr)
    sr.add_argument("--families", help="comma-separated family names (default: all for the language)")
    sr.add_argument(
        "--pool",
        action="store_true",
        help="pool the selected families into one score, combining arm 1 of each and "
        "arm 2 of each. Almost always needed: a single family is rare enough that its "
        "interval spans chance. Only pool families whose arms mean the same thing.",
    )
    sr.add_argument(
        "--spacing",
        action="store_true",
        help="shorthand for the recommended within-corpus test: pool the four "
        "compound-spacing families (anyone/any one, everyone/every one, ...). Both arms "
        "occur inside a single read-speech corpus, so register is matched.",
    )
    sr.add_argument("--min-per-arm", type=int, default=5)
    sr.set_defaults(func=cmd_switch_rate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
