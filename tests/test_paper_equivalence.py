"""Verify this implementation reproduces the paper's published numbers.

``benchmaxx.refdis`` is an extraction of the internal script that produced the
paper's reference-disagreement results. The extraction was rewritten for
clarity, so it needs checking against the original rather than trusting that it
still means the same thing.

``repro/data/consensus/vox_en_newwl4_samples.json`` is the original run's dump.
For each flagged clip it records the exact normalized reference tokens and
hypotheses that were fed to the original, and the edits and per-model verdicts
that came out. Replaying those inputs here must reproduce every edit and every
verdict, and the per-model accept-ref rates in the matching ``_aggregate.json``.

Skipped if the repro data is not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmaxx import refdis

DATA = Path(__file__).resolve().parent.parent / "repro" / "data" / "consensus"
SAMPLES = DATA / "vox_en_newwl4_samples.json"
AGGREGATE = DATA / "vox_en_newwl4_aggregate.json"

pytestmark = pytest.mark.skipif(
    not (SAMPLES.exists() and AGGREGATE.exists()),
    reason="repro/data/consensus dumps not present",
)


@pytest.fixture(scope="module")
def published():
    agg = json.loads(AGGREGATE.read_text())
    return agg, json.loads(SAMPLES.read_text())


@pytest.fixture(scope="module")
def replayed(published):
    """Run the extracted probe over the original run's exact inputs."""
    agg, samples = published
    panel_names = agg["whitelist"]
    p = agg["params"]
    out = []
    for rec in samples:
        hyps = rec["hyps"]
        panel = {m: hyps[m] for m in panel_names if m in hyps}
        edits = refdis.find_ref_edits(
            rec["ref_tokens"],
            panel,
            hyps,
            majority=p["majority_pct"],
            min_run_len=p["min_run_len"],
            include_middle=p["include_middle"],
            min_ref_match=p["min_ref_match_pct"],
            min_consensus_cer=p["min_consensus_cer"],
            min_panel=3,
        )
        out.append((rec, edits))
    return out


def _sig_published(r):
    return (
        r["run_type"], r["position"], tuple(r["run_indices"]), tuple(r["run_tokens"]),
        r["n_wl_agree"], r["n_wl_total"], r["median_cer"],
    )


def _sig_replayed(e):
    return (
        e.kind, e.position, tuple(e.ref_indices), tuple(e.ref_tokens),
        e.n_panel_agree, e.n_panel, e.consensus_cer,
    )


def test_every_clip_yields_the_same_edits(replayed):
    mismatched = [
        rec["key"]
        for rec, edits in replayed
        if sorted(map(_sig_published, rec["runs"])) != sorted(map(_sig_replayed, edits))
    ]
    assert not mismatched, f"{len(mismatched)} clips differ, e.g. {mismatched[:3]}"


def test_every_model_verdict_matches(replayed):
    compared = 0
    diffs = []
    for rec, edits in replayed:
        by_published = {_sig_published(r): r["verdict"] for r in rec["runs"]}
        by_replayed = {_sig_replayed(e): e.verdict for e in edits}
        for sig, want in by_published.items():
            got = by_replayed.get(sig)
            if got is None:
                continue
            compared += 1
            if want != got:
                diffs.append((rec["key"], {m: (want.get(m), got.get(m)) for m in want if want.get(m) != got.get(m)}))
    assert compared > 1000, f"only {compared} verdicts compared — dump looks truncated"
    assert not diffs, f"{len(diffs)} runs have differing verdicts, e.g. {diffs[:2]}"


def test_accept_ref_leaderboard_matches(published, replayed):
    agg, _ = published
    all_edits = [e for _rec, edits in replayed for e in edits]
    assert len(all_edits) == agg["prominence"]["n_runs"]

    rates = refdis.accept_ref_rate(all_edits)
    published_rates = {r["model"]: r for r in agg["benchmaxx_leaderboard"]}
    assert set(published_rates) <= set(rates), "models missing from the replay"

    for model, row in published_rates.items():
        ours = rates[model]
        assert ours["n_eligible"] == row["eligible_runs"], model
        assert ours["n_ref"] == row["accepted_ref"], model
        # The published table is rounded to four decimals.
        assert abs(ours["rate"] - row["benchmaxx_score"]) < 5e-5, model
