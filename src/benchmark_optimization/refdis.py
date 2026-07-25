"""Reference-error reproduction: does a model follow the reference or the audio?

Some reference transcripts are wrong. VoxPopuli, whose references are derived
from official parliamentary records rather than from the audio, is the clearest
case: references routinely contain words that were not spoken and omit words
that were.

Those clips are natural probes. Where the audio and the reference disagree, a
model transcribing the audio must produce the audio's version; only a model
that has learned the benchmark's text can produce the reference's. We do not
need a hand-corrected reference to find these clips, because independent
models that agree with each other against the reference are evidence about
what was said.

The procedure, per clip:

1. A **panel** of models transcribes the clip. Where a supermajority of the
   panel makes the *same* edit to the reference, that edit is a candidate
   reference error.
2. Candidates that are normalization artifacts are dropped (see
   ``min_consensus_cer``).
3. Every model under test gets a verdict at each surviving edit:
   ``"consensus"`` if it made the panel's edit, ``"ref"`` if it reproduced the
   reference instead, ``None`` if it was not competent on the clip.

The headline number, ``accept-ref rate``, is the share of eligible edits where
a model sided with the erroneous reference. It is a *rate over disagreements*,
not a WER: a model can have excellent WER and an accept-ref rate near zero,
and the two carry different information.

The panel must be chosen independently of the models under test. In the paper
we used a panel of four models with no VoxPopuli-specific behaviour on the
other probes, and validated the flagged edits against human judgement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import align

__all__ = ["RefEdit", "find_ref_edits", "accept_ref_rate", "DEFAULTS"]

DEFAULTS = {
    # Share of the panel that must make the same edit. Read as a true
    # supermajority via ceil, so 4 panel models need 4 and never split 2-2.
    "majority": 0.8,
    # Shortest edit worth counting, in reference tokens.
    "min_run_len": 1,
    # Count edits interior to the reference, not just those at either end.
    # Interior edits are noisier: alignment ambiguity is higher away from the
    # boundaries. The paper reports boundary edits.
    "include_middle": False,
    # A model must reproduce this share of the reference to be scored on the
    # clip at all. Without it, empty and off-language outputs dominate.
    "min_ref_match": 0.5,
    # A deletion edit only counts if what the panel wrote instead is
    # character-wise far from the reference. Below this, the "error" is
    # spacing or accents, i.e. a normalization artifact.
    "min_consensus_cer": 0.30,
    # Panel members that clear ``min_ref_match``, below which the clip is
    # skipped for want of a trustworthy consensus.
    "min_panel": 3,
}


@dataclass
class RefEdit:
    """One reference error, with each model's verdict on it.

    ``kind`` is ``"delete"`` when the reference contains words that were not
    spoken, and ``"insert"`` when it omits words that were.
    """

    kind: str
    position: str
    ref_tokens: list[str]
    ref_indices: list[int]
    n_panel_agree: int
    n_panel: int
    verdict: dict[str, str | None] = field(default_factory=dict)
    consensus_cer: float | None = None

    @property
    def text(self) -> str:
        return " ".join(self.ref_tokens)


def find_ref_edits(
    ref_tokens: list[str],
    panel_hyps: dict[str, str],
    model_hyps: dict[str, str],
    *,
    majority: float = DEFAULTS["majority"],
    min_run_len: int = DEFAULTS["min_run_len"],
    include_middle: bool = DEFAULTS["include_middle"],
    min_ref_match: float = DEFAULTS["min_ref_match"],
    min_consensus_cer: float = DEFAULTS["min_consensus_cer"],
    min_panel: int = DEFAULTS["min_panel"],
) -> list[RefEdit]:
    """Find reference errors in one clip and score every model against them.

    ``ref_tokens`` must already be normalized and tokenized (see
    :func:`benchmark_optimization.normalize.tokenize`); ``panel_hyps`` and ``model_hyps`` map
    model name to a normalized hypothesis string. The panel may overlap the
    models under test — a panel member is scored like any other model, and its
    own vote is what it is.
    """
    n_ref = len(ref_tokens)
    if not n_ref or not panel_hyps:
        return []

    missed = {m: align.missed_indices(ref_tokens, h) for m, h in model_hyps.items()}
    inserted = {m: align.insertions_by_anchor(ref_tokens, h) for m, h in model_hyps.items()}
    matched = {m: align.matched_count(ref_tokens, h) for m, h in model_hyps.items()}
    min_matched = int(min_ref_match * n_ref)

    def eligible(m: str) -> bool:
        return matched.get(m, 0) >= min_matched

    # An empty or truncated panel output otherwise reads as "missed every
    # word" and drags the consensus with it.
    panel = {
        m: h for m, h in panel_hyps.items() if align.matched_count(ref_tokens, h) >= min_matched
    }
    if len(panel) < min_panel:
        return []
    n_panel = len(panel)
    threshold = max(2, math.ceil(majority * n_panel - 1e-9))

    edits: list[RefEdit] = []
    edits += _deletion_edits(
        ref_tokens, panel, model_hyps, missed, eligible,
        threshold, n_panel, min_run_len, include_middle, min_consensus_cer,
    )
    edits += _insertion_edits(
        ref_tokens, panel, model_hyps, inserted, eligible,
        threshold, n_panel, min_run_len,
    )
    return edits


def _deletion_edits(
    ref_tokens, panel, model_hyps, missed, eligible,
    threshold, n_panel, min_run_len, include_middle, min_consensus_cer,
):
    """Reference spans a supermajority of the panel refused to transcribe."""
    n_ref = len(ref_tokens)
    panel_missed = {m: align.missed_indices(ref_tokens, h) for m, h in panel.items()}

    votes = [0] * n_ref
    for s in panel_missed.values():
        for i in s:
            votes[i] += 1

    runs, cur = [], []
    for i, v in enumerate(votes):
        if v >= threshold:
            cur.append(i)
        else:
            if len(cur) >= min_run_len:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run_len:
        runs.append(cur)
    if not include_middle:
        runs = [r for r in runs if r[0] == 0 or r[-1] == n_ref - 1]

    out = []
    for run in runs:
        span = set(run)
        ref_chunk = " ".join(ref_tokens[i] for i in run)
        agreeing = [m for m in panel if panel_missed[m] & span]
        cers = sorted(
            align.cer(ref_chunk, align.substitution_at(ref_tokens, panel[m], span)) for m in agreeing
        )
        median_cer = cers[len(cers) // 2] if cers else 1.0
        if median_cer < min_consensus_cer:
            continue  # normalization artifact, not a reference error

        verdict = {}
        for m in model_hyps:
            if not eligible(m):
                verdict[m] = None
            elif missed[m] & span:
                verdict[m] = "consensus"  # also declined the unspoken words
            else:
                verdict[m] = "ref"  # reproduced words that were not said
        out.append(
            RefEdit(
                kind="delete",
                position="start" if run[0] == 0 else ("end" if run[-1] == n_ref - 1 else "middle"),
                ref_tokens=[ref_tokens[i] for i in run],
                ref_indices=run,
                n_panel_agree=votes[run[0]],
                n_panel=n_panel,
                verdict=verdict,
                consensus_cer=round(median_cer, 3),
            )
        )
    return out


def _insertion_edits(
    ref_tokens, panel, model_hyps, inserted, eligible, threshold, n_panel, min_run_len,
):
    """Words a supermajority of the panel heard that the reference omits.

    Restricted to the two reference boundaries. Interior insertions cannot be
    anchored reliably: which side of a matched token an inserted word belongs
    to is an alignment choice, not a fact about the audio.
    """
    n_ref = len(ref_tokens)
    panel_ins = {m: align.insertions_by_anchor(ref_tokens, h) for m, h in panel.items()}

    out = []
    for anchor in (0, n_ref):
        at_start = anchor == 0
        lists = [ins.get(anchor, []) for ins in panel_ins.values()]
        chunk = align.consensus_insertion(lists, at_start, threshold)
        if len(chunk) < min_run_len:
            continue
        n_agree = sum(1 for lst in lists if align.emitted_insertion(lst, chunk, at_start))
        if n_agree < threshold:
            continue

        verdict = {}
        for m in model_hyps:
            if not eligible(m):
                verdict[m] = None
            elif align.emitted_insertion(inserted[m].get(anchor, []), chunk, at_start):
                verdict[m] = "consensus"  # also transcribed the spoken words
            else:
                verdict[m] = "ref"  # omitted them, matching the deficient reference
        out.append(
            RefEdit(
                kind="insert",
                position="start" if at_start else "end",
                ref_tokens=chunk,
                ref_indices=[anchor],
                n_panel_agree=n_agree,
                n_panel=n_panel,
                verdict=verdict,
            )
        )
    return out


def accept_ref_rate(edits: list[RefEdit]) -> dict[str, dict]:
    """Aggregate per-model accept-ref rates over a collection of edits.

    Returns ``{model: {"rate", "n_ref", "n_eligible", "lo", "hi"}}``, where
    ``rate = n_ref / n_eligible`` and ``lo``/``hi`` are a 95% Wilson interval.
    Models are only charged for edits they were eligible for, so the
    denominators differ between models and must be reported alongside the rate.
    """
    tally: dict[str, list[int]] = {}
    for e in edits:
        for model, v in e.verdict.items():
            if v is None:
                continue
            k, n = tally.setdefault(model, [0, 0])
            tally[model] = [k + (v == "ref"), n + 1]

    out = {}
    for model, (k, n) in tally.items():
        p, lo, hi = wilson(k, n)
        out[model] = {"rate": p, "n_ref": k, "n_eligible": n, "lo": lo, "hi": hi}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["rate"]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for ``k`` successes in ``n``.

    Wilson rather than normal-approximation because several models sit near 0
    or 1, where the normal interval leaves the unit range.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, center - half), min(1.0, center + half)
