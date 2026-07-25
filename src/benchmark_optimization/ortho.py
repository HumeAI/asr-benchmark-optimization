"""Orthographic switch rate: does a model's spelling follow the corpus?

For a convention family whose arms sound identical (see
:mod:`benchmark_optimization.conventions`), split the clips by which arm the *reference*
uses. A model that transcribes audio has one habit and keeps it, so it scores
high on the arm matching its habit and near zero on the other. A model whose
output follows the reference scores high on both.

The switch rate is therefore the **minimum agreement across arms**:

    switch(model) = min over arms a of  P(model emits arm a | reference uses a)

A fixed-convention model is pinned near 0 whatever its habit is; only a model
that changes arm with the reference can raise the minimum. The chance level for
a two-arm family is 0.5, reached by a model picking an arm at random. Read
switch rates against 0.5, not against 0.

Two ways to obtain the arms, with different confounds:

``within-dataset``
    Both arms appear inside one corpus. This holds for compound spacing
    (:data:`benchmark_optimization.conventions.SPACING_PAIRS`) and is the cleanest test:
    speaker, register, and recording conditions are matched by construction.

``cross-dataset``
    Each arm comes from a different corpus, which is the only option for
    honorifics — a single corpus almost never uses both. Register then differs
    with the arm, so a model could be responding to register rather than to text
    convention. Prefer corpora of the same register; the paper's headline
    comparison uses two read-speech corpora that happen to have opposite
    honorific conventions, which holds register roughly fixed.

Matching runs on **raw text**, before normalization, since normalization is
what erases these distinctions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .conventions import Family, pooled_arms
from .refdis import wilson

__all__ = ["ArmTally", "SwitchResult", "switch_rate", "switch_rates", "pooled_switch_rate"]


@dataclass
class ArmTally:
    """Per-arm hits and clips for one model."""

    hits: int = 0
    n: int = 0

    @property
    def rate(self) -> float:
        return self.hits / self.n if self.n else 0.0


@dataclass
class SwitchResult:
    """Switch rate for one model on one family, with the per-arm breakdown."""

    model: str
    family: str
    switch: float
    lo: float
    hi: float
    limiting_arm: str
    arms: dict[str, ArmTally] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return sum(a.n for a in self.arms.values())

    @property
    def follows_reference(self) -> bool:
        """Whether the interval clears chance for a two-arm family."""
        return self.lo > 0.5


def _tally(
    families: list[Family],
    clips,
    arm_names: tuple[str, ...] | None,
) -> tuple[dict[str, dict[str, ArmTally]], tuple[str, ...]]:
    """Count per-model, per-arm hits over ``clips``.

    With ``arm_names`` set, arms are pooled positionally across families, so arm
    *i* of every family contributes to ``arm_names[i]``. Otherwise a single
    family's own arm labels are used.
    """
    compiled = [(f, f.compiled()) for f in families]
    labels = arm_names or families[0].arm_labels
    tally: dict[str, dict[str, ArmTally]] = defaultdict(lambda: defaultdict(ArmTally))

    for ref, hyps in clips:
        # First family that applies claims the clip, so a clip is counted once
        # even if two families overlap in it.
        for fam, pats in compiled:
            arm = fam.arm_of(ref)
            if arm is None:
                continue
            idx = fam.arm_labels.index(arm)
            want = pats[idx][1]
            role = labels[idx]
            for model, hyp in hyps.items():
                if hyp is None:
                    continue
                t = tally[model][role]
                t.n += 1
                t.hits += any(p.search(hyp) for p in want)
            break
    return tally, labels


def _score(
    tally: dict[str, dict[str, ArmTally]],
    labels: tuple[str, ...],
    name: str,
    min_per_arm: int,
) -> dict[str, SwitchResult]:
    out: dict[str, SwitchResult] = {}
    for model, arms in tally.items():
        present = {a: arms[a] for a in labels if a in arms and arms[a].n}
        # A single arm cannot distinguish a fixed habit from following the
        # reference, so the family is not measurable for this model.
        if len(present) < 2 or any(t.n < min_per_arm for t in present.values()):
            continue
        limiting = min(present, key=lambda a: present[a].rate)
        t = present[limiting]
        p, lo, hi = wilson(t.hits, t.n)
        out[model] = SwitchResult(
            model=model, family=name, switch=p, lo=lo, hi=hi,
            limiting_arm=limiting, arms=dict(present),
        )
    return dict(sorted(out.items(), key=lambda kv: -kv[1].switch))


def switch_rate(
    family: Family,
    clips: list[tuple[str, dict[str, str]]],
    *,
    min_per_arm: int = 5,
) -> dict[str, SwitchResult]:
    """Switch rate per model for one convention family.

    ``clips`` is a list of ``(reference_text, {model: hypothesis_text})``. Both
    the reference and the hypotheses must be **raw**, un-normalized text. Clips
    where the family does not appear in the reference are skipped, so it is fine
    to pass an entire corpus.

    ``min_per_arm`` drops any model without that many clips on *every* arm.
    Without the floor, a model with two clips on one arm can post 1.0.

    The returned interval is a Wilson interval on the limiting arm alone; it
    does not account for having taken a minimum over arms, so it is
    anti-conservative when the arms are close. Treat near-chance results as
    near-chance.
    """
    tally, labels = _tally([family], clips, None)
    return _score(tally, labels, family.name, min_per_arm)


def pooled_switch_rate(
    families: list[Family],
    clips: list[tuple[str, dict[str, str]]],
    *,
    arm_names: tuple[str, ...],
    name: str = "pooled",
    min_per_arm: int = 5,
) -> dict[str, SwitchResult]:
    """Switch rate over several families sharing a common arm structure.

    Arms are pooled by position, so arm *i* of every family counts toward
    ``arm_names[i]``. Use this for families that test the same convention in
    different words and are individually too rare to bound tightly — the
    spacing group (``any one``/``anyone``, ``every one``/``everyone``, …) is the
    motivating case.

    Only pool families where the positions genuinely mean the same thing.
    Pooling "abbreviated" with "spaced" would produce a number with no
    interpretation.
    """
    if len(arm_names) != pooled_arms(families):
        raise ValueError(f"got {len(arm_names)} arm names for families with {pooled_arms(families)} arms")
    tally, labels = _tally(families, clips, arm_names)
    return _score(tally, labels, name, min_per_arm)


def switch_rates(
    families: list[Family],
    clips: list[tuple[str, dict[str, str]]],
    *,
    min_per_arm: int = 5,
) -> dict[str, dict[str, SwitchResult]]:
    """Run :func:`switch_rate` over several families, each scored separately.

    Returns ``{family_name: {model: SwitchResult}}``, omitting families no model
    was measurable on. Families are kept separate rather than averaged: they
    differ in base rate and in coverage, so a pooled *average* would be
    dominated by whichever family is most frequent. To combine families that
    test the same convention, use :func:`pooled_switch_rate`.
    """
    out = {}
    for fam in families:
        res = switch_rate(fam, clips, min_per_arm=min_per_arm)
        if res:
            out[fam.name] = res
    return out
