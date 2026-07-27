"""Orthographic switch rate.

For a convention family whose arms sound identical (see
:mod:`benchmark_optimization.conventions`), partition clips by which arm the
reference uses, then per model:

    switch = min over arms a of  P(model emits arm a | reference uses a)

A model with a fixed habit is right on one arm and wrong on the other, so its
minimum is near 0; only a model that changes arm with the reference raises it.
Chance for two arms is 0.5.

Arms may come from one corpus (compound spacing) or from two with opposing
conventions (honorifics, which no single corpus uses both of). The latter varies
register along with the arm.

Matching runs on raw text; normalization erases these distinctions.
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

    ``clips`` is ``[(reference, {model: hypothesis})]``, all raw text. Clips
    where the family does not appear in the reference are skipped.

    ``min_per_arm`` drops models without that many clips on every arm. The
    interval covers the limiting arm only, so it is anti-conservative when arms
    are close.
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
    """Switch rate over several families sharing an arm structure.

    Arms pool by position: arm *i* of every family counts toward
    ``arm_names[i]``. Needed for the spacing group, whose families are each too
    rare to bound tightly. Only pool families whose positions mean the same
    thing.
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
    """Run :func:`switch_rate` per family, scored separately.

    Returns ``{family: {model: SwitchResult}}``, omitting families no model was
    measurable on. Not averaged across families: they differ in base rate and
    coverage. Use :func:`pooled_switch_rate` to combine.
    """
    out = {}
    for fam in families:
        res = switch_rate(fam, clips, min_per_arm=min_per_arm)
        if res:
            out[fam.name] = res
    return out
