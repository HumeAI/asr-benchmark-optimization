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

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .conventions import Family, pooled_arms
from .refdis import wilson

__all__ = ["ArmTally", "SwitchResult", "switch_rate", "switch_rates", "pooled_switch_rate"]

INTERVAL_METHODS = ("simultaneous", "legacy-wilson")


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF, summing from the largest term in the smaller tail.

    The log-PMF start avoids underflow at the zero-success term for large n.
    This stays stdlib-only, like the other prediction-file probes.
    """
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    if k >= n * p:
        return 1.0 - _binomial_cdf(n - k - 1, n, 1.0 - p)

    def terms():
        term = math.exp(
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p)
        )
        yield term
        for j in range(k, 0, -1):
            term *= (j / (n - j + 1)) * ((1.0 - p) / p)
            yield term

    return min(1.0, math.fsum(terms()))


def _binomial_lower(k: int, n: int, tail: float) -> float:
    """Invert P_p(X >= k) = tail for a Clopper-Pearson lower bound."""
    if k == 0:
        return 0.0
    low, high = 0.0, k / n
    for _ in range(60):
        mid = (low + high) / 2.0
        # P_p(X >= k) = P_{1-p}(X <= n-k); no small-tail subtraction.
        if _binomial_cdf(n - k, n, 1.0 - mid) < tail:
            low = mid
        else:
            high = mid
    return low


def _minimum_interval(arms: list[ArmTally]) -> tuple[float, float]:
    """95% interval for the minimum of fixed, binomial arm probabilities.

    Each two-sided Clopper-Pearson interval has error budget 0.05 / m.
    Bonferroni gives simultaneous coverage >= 95%, and taking the minimum
    of the lower and upper endpoints preserves that event. Cross-arm
    independence is unnecessary; valid marginal binomial sampling is required.
    This is conservative, not an assertion of exactly 95% coverage.
    """
    if not arms or any(t.n <= 0 or not 0 <= t.hits <= t.n for t in arms):
        raise ValueError("each required arm needs valid, nonempty binomial counts")
    tail = 0.05 / (2 * len(arms))
    lower = [_binomial_lower(t.hits, t.n, tail) for t in arms]
    upper = [1.0 - _binomial_lower(t.n - t.hits, t.n, tail) for t in arms]
    return min(lower), min(upper)


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
    interval_method: str = "simultaneous"

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
    interval_method: str,
) -> dict[str, SwitchResult]:
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(f"unknown interval method: {interval_method!r}")
    out: dict[str, SwitchResult] = {}
    for model, arms in tally.items():
        present = {a: arms[a] for a in labels if a in arms and arms[a].n}
        # A single arm cannot distinguish a fixed habit from following the
        # reference, so the family is not measurable for this model.
        if len(present) < 2 or any(t.n < min_per_arm for t in present.values()):
            continue
        if interval_method == "simultaneous" and len(present) != len(labels):
            # Silently dropping a missing arm would change the minimum being
            # estimated. Keep the historical behavior only in reproduction mode.
            continue
        limiting = min(present, key=lambda a: present[a].rate)
        t = present[limiting]
        p = t.rate
        if interval_method == "legacy-wilson":
            p, lo, hi = wilson(t.hits, t.n)
        else:
            lo, hi = _minimum_interval(list(present.values()))
        out[model] = SwitchResult(
            model=model, family=name, switch=p, lo=lo, hi=hi,
            limiting_arm=limiting, arms=dict(present), interval_method=interval_method,
        )
    return dict(sorted(out.items(), key=lambda kv: -kv[1].switch))


def switch_rate(
    family: Family,
    clips: list[tuple[str, dict[str, str]]],
    *,
    min_per_arm: int = 5,
    interval_method: str = "simultaneous",
) -> dict[str, SwitchResult]:
    """Switch rate per model for one convention family.

    ``clips`` is ``[(reference, {model: hypothesis})]``, all raw text. Clips
    where the family does not appear in the reference are skipped.

    ``min_per_arm`` drops models without that many clips on every required arm.
    The default 95% interval combines simultaneous exact binomial bounds for
    all arms. It assumes fixed arms and marginal binomial sampling, not
    correlated repeated clips or a post-hoc choice of convention families.
    ``interval_method="legacy-wilson"`` reproduces the historical interval on
    the selected limiting arm; it is not selection-adjusted.
    """
    tally, labels = _tally([family], clips, None)
    return _score(tally, labels, family.name, min_per_arm, interval_method)


def pooled_switch_rate(
    families: list[Family],
    clips: list[tuple[str, dict[str, str]]],
    *,
    arm_names: tuple[str, ...],
    name: str = "pooled",
    min_per_arm: int = 5,
    interval_method: str = "simultaneous",
) -> dict[str, SwitchResult]:
    """Switch rate over several families sharing an arm structure.

    Arms pool by position: arm *i* of every family counts toward
    ``arm_names[i]``. Needed for the spacing group, whose families are each too
    rare to bound tightly. Only pool families whose positions mean the same
    thing. ``interval_method`` has the same meaning as in :func:`switch_rate`.
    """
    if len(arm_names) != pooled_arms(families):
        raise ValueError(f"got {len(arm_names)} arm names for families with {pooled_arms(families)} arms")
    tally, labels = _tally(families, clips, arm_names)
    return _score(tally, labels, name, min_per_arm, interval_method)


def switch_rates(
    families: list[Family],
    clips: list[tuple[str, dict[str, str]]],
    *,
    min_per_arm: int = 5,
    interval_method: str = "simultaneous",
) -> dict[str, dict[str, SwitchResult]]:
    """Run :func:`switch_rate` per family, scored separately.

    Returns ``{family: {model: SwitchResult}}``, omitting families no model was
    measurable on. Not averaged across families: they differ in base rate and
    coverage. Use :func:`pooled_switch_rate` to combine.
    """
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(f"unknown interval method: {interval_method!r}")
    out = {}
    for fam in families:
        res = switch_rate(fam, clips, min_per_arm=min_per_arm, interval_method=interval_method)
        if res:
            out[fam.name] = res
    return out
