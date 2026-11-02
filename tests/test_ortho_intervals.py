"""Selection-adjusted minimum intervals; no audio, models or simulation."""

import itertools
import json
import math

import pytest

from benchmark_optimization import conventions, ortho
from benchmark_optimization.cli import main
from benchmark_optimization.refdis import wilson


def _score(counts, *, labels=None, method="simultaneous"):
    arms = {str(i): ortho.ArmTally(k, n) for i, (k, n) in enumerate(counts)}
    return ortho._score({"m": arms}, labels or tuple(arms), "test", 1, method)


def _direct_cdf(k, n, p):
    # Independent small-n polynomial, rather than the production PMF recurrence.
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k + 1))


@pytest.mark.parametrize("n", [1, 2, 7, 20])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.25, 0.5, 0.8, 0.99, 1.0])
def test_cdf_matches_direct_small_binomial(n, p):
    for k in range(n + 1):
        assert ortho._binomial_cdf(k, n, p) == pytest.approx(_direct_cdf(k, n, p), abs=2e-14)


@pytest.mark.parametrize("n", [1, 10, 1000])
def test_zero_and_all_successes_have_exact_binomial_boundary_formulas(n):
    tail = 0.05 / 4  # two arms, two tails each
    zero = _score([(0, n), (0, n)])["m"]
    one = _score([(n, n), (n, n)])["m"]
    assert zero.lo == 0.0 and one.hi == 1.0
    assert zero.hi == pytest.approx(1 - tail**(1 / n), abs=2e-12)
    assert one.lo == pytest.approx(tail**(1 / n), abs=2e-12)


def test_non_boundary_two_trial_formula():
    # For one success in two trials: P(X >= 1)=2p-p².
    tail = 0.05 / 4
    r = _score([(1, 2), (1, 2)])["m"]
    assert r.lo == pytest.approx(1 - math.sqrt(1 - tail), abs=1e-14)
    assert r.hi == pytest.approx(math.sqrt(1 - tail), abs=1e-14)


def test_tied_rates_are_invariant_to_which_arm_is_listed_first():
    counts = [(5, 10), (50, 100)]
    a = _score(counts)["m"]
    b = _score(counts, labels=("1", "0"))["m"]
    assert (a.switch, a.lo, a.hi) == (b.switch, b.lo, b.hi)
    # The descriptive limiting-arm name may break a tie by input order.
    assert a.limiting_arm != b.limiting_arm
    old_a = _score(counts, method="legacy-wilson")["m"]
    old_b = _score(counts, labels=("1", "0"), method="legacy-wilson")["m"]
    assert (old_a.lo, old_a.hi) != (old_b.lo, old_b.hi)


def test_legacy_preserves_original_wilson_values_and_point_estimates():
    counts = [(7, 10), (16, 20)]
    old = _score(counts, method="legacy-wilson")["m"]
    new = _score(counts)["m"]
    assert (old.switch, old.lo, old.hi) == wilson(7, 10)
    assert (new.switch, new.limiting_arm, new.arms) == (old.switch, old.limiting_arm, old.arms)
    assert old.interval_method == "legacy-wilson"
    assert new.interval_method == "simultaneous"


def test_missing_required_third_arm_is_not_silently_removed():
    assert _score([(5, 10), (5, 10)], labels=("0", "1", "2")) == {}
    # The explicitly requested reproduction path preserves historical behavior.
    assert "m" in _score([(5, 10), (5, 10)], labels=("0", "1", "2"), method="legacy-wilson")


def test_invalid_or_empty_counts_do_not_gain_spurious_certainty():
    for arms in ([], [ortho.ArmTally(0, 0)], [ortho.ArmTally(2, 1)]):
        with pytest.raises(ValueError):
            ortho._minimum_interval(arms)
    with pytest.raises(ValueError, match="unknown interval"):
        ortho.switch_rates([], [], interval_method="typo")


@pytest.mark.parametrize("probabilities", [(0.1, 0.1), (0.5, 0.5), (0.1, 0.8), (0.8, 0.85)])
def test_exact_enumeration_covers_ties_and_separated_arms(probabilities):
    # Enumerate every independent-binomial outcome, not a Monte Carlo sample.
    # The mathematical guarantee also allows cross-arm dependence.
    ns = (4, 7)
    target = min(probabilities)
    coverage = 0.0
    for ks in itertools.product(*(range(n + 1) for n in ns)):
        mass = math.prod(math.comb(n, k) * p**k * (1 - p)**(n - k)
                         for k, n, p in zip(ks, ns, probabilities))
        r = _score(list(zip(ks, ns)))["m"]
        if r.lo <= target <= r.hi:
            coverage += mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize("method", ortho.INTERVAL_METHODS)
def test_cli_and_pooled_api_report_the_interval_method(tmp_path, capsys, method):
    records = []
    for j, ref in enumerate(("did any one see him", "did anyone see him")):
        for i in range(10):
            records.append({"audio_filepath": f"{j}-{i}", "text": ref, "pred_text": ref})
    source = tmp_path / "m.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in records))
    output = tmp_path / "result.json"
    main(["switch-rate", "--preds", str(source), "--spacing",
          "--interval-method", method, "--out", str(output)])
    payload = json.loads(output.read_text())
    row = next(iter(next(iter(payload.values())).values()))
    assert row["interval_method"] == method
    clips = [(r["text"], {"m": r["pred_text"]}) for r in records]
    expected = ortho.pooled_switch_rate(
        list(conventions.SPACING_PAIRS), clips,
        arm_names=conventions.SPACING_ARMS, interval_method=method,
    )["m"]
    assert (row["switch"], row["lo"], row["hi"]) == (expected.switch, expected.lo, expected.hi)
    assert f"Interval method: {method}" in capsys.readouterr().out
