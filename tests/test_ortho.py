import pytest

from benchmaxx import conventions, ortho


def _clips(arm_a_ref, arm_b_ref, hyp_fn, n=10):
    """Build n clips per arm, with each model's hypothesis from ``hyp_fn``."""
    out = []
    for ref in (arm_a_ref, arm_b_ref):
        for _ in range(n):
            out.append((ref, hyp_fn(ref)))
    return out


MISTER = conventions.family("hon_mister")


def test_fixed_convention_model_scores_near_zero():
    # Always writes "mister", whatever the reference did. It is right on one
    # arm and wrong on the other, so the minimum across arms is 0.
    clips = _clips(
        "Mr. Smith spoke at length",
        "mister Smith spoke at length",
        lambda ref: {"fixed": "mister Smith spoke at length"},
    )
    r = ortho.switch_rate(MISTER, clips)["fixed"]
    assert r.switch == 0.0
    assert not r.follows_reference


def test_reference_following_model_scores_high():
    clips = _clips(
        "Mr. Smith spoke at length",
        "mister Smith spoke at length",
        lambda ref: {"copies_ref": ref},
    )
    r = ortho.switch_rate(MISTER, clips)["copies_ref"]
    assert r.switch == 1.0
    assert r.follows_reference


def test_switch_is_the_minimum_across_arms_not_the_mean():
    # Right on every "mister" clip, right on half the "Mr." clips. The mean
    # would read 0.75 and look like partial switching; the minimum reports the
    # arm the model actually fails on.
    seen = {"n": 0}

    def hyp(ref):
        if "mister" in ref:
            return {"m": "mister Smith spoke at length"}
        seen["n"] += 1
        return {"m": "Mr. Smith spoke at length" if seen["n"] % 2 else "mister Smith spoke at length"}

    r = ortho.switch_rate(MISTER, _clips("Mr. Smith spoke at length", "mister Smith spoke at length", hyp))["m"]
    assert r.switch == 0.5
    assert r.limiting_arm == "Mr"


def test_family_absent_from_reference_is_skipped():
    clips = [("nothing relevant here", {"m": "nothing relevant here"})] * 20
    assert ortho.switch_rate(MISTER, clips) == {}


def test_single_arm_is_not_measurable():
    # One arm cannot separate a fixed habit from reference-following, so the
    # model must be dropped rather than scored 1.0.
    clips = [("Mr. Smith spoke at length", {"m": "Mr. Smith spoke at length"})] * 20
    assert ortho.switch_rate(MISTER, clips) == {}


def test_min_per_arm_floor_drops_thin_models():
    clips = _clips("Mr. Smith spoke", "mister Smith spoke", lambda ref: {"m": ref}, n=3)
    assert ortho.switch_rate(MISTER, clips, min_per_arm=5) == {}
    assert "m" in ortho.switch_rate(MISTER, clips, min_per_arm=3)


def test_pointed_and_unpointed_abbreviation_share_one_arm():
    # The contrast measured is abbreviate-vs-spell-out. The trailing period is a
    # separate convention; splitting it out would turn one arm into two and
    # depress the score for a model that just omits periods.
    assert MISTER.arm_of("Mr. Smith spoke") == "Mr"
    assert MISTER.arm_of("Mr Smith spoke") == "Mr"
    assert MISTER.arm_of("mister Smith spoke") == "mister"
    assert MISTER.arm_of("nothing here") is None
    # A bare honorific with no following name is a different word (an initialism
    # or a sentence end), so it must not match.
    assert MISTER.arm_of("addressed to Mr.") is None


def test_saint_arm_requires_a_following_word():
    saint = conventions.family("hon_saint")
    assert saint.arm_of("St. Peter was named") == "St"
    # "St." for Street, sentence-final, is pronounced differently.
    assert saint.arm_of("he lived on Bay St.") is None


def test_pooled_spacing_combines_rare_families():
    # Each spacing family alone is too thin to clear min_per_arm; pooled, they
    # are measurable. This is why the paper pools them.
    fams = list(conventions.SPACING_PAIRS)
    clips = []
    for spaced, solid in [
        ("did any one see him", "did anyone see him"),
        ("has every one arrived", "has everyone arrived"),
        ("tell some one quickly", "tell someone quickly"),
    ]:
        for ref in (spaced, solid):
            clips += [(ref, {"copies_ref": ref, "always_solid": solid})] * 2

    assert ortho.switch_rates(fams, clips, min_per_arm=5) == {}
    pooled = ortho.pooled_switch_rate(fams, clips, arm_names=conventions.SPACING_ARMS, name="spacing")
    assert pooled["copies_ref"].switch == 1.0
    assert pooled["always_solid"].switch == 0.0
    assert pooled["always_solid"].limiting_arm == "spaced"
    assert set(pooled["copies_ref"].arms) == {"spaced", "solid"}
    assert pooled["copies_ref"].n_total == 12


def test_pooling_rejects_mismatched_arm_counts():
    fams = [conventions.family("sp_anyone"), conventions.family("percent")]
    # percent has 2 arms too, so pooling is structurally allowed but the names
    # must match the arm count.
    with pytest.raises(ValueError):
        ortho.pooled_switch_rate(fams, [], arm_names=("a", "b", "c"))


def test_spacing_family_has_both_arms_in_one_corpus():
    fam = conventions.family("sp_anyone")
    assert fam.arm_of("did any one see him") == "any one"
    assert fam.arm_of("did anyone see him") == "anyone"


def test_families_for_language():
    en = {f.name for f in conventions.families_for("en")}
    assert "hon_mister" in en and "sp_anyone" in en
    assert "hon_senor" not in en
    assert {f.name for f in conventions.families_for("es")} >= {"hon_senor", "percent_es"}
    # Regional subtags select the base language.
    assert conventions.families_for("en_us") == conventions.families_for("en")


def test_switch_rates_omits_unmeasurable_families():
    clips = _clips("Mr. Smith spoke", "mister Smith spoke", lambda ref: {"m": ref})
    out = ortho.switch_rates(conventions.families_for("en"), clips)
    assert set(out) == {"hon_mister"}
