"""Behavioural tests for the reference-error probe.

Each case is a hand-built clip where the right verdict is unambiguous, so a
regression in the alignment or thresholding shows up as a wrong verdict rather
than as a slightly different number.
"""

from benchmark_optimization import refdis

# Four panel models, matching the paper's panel size. The competence gate
# requires each to reproduce at least half the reference, so hypotheses are
# full sentences rather than fragments.
PANEL = ["panel_a", "panel_b", "panel_c", "panel_d"]


def _panel(*hyps):
    return dict(zip(PANEL, hyps))


def test_deletion_edit_splits_ref_followers_from_audio_followers():
    # The reference opens with "madam president", which nobody hears. The
    # panel unanimously drops it; the model under test reproduces it anyway.
    ref = "madam president the report is adopted today by a wide margin".split()
    spoken = "the report is adopted today by a wide margin"
    panel = _panel(spoken, spoken, spoken, spoken)
    models = {
        **panel,
        "follows_reference": "madam president the report is adopted today by a wide margin",
        "follows_audio": spoken,
    }

    edits = refdis.find_ref_edits(ref, panel, models)
    assert len(edits) == 1
    e = edits[0]
    assert e.kind == "delete"
    assert e.position == "start"
    assert e.ref_tokens == ["madam", "president"]
    assert e.verdict["follows_reference"] == "ref"
    assert e.verdict["follows_audio"] == "consensus"


def test_insertion_edit_when_reference_omits_spoken_words():
    # Everyone hears a closing "thank you" that the reference lacks. A model
    # that omits it is matching the deficient reference.
    ref = "i therefore support the proposal".split()
    heard = "i therefore support the proposal thank you"
    panel = _panel(heard, heard, heard, heard)
    models = {**panel, "omits": "i therefore support the proposal", "transcribes": heard}

    edits = refdis.find_ref_edits(ref, panel, models)
    assert len(edits) == 1
    e = edits[0]
    assert e.kind == "insert"
    assert e.position == "end"
    assert e.ref_tokens == ["thank", "you"]
    assert e.verdict["omits"] == "ref"
    assert e.verdict["transcribes"] == "consensus"


def test_incompetent_model_is_ineligible_not_a_ref_follower():
    # An empty hypothesis technically fails to reproduce the spurious words,
    # which would look like siding with the panel. The competence gate must
    # mark it ineligible instead.
    ref = "madam president the report is adopted today".split()
    spoken = "the report is adopted today"
    panel = _panel(spoken, spoken, spoken, spoken)
    models = {**panel, "silent": "", "off_language": "le rapport est adopte"}

    edits = refdis.find_ref_edits(ref, panel, models)
    assert len(edits) == 1
    assert edits[0].verdict["silent"] is None
    assert edits[0].verdict["off_language"] is None


def test_no_edit_without_a_supermajority():
    # A 2-2 split is not a consensus. At majority=0.8 with 4 panel models the
    # threshold is 4, so a split panel yields nothing.
    ref = "madam president the report is adopted today".split()
    spoken = "the report is adopted today"
    panel = _panel(spoken, spoken, ref_text := " ".join(ref), ref_text)
    assert refdis.find_ref_edits(ref, panel, dict(panel)) == []


def test_normalization_artifact_is_not_a_reference_error():
    # The panel writes the reference's "lactuelle" as two words. That is a
    # spacing difference, not evidence about the audio, and the CER floor must
    # discard it.
    ref = "nous devons lactuelle situation examiner de pres aujourd hui".split()
    panel_hyp = "nous devons l actuelle situation examiner de pres aujourd hui"
    panel = _panel(panel_hyp, panel_hyp, panel_hyp, panel_hyp)
    edits = refdis.find_ref_edits(ref, panel, dict(panel), include_middle=True)
    assert [e for e in edits if e.kind == "delete"] == []


def test_interior_edits_excluded_by_default():
    ref = "the report on fisheries policy is adopted today by parliament".split()
    spoken = "the report on policy is adopted today by parliament"
    panel = _panel(spoken, spoken, spoken, spoken)
    assert refdis.find_ref_edits(ref, panel, dict(panel)) == []
    interior = refdis.find_ref_edits(ref, panel, dict(panel), include_middle=True)
    assert [e.position for e in interior] == ["middle"]
    assert interior[0].ref_tokens == ["fisheries"]


def test_small_panel_is_refused():
    ref = "madam president the report is adopted".split()
    spoken = "the report is adopted"
    panel = {"a": spoken, "b": spoken}
    assert refdis.find_ref_edits(ref, panel, dict(panel)) == []


def test_accept_ref_rate_excludes_ineligible_from_denominator():
    edits = [
        refdis.RefEdit("delete", "start", ["x"], [0], 4, 4, {"a": "ref", "b": "consensus", "c": None}),
        refdis.RefEdit("delete", "end", ["y"], [3], 4, 4, {"a": "ref", "b": "ref", "c": None}),
    ]
    rates = refdis.accept_ref_rate(edits)
    assert rates["a"]["rate"] == 1.0
    assert rates["a"]["n_eligible"] == 2
    assert rates["b"]["rate"] == 0.5
    assert "c" not in rates
    # Sorted worst-first so the table leads with the most benchmark-optimized.
    assert list(rates) == ["a", "b"]


def test_wilson_interval_stays_in_unit_range():
    p, lo, hi = refdis.wilson(0, 10)
    assert p == 0.0 and lo == 0.0 and 0 < hi < 1
    p, lo, hi = refdis.wilson(10, 10)
    assert p == 1.0 and 0 < lo < 1 and hi == 1.0
    assert refdis.wilson(0, 0) == (0.0, 0.0, 0.0)
