from benchmark_optimization import align


def test_missed_indices_covers_delete_and_replace():
    ref = "the quick brown fox".split()
    assert align.missed_indices(ref, "the quick brown fox") == set()
    assert align.missed_indices(ref, "the quick fox") == {2}
    assert align.missed_indices(ref, "the quick red fox") == {2}


def test_matched_count_gates_incompetent_output():
    ref = "the quick brown fox".split()
    assert align.matched_count(ref, "the quick brown fox") == 4
    assert align.matched_count(ref, "") == 0
    assert align.matched_count(ref, "le renard brun rapide") == 0


def test_substitution_at_reports_replacement_but_not_deletion():
    ref = "the quick brown fox".split()
    assert align.substitution_at(ref, "the quick red fox", {2}) == "red"
    assert align.substitution_at(ref, "the quick fox", {2}) == ""


def test_insertions_anchor_at_boundaries():
    ref = "quick brown fox".split()
    assert align.insertions_by_anchor(ref, "well quick brown fox") == {0: ["well"]}
    assert align.insertions_by_anchor(ref, "quick brown fox indeed") == {3: ["indeed"]}


def test_consensus_insertion_aligns_to_boundary_not_to_longest():
    # Three models insert at the end boundary. Two agree on "thank you"; the
    # third is longer. The consensus must be what the majority share, so the
    # verbose outlier must not define it.
    lists = [["thank", "you"], ["thank", "you"], ["thank", "you", "very", "much"]]
    assert align.consensus_insertion(lists, at_start=False, threshold=2) == ["thank", "you"]
    # At the start boundary alignment is right-to-left: the token adjacent to
    # the reference is position 0, so a leading extra word must not shift it.
    lists = [["mister", "president"], ["mister", "president"], ["well", "mister", "president"]]
    assert align.consensus_insertion(lists, at_start=True, threshold=2) == ["mister", "president"]


def test_consensus_insertion_requires_threshold():
    lists = [["thank", "you"], ["merci"], ["danke"]]
    assert align.consensus_insertion(lists, at_start=False, threshold=2) == []


def test_emitted_insertion_is_boundary_anchored():
    cons = ["thank", "you"]
    assert align.emitted_insertion(["thank", "you"], cons, at_start=False)
    assert align.emitted_insertion(["thank", "you", "very", "much"], cons, at_start=False)
    # At the end boundary the consensus must be a prefix, so trailing-only
    # agreement does not count.
    assert not align.emitted_insertion(["very", "much", "thank", "you"], cons, at_start=False)
    # At the start boundary it must be a suffix.
    assert align.emitted_insertion(["well", "thank", "you"], cons, at_start=True)
    assert not align.emitted_insertion(["thank", "you", "well"], cons, at_start=True)


def test_cer_ignores_whitespace_so_spacing_is_not_an_error():
    assert align.cer("lactuelle", "l actuelle") == 0.0
    assert align.cer("", "") == 0.0
    assert align.cer("", "anything") == 1.0
    assert align.cer("abcd", "abxy") == 0.5
