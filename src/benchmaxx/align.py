"""Word-level alignment primitives shared by the probes.

Every probe reduces to one question: *at this span of the reference, did the
model side with the reference or with the audio?* Answering it needs an
alignment between reference tokens and hypothesis tokens, and a way to name
the span that a model deleted, substituted, or inserted.

We align with :class:`difflib.SequenceMatcher` rather than a WER-style
Levenshtein alignment. It is deterministic, dependency-free, and its opcode
view (``equal`` / ``replace`` / ``delete`` / ``insert``) maps directly onto the
distinctions the probes need. ``autojunk`` is disabled throughout: the
heuristic drops frequent tokens, which on real transcripts means dropping
exactly the function words that carry the reference-error signal.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

__all__ = [
    "missed_indices",
    "matched_count",
    "substitution_at",
    "insertions_by_anchor",
    "consensus_insertion",
    "emitted_insertion",
    "cer",
]


def _opcodes(ref_tokens: list[str], hyp: str):
    return SequenceMatcher(None, ref_tokens, (hyp or "").split(), autojunk=False).get_opcodes()


def missed_indices(ref_tokens: list[str], hyp: str) -> set[int]:
    """Reference positions the hypothesis did not reproduce.

    A position counts as missed under ``delete`` (the hypothesis skipped it)
    and under ``replace`` (the hypothesis put something else there).
    """
    out: set[int] = set()
    for tag, i1, i2, _j1, _j2 in _opcodes(ref_tokens, hyp):
        if tag in ("delete", "replace"):
            out.update(range(i1, i2))
    return out


def matched_count(ref_tokens: list[str], hyp: str) -> int:
    """Number of reference positions the hypothesis reproduced exactly.

    Used as a competence gate. A model that returned nothing, hallucinated, or
    answered in the wrong language "agrees" with any given span by accident;
    requiring it to have matched a decent fraction of the rest of the
    reference keeps those accidents out of the numerator.
    """
    return sum(i2 - i1 for tag, i1, i2, _j1, _j2 in _opcodes(ref_tokens, hyp) if tag == "equal")


def substitution_at(ref_tokens: list[str], hyp: str, span: set[int]) -> str:
    """The hypothesis text that stands in for reference positions ``span``.

    Only ``replace`` opcodes contribute; a pure ``delete`` contributes the
    empty string, which is the correct reading — the model emitted nothing
    there.
    """
    hyp_tokens = (hyp or "").split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in _opcodes(ref_tokens, hyp):
        if tag == "replace" and span & set(range(i1, i2)):
            out.extend(hyp_tokens[j1:j2])
    return " ".join(out)


def insertions_by_anchor(ref_tokens: list[str], hyp: str) -> dict[int, list[str]]:
    """Map each reference boundary to the hypothesis tokens inserted there.

    Anchor ``0`` is before the first reference token; anchor ``len(ref_tokens)``
    is after the last. ``replace`` opcodes are excluded: their hypothesis
    tokens are substitutions, and are already visible through
    :func:`missed_indices`.
    """
    hyp_tokens = (hyp or "").split()
    out: dict[int, list[str]] = {}
    for tag, i1, _i2, j1, j2 in _opcodes(ref_tokens, hyp):
        if tag == "insert":
            out.setdefault(i1, []).extend(hyp_tokens[j1:j2])
    return out


def consensus_insertion(insert_lists: list[list[str]], at_start: bool, threshold: int) -> list[str]:
    """The insertion that a threshold of panel models agree on, token by token.

    Insertions are aligned *to the boundary*, not to each other, and the chunk
    grows outward only while a single token holds ``threshold`` votes at the
    same distance from the boundary. At the start boundary that means aligning
    right-to-left (the token immediately before the first reference word is
    position 0); at the end boundary, left-to-right.

    Taking the longest or most common whole insertion instead would let one
    verbose outlier define the consensus.
    """
    votes: list[str] = []
    k = 0
    while True:
        at_k = [lst[-(k + 1)] if at_start else lst[k] for lst in insert_lists if k < len(lst)]
        if not at_k:
            break
        token, n = Counter(at_k).most_common(1)[0]
        if n < threshold:
            break
        votes.append(token)
        k += 1
    return list(reversed(votes)) if at_start else votes


def emitted_insertion(model_insert: list[str], consensus: list[str], at_start: bool) -> bool:
    """Did one model insert ``consensus`` at the same boundary alignment?

    Boundary-anchored, so the consensus chunk must be a suffix of the model's
    insertion at the start boundary and a prefix at the end boundary.
    """
    if not consensus:
        return False
    n = len(consensus)
    if len(model_insert) < n:
        return False
    return model_insert[-n:] == consensus if at_start else model_insert[:n] == consensus


def cer(ref: str, hyp: str) -> float:
    """Character error rate, whitespace-insensitive.

    Whitespace is stripped so that spacing- and accent-only differences
    register as near-zero rather than as real disagreement. This is what
    separates a genuine reference error from a normalization artifact: the
    panel writing "l actuelle" where the reference has "lactuelle" is not
    evidence about the audio.

    Approximated from ``SequenceMatcher`` matched-character coverage rather
    than a true edit distance; the probes only threshold it, so the
    approximation is not load-bearing.
    """
    r = "".join(ref.split())
    h = "".join(hyp.split())
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    sm = SequenceMatcher(None, r, h, autojunk=False)
    matched = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in sm.get_opcodes() if tag == "equal")
    return 1.0 - matched / len(r)
