"""Convention families whose arms sound identical.

Each family is a set of *arms*: written forms rendered the same way out loud.
"Mr Smith" and "mister Smith" are the same acoustic event, as are ``$5`` /
``five dollars`` and ``2020`` / ``twenty twenty``. The audio cannot tell a model
which arm to emit, so it falls back on a prior; corpora differ in which arm they
use, so a prior that tracks the corpus tracks the corpus's conventions.

An arm may hold several patterns. Honorifics use this: ``Mr.`` and ``Mr`` are
both *abbreviated*, since the contrast is abbreviate-versus-spell-out and the
trailing period is a separate convention.

Families rejected for failing acoustic identity, recorded because the failure is
silent — a model preferring one arm would be doing acoustics, and would score as
convention-matching:

- ``U.S.``/``United States``, ``U.K.``, ``E.U.`` — different syllable counts.
- ``two thousand twenty``/``twenty twenty`` — different word counts.
- ``yes``/``yeah``, ``cannot``/``can't``, ``do not``/``don't``,
  ``going to``/``gonna``, ``want to``/``wanna``, ``got to``/``gotta``,
  ``because``/``'cause``, ``until``/``till`` — phonetically distinct.
- Spanish ``Nº`` — ``\bn[ºo]\b`` also matches the negation "no", and VoxPopuli-ES
  has no real instances.

Patterns run against **raw** text; normalization maps ``Mr.`` and ``mister`` to
the same string. All are case-insensitive, since models vary arbitrarily in
capitalization. Honorific arms require a following word, so a sentence-final
``St.`` (Street, pronounced differently) does not match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Family", "FAMILIES", "SPACING_PAIRS", "families_for", "family", "pooled_arms"]


@dataclass(frozen=True)
class Family:
    """A convention family: a name, a language, and two or more arms.

    ``arms`` is an ordered tuple of ``(label, patterns)``. :meth:`arm_of`
    returns the first arm with any matching pattern, so order the arms
    most-specific-first.
    """

    name: str
    language: str
    arms: tuple[tuple[str, tuple[str, ...]], ...]

    def compiled(self) -> list[tuple[str, list[re.Pattern]]]:
        return [(label, [re.compile(p) for p in pats]) for label, pats in self.arms]

    def arm_of(self, text: str) -> str | None:
        """Which arm this text uses, or ``None`` if the family does not apply."""
        text = text or ""
        for label, pats in self.compiled():
            if any(p.search(text) for p in pats):
                return label
        return None

    @property
    def arm_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.arms)


def _f(name: str, language: str, arms) -> Family:
    """Build a family, accepting a bare string or a list of patterns per arm."""
    norm = tuple((label, (pats,) if isinstance(pats, str) else tuple(pats)) for label, pats in arms)
    return Family(name, language, norm)


# Honorifics: abbreviated vs spelled out. The paper's headline family. Frequent
# enough for tight intervals, and split cleanly by corpus — read-speech corpora
# spell them out, parliamentary and broadcast corpora abbreviate. The
# abbreviated arm pools the pointed and unpointed forms.
#
# All patterns are case-insensitive. Capitalization is not the convention being
# measured, and models vary arbitrarily in whether they capitalize at all — a
# model writing "mr. smith" is abbreviating just as much as one writing
# "Mr. Smith", and scoring it as neither arm would report a fixed-convention
# model as switch 0 for the wrong reason.
_HON = [
    ("hon_mister", "Mr", [r"(?i)\bmr\.\s+\w", r"(?i)\bmr\b(?!\.)\s+\w"], "mister", [r"(?i)\bmister\b"]),
    ("hon_missus", "Mrs", [r"(?i)\bmrs\.\s+\w", r"(?i)\bmrs\b(?!\.)\s+\w"], "missus", [r"(?i)\bmiss[ui]s\b"]),
    ("hon_doctor", "Dr", [r"(?i)\bdr\.\s+[^\W\d_]", r"(?i)\bdr\b(?!\.)\s+[^\W\d_]"], "doctor", [r"(?i)\bdoctor\b"]),
    ("hon_professor", "Prof", [r"(?i)\bprof\.\s+[^\W\d_]", r"(?i)\bprof\b(?!\.)\s+[^\W\d_]"], "professor", [r"(?i)\bprofessor\b"]),
    ("hon_captain", "Capt", [r"(?i)\bcapt\.\s+[^\W\d_]", r"(?i)\bcapt\b(?!\.)\s+[^\W\d_]"], "captain", [r"(?i)\bcaptain\b"]),
    # "St." abbreviates both "Saint" and "Street", which are pronounced
    # differently and distinguished only by position. Kept for completeness but
    # positionally confounded — see the paper's appendix.
    ("hon_saint", "St", [r"(?i)\bst\.\s+[^\W\d_]", r"(?i)\bst\b(?!\.)\s+[^\W\d_]"], "saint", [r"(?i)\bsaint\b"]),
]

FAMILIES: tuple[Family, ...] = (
    *(_f(n, "en", [(a, ap), (b, bp)]) for n, a, ap, b, bp in _HON),

    # ── Symbol vs word ────────────────────────────────────────────────────
    _f("percent", "en", [("%", r"\d\s*%"), ("percent", [r"(?i)\bper\s?cent\b", r"(?i)\bpercent\b"])]),
    _f("dollars", "en", [("$", r"\$\s*\d"), ("dollars", r"(?i)\bdollars?\b")]),
    _f("euros", "en", [("€", r"€\s*\d|\d\s*€"), ("euros", r"(?i)\beuros?\b")]),

    # ── Abbreviation vs spelled out, non-honorific ────────────────────────
    _f("okay", "en", [("OK", [r"(?i)\bo\.k\.", r"(?i)\bok\b(?!\.)"]), ("okay", r"(?i)\bokay\b")]),
    _f("versus", "en", [("vs", [r"(?i)\bvs\.", r"(?i)\bvs\b(?!\.)"]), ("versus", r"(?i)\bversus\b")]),
    _f("alright", "en", [("alright", r"(?i)\balright\b"), ("all right", r"(?i)\ball right\b")]),

    # ── Compound spacing and hyphenation ──────────────────────────────────
    # Unlike honorifics, both arms occur inside a single corpus, so these can be
    # measured without comparing across datasets and therefore without the
    # register confound that cross-corpus families carry.
    _f("email", "en", [("e-mail", r"(?i)\be[- ]mail\b"), ("email", r"(?i)\bemail\b")]),
    _f("online", "en", [("on-line", r"(?i)\bon[- ]line\b"), ("online", r"(?i)\bonline\b")]),
    _f("website", "en", [("web site", r"(?i)\bweb site\b"), ("website", r"(?i)\bwebsite\b")]),
    _f("database", "en", [("data base", r"(?i)\bdata base\b"), ("database", r"(?i)\bdatabase\b")]),
    _f("policymaker", "en", [("policy maker", r"(?i)\bpolicy makers?\b"), ("policymaker", r"(?i)\bpolicymakers?\b")]),
    _f("decision_making", "en", [("decision-making", r"(?i)\bdecision-making\b"), ("decision making", r"(?i)\bdecision making\b")]),
    _f("long_term", "en", [("long-term", r"(?i)\blong-term\b"), ("long term", r"(?i)\blong term\b")]),
    _f("state_of_the_art", "en", [("state-of-the-art", r"(?i)\bstate-of-the-art\b"), ("state of the art", r"(?i)\bstate of the art\b")]),
    _f("follow_up", "en", [("follow-up", r"(?i)\bfollow-up\b"), ("follow up", r"(?i)\bfollow up\b")]),
    _f("cooperate", "en", [("co-operate", r"(?i)\bco-operate\b"), ("cooperate", r"(?i)\bcooperate\b")]),
    _f("reenter", "en", [("re-enter", r"(?i)\bre-enter\b"), ("reenter", r"(?i)\breenter\b")]),
    _f("nonzero", "en", [("non-zero", r"(?i)\bnon-zero\b"), ("nonzero", r"(?i)\bnonzero\b")]),
    _f("twentieth_century", "en", [("twentieth-century", r"(?i)\btwentieth-century\b"), ("twentieth century", r"(?i)\btwentieth century\b")]),

    # ── Years: digits vs words ────────────────────────────────────────────
    # "2020" is read "twenty twenty", so the written form is free. The
    # look-ahead on the word arm keeps it from firing inside a longer number.
    _f("year_2020", "en", [("2020", r"\b2020\b"), ("twenty twenty", r"(?i)\btwenty twenty\b(?!.{0,3}\d)")]),
    _f("year_1999", "en", [("1999", r"\b1999\b"), ("nineteen ninety nine", r"(?i)\bnineteen ninety[ -]?nine\b")]),

    # ── Spanish ───────────────────────────────────────────────────────────
    _f("hon_senor", "es", [("Sr", [r"(?i)\bsr\.\s+[^\W\d_]", r"(?i)\bsr\b(?!\.)\s+[^\W\d_]"]), ("señor", r"(?i)\bse[ñn]or\b")]),
    _f("hon_senora", "es", [("Sra", [r"(?i)\bsra\.\s+[^\W\d_]", r"(?i)\bsra\b(?!\.)\s+[^\W\d_]"]), ("señora", r"(?i)\bse[ñn]ora\b")]),
    _f("hon_senorita", "es", [("Srta", r"(?i)\bsrta\."), ("señorita", r"(?i)\bse[ñn]orita\b")]),
    _f("hon_dona", "es", [("Dña", r"(?i)\bd[ñn]a\."), ("doña", r"(?i)\bdo[ñn]a\b")]),
    _f("etcetera_es", "es", [("etc.", r"(?i)\betc\."), ("etcétera", r"(?i)\betc[eé]tera\b")]),
    _f("percent_es", "es", [("%", r"\d\s?%"), ("por ciento", r"(?i)\bpor\s?ciento\b")]),
    _f("euros_es", "es", [("€", r"€\s*\d|\d\s*€"), ("euros", r"(?i)\beuros?\b")]),

    # ── French ────────────────────────────────────────────────────────────
    _f("hon_monsieur", "fr", [("M.", r"\bM\.\s+[^\W\d_]"), ("monsieur", r"(?i)\bmonsieur\b")]),
    _f("hon_madame", "fr", [("Mme", r"\bMme\.?\s+[^\W\d_]"), ("madame", r"(?i)\bmadame\b")]),
    _f("percent_fr", "fr", [("%", r"\d\s?%"), ("pour cent", r"(?i)\bpour\s?cent\b")]),

    # ── German ────────────────────────────────────────────────────────────
    _f("abbr_usw", "de", [("usw.", r"(?i)\busw\."), ("und so weiter", r"(?i)\bund so weiter\b")]),
    _f("abbr_bzw", "de", [("bzw.", r"(?i)\bbzw\."), ("beziehungsweise", r"(?i)\bbeziehungsweise\b")]),
    _f("abbr_dh", "de", [("d.h.", r"(?i)\bd\.\s?h\."), ("das heißt", r"(?i)\bdas hei[ßs]t\b")]),
    _f("percent_de", "de", [("%", r"\d\s?%"), ("prozent", r"(?i)\bprozent\b")]),

    # ── Percent, remaining languages ──────────────────────────────────────
    _f("percent_it", "it", [("%", r"\d\s?%"), ("per cento", r"(?i)\bper\s?cento\b")]),
    _f("percent_nl", "nl", [("%", r"\d\s?%"), ("procent", r"(?i)\bprocent\b")]),
    _f("percent_pl", "pl", [("%", r"\d\s?%"), ("procent", r"(?i)\bprocent")]),
    _f("percent_pt", "pt", [("%", r"\d\s?%"), ("por cento", r"(?i)\bpor\s?cento\b")]),
)

# Indefinite pronouns and quantifiers written solid or spaced. Broken out
# because both arms are common inside a single read-speech corpus, which makes
# them the cleanest within-dataset switch test available. Individually each is
# too rare for a tight interval, so they are normally pooled — see
# :data:`SPACING_ARMS` and :func:`benchmark_optimization.ortho.pooled_switch_rate`. Arms are
# ordered (spaced, solid) consistently so positional pooling is well defined.
SPACING_PAIRS: tuple[Family, ...] = (
    _f("sp_anyone", "en", [("any one", r"(?i)\bany one\b"), ("anyone", r"(?i)\banyone\b")]),
    _f("sp_everyone", "en", [("every one", r"(?i)\bevery one\b"), ("everyone", r"(?i)\beveryone\b")]),
    _f("sp_someone", "en", [("some one", r"(?i)\bsome one\b"), ("someone", r"(?i)\bsomeone\b")]),
    _f("sp_anything", "en", [("any thing", r"(?i)\bany thing\b"), ("anything", r"(?i)\banything\b")]),
)

#: Role names for the pooled spacing group, in arm order.
SPACING_ARMS = ("spaced", "solid")

_BY_NAME = {f.name: f for f in FAMILIES + SPACING_PAIRS}


def family(name: str) -> Family:
    """Look up one family by name."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown convention family {name!r}; known: {sorted(_BY_NAME)}") from None


def families_for(language: str = "en", *, include_spacing: bool = True) -> list[Family]:
    """All families defined for a language."""
    pool = FAMILIES + SPACING_PAIRS if include_spacing else FAMILIES
    lang = (language or "").lower().split("_")[0].split("-")[0]
    return [f for f in pool if f.language == lang]


def pooled_arms(families: list[Family]) -> int:
    """Number of arms shared by every family, for positional pooling.

    Raises if the families disagree, since pooling arm *i* of one family with
    arm *i* of another is only meaningful when the positions mean the same
    thing.
    """
    counts = {len(f.arms) for f in families}
    if len(counts) != 1:
        raise ValueError(f"cannot pool families with differing arm counts: { {f.name: len(f.arms) for f in families} }")
    return counts.pop()
