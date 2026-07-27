"""Text normalization, matched to the Open ASR Leaderboard.

The leaderboard scores with the Whisper normalizers shipped in
``transformers``: :class:`EnglishTextNormalizer` for English (contractions,
number words, en-GB/en-US spelling) and :class:`BasicTextNormalizer`
elsewhere. We use the same ones so that a benchmark-optimization score
computed here is directly comparable to a WER computed there.

The en-GB -> en-US spelling map in ``data/english_spelling.json`` is a
verbatim copy of ``whisper/normalizers/english.json`` from
openai/whisper@04f449b8; re-download that file to update it.
"""

from __future__ import annotations

import json
import re
import string
from functools import lru_cache
from importlib import resources

__all__ = [
    "normalize",
    "basic_normalize",
    "tokenize",
    "has_whisper_normalizer",
    "canonicalize_numbers",
]

# CJK: whitespace does not delimit words, so word-level WER is meaningless.
# Standard practice is character-level error rate, which we obtain by spacing
# out every CJK character before word-level tokenization. Applied uniformly to
# reference and hypothesis, so the comparison stays fair.
_CJK_RE = re.compile(r"([㐀-䶿一-鿿぀-ゟ゠-ヿ가-힯])")

# The Whisper BasicTextNormalizer only strips halfwidth "(...)". Chinese text
# frequently mixes a halfwidth open paren with a fullwidth close, which the
# upstream regex misses, so we pre-strip both forms.
_PAREN_RE = re.compile(r"[(（][^)）]+?[)）]")


@lru_cache(maxsize=1)
def _normalizers():
    """Return ``(english, basic)`` Whisper normalizers, or ``(None, None)``."""
    try:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer,
            EnglishTextNormalizer,
        )
    except ImportError:
        return None, None
    spelling = json.loads(
        resources.files(__package__).joinpath("data/english_spelling.json").read_text(encoding="utf-8")
    )
    return (
        EnglishTextNormalizer(english_spelling_mapping=spelling),
        BasicTextNormalizer(remove_diacritics=True),
    )


def has_whisper_normalizer() -> bool:
    """True if ``transformers`` is installed and the Whisper normalizers loaded."""
    return _normalizers()[0] is not None


def basic_normalize(text: str) -> str:
    """Lowercase, strip ASCII punctuation, collapse whitespace.

    The fallback used when ``transformers`` is unavailable. Prefer
    :func:`normalize`.
    """
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str, language: str | None = "en") -> str:
    """Normalize ``text`` for word-level comparison.

    ``language`` is an ISO code; only its primary subtag matters (``en_us`` and
    ``en-GB`` both select English). Anything other than English routes to the
    basic normalizer. CJK is character-tokenized last so downstream word-level
    metrics become character-level.
    """
    if not text:
        return ""
    english, basic = _normalizers()
    if english is None:
        return _CJK_RE.sub(r" \1 ", basic_normalize(text)).strip()
    lang = (language or "").lower().split("_")[0].split("-")[0]
    text = _PAREN_RE.sub("", text)
    out = (english if lang == "en" else basic)(text).strip()
    return re.sub(r"\s+", " ", _CJK_RE.sub(r" \1 ", out)).strip()


def canonicalize_numbers(text: str, language: str | None = "en") -> str:
    """Rewrite spelled-out numbers as digits, so ``veinte`` matches ``20``.

    Stops word-vs-digit formatting counting as a reference error. Mainly matters
    outside English; the Whisper English normalizer already digitizes.

    Requires ``text2num``; a no-op without it, or for unsupported languages.
    """
    if not text:
        return text
    try:
        from text_to_num import alpha2digit
    except ImportError:
        return text
    lang = (language or "").lower().split("_")[0].split("-")[0]
    try:
        return alpha2digit(text, lang)
    except Exception:
        # alpha2digit raises on unsupported languages rather than passing through.
        return text


def tokenize(text: str, language: str | None = "en") -> list[str]:
    """Normalize then split on whitespace."""
    return normalize(text, language).split()
