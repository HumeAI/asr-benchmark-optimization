#!/usr/bin/env python
"""Parse an EP "Verbatim report of proceedings" (CRE) XML into floor-language turns.

The CRE is the OFFICIAL gold transcript for a sitting:
    https://www.europarl.europa.eu/doceo/document/CRE-10-YYYY-MM-DD_EN.xml

Plain curl / WebFetch returns HTTP 202 / 0 bytes (bot-gated). Fetch it with
headless Chrome, which wraps the raw XML in an xml-viewer ``<div
id="webkit-xml-viewer-source-xml">``; this module recovers the inner XML from
either the wrapped or the raw form.

Each ``<INTERVENTION>`` carries one ``<ORATEUR ... LG="XX" LIB="Speaker Name">``
and ``<PARA>`` text. ``LG="EN"`` = the speaker spoke ENGLISH on the floor, so the
PARA text is the verbatim transcript of the original floor audio. Non-EN turns
are written translations and will not match the live interpreter audio, so we
keep only ``LG="EN"`` as gold for the VoxPopuli-equivalent floor-English set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

_SRC_RE = re.compile(r'<div id="webkit-xml-viewer-source-xml">(.*?)</div>\s*<div', re.S)
_INV_RE = re.compile(r"<INTERVENTION\b.*?</INTERVENTION>", re.S)
_ORA_RE = re.compile(r"<ORATEUR\b([^>]*)>")
_PARA_RE = re.compile(r"<PARA\b[^>]*>(.*?)</PARA>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Drop the leading EP role prefix + dash, e.g.
#   "President-in-Office of the Council . – Madam President, ..."
#   "on behalf of the PPE Group . – Madam President, ..."
#   "– In relation to ..."
_LEAD_DASH_RE = re.compile(r"^.{0,80}?[.]?\s*[–-]\s+")


@dataclass
class Turn:
    speaker: str
    lg: str
    text: str  # cased + punctuated gold text (CRE verbatim), role-prefix stripped


def _attr(attrs: str, name: str) -> str:
    m = re.search(rf'{name}="([^"]*)"', attrs)
    return m.group(1) if m else ""


def recover_source_xml(raw: str) -> str:
    """Return raw CRE XML from a Chrome-dumped DOM (or pass through if already raw)."""
    m = _SRC_RE.search(raw)
    return m.group(1) if m else raw


def parse_cre(path: str | Path, lang: str = "EN", strip_lead: bool = True) -> list[Turn]:
    """Parse a CRE file → list of ``Turn`` for ``ORATEUR/@LG == lang`` with non-empty text."""
    raw = Path(path).read_text()
    inner = recover_source_xml(raw)
    turns: list[Turn] = []
    for blk in _INV_RE.findall(inner):
        om = _ORA_RE.search(blk)
        if not om:
            continue
        attrs = om.group(1)
        lg = _attr(attrs, "LG")
        if lang and lg != lang:
            continue
        lib = _attr(attrs, "LIB").replace(" | ", " ").strip()
        parts: list[str] = []
        for p in _PARA_RE.findall(blk):
            t = unescape(_TAG_RE.sub(" ", p))
            t = _WS_RE.sub(" ", t).strip()
            if t:
                parts.append(t)
        text = " ".join(parts)
        if not text:
            continue
        if strip_lead:
            # Only strip when a dash actually separates a short role prefix.
            m = _LEAD_DASH_RE.match(text)
            if m and m.end() < 90:
                text = text[m.end():].strip()
        if text:
            turns.append(Turn(speaker=lib, lg=lg, text=text))
    return turns


def gold_concat(turns: list[Turn]) -> tuple[str, str, list[int]]:
    """Concatenate turn texts → (cased_gold, norm_gold, norm_idx→cased_idx map).

    ``norm_gold`` is lowercased, alnum+space only, so rapidfuzz partial_ratio
    over it is punctuation/casing invariant; ``map_idx`` recovers the cased span.
    """
    cased = "   ".join(t.text for t in turns)  # thin-space turn separator
    norm_chars: list[str] = []
    map_idx: list[int] = []
    for i, ch in enumerate(cased):
        lc = ch.lower()
        if lc.isalnum() or lc == " ":
            norm_chars.append(lc)
            map_idx.append(i)
    return cased, "".join(norm_chars), map_idx


if __name__ == "__main__":
    import sys

    turns = parse_cre(sys.argv[1])
    print(f"{len(turns)} EN turns")
    for t in turns[:5]:
        print(f"--- {t.speaker} ({len(t.text.split())} w) ---\n{t.text[:200]}")
