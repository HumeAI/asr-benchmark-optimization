"""Shared masked-number hit matcher, robust to digits-vs-spelled formatting.

The WS2 runners emit verbose forms ("article two", "G20") while the steering harness
decode emits ITN digits ("article 2", "g 20"); a raw-string regex on hidden_ref
disagrees on ~20% of clips between the two. masked_hit() scores a hit if either the
normalized surface matches or every numeric value in the target appears among the
hypothesis's parsed values (number_norm.extract_values).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from number_norm import extract_values  # noqa: E402


def _prep(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)  # g20 -> g 20
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)  # 20th stays via extract_values; 20s -> 20 s
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def masked_hit(tgt: str, hyp: str) -> bool:
    t, h = _prep(tgt), _prep(hyp)
    if not t:
        return False
    if re.search(r"(^|\W)" + re.escape(t) + r"($|\W)", h):
        return True
    tv = [m.value for m in extract_values(t)]
    if not tv:
        return False
    hv = {m.value for m in extract_values(h)}
    return all(v in hv for v in tv)
