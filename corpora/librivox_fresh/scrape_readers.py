"""Scrape fresh-LibriVox (2026) readers for the post-cutoff voice control.

Mirror of the ep-fresh scrape precedent, for LibriVox instead of EP-plenary. Builds
a roster of English readers whose LibriVox catalog history begins in 2026 (new
volunteers), so their voice timbre is post-training-cutoff. This is the LibriSpeech
analog of the ep-fresh clone control: if TTS clones of these fresh voices FIRE the
masked-number recovery like the libri-clone voices do, then the cue is register-driven;
if they land near generic/daikon, the cue reflects training exposure.

Selection rule (CRITICAL):
  - only English, SOLO-narrated books catalogued since 2026-01-01 (clean single voice)
  - prefer readers with the FEWEST prior LibriVox projects (new volunteers). We query
    each candidate reader's full catalog via the reader/get_results AJAX endpoint and
    record the total project count as prior-work metadata.
  - reader_id is a monotone join-order proxy (higher = newer volunteer); we rank
    candidates newest-first and take the ones with the smallest catalog footprint.

Light network only (catalog API + reader history). Runs on the login node under a
memory cap; polite sleeps. Downloads happen later in download_refs.sh (SLURM CPU).

Out: $BENCHMARK_OPT_SCRATCH/freshlv_readers.json
  (ulimit -v 16000000; python3 scripts/freshlv/scrape_readers.py)
"""

import json
import re
import time
import urllib.error
import urllib.request
import os
from pathlib import Path

UA = "asr-research/1.0 (theo@hume.ai; LibriVox voice-control study)"
SINCE = 1767225600  # 2026-01-01 00:00 UTC
SCRATCH = Path(os.environ.get("BENCHMARK_OPT_SCRATCH", "scratch/librivox_fresh"))
SCRATCH.mkdir(parents=True, exist_ok=True)
CATALOG = SCRATCH / "catalog_raw.json"
OUT = SCRATCH / "freshlv_readers.json"

N_READERS = 14          # distinct readers to select
N_CAND = 55             # candidate readers to fetch prior-work counts for
CLIPS_PER_READER = 3    # reference clips wanted per reader
NEW_VOL_MAX_PROJECTS = 4  # <= this many total projects => flagged new_volunteer


def _get(url, xhr=False, tries=3):
    hdr = {"User-Agent": UA}
    if xhr:
        hdr["X-Requested-With"] = "XMLHttpRequest"
    for a in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (a + 1))
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return None


def fetch_catalog():
    if CATALOG.exists():
        return json.load(open(CATALOG))
    allb, off = [], 0
    while True:
        url = (f"https://librivox.org/api/feed/audiobooks/?since={SINCE}"
               f"&format=json&limit=50&offset={off}&extended=1")
        d = _get(url)
        b = (d or {}).get("books", []) if d else []
        if not b:
            break
        allb += b
        off += 50
        time.sleep(0.4)
        if off > 4000:
            break
    json.dump(allb, open(CATALOG, "w"))
    return allb


def solo_reader(b):
    rs = {r["reader_id"] for s in (b.get("sections") or []) for r in (s.get("readers") or [])}
    return list(rs)[0] if (len(rs) == 1 and b.get("sections")) else None


def reader_project_count(rid):
    """Total LibriVox projects for a reader (<= 2 requests via pagination parse)."""
    def page(p):
        url = (f"https://librivox.org/reader/get_results?primary_key={rid}"
               f"&search_category=reader&sub_category=&search_page={p}"
               f"&search_order=catalog_date&project_type=either")
        return _get(url, xhr=True)
    d = page(1)
    if not d:
        return None
    n1 = d.get("results", "").count("catalog-result")
    pages = [int(x) for x in re.findall(r'data-page_number="(\d+)"', d.get("pagination", ""))]
    maxp = max(pages) if pages else 1
    if maxp <= 1:
        return n1
    dl = page(maxp)
    nl = dl.get("results", "").count("catalog-result") if dl else 0
    return (maxp - 1) * 25 + nl


def pick_chapters(book, want=CLIPS_PER_READER):
    """Pick sections with enough playtime for a middle-of-chapter reference clip.

    Reference clips are cut from the MIDDLE of each chapter (>=60 s in) to skip the
    standard LibriVox spoken intro. We only need one usable chapter with a few clip
    offsets, but prefer to spread across chapters when available.
    """
    secs = [s for s in (book.get("sections") or []) if int(s.get("playtime") or 0) >= 150]
    secs.sort(key=lambda s: int(s.get("section_number") or 0))
    # prefer sections after the first (first is often a dedicated title/intro track)
    body = secs[1:] or secs
    chosen = []
    for s in body:
        url = s.get("listen_url") or ""
        if not url.endswith(".mp3"):
            continue
        chosen.append({
            "section_number": int(s.get("section_number") or 0),
            "playtime": int(s.get("playtime") or 0),
            "mp3_url": url,
            "file_name": s.get("file_name"),
        })
        if len(chosen) >= max(2, want):
            break
    return chosen


def main():
    books = fetch_catalog()
    eng_solo = [b for b in books if b.get("language") == "English" and solo_reader(b)]
    print(f"catalog since 2026-01-01: {len(books)} books; english-solo: {len(eng_solo)}")

    # reader -> best solo book (most total audio) + display name
    by_reader = {}
    for b in eng_solo:
        rid = solo_reader(b)
        nm = next(r["display_name"] for s in b["sections"] for r in (s.get("readers") or [])
                  if r["reader_id"] == rid)
        tot = sum(int(s.get("playtime") or 0) for s in b["sections"])
        cur = by_reader.get(rid)
        if cur is None or tot > cur["total_audio_s"]:
            by_reader[rid] = {"reader_id": rid, "display_name": nm, "book_id": b["id"],
                              "book_title": b["title"], "num_sections": b["num_sections"],
                              "total_audio_s": tot, "_book": b}
    # newest-first (reader_id join-order proxy)
    cands = sorted(by_reader.values(), key=lambda x: -int(x["reader_id"]))[:N_CAND]
    print(f"distinct solo readers: {len(by_reader)}; probing prior-work for top {len(cands)} newest")

    enriched = []
    for i, c in enumerate(cands):
        pc = reader_project_count(c["reader_id"])
        c["prior_work_count"] = pc
        c["is_new_volunteer"] = (pc is not None and pc <= NEW_VOL_MAX_PROJECTS)
        enriched.append(c)
        print(f"  [{i+1}/{len(cands)}] reader {c['reader_id']:>6} {c['display_name'][:24]:24s} "
              f"projects={pc} audio={c['total_audio_s']}s book={c['book_title'][:30]!r}")
        time.sleep(0.4)

    # select: new volunteers first (fewest projects), require enough audio for clips
    usable = [c for c in enriched if c["total_audio_s"] >= 300 and c.get("prior_work_count")]
    usable.sort(key=lambda x: (x["prior_work_count"], -int(x["reader_id"])))
    selected = usable[:N_READERS]

    roster = []
    for c in selected:
        chapters = pick_chapters(c["_book"])
        if not chapters:
            continue
        roster.append({
            "reader_id": c["reader_id"],
            "display_name": c["display_name"],
            "prior_work_count": c["prior_work_count"],
            "is_new_volunteer": c["is_new_volunteer"],
            "book_id": c["book_id"],
            "book_title": c["book_title"],
            "total_audio_s": c["total_audio_s"],
            "chapters": chapters,
        })
    json.dump(roster, open(OUT, "w"), indent=1)
    print(f"\nSELECTED {len(roster)} readers -> {OUT}")
    for r in roster:
        print(f"  reader {r['reader_id']:>6} {r['display_name'][:24]:24s} "
              f"prior_work={r['prior_work_count']} new={r['is_new_volunteer']} "
              f"chapters={len(r['chapters'])}")


if __name__ == "__main__":
    main()
