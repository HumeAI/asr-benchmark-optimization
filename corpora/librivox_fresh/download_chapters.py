"""STAGE 1 — fetch fresh-LibriVox chapter audio + book source text for the REAL
mini-LibriSpeech masked-number anchor (librivoxfresh).

Unlike download_clips.py (which cut 16 s *reference* clips for TTS cloning), this
downloads FULL body chapters per 2026 reader plus the Gutenberg / archive.org plain
text of each book, so build_librivoxfresh.py can whisper-anchor real audio windows to
verified ground-truth text (LibriSpeech-style construction).

Per roster reader (scripts freshlv_readers.json + catalog_raw.json):
  - download up to MAX_CH body sections (skip section 1 = spoken LibriVox intro),
    capped at ~MAX_MIN minutes of cumulative audio, polite rate.
  - fetch the book's source text once:
      gutenberg.org/ebooks/<id>  -> cache/epub/<id>/pg<id>.txt  (header/footer stripped)
      archive.org/details|stream/<id> -> download/<id>/<id>_djvu.txt (OCR; noisier, fine)

Out (SCRATCH/librivoxfresh/):
  mp3/<reader>_<sec>.mp3
  book_texts/<book_id>.txt              (cleaned plain text)
  chapters_meta.json  -> [{reader_id, display_name, prior_work_count, is_new_volunteer,
                           book_id, book_title, section_number, playtime_s, mp3,
                           book_text_path, text_ok}]

  sbatch scripts/freshlv/download_chapters.sh   (CPU partition)
"""

import json
import re
import time
import urllib.error
import urllib.request
import os
from pathlib import Path

UA = "asr-research/1.0 (theo@hume.ai; LibriVox mini-LibriSpeech anchor study)"
SCRATCH = Path(os.environ.get("BENCHMARK_OPT_SCRATCH", "scratch/librivox_fresh"))
OUT = SCRATCH / "librivoxfresh"
ROSTER = SCRATCH / "freshlv_readers.json"
CATALOG = SCRATCH / "catalog_raw.json"
MP3DIR = OUT / "mp3"
TXTDIR = OUT / "book_texts"
META = OUT / "chapters_meta.json"

MAX_CH = 5           # body sections per reader (after skipping intro)
MAX_MIN = 40         # cap cumulative audio per reader (minutes)
MIN_SEC_MIN = 3.0    # skip sections shorter than this (minutes) — too little text to anchor


def _get_bytes(url, tries=4, timeout=180):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (a + 1))
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return None


def download_mp3(url, dst):
    if dst.exists() and dst.stat().st_size > 10000:
        return True
    b = _get_bytes(url)
    if not b or len(b) < 10000:
        print(f"  DL FAIL {url}")
        return False
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(b)
    tmp.rename(dst)
    time.sleep(1.0)  # polite
    return True


def strip_gutenberg(txt: str) -> str:
    m0 = re.search(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG[^\n]*\*\*\*", txt, re.I)
    m1 = re.search(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG[^\n]*\*\*\*", txt, re.I)
    s = m0.end() if m0 else 0
    e = m1.start() if m1 else len(txt)
    body = txt[s:e]
    # drop a leading "Produced by ..." / transcriber block up to the first blank-line gap
    return body.strip()


def fetch_book_text(url_text_source: str, book_id: str) -> tuple[str | None, bool]:
    """Return (plain_text, ok). ok=False when no source could be fetched."""
    src = url_text_source or ""
    # Project Gutenberg
    mg = re.search(r"gutenberg\.org/(?:ebooks|files|cache/epub)/(\d+)", src)
    if mg:
        gid = mg.group(1)
        for u in (
            f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
            f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
            f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        ):
            b = _get_bytes(u, timeout=120)
            if b:
                return strip_gutenberg(b.decode("utf-8", "ignore")), True
    # archive.org
    ma = re.search(r"archive\.org/(?:details|stream)/([^/#?]+)", src)
    if ma:
        ident = ma.group(1)
        b = _get_bytes(f"https://archive.org/download/{ident}/{ident}_djvu.txt", timeout=180)
        if b:
            return b.decode("utf-8", "ignore").strip(), True
    return None, False


def main():
    MP3DIR.mkdir(parents=True, exist_ok=True)
    TXTDIR.mkdir(parents=True, exist_ok=True)
    roster = json.load(open(ROSTER))
    catalog = {str(b["id"]): b for b in json.load(open(CATALOG))}

    meta = []
    for r in roster:
        rid = r["reader_id"]
        bid = str(r["book_id"])
        book = catalog.get(bid, {})
        # fetch book text once
        txt_path = TXTDIR / f"{bid}.txt"
        if not txt_path.exists():
            txt, ok = fetch_book_text(book.get("url_text_source", ""), bid)
            if ok and txt and len(txt) > 2000:
                txt_path.write_text(txt)
            else:
                print(f"  TEXT FAIL reader {rid} book {bid} src={book.get('url_text_source','')!r}")
        text_ok = txt_path.exists() and txt_path.stat().st_size > 2000
        if not text_ok:
            # no usable ground-truth text -> reader unusable for the anchor; skip its audio
            print(f"reader {rid} {r['display_name'][:22]:22s} SKIP (no book text)")
            continue

        secs = [s for s in (book.get("sections") or [])
                if (s.get("listen_url") or "").endswith(".mp3")
                and int(s.get("playtime") or 0) >= MIN_SEC_MIN * 60]
        secs.sort(key=lambda s: int(s.get("section_number") or 0))
        body = secs[1:] or secs  # skip section 1 (title/intro track)

        cum_s = 0
        made = 0
        for s in body:
            if made >= MAX_CH or cum_s >= MAX_MIN * 60:
                break
            sec = int(s.get("section_number") or 0)
            pt = int(s.get("playtime") or 0)
            mp3 = MP3DIR / f"{rid}_{sec}.mp3"
            if not download_mp3(s["listen_url"], mp3):
                continue
            cum_s += pt
            made += 1
            meta.append({
                "reader_id": rid,
                "display_name": r["display_name"],
                "prior_work_count": r.get("prior_work_count"),
                "is_new_volunteer": r.get("is_new_volunteer"),
                "book_id": bid,
                "book_title": r.get("book_title"),
                "section_number": sec,
                "playtime_s": pt,
                "mp3": str(mp3),
                "book_text_path": str(txt_path),
                "text_ok": True,
            })
        print(f"reader {rid} {r['display_name'][:22]:22s} chapters={made} "
              f"audio={cum_s/60:.1f}min book={bid} textbytes={txt_path.stat().st_size}")

    json.dump(meta, open(META, "w"), indent=1)
    nr = len({m["reader_id"] for m in meta})
    tot_min = sum(m["playtime_s"] for m in meta) / 60
    print(f"\nDOWNLOAD_CHAPTERS_DONE {len(meta)} chapters from {nr} readers "
          f"({tot_min:.0f} min audio) -> {META}")


if __name__ == "__main__":
    main()
