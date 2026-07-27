"""STAGE 2 — construct the REAL mini-LibriSpeech corpus `librivoxfresh`.

For each downloaded 2026-LibriVox chapter (download_chapters.py):
  1. mp3 -> 16 kHz mono float array.
  2. whisper-large-v3 word-timestamp transcription (chunked long-form; cached).
  3. ANCHOR whisper words to the book's plain text with a moving-pointer local
     alignment (difflib on a small book window per whisper segment) -> each whisper
     word gets a book word index; the book text is the ground truth.
  4. CUT clips: merge consecutive whisper words into 4-28 s windows ending at a
     book sentence boundary; the clip's reference text is the ORIGINAL book span
     (readable, numbers kept as written). QC: fraction of NON-NUMBER book words in
     the span that align (SequenceMatcher 'equal') to a whisper word must be
     >= MATCH_MIN (LibriSpeech manual-verification substitute); record match_score.
  5. write datasets/librivoxfresh/test/{manifest.parquet, wavs/en/<key>.wav}.

Ground-truth text normalization mirrors LibriSpeech: matching is uppercase-insensitive
and punctuation-stripped; the stored `text` keeps a readable, uppercased rendering
(consistent with librispeech-clean manifests) with number words preserved.

  srun --partition=gpu --gres=gpu:1 --time=4:00:00 \
    bash scripts/freshlv/build_librivoxfresh.sh   (submitted via sbatch)
"""

import json
import re
import sys
from difflib import SequenceMatcher
from math import gcd
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SCRATCH = Path(os.environ.get("BENCHMARK_OPT_SCRATCH", "scratch/librivox_fresh"))
CORPUS = SCRATCH / "librivoxfresh"
META = CORPUS / "chapters_meta.json"
WHISPER_CACHE = CORPUS / "whisper_cache"
DR = Path(os.environ.get("BENCHMARK_OPT_DATA", "data"))
OUT = DR / "datasets/librivoxfresh/test"
QC_OUT = DR / "analysis/voxmode/vmt/librivoxfresh_corpus.json"

SR = 16000
MIN_CLIP_S = 4.0
MAX_CLIP_S = 28.0
TARGET_MIN_S = 8.0     # start allowing a sentence-boundary close past this
MATCH_MIN = 0.95       # non-number-word agreement whisper vs book
WORD_RE = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d")
_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    # ordinals commonly spoken
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
}


def norm_word(w: str) -> str:
    return "".join(WORD_RE.findall(w.lower()))


def is_number_word(w: str) -> bool:
    c = norm_word(w)
    return bool(_NUM.search(c)) or c in _NUMWORDS


def load_mp3_16k(path: str) -> np.ndarray:
    import torch
    import torchaudio
    a, sr = torchaudio.load(path)
    a = a.mean(0)  # mono
    if sr != SR:
        a = torchaudio.functional.resample(a, sr, SR)
    return a.numpy().astype(np.float32)


# ---------------------------------------------------------------- book tokens
def tokenize_book(text: str):
    """Return (norm_words, readable_words, sentence_end_flag) aligned lists.

    readable_words keep original casing/number spelling; sentence_end_flag[i]=True
    when the readable token ends a sentence (trailing . ! ? possibly + quote)."""
    # hyphen/dash -> space so spelled compounds ("twenty-three") split into maskable
    # number tokens and the reference matches LibriSpeech's spaced convention.
    text = text.replace("\r", " ").replace("\n", " ").replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-‐-―]", " ", text)
    raw = re.split(r"\s+", text.strip())
    norm, read, send = [], [], []
    for tok in raw:
        nw = norm_word(tok)
        if not nw:
            continue
        norm.append(nw)
        # readable: clean to LibriSpeech style (uppercase alnum + apostrophes) but the
        # sentence-end flag is read from the ORIGINAL punctuation before stripping.
        read.append(re.sub(r"[^A-Za-z0-9']", "", tok))
        send.append(bool(re.search(r'[.!?]["\')\]]*$', tok)))
    return norm, read, send


# ---------------------------------------------------------------- whisper
def transcribe_chapter(pipe, wav: np.ndarray, cache_path: Path):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    out = pipe(
        {"array": wav, "sampling_rate": SR},
        return_timestamps="word",
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=4,
        generate_kwargs={"num_beams": 1, "language": "en", "task": "transcribe"},
    )
    words = []
    for ch in out.get("chunks", []):
        ts = ch.get("timestamp") or (None, None)
        w = (ch.get("text") or "").strip()
        if not w or ts[0] is None:
            continue
        t1 = ts[1] if ts[1] is not None else ts[0]
        words.append({"w": w, "t0": float(ts[0]), "t1": float(t1)})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"words": words}))
    return {"words": words}


# ---------------------------------------------------------------- anchoring
K_SEED = 6


def build_ngram_index(book_norm, k=K_SEED):
    idx = {}
    for i in range(len(book_norm) - k + 1):
        idx.setdefault(tuple(book_norm[i:i + k]), []).append(i)
    return idx


def seed(wnorm, i, ng_idx, min_pos, k=K_SEED, lookahead=60):
    """Find a book position for whisper words at/after index i via a k-gram match.

    Chapter audio maps to an arbitrary region of the full book, and each chapter
    opens with LibriVox boilerplate ("chapter three of ... by ...") absent from the
    book text — so a purely forward local aligner from position 0 never lands. This
    scans the next `lookahead` whisper k-grams for one present in the book, preferring
    the closest occurrence at/after min_pos (falling back to the global-closest).
    Returns (whisper_index, book_pos) or None."""
    n = len(wnorm)
    for s in range(i, min(i + lookahead, n - k + 1)):
        key = tuple(wnorm[s:s + k])
        cands = ng_idx.get(key)
        if not cands:
            continue
        ge = [p for p in cands if p >= min_pos]
        return s, (ge[0] if ge else min(cands, key=lambda p: abs(p - min_pos)))
    return None


def anchor(whisper_words, book_norm, ng_idx):
    """Assign each whisper word a book index via k-gram-located global alignment.

    A chapter's audio maps to a CONTIGUOUS region of the full book, opening with
    LibriVox boilerplate ("chapter three of ... by ...") absent from the book text.
    Step 1: k-gram-seed the chapter's book start (skipping the boilerplate). Step 2:
    difflib the WHOLE whisper word list against the located book slice (length ~2.6x
    the whisper length + margin), mapping every 'equal' block to book indices. Global
    alignment recovers far more words than a forward-only local aligner and needs no
    fragile re-seeding. Returns book_idx (len == len(whisper_words)); -1 unmatched."""
    wnorm = [norm_word(x["w"]) for x in whisper_words]
    n_w, n_b = len(wnorm), len(book_norm)
    book_idx = [-1] * n_w
    # Multi-point consensus seed: a single opening k-gram can collide with the book's
    # TITLE PAGE (the LibriVox boilerplate speaks the book title), stranding the whole
    # chapter in front matter (20323_4 failure mode). Sample k-grams at several whisper
    # offsets, convert each hit to an implied book start (book_pos - whisper_offset),
    # and take the median of the largest agreeing cluster (tolerance 600 words).
    implied = []
    for frac in (0.02, 0.15, 0.3, 0.5, 0.7, 0.85):
        off = int(n_w * frac)
        sd = seed(wnorm, off, ng_idx, 0, lookahead=120)
        if sd is not None:
            ws, bpos = sd
            implied.append(bpos - ws)
    if not implied:
        return book_idx
    best_cluster = max(([x for x in implied if abs(x - c) <= 600] for c in implied), key=len)
    s0 = max(0, int(np.median(best_cluster)) - 50)
    s1 = min(n_b, s0 + int(n_w * 2.6) + 400)
    bslice = book_norm[s0:s1]
    sm = SequenceMatcher(None, wnorm, bslice, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for kk in range(size):
            wi = a + kk
            if wi < n_w:
                book_idx[wi] = s0 + b + kk
    return book_idx


def build_clips(whisper_words, book_idx, book_norm, book_read, book_send, base_key):
    """Greedy sentence-boundary clip cutting with QC. Yields clip dicts."""
    clips = []
    n = len(whisper_words)
    i = 0
    while i < n:
        # skip unanchored leading words
        if book_idx[i] < 0:
            i += 1
            continue
        start_t = whisper_words[i]["t0"]
        b_start = book_idx[i]
        j = i
        last_valid = None  # (j_end, b_end, dur) at a sentence boundary in-range
        while j < n:
            dur = whisper_words[j]["t1"] - start_t
            if dur > MAX_CLIP_S:
                break
            bj = book_idx[j]
            if bj >= 0 and dur >= MIN_CLIP_S and book_send[bj]:
                if dur >= TARGET_MIN_S:
                    last_valid = (j, bj, dur)
                    break
                last_valid = (j, bj, dur)
            j += 1
        if last_valid is None:
            i += 1
            continue
        j_end, b_end, dur = last_valid
        # book span
        if b_end < b_start:
            i = j_end + 1
            continue
        span_norm = book_norm[b_start:b_end + 1]
        span_read = book_read[b_start:b_end + 1]
        # QC: non-number book words that got an aligned whisper 'equal' match
        matched_book = set()
        for wi in range(i, j_end + 1):
            bi = book_idx[wi]
            if bi >= 0 and b_start <= bi <= b_end and norm_word(whisper_words[wi]["w"]) == book_norm[bi]:
                matched_book.add(bi)
        denom = [bi for bi in range(b_start, b_end + 1) if not is_number_word(book_norm[bi])]
        num = sum(1 for bi in denom if bi in matched_book)
        score = (num / len(denom)) if denom else 0.0
        end_t = whisper_words[j_end]["t1"]
        clip_dur = end_t - start_t
        if score >= MATCH_MIN and MIN_CLIP_S <= clip_dur <= MAX_CLIP_S and len(span_norm) >= 6:
            text = " ".join(span_read).upper()
            text = re.sub(r"\s+", " ", text).strip()
            clips.append({
                "t0": round(start_t, 3),
                "t1": round(end_t, 3),
                "text": text,
                "match_score": round(score, 4),
                "n_words": len(span_norm),
                "has_number": any(is_number_word(w) for w in span_norm),
            })
            i = j_end + 1
        else:
            i += 1
    return clips


def main():
    meta = json.load(open(META))
    print(f"chapters: {len(meta)}", flush=True)
    # load whisper pipeline
    import torch
    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    pipe.model.generation_config.language = "en"
    pipe.model.generation_config.task = "transcribe"

    wav_dir = OUT / "wavs/en"
    wav_dir.mkdir(parents=True, exist_ok=True)
    book_cache = {}
    rows = []
    per_ch_stats = []
    for ci, ch in enumerate(meta):
        rid = ch["reader_id"]
        sec = ch["section_number"]
        bid = ch["book_id"]
        base_key = f"librivoxfresh_test_{rid}-{bid}-{sec}"
        # book tokens (cache per book)
        if bid not in book_cache:
            btxt = Path(ch["book_text_path"]).read_text(errors="ignore")
            bn, br, bs = tokenize_book(btxt)
            book_cache[bid] = (bn, br, bs, build_ngram_index(bn))
        book_norm, book_read, book_send, ng_idx = book_cache[bid]
        try:
            wav = load_mp3_16k(ch["mp3"])
        except Exception as e:  # noqa: BLE001
            print(f"  [{ci+1}/{len(meta)}] {base_key} MP3 FAIL {e}", flush=True)
            continue
        cache_path = WHISPER_CACHE / f"{rid}_{sec}.json"
        tr = transcribe_chapter(pipe, wav, cache_path)
        ww = tr["words"]
        if len(ww) < 20:
            print(f"  [{ci+1}/{len(meta)}] {base_key} too few whisper words ({len(ww)})", flush=True)
            continue
        bidx = anchor(ww, book_norm, ng_idx)
        clips = build_clips(ww, bidx, book_norm, book_read, book_send, base_key)
        n_kept = 0
        for ck, c in enumerate(clips):
            s = max(0, int(c["t0"] * SR))
            e = min(len(wav), int(c["t1"] * SR))
            if e - s < int(MIN_CLIP_S * SR):
                continue
            key = f"{base_key}-{ck:04d}"
            rel = f"wavs/en/{key}.wav"
            sf.write(str(OUT / rel), wav[s:e], SR, subtype="PCM_16")
            rows.append({
                "__key__": key, "path": rel, "text": c["text"],
                "language": "en", "dataset": "librivoxfresh", "split": "test",
                "duration": round((e - s) / SR, 4), "sample_rate": SR,
                "speaker_id": str(rid), "book_id": bid, "section_number": sec,
                "match_score": c["match_score"], "has_number": c["has_number"],
                "prior_work_count": ch.get("prior_work_count"),
                "is_new_volunteer": ch.get("is_new_volunteer"),
                "display_name": ch.get("display_name"),
            })
            n_kept += 1
        n_num = sum(1 for c in clips if c["has_number"])
        per_ch_stats.append({"key": base_key, "reader": rid, "book": bid, "section": sec,
                             "whisper_words": len(ww), "clips": len(clips),
                             "kept": n_kept, "number_clips": n_num})
        print(f"  [{ci+1}/{len(meta)}] {base_key} wwords={len(ww)} clips={len(clips)} "
              f"number_clips={n_num}", flush=True)

    if not rows:
        raise SystemExit("no clips built")
    tbl = pa.Table.from_pylist(sorted(rows, key=lambda r: r["__key__"]))
    pq.write_table(tbl, str(OUT / "manifest.parquet"))
    n_num = sum(1 for r in rows if r["has_number"])
    n_read = len({r["speaker_id"] for r in rows})
    tot_h = sum(r["duration"] for r in rows) / 3600
    qc = {
        "n_clips": len(rows), "n_number_clips": n_num, "n_readers": n_read,
        "total_hours": round(tot_h, 3),
        "match_score_mean": round(float(np.mean([r["match_score"] for r in rows])), 4),
        "match_min": MATCH_MIN, "clip_range_s": [MIN_CLIP_S, MAX_CLIP_S],
        "per_chapter": per_ch_stats,
    }
    QC_OUT.parent.mkdir(parents=True, exist_ok=True)
    QC_OUT.write_text(json.dumps(qc, indent=1))
    print(f"\nBUILD_LIBRIVOXFRESH_DONE clips={len(rows)} number_clips={n_num} readers={n_read} "
          f"hours={tot_h:.2f} -> {OUT}/manifest.parquet")
    print(f"QC -> {QC_OUT}")


if __name__ == "__main__":
    main()
