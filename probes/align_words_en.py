"""Word-level forced alignment of EN references via torchaudio MMS_FA.

For each (dataset, key) in a source manifest, forced-aligns the reference
transcript to the audio and emits per-word time spans::

    {__key__, dataset, duration, n_words, words: [{idx, w, t0, t1}, ...]}

``idx`` indexes the original whitespace-split reference words (1:1), so a
downstream truncation builder can map a cut time → "first N reference words
whose audio survived" without re-tokenizing. Words that normalize to empty
(pure punctuation) inherit a zero-width span at the previous boundary so the
index stays aligned with ``text.split()``.

Runs on GPU. Drives off a source dataset's ``manifest.parquet``; optionally
filtered to a minimum duration / word count so 50%-truncation downstream stays
fair (a 1-word clip can't be cut in half).
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import polars as pl
import soundfile as sf
import torch
import torchaudio

import roots as _paths  # data/model roots from BENCHMARK_OPT_DATA / BENCHMARK_OPT_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("align_words")

TARGET_SR = 16_000
_NORM_RE = re.compile(r"[^a-z']")


def normalize_word(w: str) -> str:
    """Lowercase + strip to the MMS_FA latin label set ([a-z']). May be empty."""
    return _NORM_RE.sub("", w.lower())


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav)
    if sr != TARGET_SR:
        t = torchaudio.functional.resample(t.unsqueeze(0), sr, TARGET_SR).squeeze(0)
    return t


def align_one(model, tokenizer, aligner, device, wav: torch.Tensor, words: list[str]) -> list[dict] | None:
    """Forced-align ``words`` against ``wav``. Returns per-original-word spans.

    Returns None if alignment is infeasible (no alignable words, audio too short).
    """
    n_samples = wav.shape[-1]
    # Map each original word → normalized form; align only the non-empty ones.
    norm = [normalize_word(w) for w in words]
    align_idx = [i for i, nw in enumerate(norm) if nw]
    if not align_idx:
        return None
    transcript = [norm[i] for i in align_idx]

    with torch.inference_mode():
        emission, _ = model(wav.unsqueeze(0).to(device))
    num_frames = emission.shape[1]
    if num_frames < len(transcript):
        return None
    ratio = n_samples / num_frames / TARGET_SR  # frame index → seconds

    try:
        token_spans = aligner(emission[0], tokenizer(transcript))
    except Exception as e:  # noqa: BLE001 - alignment can fail on degenerate input
        log.warning("aligner failed: %s", e)
        return None
    if len(token_spans) != len(transcript):
        return None

    # Per aligned word → (t0, t1) seconds.
    aligned_times: dict[int, tuple[float, float]] = {}
    for local_i, spans in enumerate(token_spans):
        if not spans:
            continue
        t0 = spans[0].start * ratio
        t1 = spans[-1].end * ratio
        aligned_times[align_idx[local_i]] = (float(t0), float(t1))

    # Forward-fill original-index spans; dropped words get a zero-width span.
    out: list[dict] = []
    prev_end = 0.0
    for i, w in enumerate(words):
        if i in aligned_times:
            t0, t1 = aligned_times[i]
            prev_end = t1
        else:
            t0 = t1 = prev_end
        out.append({"idx": i, "w": w, "t0": round(t0, 4), "t1": round(t1, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Source dataset name under DATA_ROOT/datasets.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--min-duration", type=float, default=10.0, help="Drop clips shorter than this (s).")
    ap.add_argument("--min-words", type=int, default=15, help="Drop clips with fewer reference words.")
    ap.add_argument("--max-duration", type=float, default=None, help="Drop clips longer than this (s) — aligner OOMs on long audio.")
    ap.add_argument("--language", default=None, help="Filter manifest to this language code (e.g. en).")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of samples (after filtering).")
    ap.add_argument("--text-col", default="text", help="Reference column to align.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_dir = Path(_paths.DATASETS_ROOT) / args.dataset / args.split
    manifest = split_dir / "manifest.parquet"
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")

    cols = ["__key__", "path", args.text_col, "duration"]
    if args.language:
        cols.append("language")
    m = pl.read_parquet(manifest, columns=cols)
    if args.language:
        m = m.filter(pl.col("language") == args.language)
        log.info("language=%s → %d rows", args.language, m.height)
    m = m.with_columns(pl.col(args.text_col).str.split(" ").list.len().alias("_nwords"))
    before = m.height
    m = m.filter((pl.col("duration") >= args.min_duration) & (pl.col("_nwords") >= args.min_words))
    if args.max_duration:
        m = m.filter(pl.col("duration") <= args.max_duration)
    log.info("filtered %d → %d samples (dur>=%.1fs, words>=%d)", before, m.height, args.min_duration, args.min_words)
    if args.limit:
        m = m.head(args.limit)
        log.info("limited to %d samples", m.height)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("must run on GPU")
    log.info("device: %s", device)

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    log.info("loaded MMS_FA (%d labels)", len(bundle.get_dict()))

    rows_out: list[dict] = []
    skipped = 0
    t0 = time.time()
    last_log = t0
    for i, row in enumerate(m.iter_rows(named=True)):
        wp = split_dir / row["path"]
        if not wp.exists():
            skipped += 1
            continue
        try:
            wav = load_wav(wp)
        except Exception:
            skipped += 1
            continue
        words = row[args.text_col].split(" ")
        spans = align_one(model, tokenizer, aligner, device, wav, words)
        if spans is None:
            skipped += 1
            continue
        rows_out.append({
            "__key__": row["__key__"],
            "dataset": args.dataset,
            "duration": float(row["duration"]),
            "n_words": len(words),
            "words": spans,
        })
        if time.time() - last_log > 30:
            log.info("  aligned %d / %d (skipped %d)", len(rows_out), m.height, skipped)
            last_log = time.time()

    log.info("aligned %d in %.1fs (skipped %d)", len(rows_out), time.time() - t0, skipped)
    df = pl.DataFrame(rows_out, strict=False)
    df.write_parquet(out_path)
    log.info("wrote %s (%d rows)", out_path, df.height)


if __name__ == "__main__":
    main()
