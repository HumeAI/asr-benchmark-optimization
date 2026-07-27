"""Onboard segmented EP-fresh clips into the wav + manifest dataset layout.

Reads a staging directory produced by ``scripts/ep_fresh/segment_ep.py``:

    <staging>/clips/<key>.wav        # 16 kHz mono, single-speaker English
    <staging>/clips.jsonl            # one row per clip (see fields below)

and writes the canonical dataset:

    DATA_ROOT/datasets/ep-fresh-2026/test/manifest.parquet
    DATA_ROOT/datasets/ep-fresh-2026/test/wavs/en/<key>.wav

clips.jsonl row fields consumed: key, video_id, speaker, language, duration,
text (provisional/silver whisper transcript). The whisper transcript is stored
as ``text`` (and normalized into ``text_normalized``) only as a placeholder
reference -- the contamination probe scores models against a *consensus*
reference, not this silver text. ``session_id`` carries the YouTube video id and
``speaker_id`` the diarization label so clips can be traced back to source.

Run on a SLURM cpu node (or login node under `ulimit -v 16000000`); CPU-only.
Uses the project's ``.venv-data`` env.

    .venv-data/bin/python scripts/ep_fresh/onboard_ep_fresh.py \
        --staging $BENCHMARK_OPT_DATA/datasets/ep-fresh-2026/_staging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

# Data roots resolve through corpora/roots.py (BENCHMARK_OPT_DATA);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roots import DATASETS_ROOT  # noqa: E402
from benchmark_optimization.normalize import normalize as normalize_for_wer  # noqa: E402
from wav_store import WavStoreWriter, has_manifest, read_manifest  # noqa: E402

DATASET = "ep-fresh-2026"
SPLIT = "test"
LANG = "en"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, help="dir with clips/ and clips.jsonl")
    ap.add_argument(
        "--datasets-root",
        default=None,
        help="override DATASETS_ROOT (default: $ASR_BENCHMARKING_DATA_ROOT/datasets)",
    )
    ap.add_argument("--overwrite", action="store_true", help="ignore any existing manifest")
    args = ap.parse_args()

    staging = Path(args.staging)
    clips_jsonl = staging / "clips.jsonl"
    clips_dir = staging / "clips"
    if not clips_jsonl.exists():
        raise SystemExit(f"missing {clips_jsonl}")

    datasets_root = Path(args.datasets_root) if args.datasets_root else DATASETS_ROOT
    dataset_dir = datasets_root / DATASET / SPLIT
    writer = WavStoreWriter(dataset_dir)

    # Additive ingest: keep any previously-onboarded clips unless --overwrite.
    if not args.overwrite and has_manifest(dataset_dir):
        for row in read_manifest(dataset_dir).to_pylist():
            writer.stage_row(row)
        print(f"merged {len(writer._seen_keys)} existing rows")

    added = skipped = missing = 0
    for line in clips_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = rec["key"]
        if writer.has_key(key):
            skipped += 1
            continue
        wav = clips_dir / f"{key}.wav"
        if not wav.exists():
            print(f"WARN no wav for {key}: {wav}")
            missing += 1
            continue
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:  # safety: downmix if somehow stereo
            audio = audio.mean(axis=1)
        text = (rec.get("text") or "").strip()
        row = {
            "__key__": key,
            "text": text,
            "text_normalized": normalize_for_wer(text, LANG),
            "language": LANG,
            "dataset": DATASET,
            "split": SPLIT,
            "duration": float(rec.get("duration", len(audio) / sr)),
            "speaker_id": str(rec.get("speaker", "")),
            "session_id": str(rec.get("video_id", "")),
            "topic": "europarl-plenary",
        }
        if writer.write_sample(row, audio, sr):
            added += 1

    out = writer.finalize()
    print(f"added={added} skipped={skipped} missing={missing}")
    print(f"manifest -> {out} ({len(writer._seen_keys)} total clips)")


if __name__ == "__main__":
    main()
