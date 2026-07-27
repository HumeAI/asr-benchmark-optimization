"""Download European-Parliament audio from YouTube -> 16 kHz mono WAV.

Stage 1 of the ep-fresh-2026 contamination-control pipeline. Pulls audio from
the official EP YouTube channel (or any playlist / explicit video ids), converts
to 16 kHz mono PCM_16, and records per-video metadata so downstream stages can
filter by upload date (we want post-training-cutoff material).

Scaling: point it at the EP plenary playlist and it pulls everything; the EP
channel uploads new plenaries continuously, so this is an effectively unbounded
fresh source. Use --after to keep only recent uploads, --max-videos to cap.

Network-bound (light CPU). Fine on a SLURM cpu node or, for a handful of
videos, the login node. Uses .venv-data (has yt-dlp + ffmpeg on PATH).

    .venv-data/bin/python scripts/ep_fresh/download_ep.py \
        --playlist "https://www.youtube.com/playlist?list=PLHQxK2YVsFVuqIXyec7M__H3X8wdYwhC3" \
        --after 20210101 --max-videos 20 \
        --out-dir $BENCHMARK_OPT_DATA/datasets/ep-fresh-2026/_staging/raw
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yt_dlp

TARGET_SR = 16000


def list_videos(playlist: str | None, video_ids: list[str]) -> list[str]:
    if video_ids:
        return video_ids
    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist, download=False)
    return [e["id"] for e in info.get("entries", []) if e.get("id")]


def probe(video_id: str) -> dict:
    opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return {
        "video_id": video_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "duration": info.get("duration"),
    }


def download_audio(video_id: str, out_dir: Path) -> Path:
    """Download bestaudio then transcode to 16 kHz mono PCM_16. Returns wav path."""
    raw_tmpl = str(out_dir / f"_raw_{video_id}.%(ext)s")
    opts = {
        "quiet": True,
        "format": "bestaudio",
        "outtmpl": raw_tmpl,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    raw_wav = out_dir / f"_raw_{video_id}.wav"
    final = out_dir / f"ep16k_{video_id}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_wav), "-ar", str(TARGET_SR), "-ac", "1",
         "-c:a", "pcm_s16le", str(final)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    raw_wav.unlink(missing_ok=True)
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--playlist", help="YouTube playlist URL")
    src.add_argument("--video-ids", nargs="+", help="explicit video ids")
    src.add_argument("--ids-file", help="file with one video id per line (handles ids starting with '-')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--after", default=None, help="keep uploads on/after YYYYMMDD")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--min-duration", type=float, default=60.0, help="skip clips shorter than N s")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "videos.jsonl"

    explicit_ids = list(args.video_ids or [])
    if args.ids_file:
        explicit_ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
    ids = list_videos(args.playlist, explicit_ids)
    print(f"{len(ids)} videos to consider")

    done = 0
    with meta_path.open("a") as mf:
        for vid in ids:
            if args.max_videos and done >= args.max_videos:
                break
            final = out_dir / f"ep16k_{vid}.wav"
            try:
                meta = probe(vid)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {vid}: probe failed: {e}")
                continue
            ud = meta.get("upload_date") or ""
            if args.after and ud and ud < args.after:
                print(f"skip {vid}: upload_date {ud} < {args.after}")
                continue
            if (meta.get("duration") or 0) < args.min_duration:
                print(f"skip {vid}: duration {meta.get('duration')}s < {args.min_duration}")
                continue
            if final.exists():
                print(f"have {vid} already")
            else:
                try:
                    download_audio(vid, out_dir)
                except Exception as e:  # noqa: BLE001
                    print(f"SKIP {vid}: download failed: {e}")
                    continue
            meta["wav"] = str(final)
            mf.write(json.dumps(meta) + "\n")
            mf.flush()
            done += 1
            print(f"[{done}] {vid} {ud} {meta.get('title','')[:70]}")

    print(f"downloaded {done} videos -> {out_dir}")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
