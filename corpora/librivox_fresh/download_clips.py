"""Download fresh-LibriVox chapter mp3s and cut clean reference clips.

Reads freshlv_readers.json (from scrape_readers.py). For each selected reader:
  - download the chosen chapter mp3(s) (polite, cached)
  - cut up to CLIPS_PER_READER reference clips of REF_DUR s each, taken from the
    MIDDLE of a chapter (first offset >= 60 s) to skip the spoken LibriVox intro,
    resampled to 16 kHz mono wav.

Runs on a SLURM CPU node (network + ffmpeg). Writes:
  clips/<reader_id>__<n>.wav
  clips_meta.json  -> [{reader_id, display_name, prior_work_count, is_new_volunteer,
                        clip_id, wav, book_id, section_number, offset_s, dur_s}]

  sbatch scripts/freshlv/download_clips.sh
"""

import json
import subprocess
import time
import urllib.request
import os
from pathlib import Path

UA = "asr-research/1.0 (theo@hume.ai; LibriVox voice-control study)"
SCRATCH = Path(os.environ.get("BENCHMARK_OPT_SCRATCH", "scratch/librivox_fresh"))
ROSTER = SCRATCH / "freshlv_readers.json"
MP3DIR = SCRATCH / "mp3"
CLIPDIR = SCRATCH / "clips"
META = SCRATCH / "clips_meta.json"
REF_DUR = 16          # seconds per reference clip
CLIPS_PER_READER = 3
# clip start offsets (s) into a chapter; middle regions avoid intro boilerplate
OFFSETS = [75, 170, 265, 360, 460]
SR = 16000


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 10000:
        return True
    tmp = dst.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(dst)
        time.sleep(1.0)  # polite
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  DL FAIL {url}: {e}")
        return False


def duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return 0.0


def cut(src, dst, offset, dur):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(offset), "-t", str(dur), "-i", str(src),
           "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 10000


def main():
    MP3DIR.mkdir(parents=True, exist_ok=True)
    CLIPDIR.mkdir(parents=True, exist_ok=True)
    roster = json.load(open(ROSTER))
    meta = []
    for r in roster:
        rid = r["reader_id"]
        made = 0
        for ch in r["chapters"]:
            if made >= CLIPS_PER_READER:
                break
            mp3 = MP3DIR / f"{rid}_{ch['section_number']}.mp3"
            if not download(ch["mp3_url"], mp3):
                continue
            dur_total = duration(mp3)
            for off in OFFSETS:
                if made >= CLIPS_PER_READER:
                    break
                if off + REF_DUR + 5 > dur_total:
                    continue
                clip_id = f"{rid}__{made}"
                wav = CLIPDIR / f"{clip_id}.wav"
                if cut(mp3, wav, off, REF_DUR):
                    meta.append({
                        "reader_id": rid,
                        "display_name": r["display_name"],
                        "prior_work_count": r["prior_work_count"],
                        "is_new_volunteer": r["is_new_volunteer"],
                        "clip_id": clip_id,
                        "wav": str(wav),
                        "book_id": r["book_id"],
                        "section_number": ch["section_number"],
                        "offset_s": off,
                        "dur_s": REF_DUR,
                    })
                    made += 1
        print(f"reader {rid} {r['display_name'][:22]:22s} clips={made} "
              f"prior_work={r['prior_work_count']}")
    json.dump(meta, open(META, "w"), indent=1)
    nr = len({m['reader_id'] for m in meta})
    print(f"\nDOWNLOAD_CLIPS_DONE {len(meta)} clips from {nr} readers -> {META}")


if __name__ == "__main__":
    main()
