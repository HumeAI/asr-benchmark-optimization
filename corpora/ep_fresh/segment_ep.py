#!/usr/bin/env python
"""Segment a long-form EP plenary WAV into short single-speaker English clips.

Pipeline (one wav per invocation):
  1. Speaker diarization (pyannote/speaker-diarization-3.1) -> (start, end, speaker) turns.
  2. Drop turns < MIN_DUR. Split turns > MAX_DUR at silence boundaries using silero VAD
     speech timestamps (never cut mid-speech).
  3. Language-ID each candidate clip with WhisperX/faster-whisper large-v3 detect_language.
     Keep ONLY English clips with lang_prob >= LANG_THRESH.
  4. Transcribe each kept English clip with WhisperX large-v3 (silver reference).
  5. Cut each kept clip as 16k mono PCM_16 wav into <out-dir>/clips/<key>.wav and write
     <out-dir>/clips.jsonl.

Run on a GPU node (diarization + whisper are GPU). VAD/cutting are CPU but run inline here.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

SR = 16000
HF_TOKEN = "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"  # reused from media-pipeline diarization.py


def log(*a):
    print(*a, flush=True)


def load_wav_mono16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz wav, got {sr}. Resample upstream.")
    return np.ascontiguousarray(audio.astype(np.float32))


# ----------------------------------------------------------------------------
# 1. Diarization
# ----------------------------------------------------------------------------
def diarize(audio, device):
    from pyannote.audio import Pipeline

    log("[diarize] loading pyannote/speaker-diarization-3.1 ...")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipe.to(torch.device(device))
    wav_t = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    log("[diarize] running diarization ...")
    out = pipe({"waveform": wav_t, "sample_rate": SR})
    turns = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in out.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])
    log(f"[diarize] {len(turns)} raw turns")
    del pipe
    torch.cuda.empty_cache()
    return turns


# ----------------------------------------------------------------------------
# 2. Silero VAD + turn splitting
# ----------------------------------------------------------------------------
def load_silero():
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def vad_speech_segments(audio, model, get_speech_timestamps):
    """Return list of (start_sec, end_sec) silero speech segments over whole file."""
    ts = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=SR,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=50,
        return_seconds=True,
    )
    return [(float(d["start"]), float(d["end"])) for d in ts]


def split_turn(turn_start, turn_end, speech_segs, min_dur, max_dur):
    """Split a long turn into <=max_dur sub-clips at silence boundaries.

    speech_segs: global list of (s,e) silero speech segments.
    Strategy: collect speech segments overlapping the turn; greedily accumulate them into
    sub-clips, breaking at a silence gap (between consecutive speech segments) whenever
    adding the next segment would exceed max_dur. Clip boundaries land in silence (the
    midpoint of the gap between two speech segments), so we never cut mid-speech.
    Sub-clips shorter than min_dur are dropped.
    """
    # speech segments overlapping this turn, clipped to turn extent
    local = []
    for s, e in speech_segs:
        if e <= turn_start or s >= turn_end:
            continue
        local.append((max(s, turn_start), min(e, turn_end)))
    if not local:
        return []
    local.sort()

    subclips = []
    cur_start = local[0][0]
    cur_end = local[0][1]
    for i in range(1, len(local)):
        s, e = local[i]
        prospective = e - cur_start
        if prospective > max_dur:
            # close current sub-clip at a silence boundary = midpoint of gap to next speech seg
            gap_mid = (cur_end + s) / 2.0
            subclips.append((cur_start, gap_mid))
            cur_start = s
            cur_end = e
        else:
            cur_end = e
    subclips.append((cur_start, cur_end))

    # If a sub-clip is still > max_dur (single very long speech run with no internal silence),
    # hard-split it into <=max_dur chunks as a fallback (rare for floor speech).
    final = []
    for s, e in subclips:
        if e - s <= max_dur:
            final.append((s, e))
        else:
            n = int(np.ceil((e - s) / max_dur))
            step = (e - s) / n
            for k in range(n):
                final.append((s + k * step, min(s + (k + 1) * step, e)))

    return [(s, e) for s, e in final if (e - s) >= min_dur]


def build_candidates(turns, speech_segs, min_dur, max_dur):
    """Turn diarization turns into candidate (start, end, speaker) clips in 5-20s range."""
    candidates = []
    for ts, te, spk in turns:
        dur = te - ts
        if dur < min_dur:
            continue
        if dur <= max_dur:
            candidates.append((ts, te, spk))
        else:
            for s, e in split_turn(ts, te, speech_segs, min_dur, max_dur):
                candidates.append((s, e, spk))
    return candidates


# ----------------------------------------------------------------------------
# 3 + 4. WhisperX language ID + transcription
# ----------------------------------------------------------------------------
def load_whisper(device):
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
        asr_options=dict(
            max_new_tokens=None,
            clip_timestamps=None,
            hallucination_silence_threshold=None,
            hotwords=None,
        ),
    )
    return whisperx, model


def detect_language(whisperx, model, clip_audio):
    """Return (language, probability) for a clip using faster-whisper detect_language."""
    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    audio = clip_audio
    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]
    return language, float(language_probability)


def transcribe_clip(model, clip_audio):
    out = model.transcribe(clip_audio, batch_size=1, language="en", task="transcribe")
    segs = out.get("segments", [])
    text = " ".join(s["text"].strip() for s in segs if s.get("text"))
    return " ".join(text.split())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--lang-thresh", type=float, default=0.7)
    ap.add_argument("--target-lang", default="en")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of candidate clips processed")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[main] device={device} wav={args.wav} video_id={args.video_id}")

    out_dir = args.out_dir
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    log("[main] loading audio ...")
    audio = load_wav_mono16k(args.wav)
    total_dur = len(audio) / SR
    log(f"[main] audio loaded: {total_dur:.1f}s")

    # 1. diarization
    turns = diarize(audio, device)

    # 2. silero VAD speech segments over whole file
    log("[main] loading silero VAD ...")
    vad_model, get_speech_timestamps = load_silero()
    log("[main] running silero VAD over full file ...")
    speech_segs = vad_speech_segments(audio, vad_model, get_speech_timestamps)
    log(f"[main] {len(speech_segs)} silero speech segments")

    candidates = build_candidates(turns, speech_segs, args.min_dur, args.max_dur)
    log(f"[main] {len(candidates)} candidate clips (5-20s) after split/filter")
    if args.limit:
        candidates = candidates[: args.limit]
        log(f"[main] limited to {len(candidates)} candidates")

    # 3 + 4. whisper LID + transcription
    whisperx, wmodel = load_whisper(device)

    kept = []
    n_lang_other = 0
    n_lowprob = 0
    for idx, (s, e, spk) in enumerate(candidates):
        f1 = int(round(s * SR))
        f2 = int(round(e * SR))
        clip = np.ascontiguousarray(audio[f1:f2])
        if clip.shape[0] < int(args.min_dur * SR):
            continue
        lang, prob = detect_language(whisperx, wmodel, clip)
        if lang != args.target_lang:
            n_lang_other += 1
            continue
        if prob < args.lang_thresh:
            n_lowprob += 1
            continue
        text = transcribe_clip(wmodel, clip)
        if not text:
            continue
        key = f"ep_{args.video_id}_{len(kept):04d}"
        clip_path = os.path.join(clips_dir, f"{key}.wav")
        sf.write(clip_path, clip, SR, subtype="PCM_16")
        rec = {
            "key": key,
            "video_id": args.video_id,
            "start": round(float(s), 3),
            "end": round(float(e), 3),
            "speaker": spk,
            "language": lang,
            "lang_prob": round(prob, 4),
            "duration": round(float(e - s), 3),
            "text": text,
        }
        kept.append(rec)
        if (idx + 1) % 25 == 0 or idx == 0:
            log(f"[main] processed {idx+1}/{len(candidates)} candidates, kept {len(kept)}")

    jsonl_path = os.path.join(out_dir, "clips.jsonl")
    # append-safe per-video: write video-specific then merge handled by caller; here we
    # write a per-video jsonl and also append to the shared clips.jsonl.
    per_video = os.path.join(out_dir, f"clips_{args.video_id}.jsonl")
    with open(per_video, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    with open(jsonl_path, "a") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    log("=" * 60)
    log(f"[done] video_id={args.video_id}")
    log(f"[done] candidates={len(candidates)} kept_english={len(kept)} "
        f"dropped_lang!=en={n_lang_other} dropped_lowprob={n_lowprob}")
    log(f"[done] per-video jsonl: {per_video}")
    log(f"[done] shared jsonl:    {jsonl_path}")
    log(f"[done] clips dir:       {clips_dir}")
    if kept:
        durs = np.array([r["duration"] for r in kept])
        log(f"[done] duration min={durs.min():.1f} max={durs.max():.1f} "
            f"mean={durs.mean():.1f} median={np.median(durs):.1f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
