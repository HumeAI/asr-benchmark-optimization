"""Regenerate the manifests in this directory, or build your own.

The shipped example set came from:

    python examples/generate_predictions.py \
        --dataset facebook/voxpopuli --config en --split test \
        --streaming --limit 150 \
        --models facebook/wav2vec2-base-960h openai/whisper-tiny.en \
                 openai/whisper-base.en openai/whisper-small.en \
                 distil-whisper/distil-small.en

Any Hugging Face speech-recognition model and any audio dataset work. Audio is
streamed once and reused across models, so cost scales with models, not with
models x downloads.

    pip install torch torchaudio transformers "datasets[audio]"
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import pipeline

# Datasets name the reference transcript differently; take the first that appears.
TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "raw_text", "normalized_text")
ID_COLUMNS = ("id", "audio_id", "path")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--out", default="examples/predictions", type=Path)
parser.add_argument("--dataset", default="facebook/voxpopuli")
parser.add_argument("--config", default="en")
parser.add_argument("--split", default="test")
parser.add_argument("--streaming", action="store_true", help="do not download the whole corpus")
parser.add_argument("--limit", type=int, help="stop after N clips")
args = parser.parse_args()

dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
text_column = next(c for c in TEXT_COLUMNS if c in dataset.column_names)
id_column = next((c for c in ID_COLUMNS if c in dataset.column_names), None)
if args.limit:
    dataset = dataset.take(args.limit) if args.streaming else dataset.select(range(args.limit))

print(f"reading {args.dataset} ({text_column})...", flush=True)
clips = []
for i, row in enumerate(dataset):
    samples = row["audio"].get_all_samples()
    clips.append({
        # Prediction-only methods use this as the clip identifier, so it only has
        # to be stable across models -- it need not be a real path.
        "id": str(row[id_column]) if id_column else str(i),
        "reference": row[text_column],
        "array": samples.data.mean(dim=0).numpy(),  # decoded audio -> mono
        "sampling_rate": int(samples.sample_rate),
    })
print(f"{len(clips)} clips", flush=True)

args.out.mkdir(parents=True, exist_ok=True)
for model in args.models:
    started = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,  # otherwise clips past the model's window are truncated
    )
    path = args.out / f"{model.split('/')[-1]}.jsonl"
    with path.open("w") as f:
        for clip in clips:
            prediction = asr({
                "array": clip["array"],
                "sampling_rate": clip["sampling_rate"],
            })["text"]
            f.write(json.dumps({
                "audio_filepath": clip["id"],
                "text": clip["reference"],
                "pred_text": prediction,
            }) + "\n")
    print(f"{path}  ({time.time() - started:.0f}s)", flush=True)
