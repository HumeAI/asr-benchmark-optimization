# asr-benchmark-optimization

Methods for measuring whether an ASR model reproduces a benchmark's reference
text rather than transcribing the audio. From *Quantifying Benchmark Optimization
in ASR Models*.

Four methods, usable on your own corpora and models. The first two need only
per-clip predictions; the last two need audio and model weights.

| method | what it measures | needs |
|---|---|---|
| [reference-error detection](#reference-error-detection) | reference spans a panel of models agrees were never spoken, and which models reproduce them anyway | predictions |
| [orthographic switch rate](#orthographic-switch-rate) | whether a model's spelling of acoustically-invisible distinctions tracks the corpus | predictions |
| [masked-entity recovery](#masked-entity-recovery) | whether a model still emits a word after that word is silenced | audio |
| [teacher-forced NLL](#teacher-forced-nll) | which of two candidate transcripts a model assigns higher likelihood, given the audio | audio + weights |

## Quick start

Run the orthographic switch-rate probe on Whisper large-v3's published raw
predictions for the LibriSpeech clean test set:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[hub]"

hf download HumeAI/ASR-benchmark-optimization-predictions \
  predictions/librispeech-clean/whisper-large-v3.jsonl \
  --repo-type dataset --local-dir data/

benchmark-optimization switch-rate \
  --preds data/predictions/librispeech-clean/whisper-large-v3.jsonl \
  --spacing
```

This downloads one prediction manifest, not model weights, so it runs on CPU in
a few seconds after installation. The output reports whether Whisper follows
the reference's choice between forms such as `any one` and `anyone`; a score near
0 indicates a fixed spelling habit, while 0.5 is chance for the two arms. To test
another model, replace the JSONL with predictions in the same
[input format](#input-format).

## Installation

```bash
pip install -e .          # methods 1-2
pip install -e ".[probes]"  # 3-4: torch, torchaudio, soundfile
```

Python ≥3.10. `transformers` supplies the Whisper normalizers, so scores are
comparable to the [Open ASR
Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

## Reference-error detection

Finds reference spans that were probably never spoken, without a hand-corrected
transcript, then scores each model on whether it reproduced them.

Per clip, given a reference and hypotheses from several models:

1. Align each hypothesis to the reference (`difflib` opcodes, `align.py`).
2. Where ≥ a supermajority of a **panel** of models deletes the same contiguous
   reference span, or inserts the same tokens at the same reference boundary,
   that is a candidate reference error.
3. Drop candidates where the panel's substitute is character-wise close to the
   reference — those are normalization artifacts (`min_consensus_cer`), not
   evidence about the audio.
4. Verdict per model per surviving edit: `consensus` if it made the panel's
   edit, `ref` if it reproduced the reference, `None` if it reproduced under
   `min_ref_match` of the reference and so is not competent on that clip.

`accept-ref` is the share of eligible edits where a model sided with the
reference. Applies to any corpus whose references were derived from something
other than the audio — parliamentary records, subtitles, scripts.

```bash
benchmark-optimization ref-disagreement --preds predictions/ --panel a,b,c,d
```

```python
from benchmark_optimization import refdis

edits = refdis.find_ref_edits(reference_tokens, panel_hyps, all_hyps)
refdis.accept_ref_rate(edits)          # {model: {rate, n_ref, n_eligible, lo, hi}}
```

Choose the panel independently of the models under test; a panel of
benchmark-optimized models will not flag the edits of interest. Insertions are
only detected at the two reference boundaries — interior insertions cannot be
anchored, since which side of a matched token an inserted word belongs to is an
alignment choice.

## Orthographic switch rate

For distinctions that sound identical but are written differently, partition
clips by which form the reference used:

    switch = min over arms a of  P(model emits arm a | reference uses a)

A model with a fixed habit is right on one arm and wrong on the other, so its
minimum is near 0. Only a model that changes with the reference raises it.
Chance is 0.5.

```bash
benchmark-optimization switch-rate --preds predictions/ --spacing
```

Define your own conventions — the only requirement is that the arms are
acoustically identical:

```python
from benchmark_optimization import make_family, ortho

fahrenheit = make_family("fahrenheit", "en", [
    ("°F", r"\d\s*°\s*F"),
    ("degrees fahrenheit", r"(?i)\bdegrees fahrenheit\b"),
])
ortho.switch_rate(fahrenheit, clips)
```

`conventions.py` ships 31 English families plus Spanish, French, German, Italian,
Dutch, Polish and Portuguese, and records the families rejected for failing
acoustic identity. Individually rare families should be pooled
(`ortho.pooled_switch_rate`) or the interval will not clear chance. Matching runs
on raw text, since normalization is what erases these distinctions.

## Masked-entity recovery

Silence a chosen word in the audio, keep it in the hidden reference, and check
whether the model emits it anyway.

```bash
python probes/align_words_en.py --dataset <name> --split test        # word timings
python probes/build_entity_masked_dataset.py --source <name> --help  # silence + record
```

`build_entity_masked_dataset.py` selects target words (names, numbers), replaces
their audio span with silence, and writes a corpus plus `truncation_meta.parquet`
holding the removed text. Scoring is then a regex match of the hidden word
against the hypothesis.

The paper's masked corpora are published as **recipes rather than audio** under
`masks/` in the dataset above: `entity_t0`/`entity_t1` give the silenced span and
`hidden_ref` the removed word. Since the mask is `samples[t0:t1] = 0` and the
source corpora are natively 16 kHz, applying a recipe to the source clip
reproduces the masked audio exactly — 5.8 MB of recipes in place of 10.6 GB of
wavs, and no forced aligner needed.

## Teacher-forced NLL

Score candidate transcripts against the audio to see which the model prefers.

```bash
python probes/probe_decoder_memorization.py --help
```

Handles encoder-decoder and decoder-only models, returns per-token
log-probabilities over a forced target, and can mask the audio to separate the
language-model prior from the audio's contribution. Used for the reference-versus-
corrected comparison and the masked-word readouts.

## Building the held-out corpora

The paper's two held-out controls are freshly collected, post-cutoff speech.
`corpora/` holds the retrieval code rather than the audio.

| | source | scripts |
|---|---|---|
| **ep-fresh** | European Parliament plenary video | `corpora/ep_fresh/` |
| **libri-fresh** | LibriVox readers whose catalogue starts in 2026 | `corpora/librivox_fresh/` |

```bash
pip install -e ".[corpora]"
export BENCHMARK_OPT_DATA=data BENCHMARK_OPT_SCRATCH=scratch

# ep-fresh: fetch media, fetch and align the verbatim report, segment, onboard
python corpora/ep_fresh/download_ep.py --help
python corpora/ep_fresh/fetch_cre.py --help        # CRE = verbatim plenary report
python corpora/ep_fresh/assign_cre.py --help       # align report text to segments
python corpora/ep_fresh/segment_ep.py --help       # diarize + VAD + language-ID
python corpora/ep_fresh/onboard_ep_fresh.py --help # -> wav + manifest

# libri-fresh
python corpora/librivox_fresh/scrape_readers.py --help
python corpora/librivox_fresh/download_chapters.py --help
python corpora/librivox_fresh/download_clips.py --help
python corpora/librivox_fresh/build_librivoxfresh.py --help
```

`segment_ep.py` needs `pyannote/speaker-diarization-3.1` (gated — accept the
licence on the Hub) and silero VAD. Both corpora are collected from live sources,
so re-running yields a different clip set than ours; predictions for our exact
build are in the dataset above.

## Input format

Predictions for the paper's models are published at
[HumeAI/ASR-benchmark-optimization-predictions](https://huggingface.co/datasets/HumeAI/ASR-benchmark-optimization-predictions)
— 17 corpora, 308 model runs — so the paper's numbers are reproducible without
re-running inference:

```bash
hf download HumeAI/ASR-benchmark-optimization-predictions --repo-type dataset --local-dir data/
benchmark-optimization switch-rate --preds data/predictions/librispeech-clean --spacing
```

For your own models: one JSONL per model, named after the model, in the
leaderboard's manifest format:

```json
{"audio_filepath": "...", "text": "<reference>", "pred_text": "<prediction>"}
```

Text must be **raw**. `text`/`pred_text`, `reference`/`hypothesis` and `ref`/`hyp`
column names are all accepted, as are CSV and Parquet.

Or straight from the Hub:

```python
from benchmark_optimization import hub, conventions, ortho

preds = hub.load_predictions("librispeech-clean")
ortho.pooled_switch_rate(list(conventions.SPACING_PAIRS), list(preds.clips()),
                         arm_names=conventions.SPACING_ARMS)

hub.available()                                          # 17 corpora
hub.load_masks("voxpopuli-mask-num-all-numexp-silence")  # a mask recipe
```

```python
from benchmark_optimization import load_dir
preds = load_dir("predictions/")
preds.coverage()      # clips per model, catches partial runs
preds.normalized()    # for reference-error detection; NOT for switch rate
```

## Reading the output

- Both prediction-only metrics are rates over cases where the audio does not
  determine the reference. Neither measures transcription quality, and neither
  establishes that a model was trained on evaluation data.
- Denominators differ per model. Report `n_eligible` with any rate.
- Chance for a two-arm switch rate is 0.5, not 0.
- The switch-rate interval covers the limiting arm only, so it is
  anti-conservative when arms are close.

## Provenance

Inference used our own model wrappers, which are not part of this release: the
paper's roster spans eight different loaders (`hf_transformers`, `nemo`, `salm`,
`granite_speech`, `voxtral`, `phi4_multimodal`, `qwen3_asr`, plus one API). The
published predictions are the reproducibility artifact instead.

The reference-error implementation is a rewrite of the script used for the paper.
`tests/test_paper_equivalence.py` replays that run's inputs and requires identical
output: 1,338 edits, every verdict across 39 models, every published rate. Switch
rates match the original generator exactly on the paper's data.

Running the CLI over raw manifests re-normalizes from raw text where the paper
used stored normalized hypotheses, giving 1,340 edits rather than 1,338 (0.15%).

## Reproducing the paper's figures

[`repro/REPRODUCE.md`](repro/REPRODUCE.md).

## License

Apache-2.0.
