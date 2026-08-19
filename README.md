# ASR Benchmark Optimization

This repository provides methods for detecting whether an ASR model is reproducing a benchmark’s reference transcripts rather than transcribing the underlying audio.

Our white paper, *Quantifying Benchmark Optimization in ASR Models*, describes the methodology and presents results based on the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

The suite contains four methods that can be applied to your own models and datasets. The first two require only per-clip predictions. Masked-entity recovery additionally requires the audio and the ability to rerun inference, while teacher-forced NLL requires access to the model weights.

| Method | What it measures | Required input |
|---|---|---|
| [Reference-error detection](#reference-error-detection) | Identifies words or phrases in the reference transcript (reference spans) that a panel of models agrees were not spoken, then measures which models reproduce them anyway | Predictions |
| [Orthographic switch rate](#orthographic-switch-rate) | Tests whether a model’s rendering of acoustically indistinguishable spellings tracks the benchmark’s reference conventions | Predictions |
| [Masked-entity recovery](#masked-entity-recovery) | Tests whether a model still emits a word after the corresponding portion of the audio has been silenced | Audio + inference access |
| [Teacher-forced NLL](#teacher-forced-nll) | Compares which of two candidate transcripts the model considers more likely given the audio | Audio + model weights |

```bash
pip install -e .          # methods 1-2
pip install -e ".[probes]"  # 3-4: torch, torchaudio, soundfile
```

Python ≥3.10. `transformers` supplies the Whisper normalizers, so scores are
comparable to the [Open ASR
Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

## Quickstart Example

`examples/predictions/` contains predictions from five open ASR models over 150 clips of VoxPopuli English — European Parliament speech, whose references come from the written record rather than from the audio, which is the setting these methods are built for. Scoring them needs no audio, no model weights and no downloads:

```bash
benchmark-optimization ref-disagreement \
  --preds examples/predictions/ \
  --panel wav2vec2-base-960h,whisper-tiny.en,whisper-base.en,whisper-small.en
```

```text
clips: 150   panel: 4   reference errors found: 31

model               accept-ref          95% CI  n_ref  n_eligible
------------------  ----------  --------------  -----  ----------
distil-small.en          0.000  [0.000, 0.110]      0          31
wav2vec2-base-960h       0.000  [0.000, 0.125]      0          27
whisper-base.en          0.000  [0.000, 0.110]      0          31
whisper-small.en         0.000  [0.000, 0.110]      0          31
whisper-tiny.en          0.000  [0.000, 0.110]      0          31
```

The 31 reference errors are spans the whole panel agrees were never spoken, found without a hand-corrected transcript. Passing `--out edits.json` records each one with its text, kind, position, panel agreement and per-model verdict; the texts found here are chamber formalities and speaker attributions that the written record carries and the audio does not:

```text
delete start  'madam president'
insert start  'thank you'
delete start  'mister'
delete end    'eu neighborhood'
```

Every model scores 0.000, which is a result rather than an empty run: all five transcribe what was said instead of reproducing the record. A model that echoed `madam president` on most of those 31 edits would score near 1.0, and that gap is what `accept-ref` measures. Note that the panel here is four of the five scored models; as [Reference-error detection](#reference-error-detection) explains, a panel should be chosen independently of what you are testing. Denominators differ per model, so report `n_eligible` with any rate — see [Reading the output](#reading-the-output).

To score your own models, or to rebuild this example set, see [Input format](#input-format).

## Reference-error detection

This method finds reference spans that were probably never spoken, without a hand-corrected transcript, then scores each model on whether it reproduced them.

Per clip, given a reference and hypotheses from several models:

1. Align each hypothesis with the reference using `difflib` opcodes (`align.py`).
2. Identify candidate reference errors wherever a configured supermajority of the **panel**:
   - deletes the same contiguous span of the reference; or
   - inserts the same tokens at the same reference boundary.
3. Remove candidates where the panel’s alternative is character-wise similar to the reference. These are likely normalization differences rather than evidence of a reference error (`min_consensus_cer`).
4. Assign each model a verdict for every surviving edit:
   - `consensus`: the model made the same edit as the panel;
   - `ref`: the model reproduced the reference; or
   - `None`: the model’s hypothesis falls below `min_ref_match`, making it ineligible for that clip.

`accept-ref` is the share of eligible edits where a model sided with the reference. Applies to any corpus whose references were derived from something other than the audio — parliamentary records, subtitles, scripts.

```bash
benchmark-optimization ref-disagreement --preds predictions/ --panel a,b,c,d
```

```python
from benchmark_optimization import refdis

edits = refdis.find_ref_edits(reference_tokens, panel_hyps, all_hyps)
refdis.accept_ref_rate(edits)          # {model: {rate, n_ref, n_eligible, lo, hi}}
```

Choose the panel independently of the models under test; a panel of benchmark-optimized models will not flag the edits of interest. Insertions are only detected at the two reference boundaries — interior insertions cannot be anchored, since which side of a matched token an inserted word belongs to is an alignment choice.

## Orthographic switch rate

For this method, we focus on distinctions that sound identical but are written differently. Clips should be partitioned by which form the reference used:

    switch = min over arms a of  P(model emits arm a | reference uses a)

A model with a fixed habit is right on one arm and wrong on the other, so its minimum is near 0. Only a model that changes with the reference raises it.
Chance is 0.5.

```bash
benchmark-optimization switch-rate --preds predictions/ --spacing
```

Define your own conventions — the only requirement is that the arms are acoustically identical:

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

This method will silence a chosen word in the audio, keep it in the hidden reference, and check whether the model emits it anyway.

```bash
python probes/align_words_en.py --dataset <name> --split test        # word timings
python probes/build_entity_masked_dataset.py --source <name> --help  # silence + record
```

`build_entity_masked_dataset.py` selects target words (names, numbers), replaces their audio span with silence, and writes a corpus plus `truncation_meta.parquet` holding the removed text. Scoring is then a regex match of the hidden word against the hypothesis.

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

`examples/predictions/` holds the same format for five models over 150 VoxPopuli
clips, and ships in this repository, so it needs no download.

For your own models: one JSONL per model, named after the model, in the
leaderboard's manifest format:

```json
{"audio_filepath": "...", "text": "<reference>", "pred_text": "<prediction>"}
```

Text must be **raw**. `text`/`pred_text`, `reference`/`hypothesis` and `ref`/`hyp`
column names are all accepted, as are CSV and Parquet. `audio_filepath` is only a
clip identifier and need not be a real path; it has to be stable across models.

`examples/generate_predictions.py` produced the example set and will transcribe
any Hugging Face dataset with any Hugging Face speech-recognition model. Audio is
streamed once and reused across models, so cost scales with the number of models
rather than with models × downloads, and `--streaming` avoids downloading a
corpus in full:

```bash
pip install torch torchaudio transformers "datasets[audio]"

python examples/generate_predictions.py --streaming --limit 150 \
  --models openai/whisper-tiny.en facebook/wav2vec2-base-960h
```

Defaults are the VoxPopuli English test split; pass `--dataset`, `--config` and
`--split` for anything else. Any architecture works — the example set mixes CTC
(`wav2vec2`) and encoder-decoder (Whisper) models. The reference column is
detected across the usual names (`raw_text`, `text`, `transcription`), and
`torchaudio` handles resampling for corpora not already at 16 kHz.

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

## Resources

- Reproducing the paper figures: [`repro/REPRODUCE.md`](repro/REPRODUCE.md).
- Our white paper: [*RW-Voice-EQ Bench: A Real World Benchmark for Evaluating Voice AI Systems*](https://arxiv.org/abs/2607.14846) (arXiv) - the preliminary benchmark-optimization results, and the full evaluation they motivated.

## License

Apache-2.0.
