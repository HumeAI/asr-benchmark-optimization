# asr-benchmark-optimization

Code for *Quantifying Benchmark Optimization in ASR Models*.

Two metrics computed from per-clip ASR predictions. No audio, no model weights.

- **Reference disagreement** (`refdis`) — where a panel of models agrees against
  the reference transcript, the share of those disagreements where a model
  reproduces the reference instead of the panel.
- **Orthographic switch rate** (`ortho`) — for written distinctions that sound
  identical (`Mr.`/`mister`, `$5`/`five dollars`), the rate at which a model's
  spelling matches whichever form the reference used.

The paper's third probe, masked-entity recovery, needs audio and weights; see
[`repro/probes/`](repro/probes/).

## Install

```bash
pip install -e .
```

Python ≥3.10. Depends on `transformers` for the Whisper normalizers, so scores
are comparable to the [Open ASR
Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

## Input

One JSONL per model, named after the model, in the leaderboard's manifest format:

```json
{"audio_filepath": "...", "text": "<reference>", "pred_text": "<prediction>"}
```

Text must be **raw**, not normalized — the switch rate measures distinctions that
normalization removes. `text`/`pred_text`, `reference`/`hypothesis` and `ref`/`hyp`
column names are all accepted, as are CSV and Parquet.

## Use

```bash
benchmark-optimization ref-disagreement --preds predictions/ --panel a,b,c,d
benchmark-optimization switch-rate      --preds predictions/ --spacing
```

```python
from benchmark_optimization import load_dir, conventions, ortho, refdis

preds = load_dir("predictions/")

# switch rate takes raw text
ortho.pooled_switch_rate(list(conventions.SPACING_PAIRS), list(preds.clips()),
                         arm_names=conventions.SPACING_ARMS)

# reference disagreement takes normalized text
norm = preds.normalized()
edits = []
for key, ref in norm.refs.items():
    hyps = norm.hyps[key]
    panel = {m: hyps[m] for m in PANEL if m in hyps}
    edits += refdis.find_ref_edits(ref.split(), panel, hyps)
refdis.accept_ref_rate(edits)
```

## Interpreting the output

- Both are rates over cases where the audio does not determine the reference.
  Neither measures transcription quality, and neither establishes that a model was
  trained on evaluation data.
- Denominators differ per model, since a model is only scored on the clips and
  edits it was eligible for. Report `n_eligible` with any rate.
- Models reproducing under half the reference are marked ineligible rather than
  scored; an empty or off-language hypothesis otherwise agrees with a span by
  accident.
- Chance for a two-arm switch rate is 0.5, not 0. Pool families (`--spacing`):
  individually they are too rare for the interval to clear chance.
- The switch-rate interval covers the limiting arm only, so it is
  anti-conservative when arms are close.
- Choose the reference-disagreement panel independently of the models under test.
  A panel member is scored like any other model.

## Provenance

The reference-disagreement implementation is a rewrite of the script used for the
paper. `tests/test_paper_equivalence.py` replays that run's inputs and asserts
identical output: 1,338 edits, every verdict across 39 models, every published
rate. Switch rates match the original generator exactly on the paper's data (46
and 43 models, max difference 0).

Running the CLI over raw manifests re-normalizes from raw text where the paper
used stored normalized hypotheses, giving 1,340 edits rather than 1,338 (0.15%).

## Reproducing the paper

[`repro/REPRODUCE.md`](repro/REPRODUCE.md). 19 of 24 figures rebuild from data in
the repository.

## Layout

```
src/benchmark_optimization/
  align.py         word-level alignment
  refdis.py        reference disagreement
  ortho.py         switch rate
  conventions.py   convention families
  normalize.py     Whisper normalizers
  predictions.py   loading prediction files
repro/             paper reproduction
tests/
```

## Citation

```bibtex
@misc{benchmarkoptimization2026,
  title  = {Quantifying Benchmark Optimization in ASR Models},
  author = {Lebryk, Theo and Baird, Alice},
  year   = {2026}
}
```

## License

Apache-2.0. `src/benchmark_optimization/data/english_spelling.json` is from
[openai/whisper](https://github.com/openai/whisper) (MIT).
