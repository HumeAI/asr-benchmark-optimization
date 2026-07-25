# Quantifying Benchmark Optimization in ASR Models

Code for *Quantifying Benchmark Optimization in ASR Models*.

Public ASR benchmarks are public, so models can be optimized for them in ways
that do not generalize. This repository measures that, using cases where **the
audio does not determine the reference transcript**. When the audio is
ambiguous, a model transcribing sound has to guess; a model that has absorbed
the benchmark's text does not.

Two probes run on **prediction files alone** — no audio, no model weights, no
GPU:

| probe | question | needs |
|---|---|---|
| **reference disagreement** (`refdis`) | Where a panel of models agrees *against* the reference, does the model follow the panel or the reference? | predictions from a panel + the models under test |
| **orthographic switch rate** (`ortho`) | For written distinctions that sound identical (`Mr.` vs `mister`, `$5` vs `five dollars`), does a model's spelling track the corpus? | predictions on corpora with opposing conventions |

A third probe from the paper, masked-entity recovery, needs the audio and model
weights; it lives in [`repro/probes/`](repro/probes/).

Neither number is a measure of transcription quality. Both are rates over
*underdetermined* cases, so a model can be accurate and score near zero, or
accurate and score high. They answer a different question from WER, which is the
point: in the paper, on VoxPopuli-English, the six models with the **lowest**
WER were exactly the six with the **highest** reference-disagreement rate.

## Install

```bash
pip install -e .
```

Python ≥3.10. The only hard dependency is `transformers`, for the Whisper
normalizers — the same ones the [Open ASR
Leaderboard](https://github.com/huggingface/open_asr_leaderboard) scores with,
so numbers here are comparable to numbers there.

## Use it on Open ASR Leaderboard predictions

The leaderboard's evaluation scripts write a JSONL manifest per model:

```json
{"audio_filepath": "...", "duration": 3.1, "time": 0.2, "text": "<reference>", "pred_text": "<prediction>"}
```

That is the native input format here. Those manifests hold **raw** text — the
leaderboard normalizes only when computing WER, after writing the manifest —
which is what makes the switch probe possible at all, since normalization erases
the exact distinctions it measures.

Put one manifest per model in a directory, named after the model:

```
predictions/
├── whisper-large-v3.jsonl
├── parakeet-tdt-0.6b-v2.jsonl
├── canary-qwen-2.5b.jsonl
└── ...
```

### Reference disagreement

```bash
benchmark-optimization ref-disagreement \
  --preds predictions/ \
  --panel whisper-large-v3,voxtral-mini-3b,qwen3-asr-0.6b,moonshine-streaming-medium
```

```
clips: 1842   panel: 4   reference errors found: 731

model                       accept-ref            95% CI  n_ref  n_eligible
-------------------------  -----------  ----------------  -----  ----------
model-a                          0.297  [0.263, 0.333]      201         677
model-b                          0.184  [0.156, 0.216]      128         696
...
```

The panel replaces a hand-corrected reference. Independent models that agree
with each other *against* the transcript are evidence about what was said, so
the probe needs no ground-truth cleanup — though a cleaned reference such as a
corrected VoxPopuli release works too, and is stronger where one exists.

**Choose the panel independently of the models you are testing.** A panel member
is scored like any other model, and a panel stacked with benchmark-optimized
models will fail to flag the very edits of interest. In the paper we used four
models that showed no VoxPopuli-specific behaviour on the other probes, and
validated the flagged edits against human judgement.

### Orthographic switch rate

```bash
benchmark-optimization switch-rate --preds predictions/ --spacing
```

```
=== pooled(sp_anyone,sp_everyone,sp_someone,sp_anything)  (spaced / solid) ===
model                       switch          95% CI  limiting arm    n  >chance
--------------------------  ------  --------------  ------------  ---  -------
model-a                      0.812  [0.647, 0.911]        spaced  102      yes
model-b                      0.469  [0.309, 0.636]        spaced  102       no
...
```

`--spacing` pools the four compound-spacing families. **Pool by default**: any
single family is rare enough that its interval spans chance, so per-family
output looks like a null result even when the pooled effect is large. Use
`--pool` to pool an explicit `--families` list, but only pool families whose arms
mean the same thing.

Switch rate is the **minimum across arms** of *P(model emits the arm the
reference used)*. A model with a fixed habit is right on one arm and wrong on
the other, so its minimum is near 0 whatever its habit; only a model that
changes arm with the reference raises it. **Chance is 0.5**, not 0 — read the
interval against 0.5.

Two ways to get the arms, with different confounds:

- **Within-dataset** — both spellings occur in one corpus. True for compound
  spacing (`anyone`/`any one`), and the cleanest test available: speaker,
  register, and recording conditions are matched by construction.
- **Cross-dataset** — each arm comes from a different corpus, the only option
  for honorifics, since a single corpus almost never uses both. Register then
  varies with the arm, so prefer corpora of the same register. The paper's
  honorific comparison uses two read-speech corpora with opposite conventions.

Convention families live in
[`src/benchmark_optimization/conventions.py`](src/benchmark_optimization/conventions.py), with the
acoustic-identity check recorded for each and the reason each rejected family
was rejected. Adding a family is a few lines; the bar is that a fluent speaker
renders both arms identically.

### As a library

```python
from benchmark_optimization import load_dir, conventions, ortho, refdis, tokenize

preds = load_dir("predictions/")
print(preds.coverage())            # clips per model — catches partial runs

# switch rate needs RAW text
results = ortho.switch_rates(conventions.families_for("en"), list(preds.clips()))

# reference disagreement needs normalized text
norm = preds.normalized()
edits = []
for key, ref in norm.refs.items():
    hyps = norm.hyps[key]
    panel = {m: hyps[m] for m in PANEL if m in hyps}
    edits += refdis.find_ref_edits(ref.split(), panel, hyps)
print(refdis.accept_ref_rate(edits))
```

## Reading the numbers honestly

- **Denominators differ per model.** A model is only charged for clips and edits
  it was eligible for. Always report `n_eligible` alongside a rate.
- **Competence gating matters.** An empty or off-language hypothesis
  "agrees" with any span by accident. Models reproducing less than half the
  reference are marked ineligible, not scored.
- **Normalization artifacts are not reference errors.** A panel writing
  `l actuelle` where the reference has `lactuelle` says nothing about the audio,
  and is filtered by a character-error floor.
- **The switch-rate interval is on the limiting arm alone.** It does not account
  for having taken a minimum over arms, so it is anti-conservative when arms are
  close. Treat near-chance results as near-chance.
- **A high score is not proof of training-set contamination**, and this
  repository does not test for it. It shows a model reproducing benchmark
  conventions that the audio underdetermines. The paper's evidence is that this
  behaviour is gated by narrow acoustic cues associated with the released
  benchmark distributions.

## Is this the code that produced the paper?

It is a rewrite of it, checked against it rather than assumed equivalent.

- **Reference disagreement.** `tests/test_paper_equivalence.py` replays the
  published run's exact inputs and asserts identical output: all 1,338 flagged
  reference errors, every per-model verdict across 39 models, and every
  accept-ref rate in the published leaderboard.
- **Switch rate.** Checked against the original generator on the paper's data:
  46 models on pooled LibriSpeech spacing and 43 on the cross-corpus honorific
  comparison, maximum difference 0.000000.

One difference worth knowing about, if you compare printed numbers. The paper
scored models' *stored normalized* hypotheses; running the CLI over raw
leaderboard manifests re-normalizes from raw text instead, and the two
normalization paths are not quite identical — 1,340 flagged errors instead of
1,338 (0.15%), moving the top model from 0.3986 to 0.400. The algorithm is the
same; the inputs differ slightly. Re-normalizing from raw is the right choice
here, because the switch probe needs raw text and both probes should see the same
input.

## Reproducing the paper

See [`repro/REPRODUCE.md`](repro/REPRODUCE.md). Figure generators and the
derived data they consume are included, so every figure rebuilds offline in a
clean clone. The audio-side probes need model weights and the source corpora.

## Layout

```
src/benchmark_optimization/        prediction-only probes — the reusable part
  align.py            word-level alignment primitives
  refdis.py           reference-error reproduction
  ortho.py            orthographic switch rate
  conventions.py      convention families, with acoustic-identity notes
  normalize.py        Whisper normalizers, matched to the leaderboard
  predictions.py      prediction-file loading and joining
repro/                paper reproduction
  figures/            figure generators + house style
  probes/             audio-side probes (weights required)
  data/               derived data behind the figures
tests/
docs/                 leaderboard integration notes
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

Apache-2.0. The en-GB→en-US spelling map in `src/benchmark_optimization/data/` is from
[openai/whisper](https://github.com/openai/whisper) (MIT).
