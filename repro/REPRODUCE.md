# Reproducing the paper

```bash
pip install -e ".[figures]"
cd repro && make figures
```

That rebuilds **19 of the paper's 24 figures** into `repro/out`, with no inputs
beyond what is in this repository. The other five re-score raw per-model output
and need `BENCHMAXX_DATA` (below).

## What is shipped, and what is not

`repro/data` holds ~21 MB of **derived** data: the numbers behind each figure,
not the inputs that produced them. This is the usual arrangement for a paper
repository — the plotting code and the values it draws are version-controlled;
the hundreds of GB of audio and per-clip model output are not, and some corpora
are not ours to redistribute.

| directory | contents |
|---|---|
| `data/vmt/*.json` | per-figure result tables (steering ladders, splice/isolate cells, voice contrasts) |
| `data/vmt/wb/*.parquet` | per-utterance teacher-forced NLL from the white-box probe |
| `data/steer/newwl4/*.json` | low-rank steering ablate/induce results |
| `data/repaint/editlocus_*.json` | encoder/decoder patching results |
| `data/consensus/*_aggregate.json` | accept-ref leaderboards per corpus |
| `data/consensus/vox_en_*_samples.json` | every flagged reference error with per-model verdicts |
| `data/ortho_ppl*/…parquet` | per-clip NLL for the orthographic white-box arms |
| `data/ortho_switch.json` | orthographic switch rates, computed by `benchmaxx.ortho` |

Not shipped: source audio, per-clip transcription dumps, TTS renders, model
weights. `daikon` is a private held-out control — only its aggregate numbers
appear here, never its audio.

## Figure map

Every figure the paper includes, its generator, and what it needs.

| figure | generator | inputs |
|---|---|---|
| `wer_vs_badref.png` | `wer_vs_badref_fig.py` | shipped |
| `battery_consensus{,_full}.png` | `battery_panels.py` | shipped |
| `battery_masked{,_full}.png` | `battery_panels.py` | shipped |
| `battery_ablation{,_full}.png` | `battery_ablation_fig.py` | shipped |
| `battery_masked_ablation{,_full}.png` | `battery_masked_ablation_fig.py` | shipped |
| `consensus_robustness.png` | `consfull_fig.py robustness` | shipped |
| `consensus_whitebox_readouts.png` | `regen_whitebox_figs.py` | shipped |
| `isolate_gating.png` | `isolate_fig.py` | shipped |
| `libnum_voice_gap.png` | `libnum_voice_gap_fig.py` | shipped |
| `libnum_voice_ladder.png` | `libnum_voice_ladder_fig.py` | shipped |
| `nummask_lift_voice.png` | `nummask_lift_fig.py` | shipped |
| `orthohon_voice2x2.png` | `orthohon_voice2x2_fig.py` | shipped |
| `patch_dissociation.png` | `patch_dissociation_fig.py` | shipped |
| `steer_input_level.png`, `steer_activation_level.png` | `steer_causal_fig.py` | shipped |
| `masking_blackbox_datasets.png` | `masking_datasets_fig.py` | **`BENCHMAXX_DATA`** |
| `masking_voice.png` | `masking_voice_fig.py` | **`BENCHMAXX_DATA`** |
| `pair_spacing.png`, `pair_mister.png`, `pair_spacing_white.png` | `paired_grids.py` | **`BENCHMAXX_DATA`** |

The three `paired_grids.py` figures pair a switch-rate panel with a white-box
NLL panel. **The switch-rate panels are reproducible from shipped data** via
`ortho_switch_fig.py`, which recomputes them with `benchmaxx.ortho` from
`data/ortho_switch.json`; only the NLL panels need the raw root. The library's
switch rates were checked against `paired_grids.py` on the paper's data and
agree exactly (46 models on pooled spacing, 43 on the honorific comparison, max
difference 0.000000).

## Verifying the extraction

`benchmaxx.refdis` is a rewrite of the internal script behind the paper's
reference-disagreement numbers, so it is checked rather than assumed:

```bash
pytest tests/test_paper_equivalence.py -v
```

This replays the original run's exact inputs from
`data/consensus/vox_en_newwl4_samples.json` and asserts that every edit, every
per-model verdict (1,338 edits × 39 models), and every accept-ref rate in the
published leaderboard comes out identical.

## Using a full results root

Set `BENCHMAXX_DATA` to a directory laid out as:

```
$BENCHMAXX_DATA/
  datasets/{corpus}/{split}/manifest.parquet        # __key__, language, text
  datasets/{corpus}/{split}/truncation_meta.parquet # __key__, hidden_ref  (masked variants)
  results/{corpus}/{split}/{model}/{run}/*.wsds     # __key__, hyp[, hyp_raw]; DONE marker per run
  results/{corpus}/{model}/{split}/results.jsonl    # legacy layout, also read
```

`*.wsds` files are Arrow IPC. `hyp_raw` is preferred over `hyp` per row, since
the orthographic probe needs un-normalized text.

Then:

```bash
cd repro
BENCHMAXX_DATA=/path/to/results make all     # all 24 figures
BENCHMAXX_DATA=/path/to/results make cells   # re-derive data/ortho_switch.json
```

Environment variables: `BENCHMAXX_CELLS` (derived data, default `repro/data`),
`BENCHMAXX_FIGURES` (output, default `repro/out`), `BENCHMAXX_DATA` (raw root),
`BENCHMAXX_FONT_DIR` (optional directory holding the Fellix family used in the
paper; matplotlib's default font is used when unset, so figures render slightly
differently from the published PDF).

## Audio-side probes

`repro/probes/` holds the probes that need audio and model weights, and so
cannot run from prediction files:

| script | what it does |
|---|---|
| `build_entity_masked_dataset.py` | silences a chosen word (name, number) in each clip and records the hidden reference, producing the masked corpora |
| `align_words_en.py` | forced word alignment, to locate the span to silence |
| `probe_decoder_memorization.py` | the white-box engine: teacher-forces candidate transcripts and reads per-token log-probabilities |

```bash
pip install -e ".[probes]"     # torch, torchaudio, soundfile
export BENCHMAXX_DATA=/path/to/results BENCHMAXX_MODELS=/path/to/weights
python repro/probes/probe_decoder_memorization.py --help
```

Two caveats. Model loading covers the public checkpoints in the paper's roster;
the `omni` backend needed an internal loader that is not part of this release,
so its family is gated off, with the teacher-forcing logic kept as a reference
implementation. And these scripts are research code, lifted with paths
parameterized and internal imports replaced — they are less polished than
`src/benchmaxx`, which is the part meant for reuse.
