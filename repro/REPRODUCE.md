# Reproducing the paper

```bash
pip install -e ".[figures]"
cd repro && make figures
```

Rebuilds 19 of the paper's 24 figures into `repro/out`. The other five re-score
raw per-model output and need `BENCHMARK_OPT_DATA` (below).

## Data in the repository

`repro/data` holds ~21 MB of derived data — the values behind each figure, not
the inputs that produced them.

| path | contents |
|---|---|
| `data/vmt/*.json` | per-figure result tables |
| `data/vmt/wb/*.parquet` | per-utterance teacher-forced NLL |
| `data/steer/newwl4/*.json` | low-rank steering ablate/induce results |
| `data/repaint/editlocus_*.json` | encoder/decoder patching results |
| `data/consensus/*_aggregate.json` | accept-ref rates per corpus |
| `data/consensus/vox_en_*_samples.json` | flagged reference errors with per-model verdicts |
| `data/ortho_ppl*/*.parquet` | per-clip NLL for the orthographic arms |
| `data/ortho_switch.json` | switch rates, from `benchmark_optimization.ortho` |

Not included: source audio, per-clip transcription dumps, TTS renders, model
weights. `daikon` is a private held-out control; only its aggregate numbers
appear here.

## Figures

| figure | generator | inputs |
|---|---|---|
| `wer_vs_badref.png` | `wer_vs_badref_fig.py` | in repo |
| `battery_consensus{,_full}.png` | `battery_panels.py` | in repo |
| `battery_masked{,_full}.png` | `battery_panels.py` | in repo |
| `battery_ablation{,_full}.png` | `battery_ablation_fig.py` | in repo |
| `battery_masked_ablation{,_full}.png` | `battery_masked_ablation_fig.py` | in repo |
| `consensus_robustness.png` | `consfull_fig.py robustness` | in repo |
| `consensus_whitebox_readouts.png` | `regen_whitebox_figs.py` | in repo |
| `isolate_gating.png` | `isolate_fig.py` | in repo |
| `libnum_voice_gap.png` | `libnum_voice_gap_fig.py` | in repo |
| `libnum_voice_ladder.png` | `libnum_voice_ladder_fig.py` | in repo |
| `nummask_lift_voice.png` | `nummask_lift_fig.py` | in repo |
| `orthohon_voice2x2.png` | `orthohon_voice2x2_fig.py` | in repo |
| `patch_dissociation.png` | `patch_dissociation_fig.py` | in repo |
| `steer_input_level.png`, `steer_activation_level.png` | `steer_causal_fig.py` | in repo |
| `masking_blackbox_datasets.png` | `masking_datasets_fig.py` | `BENCHMARK_OPT_DATA` |
| `masking_voice.png` | `masking_voice_fig.py` | `BENCHMARK_OPT_DATA` |
| `pair_spacing.png`, `pair_mister.png`, `pair_spacing_white.png` | `paired_grids.py` | `BENCHMARK_OPT_DATA` |

`paired_grids.py` pairs a switch-rate panel with a white-box NLL panel. The
switch-rate panels rebuild from data in the repository via `ortho_switch_fig.py`,
which recomputes them with `benchmark_optimization.ortho`; only the NLL panels
need the raw root. Those switch rates match `paired_grids.py` exactly on the
paper's data (46 models pooled spacing, 43 cross-corpus honorific, max
difference 0).

## Checking the extraction

```bash
pytest tests/test_paper_equivalence.py -v
```

Replays the original reference-disagreement run's inputs from
`data/consensus/vox_en_newwl4_samples.json` and requires identical output: every
edit, every per-model verdict (1,338 edits × 39 models), every published rate.

## Published predictions

The five raw-data figures re-score per-model output. That output is published at
[HumeAI/ASR-benchmark-optimization](https://huggingface.co/datasets/HumeAI/ASR-benchmark-optimization);
`export_predictions.py` is the script that produced it.

## Full results root

Set `BENCHMARK_OPT_DATA` to a directory laid out as:

```
$BENCHMARK_OPT_DATA/
  datasets/{corpus}/{split}/manifest.parquet        # __key__, language, text
  datasets/{corpus}/{split}/truncation_meta.parquet # __key__, hidden_ref  (masked variants)
  results/{corpus}/{split}/{model}/{run}/*.wsds     # __key__, hyp[, hyp_raw]; DONE marker per run
  results/{corpus}/{model}/{split}/results.jsonl    # legacy layout, also read
```

`*.wsds` are Arrow IPC. `hyp_raw` is preferred per row, falling back to `hyp`.

```bash
cd repro
BENCHMARK_OPT_DATA=/path/to/results make all     # all 24 figures
BENCHMARK_OPT_DATA=/path/to/results make cells   # re-derive data/ortho_switch.json
```

Other variables: `BENCHMARK_OPT_CELLS` (derived data, default `repro/data`),
`BENCHMARK_OPT_FIGURES` (output, default `repro/out`), `BENCHMARK_OPT_FONT_DIR`
(Fellix, used in the paper; matplotlib's default otherwise, so figures differ
slightly from the published PDF).

## Audio-side probes

Moved to [`probes/`](../probes/) — they are methods rather than figure
reproduction. See the README.
