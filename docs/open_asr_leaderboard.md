# Adding a benchmark-optimization column to the Open ASR Leaderboard

A concrete proposal, and the reasoning behind each choice.

## Why it fits

The [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)
already produces everything needed. Its evaluation scripts write a per-clip
JSONL manifest of raw references and predictions, and normalize only afterwards
when computing WER. Both probes here consume exactly that file. No audio, no
model weights, no re-running inference — a benchmark-optimization score can be
computed from artifacts the leaderboard already has.

The scores also answer a question WER cannot. WER measures accuracy over all
clips; these measure behaviour over the subset where the audio does not
determine the reference. The two come apart: in our paper, on VoxPopuli-English,
the six lowest-WER models were the six highest on reference disagreement. A
leaderboard column would make that visible where it matters.

## What to add

Two columns, both rates in [0, 1], both reported with their denominator.

### 1. Reference disagreement (`accept-ref`)

On VoxPopuli-English. Where a panel of models agrees against the reference
transcript, the share of those disagreements where the model under test
reproduces the reference instead.

```bash
benchmark-optimization ref-disagreement --preds predictions/ --panel <four models>
```

VoxPopuli is the right corpus because its references derive from official
parliamentary records rather than from the audio, so genuine reference errors are
common enough to measure — around 4% of words and just under half of clips carry
at least one, at the settings in the paper. LibriSpeech references are clean, and
the probe correctly finds almost nothing there.

**The panel is the one real design decision.** It stands in for a corrected
reference, and a panel stacked with benchmark-optimized models will not flag the
edits that matter. Two workable routes:

- *Fixed published panel.* Name four models in the leaderboard config, chosen for
  showing no corpus-specific behaviour on the other probes, and version the
  choice. Simple and reproducible; needs periodic review as models change.
- *Leave-one-out.* Score each model against the consensus of the others, so
  there is no privileged panel. More defensible, more compute, and it degrades if
  many entrants share the same behaviour.

A third route removes the panel entirely: score against a **corrected
VoxPopuli** reference where one exists, and treat disagreement with the
correction as the signal. That is cleaner and we would prefer it; the panel
exists because we did not have a correction covering the test set. If the
leaderboard adopts a cleaned VoxPopuli release, the same code runs with the
cleaned transcript in place of the consensus — the metric is unchanged, only its
ground truth improves.

### 2. Orthographic switch rate

The share of clips where a model's spelling of an acoustically-invisible
distinction follows the reference's, minimized over arms. Chance is 0.5.

```bash
benchmark-optimization switch-rate --preds predictions/ --spacing
```

Two variants, and the choice matters:

- **Within-LibriSpeech spacing** (recommended). `anyone`/`any one` and friends
  both occur inside LibriSpeech, so speaker, register, and channel are matched by
  construction. This is the cleaner measurement and needs only one corpus already
  on the leaderboard. Pool the four families; individually each is too rare.
- **Cross-corpus honorifics.** `Mr.` in VoxPopuli versus `mister` in LibriSpeech.
  Higher-signal, but register varies with the arm, so a model could be responding
  to register rather than to text convention. If only one column is added, use
  the within-corpus version and avoid the confound.

## Implementation sketch

The integration is small. `benchmark_optimization` is pure Python with `transformers` as its
only dependency, and reads the leaderboard's manifests natively:

```python
from benchmark_optimization import load_manifests, conventions, ortho, refdis

preds = load_manifests({m: f"results/{m}/librispeech.jsonl" for m in MODELS})
spacing = ortho.pooled_switch_rate(
    list(conventions.SPACING_PAIRS), list(preds.clips()),
    arm_names=conventions.SPACING_ARMS,
)
```

For the leaderboard's own results the natural home is a post-processing step
after WER aggregation, reading the manifests already synced from each
submitter's bucket. It is CPU-only and takes seconds per model.

## Reporting them honestly

These numbers are easy to over-read, and a leaderboard column amplifies that. We
would ask for:

1. **Not summed into a rank.** A benchmark-optimization score is not a quality
   score, and averaging it with WER produces a number meaning nothing. Show it as
   its own column.
2. **Denominators next to rates.** Models are only charged for clips they were
   eligible for, so denominators differ per model. A rate over 40 disagreements
   is not a rate over 1,300.
3. **Chance marked, for switch rate.** 0.5, not 0. Without it a 0.45 reads as
   low when it is chance.
4. **Wording that does not claim contamination.** Neither probe tests for
   training-set contamination and neither can establish it. What they show is a
   model reproducing benchmark conventions the audio underdetermines. "Reference
   following" or "benchmark-optimized behaviour" is accurate; "trained on the test
   set" is not, and would invite disputes the measurement cannot settle.
5. **Config pinned in the results.** Panel membership, majority threshold, and
   family list all move the numbers. Record them alongside, as the CLI's
   `--out` JSON does.

## Open questions for the leaderboard maintainers

- Panel choice: fixed-and-versioned, leave-one-out, or wait for a cleaned
  VoxPopuli reference?
- One column or two? The switch rate is the cleaner measurement; reference
  disagreement is the more striking result.
- Multilingual: both probes carry non-English convention families and the
  reference-disagreement probe is language-agnostic given a panel. Worth
  extending to the multilingual leaderboard, or English-only first?
- Longer term, both probes are gameable once published — a model can be trained
  to abstain exactly where the probe looks. That is a reason to expect the
  specific families to need rotation, not a reason to withhold the measurement,
  but it is worth deciding up front who maintains the family list.
