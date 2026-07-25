"""Regenerate the paper white-box figures.

  consensus_whitebox_readouts.png : edit-localized reference NLL under audio, and reference
                                    audio lift after subtracting the silenced-audio prior.
  masking_blackbox.png         : black-box masked accept-ref (masked-word reproduction rate).
  masking_nll.png              : Lambda = (nll_none - nll_trunc)/chars on hard cells (nll_none>=3.5)

Black-box masking uses the full paper model panel; white-box panels include models with available
teacher-forced likelihood dumps. Writes directly into paper/figures/.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from _consensus_ppl_probe import corrected  # noqa: E402
from hume_style import HUME, apply_style, save_figure, style_h_bar_ax  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

apply_style()

WB = Path(str(P.CELLS / "vmt" / "wb"))
OUT = Path(str(P.FIGURES))
CONS = json.load(open(str(P.CELLS / "consensus" / "vox_en_3wl_test_samples.json")))
RNG = np.random.default_rng(0)
BLUE, ORANGE, GRAY = "#4c78a8", "#f58518", "#888888"
PAPER_MODELS = [
    "cohere-transcribe",
    "canary-qwen-2.5b",
    "granite-speech-4.1-2b",
    "phi4-multimodal",
    "parakeet-tdt-0.6b-v2",
    "higgs-audio-v3-8b-stt-v2",
    "moonshine-streaming-medium",
    "whisper-large-v3",
    "kimi-audio-7b",
    "qwen3-asr-0.6b",
    "voxtral-mini-3b",
]


def short(m):
    return (
        m.replace("-transcribe", "")
        .replace("-multimodal", "")
        .replace("granite-speech-4.1-2b", "granite")
        .replace("-mini-3b", "")
        .replace("-large-v3", "-lv3")
        .replace("qwen3-asr-0.6b", "qwen3")
        .replace("canary-qwen-2.5b", "canary-qwen")
        .replace("kimi-audio-7b", "kimi")
        .replace("moonshine-streaming-medium", "moonshine")
        .replace("higgs-audio-v3-8b-stt-v2", "higgs")
        .replace("parakeet-tdt-0.6b-v2", "parakeet")
        .replace("phi4", "phi-4")
    )


def boot(x, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, x.size, size=(n, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(k, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------- consensus edit-span char counts
EDIT_CHARS = {}
for s in CONS:
    ref = [t.lower() for t in s["ref_tokens"]]
    cons = corrected(ref, s["runs"])
    p = 0
    while p < min(len(ref), len(cons)) and ref[p] == cons[p]:
        p += 1
    q = 0
    while q < min(len(ref), len(cons)) - p and ref[-1 - q] == cons[-1 - q]:
        q += 1
    rc = len(" ".join(ref[p : len(ref) - q]))
    ac = len(" ".join(cons[p : len(cons) - q]))
    EDIT_CHARS[s["key"]] = (max(rc, 0), max(ac, 0))


def sumchar(values, ch):
    return (sum(values) / ch) if ch > 0 else float("nan")


# ================================================================ FIG 1: consensus white-box readouts
# restrict to clips carrying a newwl4-panel-unanimous edit (the reported edit set)
_nw = json.loads(P.cell("consensus", "vox_en_newwl4_samples.json").read_text())
NW_KEYS = {r["key"] for r in _nw if any(x["n_wl_agree"] == x["n_wl_total"] for x in r["runs"])}
rows = []
for f in sorted(WB.glob("wbAprior_consensus_*.parquet")):
    m = f.name[len("wbAprior_consensus_") : -len(".parquet")]
    if m not in PAPER_MODELS:
        continue
    d = pl.read_parquet(f).filter(pl.col("key").is_in(list(NW_KEYS)))
    if "r_lift" not in d.columns or "r_surp" not in d.columns:
        continue
    nll_ref = []
    lift_ref = []
    for r in d.iter_rows(named=True):
        rc, _ = EDIT_CHARS.get(r["key"], (0, 0))
        if rc <= 0:
            continue
        lift = list(r["r_lift"])
        prior_nll = list(r["r_surp"])
        audio_nll = [surprise - lift_value for surprise, lift_value in zip(prior_nll, lift)]
        nll_ref.append(sumchar(audio_nll, rc))
        lift_ref.append(sumchar(lift, rc))
    n_mu, n_lo, n_hi = boot(nll_ref)
    l_mu, l_lo, l_hi = boot(lift_ref)
    rows.append((m, n_mu, n_lo, n_hi, l_mu, l_lo, l_hi, len(lift_ref)))

# best-first: largest positive audio lift at the top (invert_y puts index 0 at top)
rows.sort(key=lambda r: r[4], reverse=True)
fig, ax = plt.subplots(figsize=(4.6, 4.3))
y = np.arange(len(rows))

for i, (m, _n_mu, _n_lo, _n_hi, l_mu, l_lo, l_hi, _n) in enumerate(rows):
    # coral where the audio lift is significantly positive; grey otherwise (no reliable lift)
    col = HUME["primary"] if l_lo > 0 else HUME["grid"]
    ax.barh(i, l_mu, color=col, edgecolor=HUME["ink"], lw=0.4, zorder=2)
    ax.plot([l_lo, l_hi], [i, i], color=HUME["err"], lw=1.1, zorder=3)
ax.axvline(0, color=HUME["ink"], lw=1.0, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([short(m) for m, *_ in rows])
style_h_bar_ax(ax, xlabel=r"$\lambda$: reference-error audio lift", invert_y=True)

fig.tight_layout()
save_figure(fig, str(OUT / "consensus_whitebox_readouts"))
plt.close(fig)
print("consensus readouts:", [(short(m), round(lift_mu, 3)) for m, _n, _nlo, _nhi, lift_mu, *_ in rows])


# ================================================================ FIG 2a: masking black-box retrieval rate
def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def hypothesis(row):
    return row.get("hypothesis_raw") or row.get("hypothesis") or row.get("hyp") or ""


# This panel re-scores raw per-model output, which is not shipped. Skip it when
# there is no results root so the cell-driven panels below still build.
variant = "voxpopuli-mask-num-all-numexp-silence"  # masked-NUMBER (names parked: NER too noisy)
bb_rows = []
meta = {}
if P.DATA is not None:
    meta = {
        r["__key__"]: (r["hidden_ref"] or "")
        for r in pl.read_parquet(P.data("datasets", variant, "test", "truncation_meta.parquet")).iter_rows(named=True)
    }
else:
    print("  skipping masked black-box panel: BENCHMARK_OPT_DATA not set")
for m in PAPER_MODELS if meta else []:
    p = P.data("results", variant, m, "test", "results.jsonl")
    if not p.exists():
        print(f"  missing black-box masking run: {m}")
        continue
    k = n = 0
    for r in read_jsonl(p):
        key = r.get("__key__")
        target = (meta.get(key) or "").lower().strip(".,;:!?\"'")
        if not target:
            continue
        n += 1
        if re.search(r"(^|\W)" + re.escape(target) + r"($|\W)", hypothesis(r).lower()):
            k += 1
    if n:
        lo, hi = wilson(k, n)
        bb_rows.append((m, k / n, lo, hi, n))

# Only plot this panel when the raw-output rescoring above actually ran.
if bb_rows:
    bb_rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (m, rate, lo, hi, n) in enumerate(bb_rows):
        ax.barh(i, rate, color=BLUE, edgecolor="k", lw=0.4)
        ax.plot([lo, hi], [i, i], color="k", lw=1.1)
    ax.set_xlim(0, max(0.36, max((r[3] for r in bb_rows), default=0.3) + 0.03))
    ax.set_yticks(range(len(bb_rows)))
    ax.set_yticklabels([short(m) for m, *_ in bb_rows])
    ax.set_xlabel("Masked accept-ref")
    ax.set_title("Masked accept-ref (black-box)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "masking_blackbox.png", dpi=140)
    plt.close(fig)
    print("masking blackbox:", [(short(m), round(v, 3), n) for m, v, _lo, _hi, n in bb_rows])

# ================================================================ FIG 2b: masking white-box lift Lambda
HARD = 3.5
mrows = []
for f in sorted(WB.glob("wb_entmask_realNUM_*.parquet")):
    m = f.name[len("wb_entmask_realNUM_") : -len(".parquet")]
    d = pl.read_parquet(f).filter(pl.col("nll_none") >= HARD)
    vals = []
    for r in d.iter_rows(named=True):
        ch = len((r["hidden_words"] or "").strip())
        if ch > 0:
            vals.append((r["nll_none"] - r["nll_trunc"]) / ch)
    if len(vals) < 10:  # drop degenerate cells (e.g. omni: 1 hard cell)
        print(f"  skip {m}: only {len(vals)} hard cells")
        continue
    mu, lo, hi = boot(vals)
    mrows.append((m, mu, lo, hi, len(vals)))

mrows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, (m, mu, lo, hi, n) in enumerate(mrows):
    ax.barh(i, mu, color=BLUE, edgecolor="k", lw=0.4)
    ax.plot([lo, hi], [i, i], color="k", lw=1.1)
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(range(len(mrows)))
ax.set_yticklabels([short(m) for m, _mu, _lo, _hi, n in mrows])
ax.set_xlabel(r"$\Lambda$: Masked-number audio lift")
ax.set_title("Masked-number audio lift", fontsize=10)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "masking_nll.png", dpi=140)
plt.close(fig)
print("masking:", [(short(m), round(v, 3)) for m, v, *_ in mrows])
print("wrote", OUT / "consensus_whitebox_readouts.png", OUT / "masking_blackbox.png", "and", OUT / "masking_nll.png")
