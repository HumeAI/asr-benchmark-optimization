"""Paired panels (2 per PNG) sharing the model y-axis. Four pairs:
  by-variant: [white mister | black mister], [black spacing | white spacing]
  by-box:     [white mister | white spacing], [black spacing | black mister]
black panel = switch% (50% line); white panel = ΔNLL/char two-arm (0 line).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from hume_style import HUME, apply_style, save_figure  # noqa: E402

from mine_ortho_variants import DATA, find_first_variant, load_hyps  # noqa: E402

apply_style()

OUTD = Path(str(P.FIGURES))
PAPER = Path(str(P.FIGURES))  # paper-referenced figs go here
PPL = Path(str(P.CELLS / "ortho_ppl"))
PPLM = Path(str(P.CELLS / "ortho_ppl_mister"))
ONLY = {
    "cohere-transcribe",
    "canary-qwen-2.5b",
    "qwen3-asr-0.6b",
    "granite-speech-4.1-2b",
    "phi4-multimodal",
    "whisper-large-v3",
    "voxtral-mini-3b",
    "moonshine-streaming-medium",
    "kimi-audio-7b",
    "higgs-audio-v3-8b-stt-v2",
    "parakeet-tdt-0.6b-v2",
}
LIBRI = ["librispeech-clean", "librispeech-other"]
SPF = [
    (re.compile(r"(?i)\bany one\b"), re.compile(r"(?i)\banyone\b")),
    (re.compile(r"(?i)\bevery one\b"), re.compile(r"(?i)\beveryone\b")),
    (re.compile(r"(?i)\bsome one\b"), re.compile(r"(?i)\bsomeone\b")),
    (re.compile(r"(?i)\bany thing\b"), re.compile(r"(?i)\banything\b")),
]
MRV = [("mister", re.compile(r"(?i)\bmister\b")), ("Mr", re.compile(r"(?i)\bmr\b\.?"))]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def boot(x, n=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    bs = [np.mean(np.random.choice(x, len(x), True)) for _ in range(n)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------- BLACK: switch% ----------
def black_spacing():
    kb = defaultdict(dict)
    for ds in LIBRI:
        m = pl.read_parquet(DATA / f"datasets/{ds}/test/manifest.parquet").filter(pl.col("language") == "en")
        for k, t in zip(m["__key__"].to_list(), m["text"].to_list()):
            for sp, so in SPF:
                if sp.search(t or ""):
                    kb[ds][k] = (sp, so, "spaced")
                    break
                if so.search(t or ""):
                    kb[ds][k] = (sp, so, "solid")
                    break
    st = defaultdict(lambda: {"spaced": [0, 0], "solid": [0, 0]})
    for ds in LIBRI:
        h = load_hyps(ds, set(kb[ds]))
        for k, (sp, so, arm) in kb[ds].items():
            want = sp if arm == "spaced" else so
            for mdl, hy in h.get(k, {}).items():
                st[mdl][arm][1] += 1
                if want.search(hy):
                    st[mdl][arm][0] += 1
    out = {}
    for mdl, s in st.items():
        if s["spaced"][1] < 5 or s["solid"][1] < 5:
            continue
        rs, ro = wilson(*s["spaced"]), wilson(*s["solid"])
        out[mdl] = wilson(*s["spaced"]) if rs[0] <= ro[0] else wilson(*s["solid"])
    return out  # mdl -> (switch, lo, hi)


def black_mister():
    def scan(dsets, rx):
        kk = defaultdict(set)
        for ds in dsets:
            m = pl.read_parquet(DATA / f"datasets/{ds}/test/manifest.parquet").filter(pl.col("language") == "en")
            for k, t in zip(m["__key__"].to_list(), m["text"].to_list()):
                if find_first_variant(t or "", MRV) == rx:
                    kk[ds].add(k)
        return kk

    voxk, libk = scan(["voxpopuli"], "Mr"), scan(LIBRI, "mister")
    st = defaultdict(lambda: {"vox": [0, 0], "libri": [0, 0]})
    for ds, ks in voxk.items():
        for k, hh in load_hyps(ds, ks).items():
            for mdl, hy in hh.items():
                st[mdl]["vox"][1] += 1
                if find_first_variant(hy, MRV) == "Mr":
                    st[mdl]["vox"][0] += 1
    for ds, ks in libk.items():
        for k, hh in load_hyps(ds, ks).items():
            for mdl, hy in hh.items():
                st[mdl]["libri"][1] += 1
                if find_first_variant(hy, MRV) == "mister":
                    st[mdl]["libri"][0] += 1
    out = {}
    for mdl, s in st.items():
        if s["vox"][1] < 3 or s["libri"][1] < 3:
            continue
        rv, rl = wilson(*s["vox"]), wilson(*s["libri"])
        out[mdl] = wilson(*s["vox"]) if rv[0] <= rl[0] else wilson(*s["libri"])
    return out


# ---------- WHITE: per-char audio-lift beta, two-arm ----------
# beta = char-normalized delta_iso = (silenced-arm per-char ΔNLL) removed from the
# real-audio per-char ΔNLL, so we plot the AUDIO's contribution toward the reference
# spelling (matching the reference-disagreement / masked probes' audio lift λ), not the
# raw given-audio ΔNLL. Each spelling's NLL is divided by that spelling's char length on
# BOTH the real-audio and the silenced-audio arm. Sign convention is unchanged from the
# raw-ΔNLL version (vox/spaced use +d, libri/solid use −d), so reference-tracking still
# reads as both arms above zero.
def white_spacing():
    CH = {}
    for ds in LIBRI:
        m = pl.read_parquet(DATA / f"datasets/{ds}/test/manifest.parquet").filter(pl.col("language") == "en")
        for k, t in zip(m["__key__"].to_list(), m["text"].to_list()):
            lc = (t or "").lower()
            for sp, so, sps, sos in [
                (re.compile(r"\bany one\b"), re.compile(r"\banyone\b"), "any one", "anyone"),
                (re.compile(r"\bevery one\b"), re.compile(r"\beveryone\b"), "every one", "everyone"),
                (re.compile(r"\bsome one\b"), re.compile(r"\bsomeone\b"), "some one", "someone"),
                (re.compile(r"\bany thing\b"), re.compile(r"\banything\b"), "any thing", "anything"),
            ]:
                if sp.search(lc) or so.search(lc):
                    CH[k] = (len(so.sub(sps, lc)), len(sp.sub(sos, lc)))
                    break
    out = {}
    for p in PPL.glob("ortho_ppl_*.parquet"):
        mdl = p.stem.replace("ortho_ppl_", "")
        if mdl not in ONLY:
            continue
        df = pl.read_parquet(p)
        need = ["nll_spaced", "nll_solid", "nll_spaced_none", "nll_solid_none"]
        if any(c not in df.columns for c in need):
            continue  # parquet lacks the silenced-audio prior arms — rerun with --with-prior
        df = df.drop_nulls(need)
        sp, so = [], []
        for r in df.iter_rows(named=True):
            ch = CH.get(r["key"])
            if not ch:
                continue
            # per-char ΔNLL (solid − spaced) on each arm; beta = real − silenced = audio lift.
            d_real = r["nll_solid"] / ch[1] - r["nll_spaced"] / ch[0]
            d_none = r["nll_solid_none"] / ch[1] - r["nll_spaced_none"] / ch[0]
            b = d_real - d_none
            (sp if r["variant"] == "spaced" else so).append(b if r["variant"] == "spaced" else -b)
        if len(sp) >= 5 and len(so) >= 5:
            out[mdl] = (boot(sp), boot(so))  # (spaced_arm, solid_arm)
    return out


def white_mister():
    MR = re.compile(r"(?i)\bmr\b\.?")
    MIS = re.compile(r"(?i)\bmister\b")
    CH = {}
    for ds in ["voxpopuli"] + LIBRI:
        m = pl.read_parquet(DATA / f"datasets/{ds}/test/manifest.parquet").filter(pl.col("language") == "en")
        for k, t in zip(m["__key__"].to_list(), m["text"].to_list()):
            lc = (t or "").lower()
            if MR.search(lc) or MIS.search(lc):
                CH[k] = (len(MIS.sub("mr", lc)), len(MR.sub("mister", lc)))
    out = {}
    for p in PPLM.glob("mister_ppl_*.parquet"):
        mdl = p.stem.replace("mister_ppl_", "")
        if mdl not in ONLY:
            continue
        df = pl.read_parquet(p)
        need = ["nll_mr", "nll_mister", "nll_mr_none", "nll_mister_none"]
        if any(c not in df.columns for c in need):
            continue  # parquet lacks the silenced-audio prior arms — rerun with --with-prior
        df = df.drop_nulls(need)
        vox, lib = [], []
        for r in df.iter_rows(named=True):
            ch = CH.get(r["key"])
            if not ch:
                continue
            # per-char ΔNLL (mister − mr) on each arm; beta = real − silenced = audio lift.
            d_real = r["nll_mister"] / ch[1] - r["nll_mr"] / ch[0]
            d_none = r["nll_mister_none"] / ch[1] - r["nll_mr_none"] / ch[0]
            b = d_real - d_none
            (vox if r["corpus"] == "vox" else lib).append(b if r["corpus"] == "vox" else -b)
        if len(vox) >= 5 and len(lib) >= 5:
            out[mdl] = (boot(vox), boot(lib))
    return out


# Shared role colours across the paired panels: coral = focal (significant switch / vox / spaced arm),
# blue = the contrast arm (significant keep / libri / solid), grey = not-significant.
ARM0, ARM1 = HUME["primary"], HUME["sv"]
_EKW = dict(ecolor=HUME["err"], lw=1, capsize=2.5)


def draw_black(ax, data, order, xlabel, title):
    ax.axvline(50, color=HUME["ink"], lw=1.0, ls="--", zorder=1)
    for i, m in enumerate(order):
        if m not in data:
            continue
        sw, lo, hi = (v * 100 for v in data[m])
        col = ARM0 if lo > 50 else (HUME["grid"] if hi >= 50 else ARM1)
        ax.barh(i, sw, xerr=[[sw - lo], [hi - sw]], color=col, zorder=2, error_kw=_EKW)
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def draw_white(ax, data, order, xlabel, title, labs):
    h = 0.38
    for idx, off, color, lab in [(0, h / 2, ARM0, labs[0]), (1, -h / 2, ARM1, labs[1])]:
        first = True
        for i, m in enumerate(order):
            if m not in data:
                continue
            me, lo, hi = data[m][idx]
            ax.barh(
                i + off,
                me,
                h,
                xerr=[[me - lo], [hi - me]],
                color=color,
                label=(lab if first else None),
                zorder=2,
                error_kw=_EKW,
            )
            first = False
    ax.axvline(0, color=HUME["ink"], lw=1.0, ls="--", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)


def score_black(d, m):
    return d[m][0]


def score_white(d, m):
    return min(d[m][0][0], d[m][1][0])


def pair(left, right, out):
    # left/right = ("black"/"white", data, xlabel, title, [arm labels])
    lkind, ldata = left[0], left[1]
    rkind, rdata = right[0], right[1]
    union = (set(ldata) | set(rdata)) & ONLY  # keep ALL canonical models (black has 11)
    # order by a black panel if present (it's the complete one), else the left panel
    if lkind == "black":
        sd, sk = ldata, "black"
    elif rkind == "black":
        sd, sk = rdata, "black"
    else:
        sd, sk = ldata, lkind

    def key(m):
        if m not in sd:
            return -1.0
        return score_black(sd, m) if sk == "black" else score_white(sd, m)

    order = sorted(union, key=key)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 1.4), sharey=True)
    (draw_black if lkind == "black" else draw_white)(a1, ldata, order, *left[2:])
    (draw_black if rkind == "black" else draw_white)(a2, rdata, order, *right[2:])
    a1.set_yticks(np.arange(len(order)))
    a1.set_yticklabels([PRETTY.get(m, m) for m in order])
    for ax in (a1, a2):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    plt.tight_layout()
    paths = save_figure(fig, out)
    print("wrote", *[p.name for p in paths])


# canonical row order + display names shared with masking_datasets_fig.py / battery_panels.py,
# so the two panels of the merged behavioral figure share a y-axis (Cohere at top).
CANON = [
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
PRETTY = {
    "cohere-transcribe": "Cohere-Transcribe",
    "canary-qwen-2.5b": "Canary-Qwen-2.5B",
    "granite-speech-4.1-2b": "Granite-Speech-4.1-2B",
    "phi4-multimodal": "Phi-4-Multimodal",
    "parakeet-tdt-0.6b-v2": "Parakeet-TDT-0.6B-v2",
    "omni-3b-llm": "Omni-3B-LLM",
    "moonshine-streaming-medium": "Moonshine-Streaming",
    "whisper-large-v3": "Whisper-Large-v3",
    "kimi-audio-7b": "Kimi-Audio-7B",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "voxtral-mini-3b": "Voxtral-Mini-3B",
    "higgs-audio-v3-8b-stt-v2": "Higgs-Audio-v3-8B",
}


import os as _os  # noqa: E402

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P  # noqa: E402

_sw = _os.environ.get("MODEL_SWAP")
if _sw:
    _a, _b = _sw.split(":")
    CANON = [(_b if _m == _a else _m) for _m in CANON]
    ONLY = {(_b if _m == _a else _m) for _m in ONLY}


def single(panel, order_ref, out):
    # panel = ("black"/"white", data, xlabel, title, [arm labels])
    kind, data = panel[0], panel[1]
    union = (set(data) | set(order_ref[1])) & ONLY  # keep ALL canonical models (black has 11)
    order = [m for m in reversed(CANON) if m in union]
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 0.55 * len(order) + 0.9))
    (draw_black if kind == "black" else draw_white)(ax, data, order, *panel[2:])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([PRETTY.get(m, m) for m in order])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    paths = save_figure(fig, out)
    print("wrote", *[p.name for p in paths])


if __name__ == "__main__":
    BS, BM = black_spacing(), black_mister()
    WS, WM = white_spacing(), white_mister()
    # honorific: black-box switch-rate only (white-box audio-lift panel dropped per Theo)
    single(("black", BM, "switch (%)", ""), ("black", BM), PAPER / "pair_mister.png")
    # spacing: black-box only in the main figure; white-box as a standalone appendix PNG.
    # Order both by the black panel so the appendix rows line up with the main figure.
    single(("black", BS, "switch (%)", "black box"), ("black", BS), PAPER / "pair_spacing.png")
    single(
        ("white", WS, r"audio lift $\beta$ (nats/char)", "white box", ["ref=spaced", "ref=solid"]),
        ("black", BS),
        PAPER / "pair_spacing_white.png",
    )
    # by box
    pair(
        ("white", WM, r"audio lift $\beta$ (nats/char)", "mister/Mr", ["vox→Mr", "libri→mister"]),
        ("white", WS, r"audio lift $\beta$ (nats/char)", "archaic spacing", ["ref=spaced", "ref=solid"]),
        OUTD / "pair_white.png",
    )
    pair(
        ("black", BS, "switch (%)", "archaic spacing"),
        ("black", BM, "switch (%)", "mister/Mr"),
        OUTD / "pair_black.png",
    )
