"""Build entity-masked datasets: corrupt a distinctive mid-utterance word.

Picks one distinctive capitalized proper-noun-like word per sample (skipping
sentence-initial and a stoplist of predictable institutional/temporal terms),
then corrupts the audio over that word's aligned [t0,t1] span:

  - silence   : zero the span (entity audio fully removed; context intact)
  - attenuate : drop the span volume (e.g. -20 dB)
  - noise     : attenuate + add white noise over the span (static over the word)

The entity is then UNHEARD (or degraded) but the rest of the utterance is intact.
Teacher-forcing the full reference and scoring the entity tokens under
{clean (full source audio), corrupted (this variant), zero-audio} answers: can
the model still produce a distinctive name it could not acoustically hear? For a
name (not register-predictable) that's a clean memorization / fingerprint signal.

Output mirrors the truncation builder: manifest (full text) + corrupted wavs +
truncation_meta with hidden_ref=entity, hidden_side="span".
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

import roots as _paths  # data/model roots from BENCHMARK_OPT_DATA / BENCHMARK_OPT_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("entity_mask")

# Capitalized words that are predictable in parliamentary register (not distinctive).
_STOP = {
    "the", "commission", "parliament", "council", "member", "members", "state", "states",
    "european", "europe", "union", "eu", "mr", "mrs", "ms", "madam", "president", "commissioner",
    "minister", "directive", "regulation", "treaty", "article", "house", "chamber", "report",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "i", "we", "it", "this", "that", "there", "but", "and", "if", "as", "how", "in", "on", "at",
    "for", "however", "finally", "first", "second", "today", "now", "west", "east", "north", "south",
}
_CAP = re.compile(r"^[A-Z][a-z']{2,}$")
_HON = {"president", "commissioner", "mr", "mrs", "ms", "madam", "baroness", "lord", "sir",
        "dame", "chancellor", "minister", "prime", "vice", "dr", "professor", "chairman"}

# Spanish (es) variants. Selected via --lang es; _CAP allows accented capitals
# (Señor, García, España) and the stop/honorific/number sets are localized so the
# --kind proper / number / proper+title paths behave like the EN ones on ES text.
_STOP_ES = {
    "el", "la", "los", "las", "comision", "comisión", "parlamento", "consejo", "miembro",
    "miembros", "estado", "estados", "europeo", "europea", "europeos", "europeas", "europa",
    "union", "unión", "ue", "señor", "señora", "señores", "señoras", "sr", "sra", "señorias",
    "señorías", "presidente", "presidenta", "comisario", "comisaria", "ministro", "ministra",
    "directiva", "reglamento", "tratado", "articulo", "artículo", "informe", "camara", "cámara",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
    "octubre", "noviembre", "diciembre", "lunes", "martes", "miercoles", "miércoles", "jueves",
    "viernes", "yo", "nosotros", "esto", "eso", "este", "esta", "pero", "y", "si", "como", "cómo",
    "en", "para", "por", "hoy", "ahora", "primero", "segundo", "oeste", "este", "norte", "sur",
}
# unicode-aware: capital (incl. accented) + lowercase body (incl. accented + ñ + ü)
_CAP_ES = re.compile(r"^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü']{2,}$")
_HON_ES = {"presidente", "presidenta", "comisario", "comisaria", "sr", "sra", "señor", "señora",
           "don", "doña", "ministro", "ministra", "primer", "vicepresidente", "dr", "dra",
           "profesor", "profesora", "baron", "barón", "baronesa", "lord", "sir"}
# Spanish spelled-out number words (build's --kind number also keeps any digit token).
_NUMWORDS_ES = {
    "cero", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciseis", "dieciséis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
    "ochenta", "noventa", "cien", "ciento", "cientos", "doscientos", "trescientos", "quinientos",
    "mil", "millon", "millón", "millones", "billon", "billón", "billones",
}


def _core(w: str) -> str:
    return w.strip(".,;:!?\"'")


_NUM = re.compile(r"\d")
_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
}


def _is_value(tok: str) -> bool:
    c = _core(tok).lower()
    return bool(_NUM.search(c)) or c in _NUMWORDS


def distinctive_numbers(full_words: list[str], mode: str = "distinctive") -> list[int]:
    """Indices of numeric value tokens worth masking, by ``mode``:

    distinctive : digit tokens + spelled number words EXCEPT bare ``one``/``1``
                  (default; excludes the only predictable "one moment"/"one of" case).
    digit       : HEAVY — only tokens containing an actual digit ("2011", "95",
                  "80,000"); no spelled-out words. Strictest / most distinctive.
    all         : OPEN SEASON — every number token, digits + spelled + ``one``/``1``
                  (catches benchmark transcription errors like "more than 1 amendments").
    """
    out = []
    for i in range(len(full_words)):
        c = _core(full_words[i]).lower()
        has_digit = bool(_NUM.search(c))
        if mode == "digit":
            if has_digit:
                out.append(i)
        elif mode == "all":
            if has_digit or c in _NUMWORDS:
                out.append(i)
        else:  # distinctive
            if c in {"one", "1"}:
                continue
            if has_digit or c in _NUMWORDS:
                out.append(i)
    return out


def collect_entities(full_words: list[str], words: list[dict], kind: str = "proper", number_mode: str = "distinctive") -> tuple[list[int], int | None]:
    """Return (entity-span word indices, target index to score) for ``kind``.

    kind='proper'    : distinctive capitalized proper nouns (all occurrences).
    kind='proper+title': proper nouns + honorific/title words.
    kind='number'    : numeric tokens that have a real (non-zero-width) audio span.

    Target to score: for proper kinds, a person name (proper noun after a title)
    else the longest proper noun; for numbers, the token with the longest span.
    Sentence-initial caps are ignored (uninformative)."""
    n = min(len(full_words), len(words))
    propers: list[int] = []
    titles: list[int] = []
    nums: list[int] = []
    for i in range(n):
        core = _core(full_words[i])
        low = core.lower()
        is_initial = i == 0 or (i > 0 and full_words[i - 1] and full_words[i - 1][-1] in ".!?")
        span = words[i]["t1"] - words[i]["t0"]
        if low in _HON:
            titles.append(i)
        elif _CAP.match(core) and low not in _STOP and not is_initial and span >= 0.12:
            propers.append(i)

    if kind == "number":
        nums = distinctive_numbers(full_words, mode=number_mode)
        if not nums:
            return [], None
        # target = first token of the longest consecutive distinctive-number run
        runs, cur = [], [nums[0]]
        for j in nums[1:]:
            if j == cur[-1] + 1:
                cur.append(j)
            else:
                runs.append(cur)
                cur = [j]
        runs.append(cur)
        return nums, max(runs, key=len)[0]

    spans = sorted(set(propers + titles)) if kind == "proper+title" else sorted(propers)
    if not propers:
        return spans, None
    target = next((i for i in propers if (i - 1) in titles), None)
    if target is None:
        target = max(propers, key=lambda i: len(_core(full_words[i])))
    return spans, target


def positional_target(
    full_words: list[str], words: list[dict], mode: str, key: str, seed: int = 0
) -> tuple[list[int], int | None]:
    """Pick ONE word index by POSITION (not entity type) for first/last/random masking.

    mode='first'  : the literal opening word (index 0). Weakest left-context, so the
                    cleanest place to separate acoustic-leak from a memorized opener.
    mode='last'   : the final word (index n-1). Strong right boundary; pairs with the
                    EOS-ban continuation signal (probe runs gen only on side='tail').
    mode='random' : a UNIFORMLY random word over all positions (no length/span
                    bias). The pick is deterministic given (seed, key) — md5(seed:key)
                    mod n — so it is reproducible and documentable, but unbiased over
                    word positions. Unmaskable picks (zero-width token with no gap) are
                    dropped downstream by the main-loop maskability gate, which keeps
                    the *selection* unbiased rather than pre-filtering to long words.

    Returns ([target], target) to match collect_entities' (all_idx, target) shape;
    there's only one word, so --mask-all is a no-op for positional kinds.
    """
    n = min(len(full_words), len(words))
    if n == 0:
        return [], None
    if mode == "first":
        t = 0
    elif mode == "last":
        t = n - 1
    else:  # random — uniform over ALL word positions
        h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16)
        t = h % n
    return [t], t


def corrupt_span(a: np.ndarray, sr: int, t0: float, t1: float, method: str, att_db: float, noise_amp: float):
    """Corrupt the exact [t0,t1] span of audio array ``a`` in place.

    Callers pass already-widened bounds (clamped to neighbour gaps), so no
    internal pad here."""
    s = max(0, int(round(t0 * sr)))
    e = min(len(a), int(round(t1 * sr)))
    if e <= s:
        return
    seg = a[s:e]
    if method == "silence":
        a[s:e] = 0.0
    elif method == "attenuate":
        a[s:e] = seg * (10 ** (-att_db / 20))
    else:  # noise: attenuate signal + add white noise scaled to the original span RMS
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        noise = np.random.default_rng(s).normal(0.0, rms * noise_amp, size=seg.shape).astype(np.float32)
        a[s:e] = seg * (10 ** (-att_db / 20)) + noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--methods", default="silence,noise")
    ap.add_argument("--att-db", type=float, default=20.0, help="attenuation in dB for attenuate/noise")
    ap.add_argument("--noise-amp", type=float, default=0.8, help="noise stddev as fraction of span RMS")
    ap.add_argument("--mask-pad", type=float, default=0.12,
                    help="seconds to widen the mask each side (catches onset the aligner clips); clamped to neighbour gaps.")
    ap.add_argument("--neighbor-guard", type=float, default=0.03,
                    help="stop the widened mask this many seconds short of an adjacent word.")
    ap.add_argument("--mask-all", action="store_true",
                    help="mask EVERY matching span (all references), not just the target word")
    ap.add_argument("--kind", choices=["proper", "proper+title", "number", "first", "last", "random"],
                    default="proper",
                    help="proper: names only (no titles); proper+title: + honorifics; number: numeric tokens; "
                         "first/last: the opening/closing word (position-based); random: a random word per clip")
    ap.add_argument("--number-mode", choices=["distinctive", "digit", "all"], default="distinctive",
                    help="for --kind number: distinctive (digits+spelled, no one/1); digit (HEAVY, digit tokens only); "
                         "all (OPEN SEASON, every number incl one/1)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --kind random word selection (per-key deterministic)")
    ap.add_argument("--entities-from", default=None,
                    help="Parquet from detect_entities_gliner.py {__key__, entity_indices, target_idx}; overrides --kind detection.")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--name-suffix", default=None, help="append to dataset name (e.g. 'loose') for A/B variants.")
    ap.add_argument("--lang", default="en",
                    help="reference language for the --kind proper/number/proper+title heuristics "
                         "(en|es). es swaps _CAP/_STOP/_HON/_NUMWORDS to Spanish; the gner "
                         "(--entities-from) path is language-agnostic and unaffected.")
    ap.add_argument("--src-lang", default=None,
                    help="filter the source manifest to this language code (e.g. es) before building. "
                         "Multilingual sources (voxpopuli) need this so only the target-language slice is masked.")
    ap.add_argument("--name-infix", default=None,
                    help="insert after the source name in the output dataset name "
                         "(e.g. 'es' -> voxpopuli-es-mask-num-all-silence). Mirrors the "
                         "voxpopuli-es-trainsample convention for the ES probe arm.")
    args = ap.parse_args()

    if args.lang == "es":
        global _CAP, _STOP, _HON, _NUMWORDS
        _CAP, _STOP, _HON, _NUMWORDS = _CAP_ES, _STOP_ES, _HON_ES, _NUMWORDS_ES
        log.info("lang=es: using Spanish _CAP/_STOP/_HON/_NUMWORDS heuristics")

    out_root = Path(args.out_root) if args.out_root else Path(_paths.DATASETS_ROOT)
    src_split = Path(_paths.DATASETS_ROOT) / args.source / args.split
    man = pl.read_parquet(src_split / "manifest.parquet")
    if args.src_lang and "language" in man.columns:
        before = man.height
        man = man.filter(pl.col("language") == args.src_lang)
        log.info("src-lang=%s: filtered manifest %d -> %d rows", args.src_lang, before, man.height)
    align = pl.read_parquet(args.alignment)
    text_by_key = dict(zip(man["__key__"], man["text"]))
    path_by_key = dict(zip(man["__key__"], man["path"]))

    # choose entities once (shared across methods)
    ext = None
    if args.entities_from:
        ext = {r["__key__"]: (list(r["entity_indices"]), int(r["target_idx"]))
               for r in pl.read_parquet(args.entities_from).iter_rows(named=True)}
    chosen = {}
    for rec in align.iter_rows(named=True):
        key = rec["__key__"]
        if key not in text_by_key:
            continue
        fw = text_by_key[key].split(" ")
        if ext is not None:
            if key not in ext:
                continue
            all_idx, target = ext[key]
        elif args.kind in ("first", "last", "random"):
            all_idx, target = positional_target(fw, rec["words"], args.kind, key, seed=args.seed)
        else:
            all_idx, target = collect_entities(fw, rec["words"], kind=args.kind, number_mode=args.number_mode)
        if target is not None:
            chosen[key] = (all_idx, target, rec["words"])
    log.info("chosen %d / %d aligned (kind=%s entities_from=%s mask_all=%s)",
             len(chosen), align.height, args.kind, bool(args.entities_from), args.mask_all)

    kind_tag = {"proper": "name", "proper+title": "nametitle", "number": "num",
                "first": "first", "last": "last", "random": "rand"}[args.kind]
    if args.entities_from:
        kind_tag = "gner"
    # hidden_side drives the probe's token locator: head→first occurrence (find),
    # tail→last occurrence (rfind) + EOS-ban continuation, span→first occurrence.
    hidden_side = {"first": "head", "last": "tail"}.get(args.kind, "span")
    scope = "all" if args.mask_all else "one"
    tag = f"mask-{kind_tag}-{scope}"
    if args.name_suffix:
        tag = f"{tag}-{args.name_suffix}"
    name_base = f"{args.source}-{args.name_infix}" if args.name_infix else args.source
    for method in args.methods.split(","):
        method = method.strip()
        ds_name = f"{name_base}-{tag}-{method}"
        out_split = out_root / ds_name / args.split
        if out_split.exists():
            shutil.rmtree(out_split)
        out_split.mkdir(parents=True, exist_ok=True)
        meta_rows, new_dur = [], {}
        for key, (all_idx, target, words) in chosen.items():
            rel = path_by_key[key]
            src_wav = src_split / rel
            if not src_wav.exists():
                continue
            wav, sr = sf.read(str(src_wav), dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            out_wav = wav.copy()
            spans = all_idx if args.mask_all else [target]
            dur_s = len(wav) / sr
            # if the target is a zero-width (digit) token, it's only maskable when the
            # aligner left a real gap for it — skip the sample otherwise (~20%).
            if words[target]["t1"] - words[target]["t0"] < 0.06:
                pe = words[target - 1]["t1"] if target > 0 else 0.0
                ns = words[target + 1]["t0"] if target + 1 < len(words) else dur_s
                if ns - pe < 0.08:
                    continue
            for i in spans:
                if i >= len(words):
                    continue
                t0, t1 = words[i]["t0"], words[i]["t1"]
                prev_end = words[i - 1]["t1"] if i > 0 else 0.0
                nxt_start = words[i + 1]["t0"] if i + 1 < len(words) else dur_s
                if args.kind == "first":
                    # LOOSE opening mask: clip start → a bit PAST the onset of word 2.
                    # The extra mask_pad into word 2 catches the first word's offset that
                    # the aligner clips, so no trailing syllable bleeds through.
                    lo, hi = 0.0, min(nxt_start + args.mask_pad, dur_s)
                elif args.kind == "last":
                    # LOOSE closing mask: a bit BEFORE the 2nd-to-last word's offset → end.
                    # The extra mask_pad into the previous word catches the last word's
                    # onset the aligner clips, so no leading syllable bleeds through.
                    lo, hi = max(prev_end - args.mask_pad, 0.0), dur_s
                elif t1 - t0 < 0.06:
                    # zero-width token (e.g. a digit the aligner couldn't place) — its
                    # audio is the whole inter-word gap; mask that (clamped off neighbours).
                    lo, hi = prev_end + args.neighbor_guard, nxt_start - args.neighbor_guard
                else:
                    # widen into the gaps to catch onset/offset the aligner clips, but
                    # never reach an adjacent word (clamp with a guard).
                    lo = min(t0, max(t0 - args.mask_pad, prev_end + args.neighbor_guard))
                    hi = max(t1, min(t1 + args.mask_pad, nxt_start - args.neighbor_guard))
                if hi > lo:
                    corrupt_span(out_wav, sr, lo, hi, method, args.att_db, args.noise_amp)
            dst = out_split / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), out_wav, sr)
            new_dur[key] = len(out_wav) / sr
            fr = text_by_key[key]
            fwk = fr.split(" ")
            # hidden_ref: for numbers, the full consecutive value run; else the target word.
            if args.kind == "number" and not args.entities_from:
                aset = set(all_idx)
                run = [target]
                j = target - 1
                while j in aset:
                    run.insert(0, j); j -= 1
                j = target + 1
                while j in aset:
                    run.append(j); j += 1
                hidden_ref = " ".join(_core(fwk[j]) for j in run if j < len(fwk))
            else:
                hidden_ref = _core(fwk[target]) if target < len(fwk) else ""
            meta_rows.append({
                "__key__": key, "source_dataset": args.source, "split": args.split,
                "mode": "entity", "method": method, "hidden_side": hidden_side, "mask_all": args.mask_all,
                "entity_idx": target, "entity_t0": round(words[target]["t0"], 3),
                "entity_t1": round(words[target]["t1"], 3), "n_masked_spans": len(spans),
                "n_ref_words_kept": target, "n_ref_words_total": len(fwk),
                "hidden_ref": hidden_ref, "partial_ref": fr, "full_ref": fr,
            })
        out_man = man.filter(pl.col("__key__").is_in(set(new_dur)))
        out_man = out_man.with_columns(
            pl.col("__key__").replace_strict(new_dur, default=None).alias("duration"),
            pl.lit(ds_name).alias("dataset"),
        )
        out_man.write_parquet(out_split / "manifest.parquet")
        pl.DataFrame(meta_rows, strict=False).write_parquet(out_split / "truncation_meta.parquet")
        log.info("built %s — %d wavs", ds_name, len(meta_rows))


if __name__ == "__main__":
    main()
