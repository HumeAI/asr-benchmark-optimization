"""POC: is masked-number retrieval benchmark(acoustic)-specific, not text-driven?

Same transcript + same masked number, rendered two ways:
  - REAL    : the real VoxPopuli recording (vox mode ON)
  - GENERIC : a generic-voice (Ryan) TTS clone of the same transcript (vox mode OFF)

Because the text (and thus the LM prior on the number) is identical, any gap in
masked-number retrieval between REAL and GENERIC is the ACOUSTIC/benchmark-register
shortcut, not context-predictability.

Validity gate (per the probed model, NOT voxtral): a clip counts only if the model
retrieves the number in its OWN UNMASKED rendering of that condition. This also
doubles as the generic-voice intelligibility check (if the model can't get the
number in unmasked generic, the TTS is the problem).

  python scripts/vmt/nummask_realvsgeneric_gated.py [model ...]   # default cohere-transcribe
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from maskclone_score import load_hyps, load_meta  # noqa: E402
from num2words import num2words  # noqa: E402

# value-based number matcher: match a number by VALUE, in digit / word / year form,
# so "Two"=="2", "2016"=="twenty sixteen". Form-neutral across real vs generic.
_W2I = {}
for _i in range(0, 2101):
    _W2I[num2words(_i)] = _i
    _W2I[num2words(_i).replace(" and ", " ")] = _i
# bare magnitude words (num2words only has "one hundred", not "hundred")
_MAG = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "dozen": 12,
}
_W2I.update(_MAG)


def _to_val(s):
    s = (s or "").strip().lower().strip(".,;:!?\"'").replace(",", "")
    if s.isdigit():
        return int(s)
    return _W2I.get(s)


def _forms(val):
    fs = {str(val), num2words(val), num2words(val).replace(" and ", " ")}
    if 1000 <= val <= 2099:
        fs.add(num2words(val, to="year"))
    if val >= 1000:
        fs.add(f"{val:,}")
    for _w, _v in _MAG.items():  # bare magnitude surface form
        if _v == val:
            fs.add(_w)
    return {f for f in fs if f}


def _alnum_flex(tok):
    """word-boundary regex that tolerates spaces between letter/digit runs: g20 -> 'g 20'."""
    runs = re.findall(r"[a-z]+|[0-9]+", tok.lower())
    if not runs:
        return None
    return r"(^|\W)" + r"\s*".join(re.escape(r) for r in runs) + r"($|\W)"


def hit(hidden, hyp):
    hyp_l = (hyp or "").lower()
    val = _to_val(hidden)
    if val is not None:
        if any(re.search(r"(^|\W)" + re.escape(f.lower()) + r"($|\W)", hyp_l) for f in _forms(val)):
            return True
    # non-numeric / alphanumeric (G20, COVID19) or value-miss: space-flexible alnum match
    pat = _alnum_flex((hidden or "").strip(".,;:!?\"'"))
    return bool(pat) and bool(re.search(pat, hyp_l))


def leak(meta, hyps, keys):
    n = lk = 0
    for k in keys:
        hidden = meta.get(k) or ""
        if not hidden or k not in hyps:
            continue
        n += 1
        lk += hit(hidden, hyps[k])
    return lk, n


TAG = "mask-num-all-numexp-silence"
GEN_MASKED = f"ttsclone-nummask-generic-{TAG}"
REAL_MASKED = f"voxpopuli-{TAG}"
GEN_UNMASKED = "ttsclone-nummask-generic"
REAL_UNMASKED = "voxpopuli"
EPF_MASKED = f"ttsclone-nummask-epfresh-{TAG}"
EPF_UNMASKED = "ttsclone-nummask-epfresh"
DK_MASKED = f"ttsclone-nummask-daikon-{TAG}"
DK_UNMASKED = "ttsclone-nummask-daikon"
VOX_MASKED = f"ttsclone-nummask-vox-{TAG}"  # clone of clip's OWN vox speaker
VOX_UNMASKED = "ttsclone-nummask-vox"
VOXX_MASKED = f"ttsclone-nummask-voxx-{TAG}"  # clone of a DIFFERENT sampled vox speaker
VOXX_UNMASKED = "ttsclone-nummask-voxx"
# real masked audio + each perturbation (robustness); gated by clean-unmasked real, same as REAL.
REALPERT = {
    "noise": f"voxpopuli-{TAG}-perturb-noise10db",
    "reverb": f"voxpopuli-{TAG}-perturb-reverb0.6",
    "mp3": f"voxpopuli-{TAG}-perturb-mp316k",
}


def gated_rate(meta, unmasked_hyps, masked_hyps, keys):
    """retrieval on masked audio, restricted to keys the model gets UNMASKED."""
    gate, n_gate = set(), 0
    for k in keys:
        hidden = meta.get(k) or ""
        if not hidden or k not in unmasked_hyps:
            continue
        n_gate += 1
        if hit(hidden, unmasked_hyps[k]):
            gate.add(k)
    lk, n = leak(meta, masked_hyps, gate)  # masked retrieval among unmasked-passers
    return {
        "unmasked_rate": round(len(gate) / n_gate, 3) if n_gate else None,  # intelligibility
        "n_unmasked_eval": n_gate,
        "n_gate_pass": len(gate),
        "masked_retrieval": round(lk / n, 3) if n else None,  # gated masked retrieval
        "masked_leak": lk,
        "n_gated": n,
    }


def main():
    models = sys.argv[1:] or ["cohere-transcribe"]
    meta_gen, meta_real = load_meta(GEN_MASKED), load_meta(REAL_MASKED)
    meta_epf, meta_dk = load_meta(EPF_MASKED), load_meta(DK_MASKED)
    meta_vox, meta_voxx = load_meta(VOX_MASKED), load_meta(VOXX_MASKED)
    keys = set(meta_gen or {}) & set(meta_real or {})  # same transcript in both
    print(
        f"shared nummask keys (real & generic): {len(keys)}"
        f"{'  + epfresh ready' if meta_epf else '  (epfresh not built yet)'}\n"
    )
    for m in models:
        ru, rm = load_hyps(REAL_UNMASKED, m), load_hyps(REAL_MASKED, m)
        gu, gm = load_hyps(GEN_UNMASKED, m), load_hyps(GEN_MASKED, m)
        eu, em = load_hyps(EPF_UNMASKED, m), load_hyps(EPF_MASKED, m)
        du, dm = load_hyps(DK_UNMASKED, m), load_hyps(DK_MASKED, m)
        if gu is None:
            print(f"{m}: no UNMASKED generic run yet (needed for gate)")
            continue
        real = gated_rate(meta_real, ru, rm, keys) if (ru and rm) else None
        realpert = {}  # perturbed masked, gated by clean-unmasked real (same as REAL)
        for plab, pds in REALPERT.items():
            hpm = load_hyps(pds, m)
            realpert[plab] = gated_rate(meta_real, ru, hpm, keys) if (ru and hpm) else None
        gen = gated_rate(meta_gen, gu, gm, keys)
        epf = gated_rate(meta_epf, eu, em, keys) if (meta_epf and eu and em) else None
        dk = gated_rate(meta_dk, du, dm, keys) if (meta_dk and du and dm) else None
        vxu, vxm = load_hyps(VOX_UNMASKED, m), load_hyps(VOX_MASKED, m)
        vxxu, vxxm = load_hyps(VOXX_UNMASKED, m), load_hyps(VOXX_MASKED, m)
        vox = gated_rate(meta_vox, vxu, vxm, keys) if (meta_vox and vxu and vxm) else None
        voxx = gated_rate(meta_voxx, vxxu, vxxm, keys) if (meta_voxx and vxxu and vxxm) else None
        print(f"=== {m} ===")
        print(
            f"  intelligibility (unmasked): generic(Ryan) {gen['unmasked_rate']}"
            + (f" | epfresh {epf['unmasked_rate']}" if epf else "")
            + (f" | daikon {dk['unmasked_rate']}" if dk else "")
            + (f" | real {real['unmasked_rate']}" if real else "")
        )
        if real:
            print(
                f"  REAL      masked retrieval (gated): {real['masked_retrieval']} ({real['masked_leak']}/{real['n_gated']})"
            )
        for plab in REALPERT:
            c = realpert.get(plab)
            if c:
                print(
                    f"  REAL+{plab:6s} masked retrieval (gated): {c['masked_retrieval']} ({c['masked_leak']}/{c['n_gated']})"
                )
        if vox:
            print(f"  VOX-CLONE (own spk)  (gated): {vox['masked_retrieval']} ({vox['masked_leak']}/{vox['n_gated']})")
        if voxx:
            print(
                f"  VOXX-CLONE (sampled) (gated): {voxx['masked_retrieval']} ({voxx['masked_leak']}/{voxx['n_gated']})"
            )
        print(
            f"  GENERIC   masked retrieval (gated): {gen['masked_retrieval']} ({gen['masked_leak']}/{gen['n_gated']})"
        )
        if epf:
            print(
                f"  EPFRESH  masked retrieval (gated): {epf['masked_retrieval']} ({epf['masked_leak']}/{epf['n_gated']})"
            )
        if dk:
            print(
                f"  DAIKON   masked retrieval (gated): {dk['masked_retrieval']} ({dk['masked_leak']}/{dk['n_gated']})"
            )
        for plab in REALPERT:
            c = realpert.get(plab)
            if real and c and c["masked_retrieval"] is not None:
                print(
                    f"  >>> REAL - REAL+{plab} = {real['masked_retrieval'] - c['masked_retrieval']:+.3f} (robustness)"
                )
        if real and gen["masked_retrieval"] is not None:
            print(f"  >>> REAL - GENERIC = {real['masked_retrieval'] - gen['masked_retrieval']:+.3f}")
        if real and epf:
            print(
                f"  >>> REAL - EPFRESH = {real['masked_retrieval'] - epf['masked_retrieval']:+.3f} (register-matched control)"
            )
        if real and dk:
            print(
                f"  >>> REAL - DAIKON  = {real['masked_retrieval'] - dk['masked_retrieval']:+.3f} (conversational held-out control)"
            )
        print()


if __name__ == "__main__":
    main()
