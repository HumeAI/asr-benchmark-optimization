"""White-box consensus probe: for each error-containing vox-EN utterance, teacher-force
the (erroneous) REFERENCE vs the CONSENSUS-CORRECTED transcript through the model and
record the NLL of each. A model that has memorized the reference assigns the bad ref a
LOW NLL (≈ the corrected one); an honest model finds the corrected (audio-true) seq more
likely → positive NLL gap (nll_ref - nll_cons).

Reuses Probe.ids_for_text + Probe.token_logprobs. mask_eos=False (full audio, natural
distribution — not a 'must continue' condition). Writes per-utterance parquet.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

# Data and output roots resolve through repro/paths.py (env-overridable);
# the originals hardcoded absolute paths on our cluster.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as RP  # noqa: E402

CONS = RP.CELLS / "consensus"


def _probe_module():
    """Import the white-box probe engine lazily.

    ``corrected`` below is pure Python and is what the figure code imports, but
    the probe engine pulls in torch and soundfile. Importing it at module scope
    would make rebuilding a figure require the full probe environment.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
    import probe_decoder_memorization as _probe

    return _probe


def corrected(ref, runs):
    drop, ins = set(), {}
    for r in runs:
        if r["n_wl_agree"] != r["n_wl_total"]:
            continue
        if r["run_type"] == "delete":
            drop.update(r["run_indices"])
        elif r["run_type"] == "insert":
            ins.setdefault(r["run_indices"][0], []).extend(t.lower() for t in r["run_tokens"])
    out = []
    for i, tok in enumerate(ref):
        out += ins.get(i, [])
        if i not in drop:
            out.append(tok)
    return out + ins.get(len(ref), [])


def nll(probe, audio, text):
    ids = probe.ids_for_text(text)
    lp = probe.token_logprobs(audio, ids)  # per-token logprobs, nan where not scored
    lp = lp[~np.isnan(lp)]
    return (float(-lp.sum()), int(lp.size)) if lp.size else (np.nan, 0)


def main():
    P = _probe_module()
    VOX = RP.data("datasets/voxpopuli/test")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(P.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default=None,
                    help="dataset name under datasets/ to take audio from (default: real voxpopuli). "
                         "Use a TTS-clone dataset to score ref-vs-cons NLL on cloned voices.")
    ap.add_argument("--with-prior", action="store_true",
                    help="also teacher-force ref/cons with the ENTIRE audio zeroed (the LM prior), "
                         "enabling the diff-in-diff audio-evidence metric.")
    args = ap.parse_args()

    samples = json.load(open(CONS / "vox_en_3wl_test_samples.json"))
    src = (VOX.parent.parent / args.source / "test") if args.source else VOX
    wav = {r["__key__"]: str(src / r["path"])
           for r in pq.read_table(src / "manifest.parquet", columns=["__key__", "path"]).to_pylist()}

    probe = P.Probe(args.model)
    probe.mask_eos = True  # mask EOS: the fully-masked prior is meaningless otherwise (silence dumps mass on EOS)

    rows = []
    items = samples[: args.limit] if args.limit else samples
    for s in items:
        k = s["key"]
        p = wav.get(k)
        if not p or not Path(p).exists():
            continue
        try:
            audio = P.load_wav(Path(p))
            ref_txt = " ".join(t.lower() for t in s["ref_tokens"])
            cons_txt = " ".join(corrected([t.lower() for t in s["ref_tokens"]], s["runs"]))
            nr, n_r = nll(probe, audio, ref_txt)
            nc, n_c = nll(probe, audio, cons_txt)
            row = {"key": k, "nll_ref": nr, "n_ref": n_r, "nll_cons": nc, "n_cons": n_c,
                   "ppl_ref": float(np.exp(nr / n_r)) if n_r else np.nan,
                   "ppl_cons": float(np.exp(nc / n_c)) if n_c else np.nan}
            if args.with_prior:
                # LM prior: teacher-force the same transcripts with the ENTIRE audio masked (zeroed).
                zero = np.zeros_like(audio)
                npr, npr_n = nll(probe, zero, ref_txt)
                ncp, ncp_n = nll(probe, zero, cons_txt)
                row.update(nll_ref_prior=npr, n_ref_prior=npr_n, nll_cons_prior=ncp, n_cons_prior=ncp_n,
                           ppl_ref_prior=float(np.exp(npr / npr_n)) if npr_n else np.nan,
                           ppl_cons_prior=float(np.exp(ncp / ncp_n)) if ncp_n else np.nan)
        except Exception as e:
            P.log.warning("skip %s: %s", k, type(e).__name__)
            continue
        rows.append(row)
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    g = df.drop_nulls(["nll_ref", "nll_cons"])
    # gap > 0  ⇒ corrected is more likely than the bad ref (honest); ≈0 / <0 ⇒ prefers bad ref
    tot_gap = (g["nll_ref"].sum() - g["nll_cons"].sum())
    mean_ppl_gap = (g["ppl_ref"] - g["ppl_cons"]).median()
    P.log.info("wrote %s (%d rows). total NLL(ref-cons)=%.1f  median ppl(ref-cons)=%.3f",
               args.out, df.height, tot_gap, mean_ppl_gap)


if __name__ == "__main__":
    main()
