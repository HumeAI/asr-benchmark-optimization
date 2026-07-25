"""Quantifying benchmark optimization in ASR models.

Two probes run on prediction files alone, with no audio and no model weights:

:mod:`benchmaxx.refdis`
    Reference-error reproduction. Where a panel of models agrees against the
    reference transcript, does the model under test follow the panel or the
    reference?

:mod:`benchmaxx.ortho`
    Orthographic switch rate. For written distinctions that sound identical,
    does the model's spelling track the corpus it is being evaluated on?

Both report a rate over *underdetermined* cases, where the audio does not fix
the answer, so neither is a measure of transcription quality. A model can be
accurate and score near zero, or accurate and score high; the two numbers
answer different questions.

Quick start::

    from benchmaxx import load_dir, ortho, conventions

    preds = load_dir("predictions/")           # one JSONL per model
    results = ortho.switch_rates(conventions.families_for("en"), list(preds.clips()))
"""

from __future__ import annotations

from . import align, conventions, normalize, ortho, predictions, refdis
from .conventions import Family, families_for, family
from .normalize import tokenize
from .ortho import switch_rate, switch_rates
from .predictions import PredictionSet, load_dir, load_manifest, load_manifests
from .refdis import RefEdit, accept_ref_rate, find_ref_edits

__version__ = "0.1.0"

__all__ = [
    "align",
    "conventions",
    "normalize",
    "ortho",
    "predictions",
    "refdis",
    "Family",
    "families_for",
    "family",
    "tokenize",
    "switch_rate",
    "switch_rates",
    "PredictionSet",
    "load_dir",
    "load_manifest",
    "load_manifests",
    "RefEdit",
    "accept_ref_rate",
    "find_ref_edits",
    "__version__",
]
