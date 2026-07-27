"""Quantifying benchmark optimization in ASR models.

:mod:`benchmark_optimization.refdis`
    Reference disagreement — where a panel of models agrees against the
    reference, does a model follow the panel or the reference?

:mod:`benchmark_optimization.ortho`
    Orthographic switch rate — for written distinctions that sound identical,
    does a model's spelling track the corpus?

Both are rates over cases where the audio does not determine the reference, so
neither measures transcription quality.

Quick start::

    from benchmark_optimization import load_dir, ortho, conventions

    preds = load_dir("predictions/")
    ortho.switch_rates(conventions.families_for("en"), list(preds.clips()))
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
