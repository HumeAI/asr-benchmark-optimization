"""Data and model roots for the audio-side probes.

These probes read source audio and write derived datasets, so unlike the
prediction-only probes in ``benchmark_optimization`` they need a real data layout on disk.
This module replaces the internal path module the scripts originally imported.

Environment variables, all optional:

``BENCHMARK_OPT_DATA``
    Root containing ``datasets/`` and ``results/``. Defaults to ``./data``.

``BENCHMARK_OPT_MODELS``
    Root for locally-staged model weights. Only needed for models loaded from
    disk rather than from the HuggingFace hub. Defaults to ``./models``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BENCHMARK_OPT_DATA", "data")).resolve()
MODELS_ROOT = Path(os.environ.get("BENCHMARK_OPT_MODELS", "models")).resolve()

DATASETS_ROOT = DATA_ROOT / "datasets"
RESULTS_ROOT = DATA_ROOT / "results"


def dataset_path(dataset: str, split: str = "test") -> Path:
    return DATASETS_ROOT / dataset / split


def results_path(dataset: str, model: str, split: str = "test") -> Path:
    return RESULTS_ROOT / dataset / split / model
