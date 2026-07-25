"""Where the figure generators read and write.

The generators originally hardcoded absolute paths on our cluster. They now
resolve three roots, each overridable by environment variable:

``BENCHMARK_OPT_CELLS`` (default ``repro/data``)
    Derived per-figure data shipped with the repository. Small JSON tables of
    the numbers behind each figure, plus a few small parquet files. Enough to
    rebuild most figures in a clean clone with no other inputs.

``BENCHMARK_OPT_FIGURES`` (default ``repro/out``)
    Where rendered PNGs go.

``BENCHMARK_OPT_DATA`` (default unset)
    A full results root, in the layout described in ``REPRODUCE.md``. Only
    needed to re-derive cells from raw model outputs via
    ``precompute_cells.py``; the shipped cells already cover the figures.
"""

from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CELLS = Path(os.environ.get("BENCHMARK_OPT_CELLS", _HERE / "data"))
FIGURES = Path(os.environ.get("BENCHMARK_OPT_FIGURES", _HERE / "out"))
_DATA = os.environ.get("BENCHMARK_OPT_DATA")
DATA = Path(_DATA) if _DATA else None

FIGURES.mkdir(parents=True, exist_ok=True)


def cell(*parts: str) -> Path:
    """Path to a shipped derived-data file, with a clear error if it is absent."""
    p = CELLS.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(
            f"missing derived data: {p}\n"
            f"Set BENCHMARK_OPT_CELLS to a directory containing it, or regenerate it "
            f"from a full results root with:\n"
            f"  BENCHMARK_OPT_DATA=/path/to/results python repro/precompute_cells.py"
        )
    return p


def data(*parts: str) -> Path:
    """Path inside a full results root. Requires ``BENCHMARK_OPT_DATA``."""
    return require_data().joinpath(*parts)


def require_data() -> Path:
    """The full results root, or a clear error explaining what is missing."""
    if DATA is None:
        raise SystemExit(
            "This figure re-derives its numbers from raw per-model outputs, which are\n"
            "not shipped with the repository (hundreds of GB, and some corpora are not\n"
            "redistributable). Set BENCHMARK_OPT_DATA to a results root laid out as\n"
            "described in repro/REPRODUCE.md.\n\n"
            "Figures driven by the shipped cells under repro/data need nothing extra —\n"
            "see the table in repro/REPRODUCE.md for which is which."
        )
    return DATA
