"""Loading per-clip predictions from several models into one table.

The native format is the Open ASR Leaderboard prediction manifest, one JSON
object per clip::

    {"audio_filepath": ..., "text": "<reference>", "pred_text": "<prediction>"}

Those manifests hold **raw** text — the leaderboard normalizes only when
computing WER, after writing the manifest. The switch rate measures distinctions
normalization erases, so it needs raw text; already-normalized input supports
reference disagreement only.

Clips join on ``audio_filepath``. A model missing a clip is absent from that
clip's hypotheses rather than present with an empty string, so that "did not
run" stays distinct from "transcribed nothing".
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PredictionSet", "load_manifest", "load_manifests", "load_dir"]

# Accepted column spellings, most-preferred first. Covers the leaderboard
# manifest, our own result dumps, and the usual hand-rolled CSV.
_KEY_FIELDS = ("audio_filepath", "id", "key", "__key__", "audio_id", "utt_id", "filename")
_REF_FIELDS = ("text", "reference", "ref", "ground_truth", "target")
_HYP_FIELDS = ("pred_text", "prediction", "hypothesis", "hyp", "transcription", "pred")


def _pick(row: dict, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in row and row[c] is not None:
            return c
    return None


@dataclass
class PredictionSet:
    """References and per-model hypotheses for a set of clips.

    ``refs`` maps clip key to reference text; ``hyps`` maps clip key to
    ``{model: hypothesis}``. Both hold raw text unless the source was already
    normalized.
    """

    refs: dict[str, str] = field(default_factory=dict)
    hyps: dict[str, dict[str, str]] = field(default_factory=dict)
    language: str = "en"

    @property
    def models(self) -> list[str]:
        return sorted({m for h in self.hyps.values() for m in h})

    def __len__(self) -> int:
        return len(self.refs)

    def add(self, key: str, ref: str, model: str, hyp: str) -> None:
        # First reference seen for a key wins. Two manifests for the same clip
        # should carry the same reference; if they do not, that is a dataset
        # version mismatch worth surfacing rather than silently averaging over.
        prev = self.refs.setdefault(key, ref)
        if prev != ref:
            self._ref_conflicts.add(key)
        self.hyps.setdefault(key, {})[model] = hyp

    _ref_conflicts: set[str] = field(default_factory=set, repr=False)

    @property
    def ref_conflicts(self) -> set[str]:
        """Clips whose sources disagreed on the reference text."""
        return self._ref_conflicts

    def clips(self, *, models: list[str] | None = None, require_all: bool = False):
        """Yield ``(reference, {model: hypothesis})`` pairs.

        ``models`` restricts and orders the models considered.
        ``require_all`` keeps only clips every requested model covered, which
        makes model-to-model comparison exact at the cost of dropping clips.
        Without it, each model is scored on the clips it has, and the
        per-model denominators must be reported.
        """
        wanted = models or self.models
        for key, ref in self.refs.items():
            got = {m: h for m, h in self.hyps.get(key, {}).items() if m in wanted}
            if require_all and len(got) < len(wanted):
                continue
            if got:
                yield ref, got

    def normalized(self, language: str | None = None, *, numbers: bool = True) -> PredictionSet:
        """A copy with every reference and hypothesis normalized.

        Needed for the reference-disagreement probe, which aligns tokens and so
        must not be tripped by casing or punctuation. Do **not** use this for
        the switch probe, which measures distinctions normalization erases.

        ``numbers`` additionally rewrites spelled-out numbers as digits, so a
        word/digit formatting difference does not register as a reference error.
        See :func:`benchmark_optimization.normalize.canonicalize_numbers`.
        """
        from .normalize import canonicalize_numbers, normalize

        lang = language or self.language

        def prep(text: str) -> str:
            out = normalize(text, lang)
            return canonicalize_numbers(out, lang) if numbers else out

        out = PredictionSet(language=lang)
        out.refs = {k: prep(v) for k, v in self.refs.items()}
        out.hyps = {k: {m: prep(h) for m, h in v.items()} for k, v in self.hyps.items()}
        return out

    def coverage(self) -> dict[str, int]:
        """Clips per model, for spotting a model that only partly ran."""
        out: dict[str, int] = {}
        for h in self.hyps.values():
            for m in h:
                out[m] = out.get(m, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _iter_rows(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from (payload if isinstance(payload, list) else payload.get("samples", []))
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq  # optional dependency

        yield from pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"unsupported prediction file type: {path.name}")


def load_manifest(
    path: str | Path,
    model: str | None = None,
    *,
    into: PredictionSet | None = None,
    key_field: str | None = None,
    ref_field: str | None = None,
    hyp_field: str | None = None,
    language: str = "en",
) -> PredictionSet:
    """Load one model's predictions.

    ``model`` defaults to the file stem. Field names are detected from the
    first row against the spellings in ``_KEY_FIELDS`` / ``_REF_FIELDS`` /
    ``_HYP_FIELDS``, and can be overridden. Rows with no hypothesis field or a
    null hypothesis are skipped, so a partial run loads cleanly.
    """
    path = Path(path)
    model = model or path.stem
    ps = into if into is not None else PredictionSet(language=language)

    rows = _iter_rows(path)
    try:
        first = next(rows)
    except StopIteration:
        return ps

    kf = key_field or _pick(first, _KEY_FIELDS)
    rf = ref_field or _pick(first, _REF_FIELDS)
    hf = hyp_field or _pick(first, _HYP_FIELDS)
    if hf is None or rf is None:
        raise ValueError(
            f"{path.name}: could not find reference/hypothesis columns in {sorted(first)}. "
            f"Pass ref_field=/hyp_field= explicitly."
        )

    for i, row in enumerate(_chain(first, rows)):
        hyp = row.get(hf)
        if hyp is None:
            continue
        key = str(row[kf]) if kf and row.get(kf) is not None else f"{path.stem}:{i}"
        ps.add(key, row.get(rf) or "", model, hyp)
    return ps


def _chain(first, rest):
    yield first
    yield from rest


def load_manifests(
    paths: dict[str, str | Path] | list[str | Path],
    *,
    language: str = "en",
    **kwargs,
) -> PredictionSet:
    """Load several models into one :class:`PredictionSet`.

    Pass a ``{model_name: path}`` mapping, or a list of paths whose stems name
    the models.
    """
    ps = PredictionSet(language=language)
    items = paths.items() if isinstance(paths, dict) else ((None, p) for p in paths)
    for model, path in items:
        load_manifest(path, model, into=ps, language=language, **kwargs)
    return ps


def load_dir(
    directory: str | Path,
    *,
    pattern: str = "*.jsonl",
    language: str = "en",
    **kwargs,
) -> PredictionSet:
    """Load every matching prediction file in a directory, one model per file.

    The model name is the file stem, so name files after models.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")
    return load_manifests(list(files), language=language, **kwargs)
