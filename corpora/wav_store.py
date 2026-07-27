"""Raw-wav + Parquet-manifest dataset storage (post-WSDS).

Layout::

    DATA_ROOT/datasets/{dataset}/{split}/
        manifest.parquet
        wavs/{iso_lang}/{key}.wav

`manifest.parquet` mirrors `BenchmarkSample` minus `audio`, plus a relative
`path` column. See `docs/plans/WAV_MIGRATION.md`.

Readers elsewhere should prefer `has_manifest(dir) + iter_samples(dir)` over
`ipc.open_file(...)` so the same code path works for both layouts during the
transition window.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.parquet"
WAV_SUBDIR = "wavs"
TARGET_SR = 16000

# Optional GCS fallback for pods with no local dataset mount. When set to a
# ``gs://<bucket>[/<prefix>]`` URI, ``read_manifest`` / ``has_manifest`` /
# ``read_wav_bytes`` fall through to GCS when the file is missing locally.
# The path layout mirrors the bucket structure produced by the sync script.
#
# Intended for K8s: pod sets ASR_DATASETS_GCS_URI=gs://asr-benchmarking-data
# alongside ASR_BENCHMARKING_DATA_ROOT=/nonexistent so all reads go through
# GCS. For VM / dev, leave unset and reads stay local.
DATASETS_GCS_URI = os.environ.get("ASR_DATASETS_GCS_URI", "").rstrip("/")
_gcs_bucket = None
_gcs_prefix: str | None = None


def _get_gcs_bucket():
    """Return a cached (bucket, prefix) pair for the GCS fallback path, or None."""
    global _gcs_bucket, _gcs_prefix
    if not DATASETS_GCS_URI:
        return None
    if _gcs_bucket is not None:
        return _gcs_bucket, _gcs_prefix or ""
    from google.cloud import storage  # lazy import

    without_scheme = DATASETS_GCS_URI.removeprefix("gs://")
    bucket_name, _, prefix = without_scheme.partition("/")
    client = storage.Client()
    _gcs_bucket = client.bucket(bucket_name)
    _gcs_prefix = prefix
    return _gcs_bucket, prefix


def _gcs_blob_name(dataset_dir: Path, rel_path: str) -> str:
    """Map (local dataset_dir, rel_path) → a GCS blob name under DATASETS_GCS_URI.

    Expects dataset_dir to end with ``/datasets/<dataset>/<split>`` — we keep
    the ``datasets/...`` suffix from the local path and drop anything before it.
    """
    parts = Path(dataset_dir).parts
    if "datasets" not in parts:
        return f"{rel_path}".lstrip("/")
    i = parts.index("datasets")
    tail = "/".join(parts[i:])
    prefix = _gcs_prefix or ""
    if prefix:
        return f"{prefix}/{tail}/{rel_path}"
    return f"{tail}/{rel_path}"


# Columns we write to the manifest. Order is stable for readers that pull a
# projection by name. `audio` is deliberately absent — that's the whole point.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "__key__",
    "path",
    "text",
    "text_normalized",
    "language",
    "dataset",
    "split",
    "duration",
    "sample_rate",
    "speaker_id",
    "gender",
    "accent",
    "dialect",
    "age",
    "age_group",
    "emotion",
    "topic",
    "session_id",
    "up_votes",
    "down_votes",
)


def _coerce_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def manifest_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / MANIFEST_NAME


def has_manifest(dataset_dir: Path) -> bool:
    """True if this dataset has a wav + manifest layout locally or in GCS."""
    if manifest_path(dataset_dir).exists():
        return True
    gcs = _get_gcs_bucket()
    if gcs is None:
        return False
    bucket, _ = gcs
    return bucket.blob(_gcs_blob_name(dataset_dir, MANIFEST_NAME)).exists()


def wav_relpath(iso_lang: str, key: str) -> str:
    """Relative path for a sample's wav file, written into the manifest."""
    # iso_lang may be empty for English-only datasets; group those under `_`
    # so we never end up with bare `wavs/{key}.wav` (simplifies listing).
    lang = iso_lang or "_"
    return f"{WAV_SUBDIR}/{lang}/{key}.wav"


def wav_abspath(dataset_dir: Path, iso_lang: str, key: str) -> Path:
    return Path(dataset_dir) / wav_relpath(iso_lang, key)


# ── Writer ─────────────────────────────────────────────────────────────────


@dataclass
class WavStoreWriter:
    """Streamed writer: call `write_sample(row, audio_array, sr)` per sample,
    then `finalize()` to flush the manifest.

    Holds manifest rows in memory (one dict per sample, no audio). For the
    48 GB corpus this is ~200 k rows × ~300 B = ~60 MB — fine.

    Guarantees one row per `__key__`: callers can re-submit the same key
    (HF sources occasionally surface the same audio_hash_id twice, additive
    re-ingests can replay rows, etc.) and the writer skips the duplicate
    instead of doubling up the manifest. The historical breakage where 21
    datasets shipped 2-10× duplicate __key__ rows traces back to a missing
    guard here.
    """

    dataset_dir: Path
    _rows: List[Dict[str, Any]] = None  # type: ignore[assignment]
    _seen_keys: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._seen_keys = set()

    def has_key(self, key: str) -> bool:
        """Whether `key` has already been staged. Callers use this to skip
        re-decoding/re-resampling audio for a sample they've already added.
        """
        return key in self._seen_keys

    def stage_row(self, row: Dict[str, Any]) -> bool:
        """Append a manifest row without writing audio (used when merging an
        existing manifest into a fresh ingest pass). Returns True if added,
        False if the key was already present.
        """
        key = row["__key__"]
        if key in self._seen_keys:
            return False
        self._rows.append(row)
        self._seen_keys.add(key)
        return True

    def write_sample(
        self,
        row: Dict[str, Any],
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Write the wav file and stage a manifest row.

        `row` must include `__key__` and `language`; other `MANIFEST_COLUMNS`
        fields are copied if present, or defaulted to empty string / 0.

        Returns True if the sample was added, False if the key was already
        staged (the wav is not rewritten, the manifest is not appended).
        """
        key = row["__key__"]
        if key in self._seen_keys:
            return False
        iso_lang = row.get("language", "") or ""
        rel = wav_relpath(iso_lang, key)
        out = self.dataset_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio_array, sample_rate, format="WAV", subtype="PCM_16")
        duration = float(row.get("duration", len(audio_array) / sample_rate))

        manifest_row: Dict[str, Any] = {
            "__key__": key,
            "path": rel,
            "text": row.get("text", "") or "",
            "text_normalized": row.get("text_normalized", "") or "",
            "language": iso_lang,
            "dataset": row.get("dataset", "") or "",
            "split": row.get("split", "") or "",
            "duration": round(duration, 4),
            "sample_rate": int(sample_rate),
            "speaker_id": row.get("speaker_id", "") or "",
            "gender": row.get("gender", "") or "",
            "accent": row.get("accent", "") or "",
            "dialect": row.get("dialect", "") or "",
            "age": row.get("age", "") or "",
            "age_group": row.get("age_group", "") or "",
            "emotion": row.get("emotion", "") or "",
            "topic": row.get("topic", "") or "",
            "session_id": row.get("session_id", "") or "",
            "up_votes": _coerce_int(row.get("up_votes")),
            "down_votes": _coerce_int(row.get("down_votes")),
        }
        self._rows.append(manifest_row)
        self._seen_keys.add(key)
        return True

    def finalize(self) -> Path:
        """Write `manifest.parquet`. Called once after all samples are staged.

        Re-running `finalize()` overwrites the manifest — staged rows are
        authoritative.
        """
        if not self._rows:
            raise RuntimeError(f"no samples staged for {self.dataset_dir}")

        # Build the table column-major so we get consistent pa types even
        # when optional fields are all-empty. Explicit schema keeps int/float
        # columns from collapsing to null under pyarrow's inference.
        arrays = []
        schema_fields = []
        for col in MANIFEST_COLUMNS:
            values = [r.get(col) for r in self._rows]
            if col == "duration":
                arr = pa.array(values, type=pa.float64())
                schema_fields.append(pa.field(col, pa.float64()))
            elif col == "sample_rate":
                arr = pa.array(values, type=pa.int32())
                schema_fields.append(pa.field(col, pa.int32()))
            elif col in ("up_votes", "down_votes"):
                # Nullable int — None for sources that don't expose vote counts.
                arr = pa.array([_coerce_int(v) for v in values], type=pa.int32())
                schema_fields.append(pa.field(col, pa.int32()))
            else:
                arr = pa.array([("" if v is None else str(v)) for v in values], type=pa.string())
                schema_fields.append(pa.field(col, pa.string()))
            arrays.append(arr)

        table = pa.Table.from_arrays(arrays, schema=pa.schema(schema_fields))
        out = manifest_path(self.dataset_dir)
        pq.write_table(table, str(out), compression="zstd")
        logger.info("wrote manifest: %s (%d rows)", out, len(self._rows))
        return out


# ── Reader ─────────────────────────────────────────────────────────────────


# Per-process manifest cache (keyed by dataset_dir stringification). Keeps
# the GCS-fallback path from round-tripping on every request. Manifests are
# tiny (<10 MB each) and effectively immutable within a pod's lifetime.
_manifest_cache: Dict[str, pa.Table] = {}


def read_manifest(dataset_dir: Path, columns: Optional[Iterable[str]] = None) -> pa.Table:
    """Load the manifest (optionally projecting a subset of columns).

    Reads from local disk when present; otherwise falls back to
    ``gs://<DATASETS_GCS_URI>/.../manifest.parquet``. Fully-cached per
    process — downstream column projection happens in-memory.
    """
    key = str(dataset_dir)
    if key in _manifest_cache:
        full = _manifest_cache[key]
        return full.select(list(columns)) if columns else full

    path = manifest_path(dataset_dir)
    if path.exists():
        full = pq.read_table(str(path))
    else:
        gcs = _get_gcs_bucket()
        if gcs is None:
            raise FileNotFoundError(f"no manifest at {path} and no GCS fallback configured")
        bucket, _ = gcs
        blob_name = _gcs_blob_name(dataset_dir, MANIFEST_NAME)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"no manifest at {path} or gs://{bucket.name}/{blob_name}")
        full = pq.read_table(io.BytesIO(blob.download_as_bytes()))

    _manifest_cache[key] = full
    return full.select(list(columns)) if columns else full


def manifest_keys(dataset_dir: Path) -> List[str]:
    """Return `__key__` in manifest order."""
    return read_manifest(dataset_dir, columns=["__key__"]).column("__key__").to_pylist()


def read_wav_bytes(dataset_dir: Path, rel_path: str) -> bytes:
    """Load a wav file and return its raw bytes.

    Local-first, with a GCS fallback driven by ``ASR_DATASETS_GCS_URI``. The
    leaderboard's audio route prefers 302-to-signed-URL over this path, so
    this is mostly used by inference / feature extraction when a pod is
    configured to stream audio bytes through itself.
    """
    path = Path(dataset_dir) / rel_path
    if path.exists():
        return path.read_bytes()
    gcs = _get_gcs_bucket()
    if gcs is None:
        raise FileNotFoundError(f"wav not found: {path}")
    bucket, _ = gcs
    blob_name = _gcs_blob_name(dataset_dir, rel_path)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"wav not found: {path} or gs://{bucket.name}/{blob_name}")
    return blob.download_as_bytes()


def load_audio(dataset_dir: Path, rel_path: str) -> tuple[np.ndarray, int]:
    """Load a wav file as (float32 mono array, sample_rate)."""
    path = Path(dataset_dir) / rel_path
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


def iter_samples(dataset_dir: Path, with_audio: bool = True) -> Iterator[Dict[str, Any]]:
    """Yield per-sample dicts that match the WSDS-era `iter_wsds_samples` shape.

    When `with_audio=True` (default), each dict includes an `audio` key with
    raw wav bytes — drop-in for existing decode pipelines. Pass False to stream
    only metadata (e.g. for the leaderboard source-frame build).
    """
    dataset_dir = Path(dataset_dir)
    table = read_manifest(dataset_dir)
    col_names = table.column_names
    n = table.num_rows
    # Materialize columns once; pyarrow chunked access in a tight loop is slow.
    cols = {name: table.column(name).to_pylist() for name in col_names}
    for i in range(n):
        row = {name: cols[name][i] for name in col_names}
        if with_audio:
            row["audio"] = read_wav_bytes(dataset_dir, row["path"])
        yield row
