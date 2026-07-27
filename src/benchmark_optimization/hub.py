"""Load the published predictions straight from the Hub.

    from benchmark_optimization import hub

    preds = hub.load_predictions("voxpopuli")
    masks = hub.load_masks("voxpopuli-mask-num-all-numexp-silence")

Requires ``huggingface_hub`` (``pip install "asr-benchmark-optimization[hub]"``).
"""

from __future__ import annotations

from .predictions import PredictionSet, load_manifest

DATASET = "HumeAI/ASR-benchmark-optimization-predictions"


def _api(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            'the Hub helpers need huggingface_hub: pip install "asr-benchmark-optimization[hub]"'
        ) from e
    return HfApi(token=token)


def available(token: str | None = None) -> list[str]:
    """Corpora with published predictions."""
    files = _api(token).list_repo_files(DATASET, repo_type="dataset")
    return sorted({f.split("/")[1] for f in files if f.startswith("predictions/")})


def load_predictions(
    corpus: str,
    models: list[str] | None = None,
    *,
    language: str = "en",
    token: str | None = None,
) -> PredictionSet:
    """Download and load one corpus into a :class:`PredictionSet`.

    ``models`` restricts the download; by default every model published for the
    corpus is fetched. Files are cached by ``huggingface_hub``, so repeat calls
    do not re-download.
    """
    from huggingface_hub import hf_hub_download

    api = _api(token)
    prefix = f"predictions/{corpus}/"
    names = [f for f in api.list_repo_files(DATASET, repo_type="dataset") if f.startswith(prefix)]
    if not names:
        raise ValueError(f"no published predictions for {corpus!r}; try available()")
    if models is not None:
        wanted = set(models)
        names = [f for f in names if f.rsplit("/", 1)[-1].removesuffix(".jsonl") in wanted]

    ps = PredictionSet(language=language)
    for name in sorted(names):
        path = hf_hub_download(DATASET, name, repo_type="dataset", token=token)
        load_manifest(path, name.rsplit("/", 1)[-1].removesuffix(".jsonl"), into=ps, language=language)
    return ps


def load_masks(variant: str, token: str | None = None):
    """Download one mask recipe as a DataFrame.

    ``entity_t0``/``entity_t1`` are the silenced span in seconds and
    ``hidden_ref`` the removed text. Zeroing that span in the source clip
    reproduces the masked audio.
    """
    import polars as pl
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET, f"masks/{variant}.parquet", repo_type="dataset", token=token
    )
    return pl.read_parquet(path)
