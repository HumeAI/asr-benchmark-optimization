import json

import pytest

from benchmark_optimization import predictions


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


LEADERBOARD_ROWS = [
    {"audio_filepath": "clip1", "duration": 3.0, "time": 0.1, "text": "Mr. Smith spoke", "pred_text": "mister Smith spoke"},
    {"audio_filepath": "clip2", "duration": 4.0, "time": 0.1, "text": "the report is adopted", "pred_text": "the report is adopted"},
]


def test_loads_open_asr_leaderboard_manifest(tmp_path):
    p = tmp_path / "whisper-large-v3.jsonl"
    _write_manifest(p, LEADERBOARD_ROWS)
    ps = predictions.load_manifest(p)
    assert ps.models == ["whisper-large-v3"]
    assert len(ps) == 2
    assert ps.refs["clip1"] == "Mr. Smith spoke"
    assert ps.hyps["clip1"]["whisper-large-v3"] == "mister Smith spoke"


def test_joins_models_on_clip_key(tmp_path):
    for name in ("model_a", "model_b"):
        _write_manifest(tmp_path / f"{name}.jsonl", LEADERBOARD_ROWS)
    ps = predictions.load_dir(tmp_path)
    assert ps.models == ["model_a", "model_b"]
    assert set(ps.hyps["clip1"]) == {"model_a", "model_b"}
    assert ps.coverage() == {"model_a": 2, "model_b": 2}


def test_missing_clip_is_absent_not_empty(tmp_path):
    # A partial run must not read as "transcribed nothing" — the probes treat
    # those differently.
    _write_manifest(tmp_path / "full.jsonl", LEADERBOARD_ROWS)
    _write_manifest(tmp_path / "partial.jsonl", LEADERBOARD_ROWS[:1])
    ps = predictions.load_dir(tmp_path)
    assert set(ps.hyps["clip2"]) == {"full"}
    assert ps.coverage() == {"full": 2, "partial": 1}
    # require_all drops the clip rather than inventing a hypothesis.
    assert len(list(ps.clips(require_all=True))) == 1


def test_null_hypothesis_row_is_skipped(tmp_path):
    rows = LEADERBOARD_ROWS + [{"audio_filepath": "clip3", "text": "x y z", "pred_text": None}]
    _write_manifest(tmp_path / "m.jsonl", rows)
    ps = predictions.load_manifest(tmp_path / "m.jsonl")
    assert "clip3" not in ps.refs


def test_alternate_column_names(tmp_path):
    rows = [{"id": "c1", "reference": "a b c", "hypothesis": "a b d"}]
    _write_manifest(tmp_path / "m.jsonl", rows)
    ps = predictions.load_manifest(tmp_path / "m.jsonl")
    assert ps.refs["c1"] == "a b c"
    assert ps.hyps["c1"]["m"] == "a b d"


def test_unrecognised_columns_raise_with_guidance(tmp_path):
    _write_manifest(tmp_path / "m.jsonl", [{"a": 1, "b": 2}])
    with pytest.raises(ValueError, match="ref_field"):
        predictions.load_manifest(tmp_path / "m.jsonl")


def test_reference_conflicts_are_recorded(tmp_path):
    _write_manifest(tmp_path / "a.jsonl", [{"audio_filepath": "c1", "text": "a b c", "pred_text": "a b c"}])
    _write_manifest(tmp_path / "b.jsonl", [{"audio_filepath": "c1", "text": "a b DIFFERENT", "pred_text": "a b c"}])
    ps = predictions.load_dir(tmp_path)
    assert ps.ref_conflicts == {"c1"}


def test_normalized_leaves_original_untouched(tmp_path):
    _write_manifest(tmp_path / "m.jsonl", LEADERBOARD_ROWS)
    ps = predictions.load_manifest(tmp_path / "m.jsonl")
    norm = ps.normalized()
    assert ps.refs["clip1"] == "Mr. Smith spoke"
    # Whichever normalizer is active, casing is folded and the objects differ.
    assert norm.refs["clip1"] != ps.refs["clip1"]
    assert norm.refs["clip1"] == norm.refs["clip1"].lower()


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predictions.load_dir(tmp_path)


def test_csv_source(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("audio_filepath,text,pred_text\nc1,a b c,a b d\n", encoding="utf-8")
    ps = predictions.load_manifest(p)
    assert ps.hyps["c1"]["m"] == "a b d"
