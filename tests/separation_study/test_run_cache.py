"""Tests for run-cache configuration hashing and completion detection."""


def test_run_hash_deterministic():
    from analysis.separation_study.run_cache import config_hash
    cfg = {"a": 1, "b": [1, 2, 3]}
    assert config_hash(cfg) == config_hash(cfg)


def test_run_hash_changes_with_value():
    from analysis.separation_study.run_cache import config_hash
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_hash_dict_order_independent():
    from analysis.separation_study.run_cache import config_hash
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_run_already_complete_detects_metrics_json(tmp_path):
    from analysis.separation_study.run_cache import is_run_complete
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not is_run_complete(run_dir)
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "study_metrics.json").write_text("{}")
    assert is_run_complete(run_dir)
