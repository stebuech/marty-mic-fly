import json
from pathlib import Path


def test_aggregate_long_format(tmp_path):
    from analysis.separation_study.aggregate_results import aggregate_dir
    for name, val in [("run_a", 1.0), ("run_b", 2.0)]:
        d = tmp_path / name
        d.mkdir()
        (d / "study_metrics.json").write_text(json.dumps({
            "bands": {"mid": {"spectrum_l1_db": val,
                              "drone_leakage_db_def2": val * 2,
                              "compensation_flag": False}},
        }))
        (d / "run_meta.json").write_text(json.dumps({
            "run_id": name, "config": "configs/test.yaml",
            "scenario": "S0", "overrides": {}, "method": "test",
        }))
    df = aggregate_dir(tmp_path)
    assert {"run_id", "method", "scenario", "band", "metric", "value"} \
        <= set(df.columns)
    assert (df["metric"] == "spectrum_l1_db").any()


def test_consistency_check_warns(tmp_path, caplog):
    from analysis.separation_study.aggregate_results import (
        aggregate_dir, check_consistency,
    )
    d = tmp_path / "run_x"
    d.mkdir()
    (d / "study_metrics.json").write_text(json.dumps({
        "bands": {"mid": {
            "spectrum_l1_db": -10.0, "over_subtraction_db": -20.0,
            "drone_leakage_db_def1": -5.0, "drone_leakage_db_def2": -25.0,
            "spectrum_rms_db": 1.0, "recovery_db_signed": -5.0,
            "e_excess": 0.001, "e_deficit": 0.0001, "e_gt": 1.0,
            "d_unflt": 100.0, "p_post": 0.5,
            "compensation_flag": False,
        }},
    }))
    (d / "run_meta.json").write_text(json.dumps({
        "run_id": "run_x", "config": "x.yaml", "scenario": "S0",
        "overrides": {}, "method": "test",
    }))
    df = aggregate_dir(tmp_path)
    with caplog.at_level("WARNING"):
        check_consistency(df)
    assert any("consistency" in r.message.lower() for r in caplog.records)
