import pandas as pd


def test_compute_sensitivity_max_abs():
    from analysis.separation_study.sensitivity import compute_sensitivity
    rows = [
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -20.0, "overrides": "{}", "run_id": "baseline"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -18.0, "overrides": '{"a.b": 1}', "run_id": "low"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -25.0, "overrides": '{"a.b": 2}', "run_id": "high"},
    ]
    df = pd.DataFrame(rows)
    out = compute_sensitivity(df, baseline_overrides_json="{}")
    row = out.iloc[0]
    assert row["axis"] == "a.b"
    assert abs(row["sensitivity_db"] - 5.0) < 1e-9


def test_dominant_flag_uses_threshold():
    from analysis.separation_study.sensitivity import compute_sensitivity
    rows = [
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -20.0, "overrides": "{}", "run_id": "b"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -19.95, "overrides": '{"x.y": 1}', "run_id": "low"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -22.0, "overrides": '{"a.b": 2}', "run_id": "high"},
    ]
    df = pd.DataFrame(rows)
    out = compute_sensitivity(df, baseline_overrides_json="{}",
                             welch_floor_db=0.1)
    a_b = out[out["axis"] == "a.b"].iloc[0]
    x_y = out[out["axis"] == "x.y"].iloc[0]
    assert a_b["dominant_flag"] == True
    assert x_y["dominant_flag"] == False
