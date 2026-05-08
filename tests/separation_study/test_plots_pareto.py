import pandas as pd


def test_pareto_plot_writes_html(tmp_path):
    from analysis.separation_study.plots.pareto_plot import write_pareto
    df = pd.DataFrame([
        {"axis_value": 0.3, "over_subtraction_db": -10.0,
         "drone_leakage_db_def2": -25.0, "method": "rotor_cone"},
        {"axis_value": 0.6, "over_subtraction_db": -15.0,
         "drone_leakage_db_def2": -30.0, "method": "rotor_cone"},
        {"axis_value": 0.9, "over_subtraction_db": -8.0,
         "drone_leakage_db_def2": -22.0, "method": "rotor_cone"},
    ])
    out = tmp_path / "pareto.html"
    write_pareto(df, axis_label="clean_sc.damp", out_path=out,
                welch_floor_db=-40.0, band="mid")
    assert out.exists()
    assert out.stat().st_size > 1000
