import pandas as pd


def test_interaction_plot_writes_html(tmp_path):
    from analysis.separation_study.plots.interaction_plot import write_interaction
    df = pd.DataFrame([
        {"axis_a_value": 0.3, "axis_b_value": 50, "spectrum_l1_db": -10.0},
        {"axis_a_value": 0.3, "axis_b_value": 200, "spectrum_l1_db": -12.0},
        {"axis_a_value": 0.9, "axis_b_value": 50, "spectrum_l1_db": -8.0},
        {"axis_a_value": 0.9, "axis_b_value": 200, "spectrum_l1_db": -15.0},
    ])
    out = tmp_path / "ia.html"
    write_interaction(df, axis_a="clean_sc.damp", axis_b="clean_sc.n_iter",
                     metric="spectrum_l1_db", out_path=out)
    assert out.exists()
