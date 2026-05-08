import pandas as pd


def test_sensitivity_heatmap_writes_html(tmp_path):
    from analysis.separation_study.plots.sensitivity_heatmap import write_heatmap
    sens = pd.DataFrame([
        {"method": "rotor_cone", "axis": "csm.nperseg", "metric": "spectrum_l1_db",
         "band": "mid", "sensitivity_db": 1.5, "dominant_flag": True},
        {"method": "rotor_cone", "axis": "clean_sc.damp", "metric": "spectrum_l1_db",
         "band": "mid", "sensitivity_db": 0.2, "dominant_flag": False},
    ])
    out = tmp_path / "heatmap.html"
    write_heatmap(sens, out_path=out, band="mid")
    assert out.exists()
    assert out.stat().st_size > 1000
