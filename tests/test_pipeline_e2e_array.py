"""End-to-end: notch + array_filter pipeline on the tiny composed fixture."""

import json
from pathlib import Path


def test_pipeline_e2e_notch_plus_array_filter(tmp_path):
    from martymicfly.cli.run_pipeline import main
    fx = Path("tests/fixtures")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
input:
  audio_h5: {fx / 'tiny_synth_mixed.h5'}
  mic_geom_xml: {fx / 'tiny_geom_4mic.xml'}
  ground_truth_h5: {fx / 'tiny_gt.h5'}
segment: {{mode: head, duration: 1.0}}
channels: {{selection: all}}
rotor: {{n_blades: 2, n_harmonics: 4}}
stages:
  - kind: notch
    pole_radius: {{mode: scalar, value: 0.99}}
    multichannel: false
    block_size: 1024
  - kind: array_filter
    csm: {{nperseg: 256, noverlap: 128, f_min_hz: 200.0, f_max_hz: 4000.0}}
    diagnostic_grid: {{extent_xy_m: 0.6, increment_m: 0.05, z_m: 0.0}}
    bands:
      - {{name: mid, f_min_hz: 500.0, f_max_hz: 2000.0}}
    target_point_m: [0.5, 0.0, -0.5]
    rotor_z_tolerance_m: 0.05
    clean_sc: {{damp: 0.6, n_iter: 30}}
metrics: {{welch_nperseg: 1024, welch_noverlap: 512}}
plots: {{enabled: false, spectrogram_window: 512, spectrogram_overlap: 256}}
output: {{dir: {tmp_path / 'out'}}}
""")
    rc = main(["--config", str(cfg)])
    assert rc == 0
    runs = list((tmp_path / "out").glob("*"))
    assert len(runs) == 1
    run_dir = runs[0]
    for name in ("filtered.h5", "residual_csm.h5", "beam_maps.html",
                 "target_psd.html", "metrics.json",
                 "stage1_metrics.csv", "stage2_metrics.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "stage1_notch" in metrics
    assert "stage2_array_filter" in metrics
    band = metrics["stage2_array_filter"]["bands"]["mid"]
    assert band["csm_trace_reduction_db"] > 0.0
    assert band["ground_truth"] is not None
