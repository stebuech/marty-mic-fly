import numpy as np


def test_probe_clean_sc_localization_returns_per_band_stats():
    from martymicfly.config import (
        ArrayFilterStageConfig,
        BandConfig,
        CleanScConfig,
        CsmConfig,
        DiagnosticGridConfig,
    )
    from martymicfly.eval.localization_probe import (
        format_localization_report,
        probe_clean_sc_localization,
    )
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5

    synth = load_synth_h5("tests/fixtures/tiny_synth_mixed.h5")
    mic_pos = load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml")
    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        diagnostic_grid=DiagnosticGridConfig(
            extent_xy_m=0.4, increment_m=0.1, z_min_m=-0.5, z_max_m=0.0,
        ),
        bands=[
            BandConfig(name="lo", f_min_hz=300.0, f_max_hz=900.0),
            BandConfig(name="hi", f_min_hz=900.0, f_max_hz=3500.0),
        ],
        target_point_m=(0.0, 0.0, -0.5),
        clean_sc=CleanScConfig(damp=0.6, n_iter=10),
    )
    report = probe_clean_sc_localization(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        mic_positions=mic_pos,
        platform=synth["platform"],
        stage_cfg=cfg,
        target_xyz_m=cfg.target_point_m,
    )

    assert set(report["bands"].keys()) == {"lo", "hi"}
    assert report["target_xyz_m"] == (0.0, 0.0, -0.5)
    assert len(report["grid_shape"]) == 3

    for name, b in report["bands"].items():
        assert len(b["peak_xyz_m"]) == 3
        assert len(b["z_span_10_90_m"]) == 2
        assert 0.0 <= b["frac_at_target_z"] <= 1.0
        assert 0.0 <= b["frac_at_drone_z"] <= 1.0
        assert b["total_power"] >= 0.0
        # peak_xyz_m must lie inside the configured grid extent
        if np.isfinite(b["peak_xyz_m"][0]):
            assert -0.4 <= b["peak_xyz_m"][0] <= 0.4
            assert -0.4 <= b["peak_xyz_m"][1] <= 0.4
            assert -0.5 <= b["peak_xyz_m"][2] <= 0.0

    text = format_localization_report(report)
    assert "lo" in text and "hi" in text
    assert "target=" in text
