import numpy as np


def test_compare_mask_modes_runs_all_three_on_single_fit():
    from martymicfly.cli.compare_modes import compare_mask_modes
    from martymicfly.config import (
        ArrayFilterStageConfig, BandConfig, CleanScConfig,
        CsmConfig, DiagnosticGridConfig,
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
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.0, 0.0, -0.5),
        target_box_half_extent_m=(0.15, 0.15, 0.15),
        drone_box_center_m=(0.0, 0.0, 0.0),
        drone_box_half_extent_m=(0.3, 0.3, 0.2),
        rotor_z_tolerance_m=0.05,
        clean_sc=CleanScConfig(damp=0.6, n_iter=5),
    )
    report = compare_mask_modes(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        mic_positions=mic_pos,
        platform=synth["platform"],
        stage_cfg=cfg,
    )

    # Localization payload shape
    assert "localization" in report
    assert set(report["localization"]["bands"].keys()) == {"mid"}
    # All three modes evaluated
    assert set(report["per_mode"].keys()) == {"rotor_disc", "drone_box", "target_box"}
    # Per-mode metrics block has the band keys
    for m in report["per_mode"].values():
        assert "mid" in m["bands"]
        b = m["bands"]["mid"]
        for k in ("csm_trace_reduction_db", "target_psd_reduction_db",
                  "drone_power_share_db", "ground_truth"):
            assert k in b
        # No GT was supplied, so ground_truth is None
        assert b["ground_truth"] is None
        assert np.isfinite(b["csm_trace_reduction_db"])

    # The three modes must produce different csm_red values (different masks)
    reds = sorted({round(report["per_mode"][m]["bands"]["mid"]["csm_trace_reduction_db"], 6)
                   for m in ("rotor_disc", "drone_box", "target_box")})
    assert len(reds) >= 2  # at least two distinct values


def test_compare_mask_modes_subset_of_modes():
    from martymicfly.cli.compare_modes import compare_mask_modes
    from martymicfly.config import (
        ArrayFilterStageConfig, BandConfig, CleanScConfig,
        CsmConfig, DiagnosticGridConfig,
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
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.0, 0.0, -0.5),
        clean_sc=CleanScConfig(damp=0.6, n_iter=5),
    )
    report = compare_mask_modes(
        time_data=synth["time_data"], sample_rate=synth["sample_rate"],
        mic_positions=mic_pos, platform=synth["platform"], stage_cfg=cfg,
        modes=("target_box", "rotor_disc"),
    )
    assert set(report["per_mode"].keys()) == {"target_box", "rotor_disc"}
