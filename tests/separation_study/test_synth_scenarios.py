from pathlib import Path

import yaml


def test_make_compose_config_S0_returns_none():
    """S0 ist die existierende Default-Synth-Datei — Generator gibt None zurück."""
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S0", base_drone_artifact="/dev/null",
        mic_geom_xml="/dev/null", out_synth_h5="...", out_gt_h5="...",
    )
    assert cfg is None


def test_make_compose_config_S1_offaxis_target():
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S1", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    assert tuple(cfg["external"]["position_m"]) == (0.3, 0.0, -1.5)
    assert cfg["external"]["amplitude_db"] == 0.0
    assert cfg["include_drone"] is True


def test_make_compose_config_S2_low_snr_amplitude():
    """amp×0.3 entspricht 20·log10(0.3) ≈ -10.46 dB."""
    from analysis.separation_study.synth_scenarios import make_compose_config
    import math
    cfg = make_compose_config(
        scenario="S2", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    expected = 20.0 * math.log10(0.3)
    assert abs(cfg["external"]["amplitude_db"] - expected) < 1e-6


def test_make_compose_config_S3_far_distance():
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S3", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    assert tuple(cfg["external"]["position_m"]) == (0.0, 0.0, -3.0)


def test_make_compose_config_unknown_scenario_raises():
    from analysis.separation_study.synth_scenarios import make_compose_config
    import pytest
    with pytest.raises(ValueError, match="unknown scenario"):
        make_compose_config(
            scenario="S99", base_drone_artifact="/a.h5",
            mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
        )


def test_ensure_scenarios_caches(tmp_path, monkeypatch):
    """Bestehende h5 werden nicht neu erzeugt."""
    from analysis.separation_study import synth_scenarios as ss
    out_synth = tmp_path / "synth.h5"
    out_gt = tmp_path / "gt.h5"
    out_synth.write_bytes(b"0")
    out_gt.write_bytes(b"0")
    called = {"compose": False}
    def fake_compose(*a, **kw):
        called["compose"] = True
    monkeypatch.setattr(ss, "_run_compose", fake_compose)
    ss.ensure_scenario(
        scenario="S1", base_drone_artifact="/a.h5", mic_geom_xml="/m.xml",
        out_synth_h5=out_synth, out_gt_h5=out_gt,
    )
    assert called["compose"] is False
