import numpy as np
import pytest


def _common_synth():
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5
    return (
        load_synth_h5("tests/fixtures/tiny_synth_mixed.h5"),
        load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml"),
    )


def _doa_cfg(mask_mode="target_cone", **overrides):
    from martymicfly.config import (
        ArrayFilterStageConfig, BandConfig, CleanScConfig,
        CsmConfig, DiagnosticGridConfig, DoaGridConfig,
    )
    base = dict(
        kind="array_filter",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        # diagnostic_grid is required by the schema even when doa_grid takes
        # over — the cfg is shared across both stages.
        diagnostic_grid=DiagnosticGridConfig(
            extent_xy_m=0.4, increment_m=0.1, z_min_m=-0.5, z_max_m=0.0,
        ),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.0, 0.0, -0.5),
        mask_mode=mask_mode,
        clean_sc=CleanScConfig(damp=0.6, n_iter=10),
        doa_grid=DoaGridConfig(
            focal_radius_m=0.5, azimuth_step_deg=20.0, elevation_step_deg=20.0,
            hemisphere="lower",
            rotor_cone_half_angle_deg=30.0,
            target_cone_half_angle_deg=30.0,
            drone_disk_half_width_deg=15.0,
        ),
    )
    base.update(overrides)
    return ArrayFilterStageConfig(**base)


def test_doa_stage_factory_dispatch():
    """Setting cfg.doa_grid routes the factory to DoaArrayFilterStage."""
    from martymicfly.processing.array_filter import _array_filter_factory
    from martymicfly.processing.array_filter_doa import DoaArrayFilterStage
    cfg = _doa_cfg()
    stage = _array_filter_factory(cfg)
    assert isinstance(stage, DoaArrayFilterStage)


def test_doa_stage_e2e_target_cone_runs():
    from martymicfly.processing.array_filter_doa import DoaArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext

    synth, mic_pos = _common_synth()
    cfg = _doa_cfg("target_cone")
    ctx = PipelineContext(
        time_data=synth["time_data"], sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"], mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    new_ctx = DoaArrayFilterStage(cfg).process(ctx)
    af = new_ctx.metadata["array_filter"]

    # Schema: same keys as the cartesian stage
    for k in ("csm_pre", "residual_csm", "frequencies", "source_map",
              "drone_mask", "mask_mode", "beam_maps",
              "target_psd_pre", "target_psd_post",
              "diagnostic_grid", "diagnostic_grid_shape"):
        assert k in af, f"missing key {k!r}"

    csm = af["csm_pre"]
    res = af["residual_csm"]
    assert csm.shape == res.shape
    np.testing.assert_allclose(res, res.conj().transpose(0, 2, 1), atol=1e-8)
    assert np.all(np.isfinite(af["target_psd_post"]))
    assert af["target_psd_post"].sum() >= 0.0

    # DOA-specific masks present + cartesian aliases populated
    for k in ("rotor_cone_mask", "target_cone_mask", "drone_cone_mask",
              "rotor_disc_mask", "target_box_mask", "drone_box_mask"):
        assert k in af and af[k].dtype == bool

    # reshape_hint is (n_az, n_el, 1) — last dim 1
    assert af["diagnostic_grid_shape"][2] == 1
    # All grid points lie on the focal sphere
    radii = np.linalg.norm(af["diagnostic_grid"], axis=1)
    np.testing.assert_allclose(radii, 0.5, atol=1e-9)


def test_doa_stage_rejects_box_mask_modes():
    from martymicfly.processing.array_filter_doa import DoaArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext
    synth, mic_pos = _common_synth()
    cfg = _doa_cfg("target_box")  # box mode + doa_grid is invalid
    ctx = PipelineContext(
        time_data=synth["time_data"], sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"], mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    with pytest.raises(ValueError, match="cone"):
        DoaArrayFilterStage(cfg).process(ctx)


def test_doa_stage_three_mask_modes_yield_distinct_residuals():
    from martymicfly.processing.array_filter_doa import DoaArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext
    synth, mic_pos = _common_synth()

    def run(mode):
        cfg = _doa_cfg(mode)
        ctx = PipelineContext(
            time_data=synth["time_data"], sample_rate=synth["sample_rate"],
            rpm_per_esc=synth["rpm_per_esc"], mic_positions=mic_pos,
            per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
            harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
            metadata={"platform": synth["platform"]},
        )
        return DoaArrayFilterStage(cfg).process(ctx).metadata["array_filter"]

    af_t = run("target_cone")
    af_r = run("rotor_cone")
    af_d = run("drone_cone")

    assert not np.allclose(af_t["residual_csm"], af_r["residual_csm"])
    assert not np.allclose(af_t["residual_csm"], af_d["residual_csm"])
    # rotor_cone vs drone_cone may coincide if the rotor positions and
    # cone half-angles produce the same cones; tolerate that case but
    # require *some* mode pair to differ (already established above).
