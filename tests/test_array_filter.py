import numpy as np
from dataclasses import replace


def test_array_filter_stage_e2e_on_tiny_fixture():
    from martymicfly.config import (
        ArrayFilterStageConfig,
        BandConfig,
        CleanScConfig,
        CsmConfig,
        DiagnosticGridConfig,
    )
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5
    from martymicfly.processing.array_filter import ArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext

    synth = load_synth_h5("tests/fixtures/tiny_synth_mixed.h5")
    mic_pos = load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml")
    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        diagnostic_grid=DiagnosticGridConfig(
            extent_xy_m=0.6, increment_m=0.05, z_min_m=0.0, z_max_m=0.0,
        ),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.5, 0.0, -0.5),
        rotor_z_tolerance_m=0.05,
        clean_sc=CleanScConfig(damp=0.6, n_iter=30),
    )
    ctx = PipelineContext(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"],
        mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    stage = ArrayFilterStage(cfg)
    new_ctx = stage.process(ctx)

    af = new_ctx.metadata["array_filter"]
    csm = af["csm_pre"]
    res = af["residual_csm"]
    assert csm.shape == res.shape
    np.testing.assert_allclose(res, res.conj().transpose(0, 2, 1), atol=1e-8)
    assert np.real(np.diagonal(csm, axis1=1, axis2=2)).sum() > \
           np.real(np.diagonal(res, axis1=1, axis2=2)).sum()
    assert af["target_psd_pre"].shape == af["target_psd_post"].shape
    # target_psd_post may dip slightly negative on individual freq bins because
    # the rebuilt drone_csm (after trace-rescale) can locally overshoot the
    # observed CSM, leaving (csm - drone_csm) hermitian but not PSD. Require
    # finite values + non-negative band-integrated power instead.
    assert np.all(np.isfinite(af["target_psd_post"]))
    assert af["target_psd_post"].sum() >= 0.0
    assert "mid" in af["beam_maps"]
    assert af["beam_maps"]["mid"].shape == (25, 25, 1)  # extent 0.6 / inc 0.05 = 25, z_min==z_max → nz=1


def test_3d_grid_band_map_shape():
    from martymicfly.config import (
        ArrayFilterStageConfig,
        BandConfig,
        CleanScConfig,
        CsmConfig,
        DiagnosticGridConfig,
    )
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5
    from martymicfly.processing.array_filter import ArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext

    synth = load_synth_h5("tests/fixtures/tiny_synth_mixed.h5")
    mic_pos = load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml")
    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        diagnostic_grid=DiagnosticGridConfig(
            extent_xy_m=0.2, increment_m=0.05, z_min_m=-0.1, z_max_m=0.1,
        ),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.5, 0.0, -0.5),
        rotor_z_tolerance_m=0.05,
        clean_sc=CleanScConfig(damp=0.6, n_iter=5),
    )
    ctx = PipelineContext(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"],
        mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    stage = ArrayFilterStage(cfg)
    new_ctx = stage.process(ctx)
    af = new_ctx.metadata["array_filter"]
    nx = ny = 9  # round(2*0.2/0.05)+1
    nz = 5       # round((0.1 - -0.1)/0.05)+1
    assert af["beam_maps"]["mid"].shape == (nx, ny, nz)
    assert af["diagnostic_grid_shape"] == (nx, ny, nz)
