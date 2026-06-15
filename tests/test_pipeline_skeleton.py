"""Tests for martymicfly.processing.pipeline (skeleton only)."""

from dataclasses import replace

import numpy as np

from martymicfly.processing.pipeline import PipelineContext, run_pipeline


class _DoubleStage:
    name = "double"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        return replace(ctx, time_data=ctx.time_data * 2.0)


class _LogMetadataStage:
    name = "log_meta"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        meta = dict(ctx.metadata)
        meta["log_meta_seen_shape"] = ctx.time_data.shape
        return replace(ctx, metadata=meta)


def _empty_ctx():
    return PipelineContext(
        time_data=np.ones((10, 2), dtype=np.float64),
        sample_rate=1000.0,
        rpm_per_esc={},
        mic_positions=np.zeros((2, 3), dtype=np.float64),
        per_motor_bpf=np.zeros((10, 2), dtype=np.float64),
        harm_matrix=np.zeros((10, 4), dtype=np.float64),
        metadata={},
    )


def test_run_pipeline_chains_stages_in_order():
    ctx = run_pipeline([_DoubleStage(), _LogMetadataStage()], _empty_ctx())
    np.testing.assert_allclose(ctx.time_data, 2.0)
    assert ctx.metadata["log_meta_seen_shape"] == (10, 2)


def test_run_pipeline_empty_stage_list_is_identity():
    ctx_in = _empty_ctx()
    ctx_out = run_pipeline([], ctx_in)
    assert ctx_out is ctx_in


def test_stage_registry_builds_notch_stage():
    from martymicfly.config import NotchStageConfig, PoleRadiusConfig, RotorConfig
    from martymicfly.processing.pipeline import build_stage
    cfg = NotchStageConfig(
        kind="notch",
        pole_radius=PoleRadiusConfig(mode="scalar", value=0.9994),
        multichannel=False,
        block_size=4096,
    )
    rotor = RotorConfig(n_blades=2, n_harmonics=4)
    stage = build_stage(cfg, rotor=rotor)
    assert stage.name == "notch"


def test_stage_registry_raises_on_unknown_kind():
    import pytest
    from martymicfly.processing.pipeline import build_stage

    class _Fake:
        kind = "definitely_not_a_real_stage"

    with pytest.raises(ValueError, match="no stage builder"):
        build_stage(_Fake())


def test_notch_builder_rejects_linear_mode():
    import pytest
    from martymicfly.config import NotchStageConfig, PoleRadiusConfig, RotorConfig
    from martymicfly.processing.pipeline import build_stage
    cfg = NotchStageConfig(
        kind="notch",
        pole_radius=PoleRadiusConfig(
            mode="linear", k_cover=1.5, margin_hz=5.0,
        ),
        multichannel=False,
        block_size=4096,
    )
    rotor = RotorConfig(n_blades=2, n_harmonics=4)
    with pytest.raises(NotImplementedError, match="linear"):
        build_stage(cfg, rotor=rotor)


def test_notch_builder_requires_rotor():
    import pytest
    from martymicfly.config import NotchStageConfig, PoleRadiusConfig
    from martymicfly.processing.pipeline import build_stage
    cfg = NotchStageConfig(
        kind="notch",
        pole_radius=PoleRadiusConfig(mode="scalar", value=0.99),
        multichannel=False,
        block_size=4096,
    )
    with pytest.raises(ValueError, match="rotor"):
        build_stage(cfg)
