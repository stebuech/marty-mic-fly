"""Pipeline skeleton: PipelineContext + Stage protocol + run_pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from martymicfly.config import StageConfig


@dataclass
class PipelineContext:
    """State passed between stages."""

    time_data: np.ndarray            # (N, C) float64 — current signal
    sample_rate: float
    rpm_per_esc: dict
    mic_positions: np.ndarray        # (M, 3)
    per_motor_bpf: np.ndarray        # (N, S)
    harm_matrix: np.ndarray          # (N, S*M)
    metadata: dict = field(default_factory=dict)


class Stage(Protocol):
    """Pipeline stage contract.

    Implementations consume a :class:`PipelineContext` and return a new one
    (typically via :func:`dataclasses.replace`). They MUST NOT mutate the
    input ctx in place.
    """

    name: str

    def process(self, ctx: PipelineContext) -> PipelineContext: ...


def run_pipeline(stages: list[Stage], ctx: PipelineContext) -> PipelineContext:
    """Run stages in order, threading the context through."""
    for stage in stages:
        ctx = stage.process(ctx)
    return ctx


_STAGE_BUILDERS: dict[str, Callable[..., "Stage"]] = {}


def register_stage_builder(kind: str, builder: Callable[..., "Stage"]) -> None:
    if kind in _STAGE_BUILDERS:
        raise ValueError(f"stage builder already registered for kind={kind!r}")
    _STAGE_BUILDERS[kind] = builder


def build_stage(stage_cfg: StageConfig, **kwargs) -> "Stage":
    """Look up the registered builder for stage_cfg.kind and call it with
    stage_cfg plus any additional kwargs. kwargs let callers pass the
    rotor/runtime context that some builders need (e.g., notch needs
    n_blades/n_harmonics from RotorConfig)."""
    if stage_cfg.kind not in _STAGE_BUILDERS:
        raise ValueError(
            f"no stage builder registered for kind={stage_cfg.kind!r}; "
            f"available: {sorted(_STAGE_BUILDERS)}"
        )
    return _STAGE_BUILDERS[stage_cfg.kind](stage_cfg, **kwargs)


def build_pipeline(stages_cfg: list[StageConfig], **kwargs) -> list["Stage"]:
    return [build_stage(s, **kwargs) for s in stages_cfg]
