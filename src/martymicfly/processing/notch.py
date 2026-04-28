"""NotchStage — RPM-driven cascade notch filtering."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from notchfilter.cascade import CascadeNotchFilter

from .pipeline import PipelineContext
from .sources import ArrayFreqSource, ArraySamplesGenerator


@dataclass
class NotchStageParams:
    n_blades: int
    n_harmonics: int
    pole_radius: float | np.ndarray
    multichannel: bool = False
    block_size: int = 4096


class NotchStage:
    """Wrap CascadeNotchFilter (mode='external', zero_phase=True) as a Stage."""

    name = "notch"

    def __init__(self, cfg: NotchStageParams) -> None:
        self.cfg = cfg

    def process(self, ctx: PipelineContext) -> PipelineContext:
        signal = ctx.time_data
        n, c = signal.shape
        S = ctx.per_motor_bpf.shape[1]
        M = self.cfg.n_harmonics

        f_inits = np.array(
            [[ctx.per_motor_bpf[0, s] * h for h in range(1, M + 1)] for s in range(S)],
            dtype=np.float64,
        )

        if self.cfg.multichannel:
            filtered = self._run_cascade(signal, ctx.sample_rate, ctx.harm_matrix,
                                         f_inits, S, M)
        else:
            filtered = np.empty_like(signal)
            for ch in range(c):
                filtered[:, ch:ch + 1] = self._run_cascade(
                    signal[:, ch:ch + 1], ctx.sample_rate, ctx.harm_matrix,
                    f_inits, S, M,
                )

        new_meta = dict(ctx.metadata)
        new_meta["pre_notch"] = signal
        return replace(ctx, time_data=filtered, metadata=new_meta)

    def _run_cascade(self, signal: np.ndarray, fs: float, harm_matrix: np.ndarray,
                     f_inits: np.ndarray, S: int, M: int) -> np.ndarray:
        cascade = CascadeNotchFilter(
            num_sources=S,
            harmonics_per_source=M,
            frequencies=f_inits,
            pole_radius=self.cfg.pole_radius,
            mode="external",
            zero_phase=True,
            source=ArraySamplesGenerator(signal, fs),
            freq_source=ArrayFreqSource(harm_matrix),
        )
        return np.vstack(list(cascade.result(self.cfg.block_size)))


from martymicfly.processing.pipeline import register_stage_builder


def _build_notch_stage(cfg) -> "NotchStage":
    """Translate a pydantic NotchStageConfig into the runtime NotchStageParams
    and instantiate the NotchStage. Called by the stage registry.

    Note: the pydantic NotchStageConfig does not carry n_blades / n_harmonics
    (those live on RotorConfig and are wired in by run_pipeline.py in Task 4).
    We seed sensible placeholders here; the CLI is expected to override them
    before invoking process(), or to construct NotchStageParams directly.
    Likewise pole_radius arrives as a PoleRadiusConfig: scalar mode is resolved
    here; linear mode is left for Task 19's resolver and falls back to r_max.
    """
    pr_cfg = cfg.pole_radius
    if pr_cfg.mode == "scalar":
        pole_radius: float = float(pr_cfg.value)
    else:
        # Linear mode resolution requires per-harmonic BPF info not yet in
        # scope. Fall back to r_max so the stage instantiates; Task 19 will
        # replace this with a proper resolver.
        pole_radius = float(pr_cfg.r_max)

    return NotchStage(NotchStageParams(
        n_blades=1,
        n_harmonics=1,
        pole_radius=pole_radius,
        multichannel=cfg.multichannel,
        block_size=cfg.block_size,
    ))


register_stage_builder("notch", _build_notch_stage)
