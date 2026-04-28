"""NotchStage — RPM-driven cascade notch filtering."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from notchfilter.cascade import CascadeNotchFilter

from .pipeline import PipelineContext, register_stage_builder
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


def _build_notch_stage(cfg, *, rotor=None, **_) -> "NotchStage":
    """Translate a pydantic NotchStageConfig (YAML stage entry) plus the
    top-level rotor block into a runtime NotchStage. The rotor block carries
    n_blades and n_harmonics, which the YAML stage entry does not."""
    if rotor is None:
        raise ValueError(
            "_build_notch_stage requires rotor=RotorConfig(...) — pass it via "
            "build_pipeline(cfg.stages, rotor=cfg.rotor) at the CLI boundary"
        )

    if cfg.pole_radius.mode == "scalar":
        pole_radius = float(cfg.pole_radius.value)
    else:  # 'linear'
        raise NotImplementedError(
            "pole_radius.mode='linear' is not resolvable at stage-construction time "
            "because the per-harmonic schedule depends on runtime per_motor_bpf. "
            "Use mode='scalar' until linear-mode deferred resolution lands "
            "(see plan Task 19 / handoff)."
        )

    return NotchStage(NotchStageParams(
        n_blades=rotor.n_blades,
        n_harmonics=rotor.n_harmonics,
        pole_radius=pole_radius,
        multichannel=cfg.multichannel,
        block_size=cfg.block_size,
    ))


register_stage_builder("notch", _build_notch_stage)
