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
