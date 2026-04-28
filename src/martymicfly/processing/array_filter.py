"""Stage 2 — array deconvolution stage. Body filled in Task 21."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from martymicfly.processing.algorithms.base import SourceMap


@dataclass
class BandConfig:
    name: str
    f_min_hz: float
    f_max_hz: float


def integrate_band_maps(
    source_map: SourceMap,
    bands: list[BandConfig],
    grid_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    nx, ny = grid_shape
    out: dict[str, np.ndarray] = {}
    f = source_map.frequencies
    for band in bands:
        mask = (f >= band.f_min_hz) & (f <= band.f_max_hz)
        if not mask.any():
            out[band.name] = np.zeros((nx, ny), dtype=np.float64)
            continue
        integrated = source_map.powers[mask].sum(axis=0)        # (G,)
        if integrated.size != nx * ny:
            raise ValueError(
                f"powers length {integrated.size} doesn't match grid_shape {grid_shape}"
            )
        out[band.name] = integrated.reshape(nx, ny)
    return out
