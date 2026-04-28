"""Algorithm protocol, SourceMap dataclass, default reconstruct_csm."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Protocol

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


@dataclass(frozen=True)
class SourceMap:
    positions: np.ndarray            # (G, 3) grid points
    powers: np.ndarray               # (F, G) p^2
    frequencies: np.ndarray          # (F,) Hz
    grid_shape: Optional[tuple[int, int]]
    metadata: dict

    def subset(self, mask: np.ndarray) -> "SourceMap":
        return replace(
            self,
            positions=self.positions[mask],
            powers=self.powers[:, mask],
            grid_shape=None,
        )


class Algorithm(Protocol):
    name: str
    consumes: Literal["csm", "time"]

    def fit(
        self,
        *,
        csm: Optional[np.ndarray],
        frequencies: Optional[np.ndarray],
        time_data: Optional[np.ndarray],
        sample_rate: float,
        mic_positions: np.ndarray,
        grid_positions: np.ndarray,
        params: dict,
    ) -> SourceMap: ...


def reconstruct_csm(
    source_map: SourceMap,
    mic_positions: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """drone_csm[f] = sum_g power[f, g] · h[f, g] h[f, g]^H
    h[f, g] = exp(-j 2π f r_mg / c) / (4π r_mg)
    """
    positions = source_map.positions      # (G, 3)
    powers = source_map.powers            # (F, G)
    freqs = source_map.frequencies        # (F,)
    n_f = freqs.shape[0]
    n_m = mic_positions.shape[0]

    # Distances (M, G)
    diff = mic_positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r < 1e-9, 1e-9, r)
    inv_r = 1.0 / (4.0 * np.pi * r)        # (M, G)

    csm = np.zeros((n_f, n_m, n_m), dtype=np.complex128)
    for fi, f in enumerate(freqs):
        phase = np.exp(-2j * np.pi * f * r / speed_of_sound)   # (M, G)
        h = inv_r * phase                                       # (M, G)
        # csm[fi] = h * diag(powers[fi]) * h^H
        weighted = h * powers[fi][None, :]                      # (M, G)
        csm[fi] = weighted @ h.conj().T                         # (M, M)
    return csm
