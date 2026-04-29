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


def rescale_source_map_to_csm_trace(
    source_map: SourceMap,
    csm_observed: np.ndarray,
    mic_positions: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> tuple[SourceMap, np.ndarray]:
    """Re-scale source_map.powers per-frequency so reconstruct_csm matches the
    observed CSM trace. Acoular's BeamformerCleansc.synthetic with steer_type
    'classic' returns powers off by a deterministic, geometry-dependent factor
    (the array gain Σ_m|h_target,m|²). Trace-matching corrects to Pa².

    Returns (calibrated_source_map, per_frequency_factors).
    """
    # CLEAN-SC may emit NaN/negative powers in degenerate cells; sanitize before
    # rescaling so downstream reconstruct_csm and CSM subtraction stay finite.
    sanitized_powers = np.where(
        np.isfinite(source_map.powers), source_map.powers, 0.0,
    )
    sanitized_powers = np.maximum(sanitized_powers, 0.0)
    sm_clean = replace(source_map, powers=sanitized_powers)
    csm_rebuilt = reconstruct_csm(sm_clean, mic_positions, speed_of_sound)
    n_f = sm_clean.frequencies.shape[0]
    factors = np.zeros(n_f, dtype=np.float64)
    for fi in range(n_f):
        obs = float(np.real(np.trace(csm_observed[fi])))
        rec = float(np.real(np.trace(csm_rebuilt[fi])))
        factors[fi] = obs / rec if rec > 1e-30 else 0.0
    new_powers = sm_clean.powers * factors[:, None]
    return replace(sm_clean, powers=new_powers), factors
