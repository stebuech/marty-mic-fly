"""Conventional-beamformer steering helper for the pseudo-target-PSD output."""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def steer_to_psd(
    csm: np.ndarray,             # (F, M, M)
    frequencies: np.ndarray,     # (F,)
    mic_positions: np.ndarray,   # (M, 3)
    target_point: tuple[float, float, float],
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """PSD = (1/M^2) · h^H · csm · h, with h[m] = exp(j 2π f r_m / c) (no
    1/r weighting; pure phase steering — what conventional delay-and-sum
    delivers when integrated over the mic aperture)."""
    target = np.asarray(target_point, dtype=np.float64)
    diff = mic_positions - target[None, :]
    r = np.linalg.norm(diff, axis=1)         # (M,)
    n_f = frequencies.shape[0]
    n_m = mic_positions.shape[0]
    psd = np.zeros(n_f, dtype=np.float64)
    for fi, f in enumerate(frequencies):
        h = np.exp(2j * np.pi * f * r / speed_of_sound)   # (M,)
        # quadratic form
        val = np.real(h.conj() @ csm[fi] @ h) / (n_m * n_m)
        psd[fi] = float(val)
    return psd
