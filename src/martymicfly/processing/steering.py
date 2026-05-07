"""Conventional-beamformer steering helper for the pseudo-target-PSD output."""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def range_compensation_factor(
    mic_positions: np.ndarray,                # (M, 3)
    target_point: tuple[float, float, float],
) -> float:
    """Inverse of the geometric+Greens factor introduced by phase-only DAS.

    The pipeline propagates with the physical free-field Greens 1/(4π·r);
    `steer_to_psd` is phase-only with 1/M² normalization and does not restore
    the amplitude.  For a monopole at ``target_point`` with source PSD ``S_q``
    the steered output approaches ``S_q · (⟨1/r_m⟩/(4π))²`` when the array
    aperture is small relative to the source distance.  Multiplying the
    steered PSD by this factor recovers ``S_q`` (proper Pa²/Hz at the source).
    """
    mics = np.asarray(mic_positions, dtype=np.float64)
    tgt = np.asarray(target_point, dtype=np.float64)
    r = np.linalg.norm(mics - tgt[None, :], axis=1)
    return float((4.0 * np.pi / (1.0 / r).mean()) ** 2)


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
