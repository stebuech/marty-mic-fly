"""Conventional-beamformer steering helper for the pseudo-target-PSD output."""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def steer_to_psd_matched(
    csm: np.ndarray,             # (F, M, M)
    frequencies: np.ndarray,     # (F,)
    mic_positions: np.ndarray,   # (M, 3)
    target_point: tuple[float, float, float],
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Range-corrected (matched-filter) steerer:
    h_m = (4π·r_m) · exp(+j·2π f r_m / c).

    Given scipy's csd convention CSM[i,j] = E[X_i* · X_j] = G_i* · G_j · S_q
    (with G_m = exp(-jω r_m/c) / (4π r_m) being the free-field propagator
    from target to mic m), the quadratic form ``(1/M²) · h^H · CSM · h``
    yields exactly S_q for any single point source at the target.  The
    (4π·r_m) factor inverts the propagation attenuation, recovering the
    source PSD in proper Pa²/Hz at unit reference — no separate
    range_compensation_factor needed.

    The phase sign matches the existing `steer_to_psd` (also ``+jω r/c``);
    that sign is what cancels the propagation phase under scipy's csd
    convention.  An opposite sign would introduce a frequency-dependent
    geometric ripple of the form |Σ_m e^{+2jω r_m/c}|².
    """
    target = np.asarray(target_point, dtype=np.float64)
    diff = mic_positions - target[None, :]
    r = np.linalg.norm(diff, axis=1)         # (M,)
    n_f = frequencies.shape[0]
    n_m = mic_positions.shape[0]
    psd = np.zeros(n_f, dtype=np.float64)
    amp = 4.0 * np.pi * r                     # (M,)
    for fi, f in enumerate(frequencies):
        h = amp * np.exp(2j * np.pi * f * r / speed_of_sound)   # (M,)
        val = np.real(h.conj() @ csm[fi] @ h) / (n_m * n_m)
        psd[fi] = float(val)
    return psd


def range_compensation_factor(
    mic_positions: np.ndarray,                # (M, 3)
    target_point: tuple[float, float, float],
) -> float:
    """Inverse of the geometric+Greens factor introduced by phase-only DAS.

    The pipeline propagates with the physical free-field Greens 1/(4π·r);
    `steer_to_psd` is phase-only with 1/M² normalization, missing the
    amplitude restoration.  Under scipy's csd convention CSM[i,j] = G_i*·G_j·S_q
    the propagation phase cancels exactly in ``h^H·CSM·h`` (no ripple), and
    the result becomes the constant ``S_q · (⟨1/r_m⟩/(4π))²``.  Multiplying
    the steered PSD by ``(4π/⟨1/r_m⟩)²`` recovers ``S_q`` in proper Pa²/Hz at
    unit reference.  Equivalent to (but cheaper than) using
    `steer_to_psd_matched`, which folds the same factor into the steerer.
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
