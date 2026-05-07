"""Scaling-Ladder Diagnostic.

Diagnoses the ~−27 dB external_recovery bias observed in ext_only smoke runs
by following an analytically-defined white-noise monopole through three
points in the steering chain (mic-PSD, CSM-diagonal, steered PSD).

Run via:  uv run python analysis/scaling_ladder.py [--mic-geom PATH]
"""
from __future__ import annotations

import numpy as np


def propagate_white_noise(
    *,
    n_samples: int,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,   # (M, 3)
    rng: np.random.Generator,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Propagate one white-noise monopole to M mics via free-field 1/r-Greens.

    The source signal q(t) is white noise with two-sided PSD = s_q_pa2_per_hz
    (one-sided density as used by `scipy.welch(..., scaling='density')`).
    Each mic receives `p_m(t) = q(t − r_m/c) / r_m` via fractional-sample
    delay (linear interpolation in time domain).

    Returns
    -------
    time_data : (n_samples, M) float64
    """
    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)         # (M,)

    # White noise of length n_samples + max-delay-margin so we can shift safely.
    max_delay_samples = int(np.ceil(r.max() / speed_of_sound * sample_rate)) + 4
    n_pad = n_samples + max_delay_samples

    # Generate q with the right one-sided PSD.
    # For a real Gaussian process with two-sided PSD = s_two_sided,
    # variance = s_two_sided · fs. Using one-sided density convention:
    # s_one_sided = 2 · s_two_sided (for f > 0), so variance = (s_one_sided/2) · fs.
    sigma = float(np.sqrt(s_q_pa2_per_hz * sample_rate / 2.0))
    q = rng.normal(0.0, sigma, size=n_pad)

    # Per-mic fractional delay via linear interpolation.
    n = np.arange(n_samples, dtype=np.float64)
    out = np.zeros((n_samples, mics.shape[0]), dtype=np.float64)
    for m in range(mics.shape[0]):
        delay_samples = r[m] / speed_of_sound * sample_rate
        # Source-time index for output sample n: t_src = n - delay
        # but we generated q starting at -max_delay (offset), so:
        t_src = n + (max_delay_samples - delay_samples)
        i0 = np.floor(t_src).astype(np.int64)
        frac = t_src - i0
        out[:, m] = (1.0 - frac) * q[i0] + frac * q[i0 + 1]
        out[:, m] /= r[m]
    return out
