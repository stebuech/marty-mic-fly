"""Free-space Green's function propagator.

For each source-mic pair, the source signal is delayed by r/c·fs samples
(fractional delay via FFT phase rotation) and attenuated by 1/(4π·r), then
summed across sources.
"""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def _fractional_delay_fft(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a fractional delay via phase rotation in the FFT domain.

    signal: (N,) real. delay_samples: float (positive = delay).
    Returns (N,) real. Periodic wrap is acceptable because the synthesis is
    longer than the longest path.
    """
    n = signal.shape[0]
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles/sample
    phase = np.exp(-2j * np.pi * freqs * delay_samples)
    return np.fft.irfft(spec * phase, n=n)


def greens_propagate(
    source_signals: np.ndarray,    # (S, N)
    source_positions: np.ndarray,  # (S, 3)
    mic_positions: np.ndarray,     # (M, 3)
    sample_rate: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Propagate S point sources to M mics, return (N, M)."""
    src = np.asarray(source_signals, dtype=np.float64)
    src_pos = np.asarray(source_positions, dtype=np.float64)
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if src.ndim != 2 or src.shape[0] != src_pos.shape[0]:
        raise ValueError("source_signals shape must be (S, N) matching source_positions (S, 3)")
    n_samples = src.shape[1]
    out = np.zeros((n_samples, mic_pos.shape[0]), dtype=np.float64)
    for s in range(src.shape[0]):
        for m in range(mic_pos.shape[0]):
            r = float(np.linalg.norm(mic_pos[m] - src_pos[s]))
            if r < 1e-9:
                shifted = src[s].copy()
                amp = 1.0
            else:
                delay = r / speed_of_sound * sample_rate
                shifted = _fractional_delay_fft(src[s], delay)
                amp = 1.0 / (4.0 * np.pi * r)
            out[:, m] += amp * shifted
    return out
