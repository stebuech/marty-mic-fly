"""RPM-to-BPF interpolation, harmonic matrix construction, r-schedule."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def interpolate_per_motor_bpf(
    rpm_per_esc: dict[str, dict[str, np.ndarray]],
    t_start: float,
    n_samples: int,
    sample_rate: float,
    n_blades: int,
) -> np.ndarray:
    """Interpolate ESC RPM onto an audio sample grid, convert to BPF.

    Parameters
    ----------
    rpm_per_esc : dict
        Returned by :func:`martymicfly.io.synth_h5.load_synth_h5`.
    t_start : float
        Start time of the audio segment in seconds (audio time = ESC time
        for synth data; no sync offset).
    n_samples : int
        Audio samples in the segment.
    sample_rate : float
        Audio sample rate in Hz.
    n_blades : int
        Blades per rotor (BPF = mechanical_rpm * n_blades / 60).

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, S)``, ESCs in alphabetical order.
    """
    esc_names = sorted(rpm_per_esc.keys())
    audio_times = np.arange(n_samples) / sample_rate + t_start
    out = np.empty((n_samples, len(esc_names)), dtype=np.float64)
    for s, name in enumerate(esc_names):
        esc = rpm_per_esc[name]
        f = interp1d(
            esc["timestamp"], esc["rpm"],
            kind="linear", fill_value="extrapolate", assume_sorted=True,
        )
        out[:, s] = f(audio_times) * n_blades / 60.0
    return out


def build_harmonic_matrix(per_motor_bpf: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Build ``(N, S*M)`` harmonic frequency matrix.

    Column index ``s*M + (h-1)`` for source ``s`` (0-based) and harmonic ``h``
    (1-based). Identical convention as
    ``notchfilter.cascade.CascadeNotchFilter`` and
    ``NotchFilter/validate/run_filter_comparison.py:build_harmonic_matrix``.
    """
    n_samples, n_sources = per_motor_bpf.shape
    out = np.empty((n_samples, n_sources * n_harmonics), dtype=np.float64)
    for s in range(n_sources):
        for h in range(1, n_harmonics + 1):
            out[:, s * n_harmonics + (h - 1)] = h * per_motor_bpf[:, s]
    return out


def linear_r_schedule(
    n_harmonics: int,
    fs: float,
    delta_bpf_hz: float,
    k_cover: float,
    margin_hz: float,
    r_min: float = 0.90,
    r_max: float = 0.9995,
) -> np.ndarray:
    """Return ``(M,)`` per-harmonic pole radii.

    ``BW(h) = k_cover * h * delta_bpf_hz + margin_hz``,
    ``r(h) = clip(1 - BW(h) * pi / fs, r_min, r_max)``.
    """
    h = np.arange(1, n_harmonics + 1, dtype=np.float64)
    bw = k_cover * h * delta_bpf_hz + margin_hz
    return np.clip(1.0 - bw * np.pi / fs, r_min, r_max)
