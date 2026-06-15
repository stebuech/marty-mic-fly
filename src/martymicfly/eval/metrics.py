"""Compute tonal and broadband energy metrics pre/post notch filtering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import welch


def _bw_from_pole_radius(pole_radius: float | np.ndarray, fs: float,
                        n_harmonics: int) -> np.ndarray:
    """Per-harmonic bandwidth in Hz: BW = (1 - r) * fs / pi."""
    if np.isscalar(pole_radius):
        r = np.full(n_harmonics, float(pole_radius))
    else:
        r = np.asarray(pole_radius, dtype=np.float64)
        if r.ndim == 1 and r.shape[0] == n_harmonics:
            pass
        elif r.ndim == 2:
            # (S, M) — collapse to per-harmonic by mean across S
            r = r.mean(axis=0)
        else:
            raise ValueError(
                f"pole_radius shape {r.shape} not understood for n_harmonics={n_harmonics}"
            )
    return (1.0 - r) * fs / np.pi


def _band_energy(f: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    if f_hi <= f_lo:
        return 0.0
    mask = (f >= f_lo) & (f <= f_hi)
    if not mask.any():
        # Schmale Bänder kleiner als Welch-Bin — ein Bin mitnehmen
        idx = np.argmin(np.abs(f - 0.5 * (f_lo + f_hi)))
        df = f[1] - f[0] if f.size > 1 else 1.0
        return float(psd[idx] * df)
    return float(np.trapz(psd[mask], f[mask]))


def _to_db(p: float, ref: float = 1.0) -> float:
    return 10.0 * np.log10(max(p, 1e-30) / ref ** 2)


def compute_metrics(
    *,
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    per_motor_bpf: np.ndarray,
    n_harmonics: int,
    pole_radius: float | np.ndarray,
    channels: Iterable[int],
    welch_nperseg: int,
    welch_noverlap: int,
    bandwidth_factor: float,
    broadband_low_hz: float | None,
    fmax_hz: float,
) -> dict:
    """Compute pre/post tonal and broadband metrics. dB are ref=1 (units²)."""
    channels = list(channels)
    n, c = pre.shape
    assert post.shape == pre.shape
    S = per_motor_bpf.shape[1]
    bw_per_h = _bw_from_pole_radius(pole_radius, fs, n_harmonics) * bandwidth_factor
    f_per_sh = np.array(
        [[per_motor_bpf[:, s].mean() * h for h in range(1, n_harmonics + 1)]
         for s in range(S)],
        dtype=np.float64,
    )

    if broadband_low_hz is None:
        broadband_low_hz = 0.5 * float(per_motor_bpf.min())

    esc_names = [f"ESC{s+1}" for s in range(S)]  # bookkeeping label only

    out_channels: list[dict] = []
    for ch in channels:
        f_pre, psd_pre = welch(pre[:, ch], fs=fs, window="hann",
                               nperseg=welch_nperseg, noverlap=welch_noverlap,
                               scaling="density")
        f_post, psd_post = welch(post[:, ch], fs=fs, window="hann",
                                 nperseg=welch_nperseg, noverlap=welch_noverlap,
                                 scaling="density")

        tonal_per_h: list[dict] = []
        tonal_pre_sum = 0.0
        tonal_post_sum = 0.0
        for s in range(S):
            for h in range(1, n_harmonics + 1):
                f_target = float(f_per_sh[s, h - 1])
                half = 0.5 * float(bw_per_h[h - 1])
                e_pre = _band_energy(f_pre, psd_pre, f_target - half, f_target + half)
                e_post = _band_energy(f_post, psd_post, f_target - half, f_target + half)
                tonal_pre_sum += e_pre
                tonal_post_sum += e_post
                tonal_per_h.append({
                    "motor": esc_names[s],
                    "h": h,
                    "f_hz": f_target,
                    "pre_db": _to_db(e_pre),
                    "post_db": _to_db(e_post),
                    "delta_db": _to_db(e_post) - _to_db(e_pre),
                })

        # Broadband = total energy in [low, fmax] minus all tonal bands
        full_pre = _band_energy(f_pre, psd_pre, broadband_low_hz, fmax_hz)
        full_post = _band_energy(f_post, psd_post, broadband_low_hz, fmax_hz)
        broad_pre = max(full_pre - tonal_pre_sum, 0.0)
        broad_post = max(full_post - tonal_post_sum, 0.0)

        out_channels.append({
            "channel": int(ch),
            "broadband_pre_db": _to_db(broad_pre),
            "broadband_post_db": _to_db(broad_post),
            "broadband_delta_db": _to_db(broad_post) - _to_db(broad_pre),
            "tonal_total_pre_db": _to_db(tonal_pre_sum),
            "tonal_total_post_db": _to_db(tonal_post_sum),
            "tonal_reduction_db": _to_db(tonal_pre_sum) - _to_db(tonal_post_sum),
            "tonal_per_harmonic": tonal_per_h,
        })

    return {
        "sample_rate": fs,
        "n_motors": S,
        "n_harmonics": n_harmonics,
        "per_motor_bpf_summary": [
            {
                "motor": esc_names[s],
                "bpf_min_hz": float(per_motor_bpf[:, s].min()),
                "bpf_max_hz": float(per_motor_bpf[:, s].max()),
                "bpf_mean_hz": float(per_motor_bpf[:, s].mean()),
            }
            for s in range(S)
        ],
        "broadband_low_hz": float(broadband_low_hz),
        "fmax_hz": float(fmax_hz),
        "db_reference": "1 (units squared) — relative, not calibrated to Pa",
        "channels": out_channels,
    }
