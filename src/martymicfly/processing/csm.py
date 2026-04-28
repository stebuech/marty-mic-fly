"""Welch-style cross-spectral matrix from multi-channel time data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import csd


@dataclass
class CsmConfig:
    nperseg: int = 512
    noverlap: int = 256
    window: str = "hann"
    diag_loading_rel: float = 1e-6
    f_min_hz: float = 200.0
    f_max_hz: float = 6000.0


def build_measurement_csm(
    time_data: np.ndarray,         # (N, M)
    sample_rate: float,
    cfg: CsmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (csm, freqs) where csm[f] is the (M, M) hermitian CSM at freqs[f].

    Frequencies outside [f_min_hz, f_max_hz] are dropped. Diagonal loading
    is applied as `cfg.diag_loading_rel * max(|diag|)` added to the identity.
    """
    if time_data.ndim != 2:
        raise ValueError("time_data must be 2D (N, M)")
    n_total, n_ch = time_data.shape

    f, _ = csd(time_data[:, 0], time_data[:, 0],
               fs=sample_rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
               window=cfg.window, scaling="density")
    mask = (f >= cfg.f_min_hz) & (f <= cfg.f_max_hz)
    freqs = f[mask]
    n_f = freqs.shape[0]
    csm = np.zeros((n_f, n_ch, n_ch), dtype=np.complex128)
    for i in range(n_ch):
        for j in range(i, n_ch):
            _, c_ij = csd(time_data[:, i], time_data[:, j],
                           fs=sample_rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
                           window=cfg.window, scaling="density")
            c_ij = c_ij[mask]
            csm[:, i, j] = c_ij
            if i != j:
                csm[:, j, i] = np.conj(c_ij)

    if cfg.diag_loading_rel > 0.0:
        diag_mag = np.abs(np.diagonal(csm, axis1=1, axis2=2))
        peak = float(diag_mag.max()) if diag_mag.size else 0.0
        load = cfg.diag_loading_rel * peak
        eye = np.eye(n_ch, dtype=np.complex128)
        csm = csm + load * eye[None, :, :]

    return csm, freqs
