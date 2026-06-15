"""Time-Domain-Subtraktion mixed - ext_only liefert drone-only-Signal,
ohne separate drone-only synth-Datei zu brauchen (Synth ist linear)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch


def drone_only_at_target_from_files(
    mixed_gt_h5: Path, ext_only_gt_h5: Path,
) -> tuple[np.ndarray, float]:
    """drone_only(t) = mixed_gt(t) - ext_only_gt(t).

    Beide Dateien müssen identische Sample-Rate, Länge und Anzahl Kanäle haben.
    Liefert (signal_(N, C), sample_rate).
    """
    with h5py.File(mixed_gt_h5, "r") as fm, h5py.File(ext_only_gt_h5, "r") as fe:
        mix = np.asarray(fm["time_data"][:], dtype=np.float64)
        ext = np.asarray(fe["time_data"][:], dtype=np.float64)
        fs_m = float(fm["time_data"].attrs["sample_freq"])
        fs_e = float(fe["time_data"].attrs["sample_freq"])
    if fs_m != fs_e:
        raise ValueError(f"sample_freq mismatch: mixed={fs_m}, ext={fs_e}")
    if mix.shape != ext.shape:
        raise ValueError(f"shape mismatch: mixed={mix.shape}, ext={ext.shape}")
    return mix - ext, fs_m


def welch_psd_at_target(
    mixed_gt_h5: Path, ext_only_gt_h5: Path,
    *, nperseg: int, noverlap: int, window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch-PSD des drone-only-Signals am Target. Channel 0 used."""
    drone, fs = drone_only_at_target_from_files(mixed_gt_h5, ext_only_gt_h5)
    f, psd = welch(
        drone[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        scaling="density",
    )
    return f, psd
