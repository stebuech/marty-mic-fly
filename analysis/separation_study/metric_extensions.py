"""Kompensationsfreie Separation-Metriken für mixed-Studie.

Per-Bin-Zerlegung: excess(f) = max(psd_post - ext_GT, 0),
                   deficit(f) = max(ext_GT - psd_post, 0).
Bandintegrale dieser Komponenten verhindern, dass Over- und Undersubtraktion
in benachbarten Frequenzen sich gegenseitig auslöschen."""
from __future__ import annotations

import numpy as np


def decompose_residual(
    psd_post: np.ndarray, ext_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-Bin Zerlegung in (excess, deficit). Beide ≥ 0, an jedem f genau einer ≠ 0."""
    diff = psd_post - ext_gt
    excess = np.maximum(diff, 0.0)
    deficit = np.maximum(-diff, 0.0)
    return excess, deficit


def _db(x: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(x)


def band_metrics(
    psd_post: np.ndarray,
    ext_gt: np.ndarray,
    *,
    d_ref: np.ndarray,
    delta_f: float,
    welch_floor_db: float = -50.0,
) -> dict:
    """Alle 6 Studien-Metriken + compensation_flag für ein bereits gemaskeds Band.

    Argumente sind 1D-Arrays gleicher Länge (alle nach band-mask gefiltert).
    `delta_f` ist der Welch-Bin-Abstand (für Σ·Δf-Integration).
    `welch_floor_db` setzt die Schwelle für `compensation_flag`.
    """
    assert psd_post.shape == ext_gt.shape == d_ref.shape, "shape mismatch"
    excess, deficit = decompose_residual(psd_post, ext_gt)
    e_excess = float(excess.sum() * delta_f)
    e_deficit = float(deficit.sum() * delta_f)
    e_gt = float(ext_gt.sum() * delta_f)
    p_post = float(psd_post.sum() * delta_f)
    d_unflt = float(d_ref.sum() * delta_f)

    spectrum_l1_db = _db((e_excess + e_deficit) / e_gt) if e_gt > 0 else float("-inf")
    over_subtraction_db = _db(e_deficit / e_gt) if e_gt > 0 else float("-inf")
    drone_leakage_db_def1 = _db(p_post / d_unflt) if d_unflt > 0 else float("inf")
    drone_leakage_db_def2 = _db(e_excess / d_unflt) if d_unflt > 0 else float("inf")
    recovery_db_signed = _db(p_post / e_gt) if e_gt > 0 else float("-inf")

    safe_post = np.maximum(psd_post, 1e-30)
    safe_gt = np.maximum(ext_gt, 1e-30)
    log_ratio_per_bin = 10.0 * np.log10(safe_post / safe_gt)
    spectrum_rms_db = float(np.sqrt(np.mean(log_ratio_per_bin ** 2)))

    excess_db = _db(e_excess / e_gt) if e_gt > 0 else float("-inf")
    compensation_flag = (
        over_subtraction_db > welch_floor_db + 3.0
        and excess_db > welch_floor_db + 3.0
    )

    return {
        "spectrum_l1_db": spectrum_l1_db,
        "over_subtraction_db": over_subtraction_db,
        "drone_leakage_db_def1": drone_leakage_db_def1,
        "drone_leakage_db_def2": drone_leakage_db_def2,
        "spectrum_rms_db": spectrum_rms_db,
        "recovery_db_signed": recovery_db_signed,
        "compensation_flag": bool(compensation_flag),
        "e_excess": e_excess,
        "e_deficit": e_deficit,
        "e_gt": e_gt,
        "d_unflt": d_unflt,
        "p_post": p_post,
    }


import h5py
from pathlib import Path
from scipy.signal import welch as _scipy_welch


def _welch_signal_h5(path: Path, *, nperseg: int, noverlap: int, window: str,
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as f:
        td = np.asarray(f["time_data"][:], dtype=np.float64)
        fs = float(f["time_data"].attrs["sample_freq"])
    sig = td[:, 0] if td.ndim == 2 else td
    f_w, psd = _scipy_welch(sig, fs=fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, scaling="density")
    return f_w, psd, fs


def compute_run_metrics(
    *,
    psd_post: np.ndarray,
    frequencies: np.ndarray,
    ext_gt_h5: Path,
    mixed_gt_h5: Path,
    welch_nperseg: int,
    welch_noverlap: int,
    window: str,
    bands: list[dict],
    welch_floor_db: float,
) -> dict:
    """Erzeuge alle Studien-Metriken pro Band aus psd_post + GT-h5-Pfaden.

    `psd_post` und `frequencies` müssen konsistent sein (gleiche Welch-Parameter
    wie für ext_GT/D_ref). Aufrufer ist verantwortlich für `range_compensation_factor`-
    Anwendung auf psd_post."""
    f_ext, psd_ext_gt, fs_ext = _welch_signal_h5(
        ext_gt_h5, nperseg=welch_nperseg, noverlap=welch_noverlap, window=window,
    )
    f_mix, psd_mix_gt, fs_mix = _welch_signal_h5(
        mixed_gt_h5, nperseg=welch_nperseg, noverlap=welch_noverlap, window=window,
    )
    if not np.allclose(f_ext, f_mix):
        raise ValueError("freq grid mismatch ext_gt vs mixed_gt")
    psd_d_ref = psd_mix_gt - psd_ext_gt   # Linearität: PSD(mix) - PSD(ext) ≈ PSD(drone)
    # Für negative Werte aus Welch-Streuung clamped auf 0
    psd_d_ref = np.maximum(psd_d_ref, 0.0)

    # psd_post ist auf einem (i.d.R. groberen) Frequenzgitter — interpoliere
    # ext/d_ref auf das psd_post-Gitter.
    psd_ext_on_post = np.interp(frequencies, f_ext, psd_ext_gt)
    psd_d_on_post = np.interp(frequencies, f_ext, psd_d_ref)

    delta_f_post = float(frequencies[1] - frequencies[0])

    out: dict = {"bands": {}, "frequency_resolved": {
        "frequencies": frequencies,
        "psd_post": psd_post,
        "ext_gt": psd_ext_on_post,
        "d_ref": psd_d_on_post,
    }}
    for band in bands:
        name = band["name"]
        f_lo = float(band["f_min_hz"])
        f_hi = float(band["f_max_hz"])
        mask = (frequencies >= f_lo) & (frequencies <= f_hi)
        if not mask.any():
            continue
        m = band_metrics(
            psd_post=psd_post[mask],
            ext_gt=psd_ext_on_post[mask],
            d_ref=psd_d_on_post[mask],
            delta_f=delta_f_post,
            welch_floor_db=welch_floor_db,
        )
        out["bands"][name] = m
    return out
