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

    # Normalise L1 by the dominant component (max of excess vs deficit) so that
    # a symmetric over/under-subtraction reads as > 0 dB (factor 2 = +3 dB) and
    # a pure one-sided subtraction reads as 0 dB.  Returns -inf when both are 0.
    _l1_max = max(e_excess, e_deficit)
    spectrum_l1_db = _db((e_excess + e_deficit) / _l1_max) if _l1_max > 0 else float("-inf")
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
