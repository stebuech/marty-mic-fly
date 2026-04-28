"""Stage-2 metrics: CSM-trace reductions, drone power share, target PSD,
and (when ground-truth is supplied) external recovery / spectrum MAE."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _db(x: float, eps: float = 1e-30) -> float:
    return float(10.0 * np.log10(max(x, eps)))


def _band_mask(freqs, f_min, f_max):
    return (freqs >= f_min) & (freqs <= f_max)


def compute_array_metrics(
    *,
    csm_pre: np.ndarray,                  # (F, M, M)
    residual_csm: np.ndarray,             # (F, M, M)
    frequencies: np.ndarray,              # (F,)
    psd_pre: np.ndarray,                  # (F,)
    psd_post: np.ndarray,                 # (F,)
    source_map_powers: np.ndarray,        # (F, G)
    drone_mask: np.ndarray,               # (G,) bool
    bands: list,                          # list of dict or BandConfig
    ground_truth: Optional[dict] = None,  # {"psd_at_target": (F,), "frequencies": (F,)}
) -> dict:
    metrics: dict = {"bands": {}, "global": {}}

    trace_pre = np.real(np.diagonal(csm_pre, axis1=1, axis2=2)).sum(axis=1)         # (F,)
    trace_post = np.real(np.diagonal(residual_csm, axis1=1, axis2=2)).sum(axis=1)   # (F,)

    drone_power_per_freq = source_map_powers[:, drone_mask].sum(axis=1)
    total_power_per_freq = source_map_powers.sum(axis=1)

    for band in bands:
        if isinstance(band, dict):
            name, f_lo, f_hi = band["name"], band["f_min_hz"], band["f_max_hz"]
        else:
            name, f_lo, f_hi = band.name, band.f_min_hz, band.f_max_hz
        mask = _band_mask(frequencies, f_lo, f_hi)
        if not mask.any():
            continue
        tr_pre = float(trace_pre[mask].sum())
        tr_post = float(trace_post[mask].sum())
        psd_pre_band = float(psd_pre[mask].sum())
        psd_post_band = float(psd_post[mask].sum())
        share = float(drone_power_per_freq[mask].sum() /
                      max(total_power_per_freq[mask].sum(), 1e-30))

        gt_block = None
        if ground_truth is not None:
            gt_psd = np.asarray(ground_truth["psd_at_target"])
            gt_freq = np.asarray(ground_truth["frequencies"])
            gt_mask = (gt_freq >= f_lo) & (gt_freq <= f_hi)
            gt_band = float(gt_psd[gt_mask].sum())
            recovery_db = _db(psd_post_band) - _db(gt_band)
            spectrum_mae_db = float(np.mean(np.abs(
                10 * np.log10(np.maximum(psd_post[mask], 1e-30)) -
                10 * np.log10(np.maximum(gt_psd[gt_mask], 1e-30))
            )))
            gt_block = {
                "external_recovery_db": recovery_db,
                "spectrum_mae_db": spectrum_mae_db,
            }

        metrics["bands"][name] = {
            "csm_trace_pre_db": _db(tr_pre),
            "csm_trace_post_db": _db(tr_post),
            "csm_trace_reduction_db": _db(tr_pre) - _db(tr_post),
            "target_psd_pre_db": _db(psd_pre_band),
            "target_psd_post_db": _db(psd_post_band),
            "target_psd_reduction_db": _db(psd_pre_band) - _db(psd_post_band),
            "drone_power_share_db": _db(share),
            "ground_truth": gt_block,
        }

    metrics["global"] = {
        "drone_power_share_total_db": _db(
            float(drone_power_per_freq.sum() / max(total_power_per_freq.sum(), 1e-30))
        ),
    }
    return metrics
