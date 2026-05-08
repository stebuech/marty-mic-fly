"""Berechnet Sensitivities aus dem Phase-1-Aggregate.

sensitivity = max(|metric(low) - metric(baseline)|, |metric(high) - metric(baseline)|).
dominant_flag wird gesetzt wenn sensitivity > 3 · welch_floor_db."""
from __future__ import annotations

import json
import logging

import pandas as pd

log = logging.getLogger("separation_study.sensitivity")


def _axis_from_overrides(ov_json: str) -> str | None:
    ov = json.loads(ov_json)
    if not ov:
        return None
    if len(ov) != 1:
        raise ValueError(f"expected single-axis override, got {ov}")
    return next(iter(ov.keys()))


def compute_sensitivity(
    df: pd.DataFrame, *,
    baseline_overrides_json: str = "{}",
    welch_floor_db: float = 0.1,
) -> pd.DataFrame:
    """Pro (method, scenario, band, metric, axis): sensitivity_db + dominant_flag."""
    out_rows: list[dict] = []
    df = df.copy()
    df["axis"] = df["overrides"].map(_axis_from_overrides)
    for (method, scenario, band, metric), grp in df.groupby(
        ["method", "scenario", "band", "metric"], dropna=False,
    ):
        baseline_rows = grp[grp["overrides"] == baseline_overrides_json]
        if baseline_rows.empty:
            continue
        baseline = float(baseline_rows["value"].iloc[0])
        for axis, ax_grp in grp[grp["axis"].notna()].groupby("axis"):
            deltas = (ax_grp["value"] - baseline).abs()
            sens = float(deltas.max())
            out_rows.append({
                "method": method, "scenario": scenario, "band": band,
                "metric": metric, "axis": axis,
                "baseline": baseline,
                "n_axis_points": len(ax_grp),
                "sensitivity_db": sens,
                "dominant_flag": bool(sens > 3.0 * welch_floor_db),
            })
    return pd.DataFrame(out_rows)


def top_axes_per_metric(sens_df: pd.DataFrame, *, k: int = 3) -> pd.DataFrame:
    """Pro (method, metric, band) die top-k Achsen nach sensitivity_db."""
    return (sens_df.sort_values("sensitivity_db", ascending=False)
                  .groupby(["method", "metric", "band"], as_index=False)
                  .head(k))
