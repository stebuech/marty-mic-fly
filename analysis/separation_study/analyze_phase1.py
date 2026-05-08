"""Phase-1 sensitivity analysis: compute_sensitivity + heatmap + dominant-axes ranking.

Usage:
    uv run python -m analysis.separation_study.analyze_phase1 \
        --parquet results/separation_study/phase1_sensitivity/phase1_sensitivity.parquet \
        --welch-floor results/separation_study/welch_floor.json \
        --out-dir results/separation_study/phase1_sensitivity
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from analysis.separation_study.sensitivity import (
    compute_sensitivity, top_axes_per_metric,
)
from analysis.separation_study.plots.sensitivity_heatmap import write_heatmap

log = logging.getLogger("phase1_analysis")


# Metrics we want to track sensitivity on (dB-domain only — exclude raw power)
SENSITIVITY_METRICS = (
    "spectrum_l1_db", "over_subtraction_db",
    "drone_leakage_db_def1", "drone_leakage_db_def2",
    "spectrum_rms_db", "recovery_db_signed",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--welch-floor", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.parquet)
    log.info("loaded %d rows from %s", len(df), args.parquet)

    floor = json.loads(args.welch_floor.read_text())
    # Conservative threshold: min over all bands of the spectrum_l1_db floor
    floor_l1 = floor.get("spectrum_l1_db", {})
    if floor_l1:
        welch_floor_db = abs(min(floor_l1.values()))
    else:
        welch_floor_db = 0.5
    log.info("welch_floor_db (min over bands of |spectrum_l1_db|) = %.2f", welch_floor_db)

    # Filter to only metrics we want for sensitivity (drop raw powers)
    sens_df_in = df[df["metric"].isin(SENSITIVITY_METRICS) & df["value"].apply(
        lambda v: pd.notna(v) and v not in (float("inf"), float("-inf"))
    )]

    sens = compute_sensitivity(sens_df_in, welch_floor_db=welch_floor_db)
    log.info("computed sensitivity for %d (method, scenario, band, metric, axis) entries",
             len(sens))

    out_csv = args.out_dir / "phase1_sensitivity_table.csv"
    sens.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    top = top_axes_per_metric(sens, k=3)
    out_top = args.out_dir / "phase1_dominant_axes.csv"
    top.to_csv(out_top, index=False)
    log.info("wrote %s", out_top)

    # Heatmap per band
    for band in ("low", "mid", "high"):
        out_html = args.out_dir / f"phase1_heatmap_{band}.html"
        write_heatmap(sens, out_path=out_html, band=band)
        log.info("wrote %s", out_html)

    # Print summary: top-5 axes by mean sensitivity over all (method, metric, band)
    print("\n=== Phase-1 Sensitivity Summary ===\n")

    # Per-metric: which axes dominate?
    for metric in SENSITIVITY_METRICS:
        m_sens = sens[sens["metric"] == metric]
        if m_sens.empty:
            continue
        axis_avg = m_sens.groupby("axis")["sensitivity_db"].mean().sort_values(
            ascending=False)
        print(f"\n{metric} — axes by mean |Δ| over (method × band):")
        for axis, sens_val in axis_avg.head(5).items():
            short = axis.replace("array_filter.", "")
            print(f"  {short:42s}  {sens_val:+6.2f} dB")

    # Top dominant flag count per axis
    print("\n=== Dominant-Flag Count per Axis (over all metric × band × method) ===")
    dom = sens.groupby("axis")["dominant_flag"].sum().sort_values(ascending=False)
    for axis, count in dom.head(10).items():
        short = axis.replace("array_filter.", "")
        print(f"  {short:42s}  {int(count):3d} dominant rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
