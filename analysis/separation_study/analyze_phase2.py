"""Phase-2 sweep analysis: pareto plots per axis + identify Pareto-optimal points.

Usage:
    uv run python -m analysis.separation_study.analyze_phase2 \
        --parquet results/separation_study/phase2_focused_sweep/phase2_sweep.parquet \
        --welch-floor results/separation_study/welch_floor.json \
        --out-dir results/separation_study/phase2_focused_sweep
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from analysis.separation_study.plots.pareto_plot import write_pareto

log = logging.getLogger("phase2_analysis")


def _parse_axis(overrides_json: str) -> tuple[str | None, float | None]:
    ov = json.loads(overrides_json)
    if not ov:
        return (None, None)
    axis = next(iter(ov.keys()))
    val = ov[axis]
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = float("nan")
    return (axis, val)


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
    floor_l1 = floor.get("spectrum_l1_db", {})

    df[["axis", "axis_value"]] = df["overrides"].apply(
        lambda s: pd.Series(_parse_axis(s)))

    # For each axis × band, pivot to get over_subtraction_db × drone_leakage_db_def2
    # then write Pareto plot.
    for axis, axis_df in df[df["axis"].notna()].groupby("axis"):
        for band, band_df in axis_df.groupby("band"):
            pivot = (band_df[band_df["metric"].isin([
                "over_subtraction_db", "drone_leakage_db_def2"])]
                .pivot_table(index="axis_value", columns="metric",
                            values="value", aggfunc="first")
                .reset_index())
            if pivot.empty or len(pivot) < 2:
                continue
            # Replace -inf with min finite - 5 dB for plotting
            for col in ("over_subtraction_db", "drone_leakage_db_def2"):
                if col not in pivot.columns:
                    continue
                vals = pivot[col]
                finite = vals[vals != float("-inf")]
                if not finite.empty:
                    floor_val = float(finite.min()) - 5.0
                    pivot[col] = pivot[col].replace(
                        -float("inf"), floor_val)
            pivot["method"] = "doa_target_cone"
            short_axis = axis.replace("array_filter.", "")
            out = args.out_dir / f"pareto_{short_axis.replace('.', '_')}_{band}.html"
            write_pareto(pivot, axis_label=short_axis, out_path=out,
                        welch_floor_db=floor_l1.get(band, -8.0), band=band)
            log.info("wrote %s", out)

    # Print best (Pareto-optimum) per axis × band: minimize spectrum_l1_db
    print("\n=== Phase-2 Best Configs per Axis (lowest spectrum_l1_db) ===\n")
    l1 = df[df["metric"] == "spectrum_l1_db"]
    for (axis, band), grp in l1.groupby(["axis", "band"]):
        if pd.isna(axis):
            continue
        best = grp.loc[grp["value"].idxmin()]
        short = axis.replace("array_filter.", "")
        print(f"  {short:38s}  band={band:5s}  best={best['axis_value']!s:>8s}  l1={best['value']:+6.2f} dB")

    print("\n=== Per-Axis Sweep Results (mid band, lowest l1_db is best) ===")
    for axis in sorted(df["axis"].dropna().unique()):
        ax_df = df[(df["axis"] == axis) & (df["band"] == "mid")
                  & (df["metric"].isin(["spectrum_l1_db", "over_subtraction_db",
                                       "drone_leakage_db_def2", "recovery_db_signed"]))]
        if ax_df.empty:
            continue
        pivot = ax_df.pivot_table(index="axis_value", columns="metric",
                                  values="value", aggfunc="first")
        if pivot.empty:
            continue
        short = axis.replace("array_filter.", "")
        print(f"\n  {short} — mid band:")
        for v, row in pivot.iterrows():
            l1_v = row.get("spectrum_l1_db", float("nan"))
            ov_v = row.get("over_subtraction_db", float("nan"))
            d2_v = row.get("drone_leakage_db_def2", float("nan"))
            rs_v = row.get("recovery_db_signed", float("nan"))
            print(f"    {v!s:>8s}  l1={l1_v:+6.2f}  over={ov_v:+6.2f}"
                  f"  def2={d2_v:+6.2f}  rec={rs_v:+6.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
