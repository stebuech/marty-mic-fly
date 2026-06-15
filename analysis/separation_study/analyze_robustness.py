"""Robustness-Analyse: wie wirkt S0→S1/S2/S3 auf empfohlene Config?

Pivotiert das robustness.parquet auf (method, scenario, band, cfg) und
berechnet pro Methode/Band die Degradation gegen S0 sowie den Effekt von
elevation_step=3 vs baseline.

Outputs:
  - robustness_table.csv: full pivot
  - robustness_degradation.csv: Δ(scenario − S0) für key metrics
  - robustness_el3_effect.csv: Δ(el3 − baseline) je Szenario
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_METRICS = [
    "spectrum_l1_db",
    "over_subtraction_db",
    "drone_leakage_db_def2",
    "recovery_db_signed",
]


def _cfg_label(overrides_json: str) -> str:
    o = json.loads(overrides_json)
    return "el3" if o.get("array_filter.doa_grid.elevation_step_deg") == 3.0 else "baseline"


def build_pivot(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df[df["metric"].isin(KEY_METRICS)].copy()
    df["cfg"] = df["overrides"].apply(_cfg_label)
    df["method"] = df["method"].str.replace("pipeline_mixed_doa_", "", regex=False)
    piv = df.pivot_table(
        index=["method", "scenario", "band", "cfg"],
        columns="metric", values="value",
    ).reset_index()
    return piv


def degradation_vs_s0(piv: pd.DataFrame) -> pd.DataFrame:
    """Für jede (method, band, cfg) berechne Δ = scenario - S0 in den key metrics."""
    s0 = piv[piv["scenario"] == "S0"].drop(columns=["scenario"])
    s0 = s0.rename(columns={m: f"{m}_S0" for m in KEY_METRICS})
    other = piv[piv["scenario"] != "S0"]
    merged = other.merge(s0, on=["method", "band", "cfg"])
    for m in KEY_METRICS:
        merged[f"d_{m}"] = merged[m] - merged[f"{m}_S0"]
    cols = ["method", "scenario", "band", "cfg"] + [f"d_{m}" for m in KEY_METRICS]
    return merged[cols].sort_values(["method", "band", "scenario", "cfg"])


def el3_effect(piv: pd.DataFrame) -> pd.DataFrame:
    """Δ = el3 - baseline für jede (method, scenario, band)."""
    base = piv[piv["cfg"] == "baseline"].drop(columns=["cfg"])
    base = base.rename(columns={m: f"{m}_base" for m in KEY_METRICS})
    el3 = piv[piv["cfg"] == "el3"].drop(columns=["cfg"])
    merged = el3.merge(base, on=["method", "scenario", "band"])
    for m in KEY_METRICS:
        merged[f"d_{m}"] = merged[m] - merged[f"{m}_base"]
    cols = ["method", "scenario", "band"] + [f"d_{m}" for m in KEY_METRICS]
    return merged[cols].sort_values(["method", "scenario", "band"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    piv = build_pivot(args.parquet)
    piv.to_csv(args.out_dir / "robustness_table.csv", index=False, float_format="%.3f")
    degradation_vs_s0(piv).to_csv(
        args.out_dir / "robustness_degradation.csv", index=False, float_format="%.3f",
    )
    el3_effect(piv).to_csv(
        args.out_dir / "robustness_el3_effect.csv", index=False, float_format="%.3f",
    )
    print(f"wrote 3 CSVs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
