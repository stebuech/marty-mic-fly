"""Aggregiert study_metrics.json + run_meta.json über result-dirs zu long-format
parquet."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("separation_study.aggregate")


SCALAR_METRICS = (
    "spectrum_l1_db", "over_subtraction_db",
    "drone_leakage_db_def1", "drone_leakage_db_def2",
    "spectrum_rms_db", "recovery_db_signed",
    "e_excess", "e_deficit", "e_gt", "d_unflt", "p_post",
)


def aggregate_dir(root: Path) -> pd.DataFrame:
    rows = []
    for sm in sorted(Path(root).rglob("study_metrics.json")):
        meta_path = sm.parent / "run_meta.json"
        if not meta_path.exists():
            log.warning("no run_meta.json next to %s — skipped", sm)
            continue
        meta = json.loads(meta_path.read_text())
        sm_data = json.loads(sm.read_text())
        for band, vals in sm_data.get("bands", {}).items():
            for m in SCALAR_METRICS:
                if m in vals:
                    rows.append({
                        **{k: meta.get(k) for k in
                           ("run_id", "method", "scenario", "config")},
                        "overrides": json.dumps(meta.get("overrides", {})),
                        "band": band,
                        "metric": m,
                        "value": float(vals[m]),
                    })
            if "compensation_flag" in vals:
                rows.append({
                    **{k: meta.get(k) for k in
                       ("run_id", "method", "scenario", "config")},
                    "overrides": json.dumps(meta.get("overrides", {})),
                    "band": band,
                    "metric": "compensation_flag",
                    "value": float(bool(vals["compensation_flag"])),
                })
    return pd.DataFrame(rows)


def check_consistency(df: pd.DataFrame) -> None:
    """Verifiziert: recovery_signed - drone_leakage_def1 ≈ -SNR_band (±0.5 dB)."""
    import numpy as np
    if df.empty:
        return
    pivot = df.pivot_table(index=["run_id", "band"],
                          columns="metric", values="value", aggfunc="first")
    needed = {"recovery_db_signed", "drone_leakage_db_def1", "e_gt", "d_unflt"}
    if not needed.issubset(pivot.columns):
        return
    snr = 10.0 * np.log10(pivot["e_gt"] / pivot["d_unflt"])
    sanity = pivot["recovery_db_signed"] - pivot["drone_leakage_db_def1"] + snr
    bad = sanity[abs(sanity) > 0.5]
    if not bad.empty:
        log.warning(
            "consistency check failed for %d (run, band) entries: max |Δ|=%.3f dB",
            len(bad), float(bad.abs().max()),
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = aggregate_dir(args.root)
    check_consistency(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    log.info("aggregated %d rows → %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
