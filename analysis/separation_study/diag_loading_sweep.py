"""Sweep diag_loading_rel for mixed+notch S0 drone_disk case.

Goal: find a loading value that keeps the residual CSM PSD-safe (no negative
``h^H·CSM·h`` at the target) without distorting the post-PSD too much.
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch

from analysis.separation_study.pipeline_wrapper import run_pipeline_with_overrides
from martymicfly.io.ground_truth_h5 import load_ground_truth
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("diag_loading_sweep")

MIXED_CFG = "configs/pipeline_mixed_doa_drone_disk.yaml"
MIXED_AUDIO = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5"
EXT_GT      = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_external_only_gaptip_gt.h5"
MIC_XML     = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml"
TARGET = (0.0, 0.0, -1.5)
HIRES_NPERSEG, HIRES_NOVERLAP, F_MIN = 8192, 4096, 50.0

LOADINGS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]

OUT_ROOT = Path("results/separation_study/s0_diag_loading_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(load: float) -> Path:
    tag = f"diag{load:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_dir = OUT_ROOT / tag
    overrides = {
        "input.audio_h5": MIXED_AUDIO,
        "input.ground_truth_h5": EXT_GT,
        "array_filter.target_point_m": list(TARGET),
        "array_filter.doa_grid.focal_radius_m": 1.5,
        "array_filter.csm.nperseg": HIRES_NPERSEG,
        "array_filter.csm.noverlap": HIRES_NOVERLAP,
        "array_filter.csm.f_min_hz": F_MIN,
        "array_filter.csm.diag_loading_rel": float(load),
        "plots.enabled": False,
    }
    return run_pipeline_with_overrides(
        base_config_path=Path(MIXED_CFG),
        overrides=overrides, output_dir=run_dir,
    )


def report(run_dir: Path, label: str, gt) -> None:
    mic = load_mic_geom_xml(MIC_XML)
    cal = range_compensation_factor(mic, TARGET)
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs = f["frequencies"][:]
        csm = f["csm_real"][:] + 1j * f["csm_imag"][:]
    psd = steer_to_psd(csm, freqs, mic, TARGET) * cal

    fg, pg = welch(gt.signal, fs=gt.sample_rate,
                   nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, scaling="density")
    gt_grid = np.interp(freqs, fg, pg)

    bands = {"v_low": (50, 200), "low": (200, 500),
             "mid": (500, 2000), "high": (2000, 6000)}
    n_neg = int((psd < 0).sum())
    min_psd = float(psd.min())
    p_db = 10 * np.log10(np.maximum(psd, 1e-30))
    g_db = 10 * np.log10(np.maximum(gt_grid, 1e-30))
    err = p_db - g_db
    m = freqs >= F_MIN
    x, e = freqs[m], err[m]
    maes = []
    for lo, hi in bands.values():
        bm = (x >= lo) & (x <= hi)
        maes.append(float(np.mean(np.abs(e[bm]))) if bm.any() else float("nan"))
    print(f"{label:<12} n_neg={n_neg:>3}  min={min_psd:+.2e}  "
          f"v_low={maes[0]:>5.2f}  low={maes[1]:>5.2f}  "
          f"mid={maes[2]:>5.2f}  high={maes[3]:>5.2f}")


def main() -> int:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runs = {}
    for load in LOADINGS:
        log.info("sweep diag_loading_rel=%g", load)
        runs[load] = run_one(load)

    gt = load_ground_truth(EXT_GT)
    print()
    print(f"{'loading':<12} {'n_neg':>5} {'min_psd':>12}  v_low   low   mid  high  [dB MAE]")
    for load, rd in runs.items():
        report(rd, f"{load:.0e}", gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
