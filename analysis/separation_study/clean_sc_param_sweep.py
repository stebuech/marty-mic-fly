"""Sweep CLEAN-SC damp/n_iter for mixed+notch S0 drone_disk case.

Now that residual_csm is PSD-projected after subtraction, indefiniteness
is no longer a constraint on damp/n_iter. Re-running the sweep to see
whether more aggressive subtraction can actually pull drone energy out
of the target band (post/pre attenuation) without hurting post/GT MAE.
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
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("clean_sc_param_sweep")

MIXED_CFG = "configs/pipeline_mixed_doa_drone_disk.yaml"
MIXED_AUDIO = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5"
EXT_GT      = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_external_only_gaptip_gt.h5"
MIC_XML     = "/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml"
TARGET = (0.0, 0.0, -1.5)
HIRES_NPERSEG, HIRES_NOVERLAP, F_MIN = 8192, 4096, 50.0

# (damp, n_iter) sweep — baseline + aggressive variants (PSD-projection
# now decouples indefiniteness from damp/n_iter, so we can push higher).
SWEEP = [
    (0.6, 100),   # baseline
    (0.8, 100),
    (0.8, 200),
    (0.9, 200),
    (0.95, 300),
    (0.99, 500),
]

OUT_ROOT = Path("results/separation_study/s0_clean_sc_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(damp: float, n_iter: int) -> Path:
    tag = f"damp{damp:g}_iter{n_iter}"
    run_dir = OUT_ROOT / tag
    overrides = {
        "input.audio_h5": MIXED_AUDIO,
        "input.ground_truth_h5": EXT_GT,
        "array_filter.target_point_m": list(TARGET),
        "array_filter.doa_grid.focal_radius_m": 1.5,
        "array_filter.csm.nperseg": HIRES_NPERSEG,
        "array_filter.csm.noverlap": HIRES_NOVERLAP,
        "array_filter.csm.f_min_hz": F_MIN,
        "array_filter.clean_sc.damp": float(damp),
        "array_filter.clean_sc.n_iter": int(n_iter),
        "plots.enabled": False,
    }
    return run_pipeline_with_overrides(
        base_config_path=Path(MIXED_CFG),
        overrides=overrides, output_dir=run_dir,
    )


def _pre_psd(run_dir: Path, mic: np.ndarray, cal: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-array_filter PSD at target = post-notch CSM steered to target."""
    with h5py.File(run_dir / "filtered.h5", "r") as f:
        audio = f["time_data"][:].astype(np.float64)
        fs = float(f["time_data"].attrs["sample_freq"])
    cfg = CsmConfig(nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP,
                    window="hann", diag_loading_rel=1e-6,
                    f_min_hz=F_MIN, f_max_hz=6000.0)
    csm_pre, freqs = build_measurement_csm(audio, fs, cfg)
    return freqs, steer_to_psd(csm_pre, freqs, mic, TARGET) * cal


def report(run_dir: Path, label: str, gt) -> None:
    mic = load_mic_geom_xml(MIC_XML)
    cal = range_compensation_factor(mic, TARGET)
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs = f["frequencies"][:]
        csm = f["csm_real"][:] + 1j * f["csm_imag"][:]
    psd = steer_to_psd(csm, freqs, mic, TARGET) * cal
    freqs_pre, pre = _pre_psd(run_dir, mic, cal)
    if not np.allclose(freqs, freqs_pre):
        pre = np.interp(freqs, freqs_pre, pre)

    fg, pg = welch(gt.signal, fs=gt.sample_rate,
                   nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, scaling="density")
    gt_grid = np.interp(freqs, fg, pg)

    bands = {"v_low": (50, 200), "low": (200, 500),
             "mid": (500, 2000), "high": (2000, 6000)}
    n_neg = int((psd < 0).sum())
    min_psd = float(psd.min())
    p_db = 10 * np.log10(np.maximum(psd, 1e-30))
    r_db = 10 * np.log10(np.maximum(pre, 1e-30))
    g_db = 10 * np.log10(np.maximum(gt_grid, 1e-30))
    err = p_db - g_db
    filt = p_db - r_db
    m = freqs >= F_MIN
    x = freqs[m]
    maes, atts = [], []
    for lo, hi in bands.values():
        bm = (x >= lo) & (x <= hi)
        maes.append(float(np.mean(np.abs(err[m][bm]))) if bm.any() else float("nan"))
        atts.append(float(np.mean(filt[m][bm])) if bm.any() else float("nan"))
    print(f"{label:<14}  n_neg={n_neg:>2}  "
          f"MAE[v_low/low/mid/high]={maes[0]:>5.2f}/{maes[1]:>5.2f}/{maes[2]:>5.2f}/{maes[3]:>5.2f}  "
          f"atten={atts[0]:>+5.2f}/{atts[1]:>+5.2f}/{atts[2]:>+5.2f}/{atts[3]:>+5.2f}")


def main() -> int:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runs = {}
    for damp, n_iter in SWEEP:
        log.info("sweep damp=%g n_iter=%d", damp, n_iter)
        runs[(damp, n_iter)] = run_one(damp, n_iter)

    gt = load_ground_truth(EXT_GT)
    print()
    print(f"{'config':<16} {'n_neg':>5} {'min_psd':>12}  v_low   low   mid  high  [dB MAE]")
    for (damp, n_iter), rd in runs.items():
        report(rd, f"d{damp:g}/it{n_iter}", gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
