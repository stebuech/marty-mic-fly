"""Plot the effect of `range_compensation_factor` over frequency.

Loads the residual CSM from a stage-2 run, recomputes the steered PSD at the
target (raw and calibrated), and overlays the ground-truth PSD.  Three lines
should drop onto each other after the fix is applied.

Usage:
    uv run python analysis/plot_calibration_effect.py \\
        --run-dir results/pipeline/ext_only_doa_target_cone_2026-05-07T12-07-19_37a7abbf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import yaml
from scipy.signal import welch

from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import range_compensation_factor, steer_to_psd


def load_residual_csm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        csm = f["csm_real"][...] + 1j * f["csm_imag"][...]
        freqs = f["frequencies"][...]
    return csm.astype(np.complex128), freqs.astype(np.float64)


def select_middle_segment(time_data: np.ndarray, sample_rate: float, duration_s: float) -> np.ndarray:
    """Mirror run_pipeline._select_segment for mode=middle."""
    n_total = time_data.shape[0]
    n_seg = int(round(duration_s * sample_rate))
    start = (n_total - n_seg) // 2
    return time_data[start:start + n_seg]


def compute_psd_pre(
    audio_h5: Path,
    mic_positions: np.ndarray,
    target: tuple[float, float, float],
    duration_s: float,
    csm_cfg: CsmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute psd_pre by rebuilding the input CSM from the synth audio."""
    src = load_synth_h5(audio_h5)
    seg = select_middle_segment(src["time_data"], src["sample_rate"], duration_s)
    csm, freqs = build_measurement_csm(seg, src["sample_rate"], csm_cfg)
    psd = steer_to_psd(csm, freqs, mic_positions, target)
    return psd, freqs


def gt_psd_at(gt_h5: Path, freqs_target: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """Compute Welch PSD of the GT signal and interpolate onto freqs_target.

    Layout: /time_data (N, 1) with attr sample_freq.
    """
    with h5py.File(gt_h5, "r") as f:
        td = f["time_data"]
        sig = np.asarray(td[:, 0], dtype=np.float64)
        fs = float(td.attrs["sample_freq"])
    f_gt, p_gt = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       window="hann", scaling="density")
    return np.interp(freqs_target, f_gt, p_gt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A results/pipeline/<run_id>/ directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: <run-dir>/calibration_effect.html)")
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    stage_cfg = cfg["stages"][0]
    target = tuple(stage_cfg["target_point_m"])
    nperseg = int(stage_cfg["csm"]["nperseg"])
    noverlap = int(stage_cfg["csm"]["noverlap"])
    duration_s = float(cfg["segment"]["duration"])
    csm_cfg = CsmConfig(
        nperseg=nperseg, noverlap=noverlap,
        window=stage_cfg["csm"]["window"],
        diag_loading_rel=float(stage_cfg["csm"]["diag_loading_rel"]),
        f_min_hz=float(stage_cfg["csm"]["f_min_hz"]),
        f_max_hz=float(stage_cfg["csm"]["f_max_hz"]),
    )

    mic_positions = load_mic_geom_xml(cfg["input"]["mic_geom_xml"])
    cal = range_compensation_factor(mic_positions, target)

    psd_pre_raw, freqs_pre = compute_psd_pre(
        Path(cfg["input"]["audio_h5"]), mic_positions, target, duration_s, csm_cfg,
    )
    csm_post, freqs = load_residual_csm(args.run_dir / "residual_csm.h5")
    psd_post_raw = steer_to_psd(csm_post, freqs, mic_positions, target)
    assert np.allclose(freqs_pre, freqs), "pre/post frequency grids disagree"

    psd_pre_cal = psd_pre_raw * cal
    psd_post_cal = psd_post_raw * cal

    gt_psd = gt_psd_at(Path(cfg["input"]["ground_truth_h5"]), freqs, nperseg, noverlap)

    bands = stage_cfg["bands"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_pre_raw, 1e-30)),
        mode="lines", name="psd_pre (raw, ungefiltert)",
        line={"color": "#ff9896", "width": 1, "dash": "dot"},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_post_raw, 1e-30)),
        mode="lines", name="psd_post (raw, gefiltert, vor Fix)",
        line={"color": "#d62728", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_pre_cal, 1e-30)),
        mode="lines", name=f"psd_pre · {cal:.1f} (kalibriert, ungefiltert)",
        line={"color": "#aec7e8", "width": 1.5, "dash": "dot"},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_post_cal, 1e-30)),
        mode="lines", name=f"psd_post · {cal:.1f} (kalibriert, gefiltert)",
        line={"color": "#1f77b4", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(gt_psd, 1e-30)),
        mode="lines", name="ground_truth PSD",
        line={"color": "#2ca02c", "width": 2, "dash": "dash"},
    ))

    for b in bands:
        fig.add_vline(x=b["f_min_hz"], line={"color": "#888", "width": 1, "dash": "dot"})
        fig.add_vline(x=b["f_max_hz"], line={"color": "#888", "width": 1, "dash": "dot"})
        fig.add_annotation(x=(b["f_min_hz"] + b["f_max_hz"]) / 2,
                           y=1.02, yref="paper", showarrow=False,
                           text=b["name"], font={"size": 11, "color": "#666"})

    cal_db = 10 * np.log10(cal)
    fig.update_layout(
        title=(f"Effekt von range_compensation_factor — "
               f"Faktor = {cal:.2f} ({cal_db:+.2f} dB), "
               f"M={mic_positions.shape[0]} mics, target={target}"),
        xaxis_title="frequency [Hz]",
        yaxis_title="PSD [dB re 1 Pa²/Hz]",
        legend={"yanchor": "top", "y": 0.98, "xanchor": "right", "x": 0.98},
        template="plotly_white",
    )

    out_path = args.out or args.run_dir / "calibration_effect.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"wrote {out_path}")
    print(f"  cal_db = {cal_db:+.3f} dB")
    print(f"  raw  ⟨psd_pre⟩  = {10*np.log10(psd_pre_raw.mean()):+.2f} dB re 1 Pa²/Hz")
    print(f"  raw  ⟨psd_post⟩ = {10*np.log10(psd_post_raw.mean()):+.2f} dB re 1 Pa²/Hz")
    print(f"  cal  ⟨psd_pre⟩  = {10*np.log10(psd_pre_cal.mean()):+.2f} dB re 1 Pa²/Hz")
    print(f"  cal  ⟨psd_post⟩ = {10*np.log10(psd_post_cal.mean()):+.2f} dB re 1 Pa²/Hz")
    print(f"  gt   ⟨psd⟩      = {10*np.log10(gt_psd.mean()):+.2f} dB re 1 Pa²/Hz")
    print(f"  pre→GT  Δ = {10*np.log10(psd_pre_cal.mean()/gt_psd.mean()):+.3f} dB  (Eingang vs. GT — Skalierungs-Sanity)")
    print(f"  post→GT Δ = {10*np.log10(psd_post_cal.mean()/gt_psd.mean()):+.3f} dB  (gefiltert vs. GT — Recovery)")
    print(f"  pre→post Δ = {10*np.log10(psd_pre_cal.mean()/psd_post_cal.mean()):+.3f} dB  (Algorithmus-Subtraktion)")


if __name__ == "__main__":
    main()
