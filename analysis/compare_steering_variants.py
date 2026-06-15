"""Overlay phase-only steering (current) vs range-corrected matched-filter
steering on the same ext_only run, to visualize how the geometric ripple is
resolved.

Three lines:
  - phase-only psd_pre · range_compensation_factor  (rippled, calibrated)
  - matched     psd_pre                              (smooth, in source units)
  - ground_truth                                     (reference)

Plus the same triplet for the chosen post-stage (CLEAN-SC residual).

Usage:
    uv run python analysis/compare_steering_variants.py \\
        --run-dir results/pipeline/ext_only_doa_target_cone_2026-05-07T12-07-19_37a7abbf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import yaml

from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd, steer_to_psd_matched,
)
from martymicfly.io.synth_h5 import load_synth_h5

import importlib.util
import sys
_SCRIPT = Path(__file__).resolve().parent / "plot_calibration_effect.py"
_spec = importlib.util.spec_from_file_location("plot_calibration_effect", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["plot_calibration_effect"] = _mod
_spec.loader.exec_module(_mod)


def db(x): return 10 * np.log10(np.maximum(x, 1e-30))
def rms_db(x): return float(np.std(db(x)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    stage_cfg = cfg["stages"][0]
    target = tuple(stage_cfg["target_point_m"])
    csm_cfg = CsmConfig(
        nperseg=int(stage_cfg["csm"]["nperseg"]),
        noverlap=int(stage_cfg["csm"]["noverlap"]),
        window=stage_cfg["csm"]["window"],
        diag_loading_rel=float(stage_cfg["csm"]["diag_loading_rel"]),
        f_min_hz=float(stage_cfg["csm"]["f_min_hz"]),
        f_max_hz=float(stage_cfg["csm"]["f_max_hz"]),
    )
    duration_s = float(cfg["segment"]["duration"])
    mic_positions = load_mic_geom_xml(cfg["input"]["mic_geom_xml"])
    cal = range_compensation_factor(mic_positions, target)

    # PRE: rebuild CSM from input audio
    src = load_synth_h5(cfg["input"]["audio_h5"])
    seg = _mod.select_middle_segment(src["time_data"], src["sample_rate"], duration_s)
    csm_pre, freqs = build_measurement_csm(seg, src["sample_rate"], csm_cfg)

    pre_phase = steer_to_psd(csm_pre, freqs, mic_positions, target) * cal
    pre_match = steer_to_psd_matched(csm_pre, freqs, mic_positions, target)

    # POST: residual_csm from the run
    csm_post, freqs_post = _mod.load_residual_csm(args.run_dir / "residual_csm.h5")
    assert np.allclose(freqs_post, freqs)
    post_phase = steer_to_psd(csm_post, freqs, mic_positions, target) * cal
    post_match = steer_to_psd_matched(csm_post, freqs, mic_positions, target)

    gt = _mod.gt_psd_at(
        Path(cfg["input"]["ground_truth_h5"]), freqs, csm_cfg.nperseg, csm_cfg.noverlap,
    )

    fig = sp.make_subplots(
        rows=2, cols=1, vertical_spacing=0.10,
        subplot_titles=(
            "psd_pre — input CSM (no algorithm applied)",
            "psd_post — CLEAN-SC residual (DOA-target-cone)",
        ),
    )
    for row, (phase_curve, match_curve, label) in enumerate([
        (pre_phase, pre_match, "pre"),
        (post_phase, post_match, "post"),
    ], start=1):
        fig.add_trace(go.Scatter(
            x=freqs, y=db(phase_curve),
            mode="lines",
            name=f"{label}: phase-only · {cal:.1f} (kalibriert) — RMS {rms_db(phase_curve):.2f} dB",
            line={"color": "#d62728", "width": 1.5},
            legendgroup=label,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=db(match_curve),
            mode="lines",
            name=f"{label}: matched-filter (4π·r-amp, conj-phase) — RMS {rms_db(match_curve):.2f} dB",
            line={"color": "#1f77b4", "width": 2},
            legendgroup=label,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=db(gt),
            mode="lines", name=f"{label}: ground_truth",
            line={"color": "#2ca02c", "width": 2, "dash": "dash"},
            legendgroup=label, showlegend=(row == 1),
        ), row=row, col=1)

    for b in stage_cfg["bands"]:
        for row in (1, 2):
            fig.add_vline(x=b["f_min_hz"], line={"color": "#888", "width": 1, "dash": "dot"}, row=row, col=1)
            fig.add_vline(x=b["f_max_hz"], line={"color": "#888", "width": 1, "dash": "dot"}, row=row, col=1)

    fig.update_xaxes(title_text="frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="PSD [dB re 1 Pa²/Hz]", row=1, col=1)
    fig.update_yaxes(title_text="PSD [dB re 1 Pa²/Hz]", row=2, col=1)
    fig.update_layout(
        title=f"Steering-Variants comparison — target={target}, M={mic_positions.shape[0]}",
        height=900, template="plotly_white",
        legend={"yanchor": "top", "y": 0.98, "xanchor": "right", "x": 0.98, "groupclick": "toggleitem"},
    )
    out = args.out or args.run_dir / "steering_variants.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    print()
    print("                            phase-only(cal)    matched-filter")
    print(f"  ⟨psd_pre⟩  [dB]:           {db(pre_phase.mean()):>+10.2f}        {db(pre_match.mean()):>+10.2f}")
    print(f"  RMS-Welligkeit [dB]:       {rms_db(pre_phase):>10.2f}        {rms_db(pre_match):>10.2f}")
    print(f"  Δ vs GT⟨⟩ [dB]:            {db(pre_phase.mean()/gt.mean()):>+10.2f}        {db(pre_match.mean()/gt.mean()):>+10.2f}")
    print()
    print(f"  ⟨psd_post⟩ [dB]:           {db(post_phase.mean()):>+10.2f}        {db(post_match.mean()):>+10.2f}")
    print(f"  RMS-Welligkeit [dB]:       {rms_db(post_phase):>10.2f}        {rms_db(post_match):>10.2f}")
    print(f"  Δ vs GT⟨⟩ [dB]:            {db(post_phase.mean()/gt.mean()):>+10.2f}        {db(post_match.mean()/gt.mean()):>+10.2f}")
    print(f"  GT⟨⟩       [dB]:            {db(gt.mean()):>+10.2f}")


if __name__ == "__main__":
    main()
