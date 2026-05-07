"""Verify that frequency-domain spikes in steered psd_pre come from the array
geometry, not the source signal.

Computes T(f) = |Σ_m exp(-2j·2πf r_m/c) / r_m|² / M² — the array transfer
function for a unit point source at the target under phase-only DAS — and
overlays it with the measured psd_pre normalized by the source PSD and the
1/(4π)² Greens-convention factor.

If the two curves overlap, the ripples are purely geometric.

Usage:
    uv run python analysis/array_transfer_at_target.py \\
        --run-dir results/pipeline/ext_only_doa_target_cone_2026-05-07T12-07-19_37a7abbf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml

from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.csm import CsmConfig
from martymicfly.processing.steering import steer_to_psd

import importlib.util
import sys
_SCRIPT = Path(__file__).resolve().parent / "plot_calibration_effect.py"
_spec = importlib.util.spec_from_file_location("plot_calibration_effect", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["plot_calibration_effect"] = _mod
_spec.loader.exec_module(_mod)

SPEED_OF_SOUND = 343.0


def array_transfer_at_target(
    mic_positions: np.ndarray,
    target_point: tuple[float, float, float],
    frequencies: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """T(f) = |Σ_m exp(-2j·2πf r_m/c) / r_m|² / M² (purely geometric)."""
    mics = np.asarray(mic_positions, dtype=np.float64)
    tgt = np.asarray(target_point, dtype=np.float64)
    r = np.linalg.norm(mics - tgt[None, :], axis=1)              # (M,)
    M = mics.shape[0]
    out = np.zeros(frequencies.shape, dtype=np.float64)
    inv_r = 1.0 / r
    for fi, f in enumerate(frequencies):
        phase = np.exp(-2j * 2.0 * np.pi * f * r / speed_of_sound)
        out[fi] = float(np.abs((phase * inv_r).sum()) ** 2) / (M * M)
    return out


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

    # Measured pre-stage steered PSD (input — methodenunabhängig)
    psd_pre, freqs = _mod.compute_psd_pre(
        Path(cfg["input"]["audio_h5"]), mic_positions, target, duration_s, csm_cfg,
    )
    gt_psd = _mod.gt_psd_at(
        Path(cfg["input"]["ground_truth_h5"]), freqs, csm_cfg.nperseg, csm_cfg.noverlap,
    )
    # Method comparison: steered psd_post for each available residual_csm we find.
    method_psds: dict[str, np.ndarray] = {}
    for sibling in args.run_dir.parent.glob("ext_only_*/residual_csm.h5"):
        run = sibling.parent
        if "2026-05-07" not in run.name:
            continue
        method = run.name.replace("ext_only_", "").rsplit("_2026-", 1)[0]
        csm, fr = _mod.load_residual_csm(sibling)
        if not np.allclose(fr, freqs):
            continue
        method_psds[method] = steer_to_psd(csm, fr, mic_positions, target)

    # Analytical geometric transfer function
    T = array_transfer_at_target(mic_positions, target, freqs)
    # Predicted psd_pre = S_q · T(f) / (4π)² where S_q ≈ <gt_psd>.
    s_q = float(gt_psd.mean())
    pred_pre = s_q * T / (4.0 * np.pi) ** 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_pre, 1e-30)),
        mode="lines", name="measured psd_pre (raw, vor Fix)",
        line={"color": "#1f77b4", "width": 2.5},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(pred_pre, 1e-30)),
        mode="lines",
        name=f"analytisch S_q · T(f) / (4π)²    [S_q = ⟨GT⟩ = {10*np.log10(s_q):+.1f} dB]",
        line={"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
    ))
    palette = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for i, (method, psd) in enumerate(sorted(method_psds.items())):
        fig.add_trace(go.Scatter(
            x=freqs, y=10 * np.log10(np.maximum(psd, 1e-30)),
            mode="lines", name=f"psd_post {method}",
            line={"color": palette[i % len(palette)], "width": 1, "dash": "dot"},
            opacity=0.7,
        ))

    # The thing the user wants to see: identical-shape ripple structure across
    # all "raw" lines (psd_pre + every psd_post + analytical T(f)) — proof that
    # the spikes are geometric.
    cal_db = 10 * np.log10((4 * np.pi / (1.0 / np.linalg.norm(mic_positions - np.asarray(target), axis=1)).mean()) ** 2)
    fig.update_layout(
        title=(f"Array transfer at target — methodenunabhängige Spikes "
               f"(M={mic_positions.shape[0]}, target={target})"),
        xaxis_title="frequency [Hz]",
        yaxis_title="PSD [dB re 1 Pa²/Hz]",
        legend={"yanchor": "bottom", "y": 0.02, "xanchor": "right", "x": 0.98},
        template="plotly_white",
    )

    out = args.out or args.run_dir / "array_transfer_overlay.html"
    fig.write_html(out, include_plotlyjs="cdn")

    # Numeric summary: correlation between curves on the dB scale
    def db(x): return 10 * np.log10(np.maximum(x, 1e-30))
    def corr(a, b): return float(np.corrcoef(db(a), db(b))[0, 1])
    print(f"wrote {out}")
    print(f"  corr(psd_pre, T·S_q/(4π)²)  = {corr(psd_pre, pred_pre):.4f}  (≈ 1 ⇒ ripple is geometric)")
    for method, psd in sorted(method_psds.items()):
        print(f"  corr(psd_post[{method:18s}], psd_pre) = {corr(psd, psd_pre):.4f}")
    rms = float(np.std(db(psd_pre) - db(pred_pre)))
    print(f"  RMS(db(psd_pre) - db(pred))  = {rms:.3f} dB  (Welch streuung-bedingt)")


if __name__ == "__main__":
    main()
