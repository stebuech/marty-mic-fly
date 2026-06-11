"""LCMV sanity-check on S0: linear-constraint MVDR with target-pass + rotor-null.

Analytical spatial filter (no iteration). Per frequency bin:

    minimize   w^H R w
    subject to C^H w = g       with C = [a_target, a_rot1, ..., a_rotN],
                                    g = [1, 0, ..., 0]

→ w = R^-1 C (C^H R^-1 C)^-1 g
→ output PSD at target = w^H R w

The matched-filter steering vector convention is

    a_m(p, f) = (4π·r_m(p)) / M · exp(+j·2π·f·r_m(p)/c)

with r_m(p) = ‖mic_m − p‖.  With this normalization, a^H·R·a returns the
*source PSD at p* (Pa²/Hz at unit reference) under scipy's CSM convention,
so the LCMV output is directly comparable to the synthetic ground truth.

What this answers: does an *optimal* analytical spatial filter (constraints
on known rotor DoAs) remove more drone energy than CLEAN-SC?  If yes →
CLEAN-SC needs an algorithmic improvement.  If LCMV also barely moves the
needle → the array aperture is the binding constraint and we must exploit
spectral structure (rotor harmonics, RPM telemetry) instead.

Layout matches s0_notch_compare_plot.py:
  - top: PSD pre/post/GT (log freq)
  - mid: error 10·log10(post/GT) solid + 10·log10(post/pre) dotted
  - bottom: MAE per band
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.signal import welch

from martymicfly.constants import SPEED_OF_SOUND
from martymicfly.io.ground_truth_h5 import load_ground_truth
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("lcmv_compare")

CASE_COLORS = {
    "mixed_pre":    "#d62728",   # red — drone+ext, no spatial filter
    "mixed_lcmv":   "#9467bd",   # purple — LCMV with rotor nulls
    "ext_only_pre": "#1f77b4",   # blue — achievable target (no drone)
}
CASE_LABELS = {
    "mixed_pre":    "mixed (no filter)",
    "mixed_lcmv":   "mixed → LCMV",
    "ext_only_pre": "ext_only (no filter)",
}

HIRES_NPERSEG = 8192
HIRES_NOVERLAP = 4096
F_MIN_HZ = 50.0
F_MAX_HZ = 6000.0
SEG_S = 10.0

MAE_BANDS = {
    "very_low (50-200 Hz)": (50.0, 200.0),
    "low (200-500 Hz)":     (200.0, 500.0),
    "mid (500-2000 Hz)":    (500.0, 2000.0),
    "high (2-6 kHz)":       (2000.0, 6000.0),
}

# Diagonal loading for LCMV: α · trace(R)/M.  Same scale as the pipeline's
# `diag_loading_rel` default but applied per-bin (so high-frequency bins with
# small trace get proportionally small loading instead of being dominated by
# the global peak — that was a known issue with the pipeline CSM).
LCMV_LOAD_REL = 1e-4


def _steering_vec(
    point: np.ndarray, mic_pos: np.ndarray, freq: float,
    c: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """v_m = exp(+j·2π·f·r_m/c) / (4π·r_m) — LCMV/MVDR steering vector (M,).

    Empirically validated convention for this codebase: with this sign,
    MVDR (w = R⁻¹v / (v^H R⁻¹v)) yields w^H R w that tracks the synthetic
    ground-truth source PSD at the target.  The +j sign matches the
    convention already used in steer_to_psd / steer_to_psd_matched and is
    the conjugate of the physical free-field Green's function; the sign
    is set by scipy.signal.csd's actual cross-spectrum convention.
    """
    r = np.linalg.norm(mic_pos - point[None, :], axis=1)
    return np.exp(2j * np.pi * freq * r / c) / (4.0 * np.pi * r)


def _build_constraint_matrix(
    target: np.ndarray, rotor_pos: np.ndarray,
    mic_pos: np.ndarray, freq: float,
) -> np.ndarray:
    """C ∈ ℂ^{M×K} with K = 1 + n_rotors; column 0 is the target Green's vec."""
    cols = [_steering_vec(target, mic_pos, freq)]
    for r in range(rotor_pos.shape[0]):
        cols.append(_steering_vec(rotor_pos[r], mic_pos, freq))
    return np.stack(cols, axis=1)


def _lcmv_output_psd(
    csm: np.ndarray, freqs: np.ndarray, mic_pos: np.ndarray,
    target: tuple[float, float, float], rotor_pos: np.ndarray,
) -> np.ndarray:
    """Per-bin LCMV output PSD at target with nulls at each rotor DoA."""
    n_f, M, _ = csm.shape
    K = 1 + rotor_pos.shape[0]
    g = np.zeros(K, dtype=np.complex128)
    g[0] = 1.0
    eye = np.eye(M, dtype=np.complex128)
    out = np.zeros(n_f, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    for fi, f in enumerate(freqs):
        R = csm[fi]
        tr_over_M = float(np.real(np.trace(R))) / M
        R_loaded = R + (LCMV_LOAD_REL * tr_over_M) * eye
        C = _build_constraint_matrix(tgt, rotor_pos, mic_pos, float(f))
        # w = R^-1 C (C^H R^-1 C)^-1 g, solved without forming inverses.
        R_inv_C = np.linalg.solve(R_loaded, C)           # (M, K)
        Gram = C.conj().T @ R_inv_C                       # (K, K)
        u = np.linalg.solve(Gram, g)                      # (K,)
        w = R_inv_C @ u                                   # (M,)
        out[fi] = float(np.real(w.conj() @ R @ w))
    return out


def _middle_segment(audio: np.ndarray, fs: float, seg_s: float = SEG_S) -> np.ndarray:
    n_seg = int(round(seg_s * fs))
    if audio.shape[0] <= n_seg:
        return audio.astype(np.float64)
    start = (audio.shape[0] - n_seg) // 2
    return audio[start:start + n_seg].astype(np.float64)


def _build_csm(audio: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = CsmConfig(
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, window="hann",
        diag_loading_rel=1e-6,  # tiny — LCMV does its own loading per bin
        f_min_hz=F_MIN_HZ, f_max_hz=F_MAX_HZ,
    )
    return build_measurement_csm(audio, sample_rate=fs, cfg=cfg)


def _das_psd(csm: np.ndarray, freqs: np.ndarray,
             mic_pos: np.ndarray, target: tuple) -> np.ndarray:
    cal = range_compensation_factor(mic_pos, target)
    return steer_to_psd(csm, freqs, mic_pos, target) * cal


def _gt_on_grid(gt_h5: str, freqs: np.ndarray) -> np.ndarray:
    gt = load_ground_truth(gt_h5)
    fg, pg = welch(gt.signal, fs=gt.sample_rate,
                   nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP,
                   scaling="density")
    return np.interp(freqs, fg, pg)


def _s0_inputs(sp_yaml: Path) -> dict:
    sp = yaml.safe_load(sp_yaml.read_text())
    s = sp["S0"]
    target = tuple(s.get("config_overrides", {}).get(
        "array_filter.target_point_m", [0.0, 0.0, -1.5]))
    return {
        "ext_only_audio_h5": s["ext_only_audio_h5"],
        "mixed_audio_h5":    s["mixed_audio_h5"],
        "ext_only_gt_h5":    s["ext_only_gt_h5"],
        "mic_geom_xml":      s["mic_geom_xml"],
        "target_point_m":    target,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-paths",
                   default="analysis/separation_study/studies/scenario_paths.yaml",
                   type=Path)
    p.add_argument("--out-dir",
                   default="results/separation_study/s0_lcmv_compare",
                   type=Path)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = _s0_inputs(args.scenario_paths)
    target = s["target_point_m"]
    mic_pos = load_mic_geom_xml(s["mic_geom_xml"])
    log.info("mic geom: %d positions", mic_pos.shape[0])

    # Pull rotor positions out of the mixed-audio H5 platform metadata.
    d_mixed = load_synth_h5(s["mixed_audio_h5"])
    plat = d_mixed["platform"]
    rotor_pos_raw = np.asarray(plat["rotor_positions"], dtype=np.float64)
    rotor_pos = rotor_pos_raw.T if rotor_pos_raw.shape[0] == 3 else rotor_pos_raw
    log.info("rotor positions (R, 3) = %s", rotor_pos.shape)
    log.info("rotor coords:\n%s", np.array2string(rotor_pos, precision=3))

    # Mixed audio: CSM + DAS-pre + LCMV.
    fs_mixed = float(d_mixed["sample_rate"])
    audio_mixed = _middle_segment(d_mixed["time_data"], fs_mixed)
    log.info("mixed segment: %d samples @ %.0f Hz", audio_mixed.shape[0], fs_mixed)
    csm_mixed, freqs = _build_csm(audio_mixed, fs_mixed)
    log.info("CSM: %d bins, %.1f-%.1f Hz", freqs.size, freqs[0], freqs[-1])

    pre_mixed = _das_psd(csm_mixed, freqs, mic_pos, target)
    log.info("computing LCMV (%d bins, K=%d constraints)...",
             freqs.size, 1 + rotor_pos.shape[0])
    lcmv_mixed = _lcmv_output_psd(csm_mixed, freqs, mic_pos, target, rotor_pos)

    # ext_only audio: just DAS-pre at the target (no drone to filter).
    d_ext = load_synth_h5(s["ext_only_audio_h5"])
    fs_ext = float(d_ext["sample_rate"])
    audio_ext = _middle_segment(d_ext["time_data"], fs_ext)
    csm_ext, freqs_ext = _build_csm(audio_ext, fs_ext)
    if not np.allclose(freqs, freqs_ext):
        raise RuntimeError("frequency grids differ between ext_only and mixed")
    pre_ext = _das_psd(csm_ext, freqs, mic_pos, target)

    gt = _gt_on_grid(s["ext_only_gt_h5"], freqs)

    # Plotting.
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.42, 0.16],
        shared_xaxes=False, vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=(
            "PSD at target [dB]: mixed pre / mixed LCMV / ext_only pre / GT",
            "PSD error [dB]: 10·log10(post/GT) solid · 10·log10(post/pre_mixed) dotted",
            "MAE per band [dB] vs ground truth",
        ),
    )

    mask = freqs >= F_MIN_HZ
    x = freqs[mask]
    series = {
        "mixed_pre":    pre_mixed,
        "mixed_lcmv":   lcmv_mixed,
        "ext_only_pre": pre_ext,
    }
    series_db = {k: 10.0 * np.log10(np.maximum(v[mask], 1e-30))
                 for k, v in series.items()}
    gt_db = 10.0 * np.log10(np.maximum(gt[mask], 1e-30))
    pre_ref_db = series_db["mixed_pre"]   # for the post/pre dotted comparison

    for case in ("mixed_pre", "mixed_lcmv", "ext_only_pre"):
        color = CASE_COLORS[case]
        label = CASE_LABELS[case]
        fig.add_trace(go.Scatter(
            x=x, y=series_db[case], mode="lines",
            name=label, legendgroup=case,
            line=dict(color=color, width=1.5),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=gt_db, mode="lines",
        name="ground truth", legendgroup="gt",
        line=dict(color="#444", width=1.0, dash="dash"),
    ), row=1, col=1)

    mae_rows: list[list] = []
    for case in ("mixed_pre", "mixed_lcmv", "ext_only_pre"):
        color = CASE_COLORS[case]
        label = CASE_LABELS[case]
        err_db = series_db[case] - gt_db
        fig.add_trace(go.Scatter(
            x=x, y=err_db, mode="lines",
            name=f"{label} post/GT",
            legendgroup=case, showlegend=False,
            line=dict(color=color, width=1.5),
        ), row=2, col=1)
        # post/pre only meaningful when "pre" is the same input; show it for
        # the LCMV trace (mixed input) and ext_only (different input, so it's
        # the gain ext_only DAS has over mixed DAS — also informative).
        if case != "mixed_pre":
            filt_db = series_db[case] - pre_ref_db
            fig.add_trace(go.Scatter(
                x=x, y=filt_db, mode="lines",
                name=f"{label} post/mixed_pre",
                legendgroup=case, showlegend=False,
                line=dict(color=color, width=1.0, dash="dot"),
                opacity=0.8,
            ), row=2, col=1)
        maes = []
        for _, (f_lo, f_hi) in MAE_BANDS.items():
            bm = (x >= f_lo) & (x <= f_hi)
            maes.append(float(np.mean(np.abs(err_db[bm])))
                        if bm.any() else float("nan"))
        mae_rows.append([label, *maes])

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"),
                  row=2, col=1)
    fig.update_xaxes(type="log",
                     range=[np.log10(F_MIN_HZ), np.log10(F_MAX_HZ)],
                     row=1, col=1)
    fig.update_xaxes(type="log",
                     range=[np.log10(F_MIN_HZ), np.log10(F_MAX_HZ)],
                     row=2, col=1, title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="PSD [dB]", row=1, col=1)
    fig.update_yaxes(title_text="error [dB]", row=2, col=1)

    header = ["case", *MAE_BANDS.keys()]
    cols_t = list(zip(*mae_rows))
    case_color_col = [CASE_COLORS[c]
                      for c in ("mixed_pre", "mixed_lcmv", "ext_only_pre")]
    cell_colors = [
        case_color_col,
        *[["#f6f6f6"] * len(mae_rows) for _ in range(len(MAE_BANDS))],
    ]
    fig.add_trace(go.Table(
        header=dict(values=header, fill_color="#dddddd",
                    font=dict(size=12), align="center"),
        cells=dict(
            values=[
                cols_t[0],
                *[[f"{v:.2f}" for v in col] for col in cols_t[1:]],
            ],
            fill_color=cell_colors,
            font=dict(color=[["white"] * len(mae_rows)]
                     + [["#222"] * len(mae_rows)] * len(MAE_BANDS)),
            align="center", height=24,
        ),
    ), row=3, col=1)

    fig.update_layout(
        title=(f"S0 LCMV vs DAS (mixed pre / mixed LCMV / ext_only pre vs GT) — "
               f"nperseg={HIRES_NPERSEG}, Δf≈{51200/HIRES_NPERSEG:.2f} Hz, "
               f"K={1 + rotor_pos.shape[0]} constraints, "
               f"load_rel={LCMV_LOAD_REL:g}"),
        height=1050, width=1100, hovermode="x unified",
    )
    out_html = args.out_dir / "s0_lcmv_compare.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)

    # Console summary so it's visible without opening the HTML.
    print()
    print(f"{'case':<22}  " + "  ".join(f"{b:<22}" for b in MAE_BANDS))
    for row in mae_rows:
        print(f"{row[0]:<22}  " + "  ".join(f"{v:>22.2f}" for v in row[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
