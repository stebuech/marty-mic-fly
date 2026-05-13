"""S0 drone_cone: ext_only vs mixed (notch) vs mixed (kein notch) PSD/Error-Plot.

Fährt drei Pipeline-Konfigurationen über das S0-Szenario:
  - ext_only             : reines externes Signal, keine Drohne, kein notch
  - mixed_with_notch     : drone+ext Signal, notch-Stage entfernt BPF-Tonale
                           vor dem array_filter
  - mixed_no_notch       : drone+ext Signal, ohne notch (= ext_only-config
                           gefüttert mit mixed-Audio) — zeigt was passieren
                           würde, wenn man auf die notch-Stage verzichtet

Alle drei Fälle steuern den gleichen Target-Punkt an und werden gegen die
ext_only Ground Truth (= das externe Signal, das wir zurückgewinnen wollen)
verglichen. Hires CSM (nperseg=8192 → Δf ≈ 6.25 Hz, f_min=50 Hz).

Layout:
  - oben:   PSD [dB] post (solid, color/case) + GT (dashed, gemeinsam)
  - mitte:  PSD error 10·log10(post/GT)
  - unten:  MAE-Tabelle pro Band
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

from analysis.separation_study.pipeline_wrapper import run_pipeline_with_overrides
from martymicfly.io.ground_truth_h5 import load_ground_truth
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("s0_notch_compare")

EXT_ONLY_CFG = "configs/pipeline_external_only_doa_drone_cone.yaml"
MIXED_CFG    = "configs/pipeline_mixed_doa_drone_cone.yaml"

CASE_COLORS = {
    "ext_only":         "#1f77b4",   # blue
    "mixed_with_notch": "#2ca02c",   # green
    "mixed_no_notch":   "#d62728",   # red
}
CASE_LABELS = {
    "ext_only":         "ext_only",
    "mixed_with_notch": "mixed + notch",
    "mixed_no_notch":   "mixed, no notch",
}

HIRES_NPERSEG = 8192
HIRES_NOVERLAP = 4096
F_MIN_HZ = 50.0

MAE_BANDS = {
    "very_low (50-200 Hz)": (50.0, 200.0),
    "low (200-500 Hz)":     (200.0, 500.0),
    "mid (500-2000 Hz)":    (500.0, 2000.0),
    "high (2-6 kHz)":       (2000.0, 6000.0),
}


def _s0_inputs(sp_yaml: Path) -> dict:
    sp = yaml.safe_load(sp_yaml.read_text())
    s = sp["S0"]
    target = tuple(s.get("config_overrides", {}).get(
        "array_filter.target_point_m", [0.0, 0.0, -1.5]))
    focal_radius = float(s.get("config_overrides", {}).get(
        "array_filter.doa_grid.focal_radius_m", 1.5))
    return {
        "ext_only_audio_h5": s["ext_only_audio_h5"],
        "mixed_audio_h5":    s["mixed_audio_h5"],
        "ext_only_gt_h5":    s["ext_only_gt_h5"],
        "mic_geom_xml":      s["mic_geom_xml"],
        "target_point_m":    target,
        "focal_radius_m":    focal_radius,
    }


def _hires_overrides(audio_h5: str, gt_h5: str, target: tuple,
                     focal_radius: float) -> dict:
    return {
        "input.audio_h5": audio_h5,
        "input.ground_truth_h5": gt_h5,
        "array_filter.target_point_m": list(target),
        "array_filter.doa_grid.focal_radius_m": focal_radius,
        "array_filter.csm.nperseg": HIRES_NPERSEG,
        "array_filter.csm.noverlap": HIRES_NOVERLAP,
        "array_filter.csm.f_min_hz": F_MIN_HZ,
        "plots.enabled": False,
    }


def _run(case: str, base_cfg: Path, audio_h5: str, gt_h5: str,
         target: tuple, focal_radius: float, out_root: Path) -> Path:
    run_dir = out_root / f"s0_{case}"
    overrides = _hires_overrides(audio_h5, gt_h5, target, focal_radius)
    log.info("run %s overrides=%s", case, overrides)
    return run_pipeline_with_overrides(
        base_config_path=base_cfg, overrides=overrides, output_dir=run_dir,
    )


def _post_psd_from_run(run_dir: Path, target_point: tuple, mic_xml: Path) -> tuple:
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs = f["frequencies"][:]
        csm = f["csm_real"][:] + 1j * f["csm_imag"][:]
    mic_positions = load_mic_geom_xml(mic_xml)
    cal = range_compensation_factor(mic_positions, target_point)
    post = steer_to_psd(csm, freqs, mic_positions, target_point) * cal
    return freqs, post


def _pre_psd(audio: np.ndarray, fs: float, target_point: tuple,
             mic_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build CSM from audio (no segmenting), steer to target, range-compensate.

    audio: (N, M) already segmented to the same window as the pipeline.
    """
    cfg = CsmConfig(
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP,
        window="hann", diag_loading_rel=1e-6,
        f_min_hz=F_MIN_HZ, f_max_hz=6000.0,
    )
    csm, freqs = build_measurement_csm(audio, sample_rate=fs, cfg=cfg)
    cal = range_compensation_factor(mic_positions, target_point)
    return freqs, steer_to_psd(csm, freqs, mic_positions, target_point) * cal


def _middle_segment(audio: np.ndarray, fs: float, seg_s: float = 10.0) -> np.ndarray:
    n_seg = int(round(seg_s * fs))
    if audio.shape[0] <= n_seg:
        return audio.astype(np.float64)
    start = (audio.shape[0] - n_seg) // 2
    return audio[start:start + n_seg].astype(np.float64)


def _pre_for_case(case: str, run_dir: Path, audio_h5_raw: str,
                  target_point: tuple, mic_positions: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd_pre) — input to the array_filter stage.

    - mixed_with_notch: post-notch time_data from filtered.h5 (already a 10s
      middle segment, written by the notch stage).
    - others: raw input audio, middle 10s segment.
    """
    if case == "mixed_with_notch":
        with h5py.File(run_dir / "filtered.h5", "r") as f:
            audio = f["time_data"][:].astype(np.float64)
            fs = float(f["time_data"].attrs["sample_freq"])
        return _pre_psd(audio, fs, target_point, mic_positions)
    d = load_synth_h5(audio_h5_raw)
    audio = _middle_segment(d["time_data"], float(d["sample_rate"]))
    return _pre_psd(audio, float(d["sample_rate"]), target_point, mic_positions)


def _gt_psd_on_grid(gt_h5: Path, freqs: np.ndarray) -> np.ndarray:
    gt = load_ground_truth(str(gt_h5))
    f_gt, p_gt = welch(
        gt.signal, fs=gt.sample_rate,
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, scaling="density",
    )
    return np.interp(freqs, f_gt, p_gt)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-paths",
                   default="analysis/separation_study/studies/scenario_paths.yaml",
                   type=Path)
    p.add_argument("--out-dir",
                   default="results/separation_study/s0_notch_compare",
                   type=Path)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = _s0_inputs(args.scenario_paths)
    target = s["target_point_m"]
    fr = s["focal_radius_m"]
    gt_h5 = s["ext_only_gt_h5"]

    # 3 cases: choose config + audio_h5
    run_dirs = {
        "ext_only":         _run("ext_only", Path(EXT_ONLY_CFG),
                                 s["ext_only_audio_h5"], gt_h5, target, fr, args.out_dir),
        "mixed_with_notch": _run("mixed_with_notch", Path(MIXED_CFG),
                                 s["mixed_audio_h5"], gt_h5, target, fr, args.out_dir),
        "mixed_no_notch":   _run("mixed_no_notch", Path(EXT_ONLY_CFG),
                                 s["mixed_audio_h5"], gt_h5, target, fr, args.out_dir),
    }

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.42, 0.16],
        shared_xaxes=False, vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("PSD: pre (dotted, input to array_filter) · post (solid) · GT (dashed)",
                        "PSD error: 10·log10(post / GT) [dB]",
                        "MAE per band [dB]"),
    )

    mic_positions = load_mic_geom_xml(s["mic_geom_xml"])
    raw_audio_for_case = {
        "ext_only":         s["ext_only_audio_h5"],
        "mixed_with_notch": s["mixed_audio_h5"],   # only used as fallback path
        "mixed_no_notch":   s["mixed_audio_h5"],
    }

    mae_rows: list[list] = []
    gt_drawn = False
    for case in ("ext_only", "mixed_with_notch", "mixed_no_notch"):
        freqs, post = _post_psd_from_run(run_dirs[case], target, Path(s["mic_geom_xml"]))
        freqs_pre, pre = _pre_for_case(
            case, run_dirs[case], raw_audio_for_case[case], target, mic_positions,
        )
        if not np.allclose(freqs, freqs_pre):
            pre = np.interp(freqs, freqs_pre, pre)
        gt = _gt_psd_on_grid(Path(gt_h5), freqs)
        mask = freqs >= F_MIN_HZ
        x = freqs[mask]
        color = CASE_COLORS[case]
        label = CASE_LABELS[case]
        pre_db = 10 * np.log10(np.maximum(pre[mask], 1e-30))
        post_db = 10 * np.log10(np.maximum(post[mask], 1e-30))
        gt_db = 10 * np.log10(np.maximum(gt[mask], 1e-30))
        err_db = post_db - gt_db

        fig.add_trace(go.Scatter(
            x=x, y=pre_db, mode="lines",
            name=f"{label} pre", legendgroup=case,
            line=dict(color=color, width=1.0, dash="dot"),
            opacity=0.7,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=post_db, mode="lines",
            name=f"{label} post", legendgroup=case,
            line=dict(color=color, width=1.5),
        ), row=1, col=1)
        if not gt_drawn:
            fig.add_trace(go.Scatter(
                x=x, y=gt_db, mode="lines",
                name="ground truth", legendgroup="gt",
                line=dict(color="#444", width=1.0, dash="dash"),
            ), row=1, col=1)
            gt_drawn = True

        fig.add_trace(go.Scatter(
            x=x, y=err_db, mode="lines", name=f"{label} err",
            legendgroup=case, showlegend=False,
            line=dict(color=color, width=1.5),
        ), row=2, col=1)

        mae_per_band = []
        for _, (f_lo, f_hi) in MAE_BANDS.items():
            band_mask = (x >= f_lo) & (x <= f_hi)
            mae_per_band.append(
                float(np.mean(np.abs(err_db[band_mask])))
                if band_mask.any() else float("nan")
            )
        mae_rows.append([label, *mae_per_band])

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"),
                  row=2, col=1)
    fig.update_xaxes(type="log", range=[np.log10(F_MIN_HZ), np.log10(6000)],
                     row=1, col=1)
    fig.update_xaxes(type="log", range=[np.log10(F_MIN_HZ), np.log10(6000)],
                     row=2, col=1, title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="PSD [dB]", row=1, col=1)
    fig.update_yaxes(title_text="error [dB]", row=2, col=1)

    header = ["case", *MAE_BANDS.keys()]
    cols_t = list(zip(*mae_rows))
    case_color_col = [CASE_COLORS[c] for c in ("ext_only", "mixed_with_notch", "mixed_no_notch")]
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
        title=(f"S0 drone_cone — ext_only vs mixed(+notch) vs mixed(no notch) "
               f"(nperseg={HIRES_NPERSEG}, Δf≈{51200/HIRES_NPERSEG:.2f} Hz)"),
        height=1050, width=1100, hovermode="x unified",
    )
    out_html = args.out_dir / "s0_notch_compare.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
