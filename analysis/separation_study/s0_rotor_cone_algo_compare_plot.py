"""S0 rotor_cone — Vergleich der Entfaltungsalgorithmen clean_sc vs orth.

Schwesterskript zu s0_notch_clean_sc_compare_plot.py. Gleiche 3-Zeilen-Optik
(PSD / Error / MAE-Tabelle) und dieselben drei Notch-Fälle, aber:

  - Vergleichsachse ist der Entfaltungs-`algorithm` (clean_sc vs orth) — die
    zwei der drei AP2-A-Methoden, die als (F,G)-Map-Verfahren vorliegen.
  - `post` kommt aus der echten Pipeline (Ansatz 2): die per Maske bestimmte
    Drohnen-CSM wird subtrahiert, `residual_csm` auf den Target-Punkt
    gesteuert. Kein Map-ROI-Integral wie im 1a/1b-Skript.
  - mask_mode = rotor_cone — die Subtraktion zielt auf die *bekannten*
    Rotor-/Spaltrichtungen (inter_rotor_midpoints), nicht auf den generischen
    drone_disk-Gürtel.

Drei Fälle:
  - ext_only             : reines externes Signal, keine Drohne, kein notch
  - mixed_with_notch     : drone+ext, notch entfernt BPF-Tonale
  - mixed_no_notch       : drone+ext, ohne notch

Verglichen gegen die ext_only Ground Truth. Hires CSM (nperseg=8192).

Layout:
  - oben:   PSD [dB] post (clean_sc solid · orth dashed) + GT (grau)
  - mitte:  PSD error 10·log10(post/GT) [dB]
  - unten:  MAE-Tabelle pro Band (case × {clean_sc, orth})
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

log = logging.getLogger("s0_rotor_cone_algo_compare")

GRIDS = ("rotor_cone", "hybrid")


def _cfg(algo: str, uses_notch_stage: bool, grid: str) -> str:
    """Base config path for (algorithm, notch-stage, grid variant).

    grid 'rotor_cone' — DOA sphere + cone mask on the known rotor directions.
    grid 'hybrid'     — DOA sphere + explicit near-field drone cells (1a).
    """
    who = "mixed" if uses_notch_stage else "external_only"
    suffix = "_orth" if algo == "orth" else ""
    return f"configs/pipeline_{who}_doa_{grid}{suffix}.yaml"

ALGOS = ("clean_sc", "orth")
ALGO_DASH = {"clean_sc": "solid", "orth": "dash"}

CASES = ("ext_only", "mixed_with_notch", "mixed_no_notch")
CASE_COLORS = {
    "ext_only":         "#1f77b4",
    "mixed_with_notch": "#2ca02c",
    "mixed_no_notch":   "#d62728",
}
CASE_LABELS = {
    "ext_only":         "ext_only",
    "mixed_with_notch": "mixed + notch",
    "mixed_no_notch":   "mixed, no notch",
}
# (audio source, uses-notch-config) per case
CASE_USES_NOTCH_CFG = {
    "ext_only": False, "mixed_with_notch": True, "mixed_no_notch": False,
}

HIRES_NPERSEG = 8192
HIRES_NOVERLAP = 4096
F_MIN_HZ = 50.0
F_MAX_HZ = 6000.0

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


def _run(case: str, algo: str, grid: str, s: dict, out_root: Path) -> Path:
    base_cfg = Path(_cfg(algo, CASE_USES_NOTCH_CFG[case], grid))
    audio = s["mixed_audio_h5"] if case != "ext_only" else s["ext_only_audio_h5"]
    overrides = _hires_overrides(
        audio, s["ext_only_gt_h5"], s["target_point_m"], s["focal_radius_m"],
    )
    run_dir = out_root / f"s0_{case}__{algo}"
    log.info("run case=%s algo=%s grid=%s base=%s", case, algo, grid, base_cfg)
    return run_pipeline_with_overrides(
        base_config_path=base_cfg, overrides=overrides, output_dir=run_dir,
    )


def _post_psd_from_run(run_dir: Path, target: tuple, mic_xml: Path) -> tuple:
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs = f["frequencies"][:]
        csm = f["csm_real"][:] + 1j * f["csm_imag"][:]
    mics = load_mic_geom_xml(mic_xml)
    cal = range_compensation_factor(mics, target)
    return freqs, steer_to_psd(csm, freqs, mics, target) * cal


def _middle_segment(audio: np.ndarray, fs: float, seg_s: float = 10.0) -> np.ndarray:
    n_seg = int(round(seg_s * fs))
    if audio.shape[0] <= n_seg:
        return audio.astype(np.float64)
    start = (audio.shape[0] - n_seg) // 2
    return audio[start:start + n_seg].astype(np.float64)


def _pre_psd(audio: np.ndarray, fs: float, target: tuple,
             mics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSM from audio, steer to target — the PSD *before* spatial filtering."""
    cfg = CsmConfig(
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, window="hann",
        diag_loading_rel=1e-6, f_min_hz=F_MIN_HZ, f_max_hz=F_MAX_HZ,
    )
    csm, freqs = build_measurement_csm(audio, sample_rate=fs, cfg=cfg)
    cal = range_compensation_factor(mics, target)
    return freqs, steer_to_psd(csm, freqs, mics, target) * cal


def _pre_for_case(case: str, run_dir: Path, audio_h5_raw: str,
                  target: tuple, mics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSD fed into the array_filter stage (same for both algorithms).

    mixed_with_notch: post-notch time_data from filtered.h5; else the raw
    input audio, middle 10s segment.
    """
    if case == "mixed_with_notch":
        with h5py.File(run_dir / "filtered.h5", "r") as f:
            audio = f["time_data"][:].astype(np.float64)
            fs = float(f["time_data"].attrs["sample_freq"])
    else:
        d = load_synth_h5(audio_h5_raw)
        fs = float(d["sample_rate"])
        audio = _middle_segment(d["time_data"], fs)
    return _pre_psd(audio, fs, target, mics)


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
    p.add_argument("--grid", choices=GRIDS, default="rotor_cone",
                   help="DOA grid variant: rotor_cone (cone mask) or hybrid "
                        "(sphere + explicit near-field drone cells, 1a).")
    p.add_argument("--out-dir", default=None, type=Path,
                   help="default: results/separation_study/s0_<grid>_algo_compare")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = (args.out_dir if args.out_dir is not None
               else Path(f"results/separation_study/s0_{args.grid}_algo_compare"))
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _s0_inputs(args.scenario_paths)
    gt_h5 = Path(s["ext_only_gt_h5"])
    mic_xml = Path(s["mic_geom_xml"])
    target = s["target_point_m"]

    run_dirs = {(case, algo): _run(case, algo, args.grid, s, out_dir)
                for case in CASES for algo in ALGOS}

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.42, 0.16],
        shared_xaxes=False, vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("PSD [dB]: pre (dotted) · post clean_sc (solid) · post orth (dashed) · GT (grau)",
                        "PSD error 10·log10(post / GT) [dB]",
                        "MAE per band [dB]"),
    )

    mics = load_mic_geom_xml(mic_xml)
    raw_audio = {
        "ext_only":       s["ext_only_audio_h5"],
        "mixed_no_notch": s["mixed_audio_h5"],
        "mixed_with_notch": s["mixed_audio_h5"],  # unused — pre uses filtered.h5
    }

    mae_rows: list[list] = []
    gt_drawn = False
    for case in CASES:
        color = CASE_COLORS[case]
        label = CASE_LABELS[case]

        # pre — PSD into the array_filter stage; identical for both algorithms.
        f_pre, pre = _pre_for_case(
            case, run_dirs[(case, "clean_sc")], raw_audio[case], target, mics,
        )
        m_pre = f_pre >= F_MIN_HZ
        fig.add_trace(go.Scatter(
            x=f_pre[m_pre],
            y=10 * np.log10(np.maximum(pre[m_pre], 1e-30)),
            mode="lines", legendgroup=f"{case}pre", name=f"{label} · pre",
            line=dict(color=color, width=1.0, dash="dot"), opacity=0.7,
        ), row=1, col=1)

        for algo in ALGOS:
            freqs, post = _post_psd_from_run(run_dirs[(case, algo)], target, mic_xml)
            gt = _gt_psd_on_grid(gt_h5, freqs)
            mask = freqs >= F_MIN_HZ
            x = freqs[mask]
            post_db = 10 * np.log10(np.maximum(post[mask], 1e-30))
            gt_db = 10 * np.log10(np.maximum(gt[mask], 1e-30))
            err_db = post_db - gt_db
            dash = ALGO_DASH[algo]

            fig.add_trace(go.Scatter(
                x=x, y=post_db, mode="lines", legendgroup=f"{case}{algo}",
                name=f"{label} · {algo}",
                line=dict(color=color, width=1.6, dash=dash),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=err_db, mode="lines", legendgroup=f"{case}{algo}",
                name=f"{label} · {algo}", showlegend=False,
                line=dict(color=color, width=1.6, dash=dash),
            ), row=2, col=1)

            if not gt_drawn:
                fig.add_trace(go.Scatter(
                    x=x, y=gt_db, mode="lines", name="ground truth",
                    legendgroup="gt", line=dict(color="#888", width=2.0),
                ), row=1, col=1)
                gt_drawn = True

            mae_per_band = []
            for _, (f_lo, f_hi) in MAE_BANDS.items():
                bm = (x >= f_lo) & (x <= f_hi)
                mae_per_band.append(
                    float(np.mean(np.abs(err_db[bm]))) if bm.any() else float("nan")
                )
            mae_rows.append([f"{label} · {algo}", *mae_per_band])

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"),
                  row=2, col=1)
    for row in (1, 2):
        fig.update_xaxes(type="log", range=[np.log10(F_MIN_HZ), np.log10(6000)],
                         row=row, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="PSD [dB]", row=1, col=1)
    fig.update_yaxes(title_text="error [dB]", row=2, col=1)

    header = ["case · algorithm", *MAE_BANDS.keys()]
    cols_t = list(zip(*mae_rows))
    row_color = []
    for case in CASES:
        row_color += [CASE_COLORS[case], CASE_COLORS[case]]
    cell_colors = [
        row_color,
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
        title=(f"S0 {args.grid} — clean_sc vs orth (Ansatz 2: residual_csm → target) "
               f"(nperseg={HIRES_NPERSEG}, Δf≈{51200/HIRES_NPERSEG:.2f} Hz)"),
        height=1080, width=1150, hovermode="x unified",
    )
    out_html = out_dir / f"s0_{args.grid}_algo_compare.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
