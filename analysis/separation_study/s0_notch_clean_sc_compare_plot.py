"""S0 drone_disk — Ansatz 1a vs 1b: räumliche Filterung per Map + ROI.

Schwesterskript zu s0_notch_compare_plot.py (Ansatz 2 = CSM-Subtraktion).
Hier wird die Drohne *im Map-Raum* über die drone_disk-Maske ausgeblendet:

    post(f) = Σ_{r ∉ drone_disk}  Map(r, f)
    pre(f)  = Σ_{alle r}          Map(r, f)

Zwei Maps werden verglichen:
  - Ansatz 1a — konventionelle "dirty" Beamforming-Map (range-kompensiert via
    steer_to_psd_matched). Keine Deconvolution; PSF-Leckage inflationiert das
    ROI-Integral.
  - Ansatz 1b — CLEAN-SC deconvolved source_map (aus source_map.h5).
    Sidelobe-frei, aber Lokalisierung kann driften.

Kalibrierung (Trace-Anker): die CLEAN-SC source_map ist trace-normiert
(Σ Zellen ≈ trace(CSM)). Der validierte Pegel-Pfad ist
steer_to_psd(CSM, target)·range_compensation. Der geometrische Konstantfaktor
K = (steer·rcf)/Σpowers wird aus dem ext_only-Fall (saubere Einzelquelle)
bestimmt und auf alle drei Fälle angewandt. Ansatz 1a braucht keinen Anker —
steer_to_psd_matched liefert S_q direkt pro Zelle.

Drei Pipeline-Konfigurationen über S0 (wie im Schwesterskript):
  - ext_only             : reines externes Signal, keine Drohne, kein notch
  - mixed_with_notch     : drone+ext, notch entfernt BPF-Tonale
  - mixed_no_notch       : drone+ext, ohne notch

Verglichen gegen die ext_only Ground Truth. Hires CSM (nperseg=8192).

Layout:
  - oben:   PSD [dB] — 1a post (solid) · 1b post (dashed) · GT (grau)
  - mitte:  PSD error 10·log10(post/GT) [dB] — 1a solid · 1b dashed
  - unten:  MAE-Tabelle pro Band (case × {1a, 1b})
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
from martymicfly.io.source_map_h5 import read_source_map
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("s0_notch_clean_sc_compare")

EXT_ONLY_CFG = "configs/pipeline_external_only_doa_drone_disk.yaml"
MIXED_CFG    = "configs/pipeline_mixed_doa_drone_disk.yaml"

SPEED_OF_SOUND = 343.0

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
CASES = ("ext_only", "mixed_with_notch", "mixed_no_notch")

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


def _run(case: str, base_cfg: Path, audio_h5: str, gt_h5: str,
         target: tuple, focal_radius: float, out_root: Path) -> Path:
    run_dir = out_root / f"s0_{case}"
    overrides = _hires_overrides(audio_h5, gt_h5, target, focal_radius)
    log.info("run %s overrides=%s", case, overrides)
    return run_pipeline_with_overrides(
        base_config_path=base_cfg, overrides=overrides, output_dir=run_dir,
    )


def _middle_segment(audio: np.ndarray, fs: float, seg_s: float = 10.0) -> np.ndarray:
    n_seg = int(round(seg_s * fs))
    if audio.shape[0] <= n_seg:
        return audio.astype(np.float64)
    start = (audio.shape[0] - n_seg) // 2
    return audio[start:start + n_seg].astype(np.float64)


def _measurement_csm(case: str, run_dir: Path, audio_h5_raw: str) -> tuple:
    """CSM that the array_filter stage saw — same segment/config as the run.

    - mixed_with_notch: post-notch time_data from filtered.h5.
    - others: raw input audio, middle 10s segment.
    """
    if case == "mixed_with_notch":
        with h5py.File(run_dir / "filtered.h5", "r") as f:
            audio = f["time_data"][:].astype(np.float64)
            fs = float(f["time_data"].attrs["sample_freq"])
    else:
        d = load_synth_h5(audio_h5_raw)
        fs = float(d["sample_rate"])
        audio = _middle_segment(d["time_data"], fs)
    cfg = CsmConfig(
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP,
        window="hann", diag_loading_rel=1e-6,
        f_min_hz=F_MIN_HZ, f_max_hz=F_MAX_HZ,
    )
    csm, freqs = build_measurement_csm(audio, sample_rate=fs, cfg=cfg)
    return csm, freqs


def _matched_map(csm: np.ndarray, freqs: np.ndarray, mics: np.ndarray,
                 positions: np.ndarray) -> np.ndarray:
    """Conventional range-compensated beamforming map B(f, g) in S_q units.

    Vectorized form of steer_to_psd_matched evaluated at every grid cell:
    h_m = (4π r_m)·exp(+j 2π f r_m / c),  B = (1/M²)·hᴴ·CSM·h.
    """
    diff = mics[None, :, :] - positions[:, None, :]   # (G, M, 3)
    r = np.linalg.norm(diff, axis=2)                  # (G, M)
    amp = 4.0 * np.pi * r                             # (G, M)
    n_m = mics.shape[0]
    B = np.empty((freqs.shape[0], positions.shape[0]), dtype=np.float64)
    for fi, f in enumerate(freqs):
        h = amp * np.exp(2j * np.pi * f * r / SPEED_OF_SOUND)   # (G, M)
        ch = np.einsum("mn,gn->gm", csm[fi], h)                 # CSM·h per cell
        B[fi] = np.real(np.einsum("gm,gm->g", h.conj(), ch)) / (n_m * n_m)
    return B


def _interp_to(freqs_dst: np.ndarray, freqs_src: np.ndarray,
               y_src: np.ndarray) -> np.ndarray:
    if freqs_dst.shape == freqs_src.shape and np.allclose(freqs_dst, freqs_src):
        return y_src
    return np.interp(freqs_dst, freqs_src, y_src)


def _trace_anchor_K(ext_run: Path, ext_audio_h5: str, mics: np.ndarray,
                    target: tuple) -> float:
    """Geometric source_map→PSD constant from the clean ext_only case.

    K = median_f[ steer_to_psd(CSM,target)·range_compensation / Σ_cells powers ].
    For a single source at target this ratio is frequency-flat geometry; the
    median guards against bins where the deconvolved Σpowers ≈ 0.
    """
    csm, f_csm = _measurement_csm("ext_only", ext_run, ext_audio_h5)
    rcf = range_compensation_factor(mics, target)
    psd_steer = steer_to_psd(csm, f_csm, mics, target) * rcf
    sm = read_source_map(str(ext_run / "source_map.h5"))
    total = _interp_to(f_csm, sm["frequencies"], sm["powers"].sum(axis=1))
    valid = total > 0
    K = float(np.median(psd_steer[valid] / total[valid]))
    log.info("trace-anchor K = %.4e (ext_only)", K)
    return K


def _gt_psd_on_grid(gt_h5: Path, freqs: np.ndarray) -> np.ndarray:
    gt = load_ground_truth(str(gt_h5))
    f_gt, p_gt = welch(
        gt.signal, fs=gt.sample_rate,
        nperseg=HIRES_NPERSEG, noverlap=HIRES_NOVERLAP, scaling="density",
    )
    return np.interp(freqs, f_gt, p_gt)


def _case_curves(case: str, run_dir: Path, audio_h5_raw: str,
                 mics: np.ndarray, target: tuple, K: float) -> dict:
    """Return per-case freqs + 1a/1b pre/post PSD curves (source-PSD units)."""
    sm = read_source_map(str(run_dir / "source_map.h5"))
    freqs = sm["frequencies"]
    powers = sm["powers"]                       # (F, G)
    positions = sm["positions"]                 # (G, 3)
    keep = ~sm["drone_mask"]                    # cells NOT in drone_disk

    # Ansatz 1b — CLEAN-SC deconvolved map, trace-anchored.
    pre_1b = powers.sum(axis=1) * K
    post_1b = powers[:, keep].sum(axis=1) * K

    # Ansatz 1a — conventional dirty map, matched-filter calibrated per cell.
    csm, f_csm = _measurement_csm(case, run_dir, audio_h5_raw)
    bmap = _matched_map(csm, f_csm, mics, positions)        # (F_csm, G)
    pre_1a = _interp_to(freqs, f_csm, bmap.sum(axis=1))
    post_1a = _interp_to(freqs, f_csm, bmap[:, keep].sum(axis=1))

    return {
        "freqs": freqs,
        "pre_1a": pre_1a, "post_1a": post_1a,
        "pre_1b": pre_1b, "post_1b": post_1b,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-paths",
                   default="analysis/separation_study/studies/scenario_paths.yaml",
                   type=Path)
    p.add_argument("--out-dir",
                   default="results/separation_study/s0_notch_clean_sc_compare",
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
    mics = load_mic_geom_xml(s["mic_geom_xml"])

    raw_audio = {
        "ext_only":         s["ext_only_audio_h5"],
        "mixed_with_notch": s["mixed_audio_h5"],
        "mixed_no_notch":   s["mixed_audio_h5"],
    }
    run_dirs = {
        "ext_only":         _run("ext_only", Path(EXT_ONLY_CFG),
                                 s["ext_only_audio_h5"], gt_h5, target, fr, args.out_dir),
        "mixed_with_notch": _run("mixed_with_notch", Path(MIXED_CFG),
                                 s["mixed_audio_h5"], gt_h5, target, fr, args.out_dir),
        "mixed_no_notch":   _run("mixed_no_notch", Path(EXT_ONLY_CFG),
                                 s["mixed_audio_h5"], gt_h5, target, fr, args.out_dir),
    }

    K = _trace_anchor_K(run_dirs["ext_only"], raw_audio["ext_only"], mics, target)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.42, 0.16],
        shared_xaxes=False, vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("PSD [dB]: 1a dirty-map (solid) · 1b CLEAN-SC (dashed) · GT (grau) — post = Σ ohne drone_disk",
                        "PSD error 10·log10(post / GT) [dB]: 1a solid · 1b dashed",
                        "MAE per band [dB]"),
    )

    mae_rows: list[list] = []
    gt_drawn = False
    for case in CASES:
        c = _case_curves(case, run_dirs[case], raw_audio[case], mics, target, K)
        freqs = c["freqs"]
        mask = freqs >= F_MIN_HZ
        x = freqs[mask]
        color = CASE_COLORS[case]
        label = CASE_LABELS[case]
        gt = _gt_psd_on_grid(Path(gt_h5), freqs)
        gt_db = 10 * np.log10(np.maximum(gt[mask], 1e-30))

        for method, dash, width in (("1a", "solid", 1.6), ("1b", "dash", 1.6)):
            post = c[f"post_{method}"]
            post_db = 10 * np.log10(np.maximum(post[mask], 1e-30))
            err_db = post_db - gt_db
            fig.add_trace(go.Scatter(
                x=x, y=post_db, mode="lines", legendgroup=f"{case}{method}",
                name=f"{label} · {method}",
                line=dict(color=color, width=width, dash=dash),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=err_db, mode="lines", legendgroup=f"{case}{method}",
                name=f"{label} · {method}", showlegend=False,
                line=dict(color=color, width=width, dash=dash),
            ), row=2, col=1)

            mae_per_band = []
            for _, (f_lo, f_hi) in MAE_BANDS.items():
                band_mask = (x >= f_lo) & (x <= f_hi)
                mae_per_band.append(
                    float(np.mean(np.abs(err_db[band_mask])))
                    if band_mask.any() else float("nan")
                )
            mae_rows.append([f"{label} · {method}", *mae_per_band])

        if not gt_drawn:
            fig.add_trace(go.Scatter(
                x=x, y=gt_db, mode="lines", name="ground truth",
                legendgroup="gt", line=dict(color="#888", width=2.0),
            ), row=1, col=1)
            gt_drawn = True

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"),
                  row=2, col=1)
    for row in (1, 2):
        fig.update_xaxes(type="log", range=[np.log10(F_MIN_HZ), np.log10(6000)],
                         row=row, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="PSD [dB]", row=1, col=1)
    fig.update_yaxes(title_text="error [dB]", row=2, col=1)

    header = ["case · Ansatz", *MAE_BANDS.keys()]
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
        title=(f"S0 drone_disk — Ansatz 1a (dirty-map) vs 1b (CLEAN-SC), Map+ROI Σ ohne drone_disk "
               f"(nperseg={HIRES_NPERSEG}, Δf≈{51200/HIRES_NPERSEG:.2f} Hz, trace-anchor K)"),
        height=1080, width=1150, hovermode="x unified",
    )
    out_html = args.out_dir / "s0_notch_clean_sc_compare.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
