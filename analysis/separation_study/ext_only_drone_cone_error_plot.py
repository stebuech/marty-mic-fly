"""ext_only drone_cone PSD + PSD-Error-zu-GT Plot über S0/S1/S2/S3.

Fährt drone_cone × 4 Szenarien mit höherer CSM-Auflösung (nperseg=8192 →
Δf ≈ 6.25 Hz statt 100 Hz) und f_min=50 Hz. Erzeugt einen 2-row-Plot:
  - oben: absolute PSD-Kurven (post solid, GT dashed) pro Szenario
  - unten: 10·log10(post / GT) pro Szenario

Positive Error-Werte = Pipeline überschätzt PSD (Restenergie aus Side-Lobes
etc.), negative Werte = Über-Subtraktion oder Range-Calibration-Drift.
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
from analysis.separation_study.synth_scenarios import ensure_scenario_pair
from martymicfly.io.ground_truth_h5 import load_ground_truth
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("ext_only_drone_cone_err")

BASE_CFG = "configs/pipeline_external_only_doa_drone_cone.yaml"

SCENARIO_COLORS = {
    "S0": "#1f77b4",   # blue
    "S1": "#ff7f0e",   # orange
    "S2": "#d62728",   # red
    "S3": "#2ca02c",   # green
}

# High-resolution Welch / CSM settings.
HIRES_NPERSEG = 8192
HIRES_NOVERLAP = 4096
F_MIN_HZ = 50.0


def _scenario_inputs(sp_yaml: Path) -> dict:
    sp = yaml.safe_load(sp_yaml.read_text())
    out: dict = {}
    for s in ("S0", "S1", "S2", "S3"):
        d = sp[s]
        target = tuple(d.get("config_overrides", {}).get(
            "array_filter.target_point_m", [0.0, 0.0, -1.5]))
        focal_radius = float(d.get("config_overrides", {}).get(
            "array_filter.doa_grid.focal_radius_m", 1.5))
        out[s] = {
            "ext_only_audio_h5": d["ext_only_audio_h5"],
            "ext_only_gt_h5": d["ext_only_gt_h5"],
            "target_point_m": target,
            "focal_radius_m": focal_radius,
            "drone_artifact": d["drone_artifact"],
            "mic_geom_xml": d["mic_geom_xml"],
            "out_mixed_h5": d.get("audio_h5"),
            "out_mixed_gt_h5": d.get("ground_truth_h5"),
            "scenario_key": s,
        }
    return out


def _ensure_synth(s: dict) -> None:
    if s["scenario_key"] == "S0":
        return
    ensure_scenario_pair(
        scenario=s["scenario_key"],
        base_drone_artifact=s["drone_artifact"],
        mic_geom_xml=s["mic_geom_xml"],
        out_mixed_h5=Path(s["out_mixed_h5"]),
        out_mixed_gt_h5=Path(s["out_mixed_gt_h5"]),
        out_ext_only_h5=Path(s["ext_only_audio_h5"]),
        out_ext_only_gt_h5=Path(s["ext_only_gt_h5"]),
    )


def _run(scenario: str, base_cfg: Path, s: dict, out_root: Path) -> Path:
    run_dir = out_root / f"drone_cone__{scenario}"
    overrides = {
        "input.audio_h5": s["ext_only_audio_h5"],
        "input.ground_truth_h5": s["ext_only_gt_h5"],
        "array_filter.target_point_m": list(s["target_point_m"]),
        "array_filter.doa_grid.focal_radius_m": s["focal_radius_m"],
        "array_filter.csm.nperseg": HIRES_NPERSEG,
        "array_filter.csm.noverlap": HIRES_NOVERLAP,
        "array_filter.csm.f_min_hz": F_MIN_HZ,
        "plots.enabled": False,
    }
    log.info("run drone_cone/%s overrides=%s", scenario, overrides)
    return run_pipeline_with_overrides(
        base_config_path=base_cfg, overrides=overrides, output_dir=run_dir,
    )


def _post_psd_from_run(run_dir: Path, target_point: tuple, mic_xml: Path) -> tuple:
    """Return (freqs, post_psd) in source-PSD units (range-compensated)."""
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs = f["frequencies"][:]
        csm = f["csm_real"][:] + 1j * f["csm_imag"][:]
    mic_positions = load_mic_geom_xml(mic_xml)
    cal = range_compensation_factor(mic_positions, target_point)
    post = steer_to_psd(csm, freqs, mic_positions, target_point) * cal
    return freqs, post


def _gt_psd_on_grid(gt_h5: Path, freqs: np.ndarray) -> np.ndarray:
    """Welch-PSD der GT auf gleicher Frequenz-Auflösung wie CSM (8192/4096)."""
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
                   default="results/separation_study/ext_only_robustness",
                   type=Path)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_inputs(args.scenario_paths)

    run_dirs: dict = {}
    for sname, s in scenarios.items():
        _ensure_synth(s)
        run_dirs[sname] = _run(sname, Path(BASE_CFG), s, args.out_dir)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("PSD: post (solid) vs ground truth (dashed)",
                        "PSD error: 10·log10(post / GT) [dB]"),
    )
    for sname, s in scenarios.items():
        freqs, post = _post_psd_from_run(
            run_dirs[sname], s["target_point_m"], Path(s["mic_geom_xml"]),
        )
        gt = _gt_psd_on_grid(Path(s["ext_only_gt_h5"]), freqs)
        mask = freqs >= F_MIN_HZ
        x = freqs[mask]
        color = SCENARIO_COLORS[sname]
        # Top: absolute PSDs (post + GT)
        fig.add_trace(go.Scatter(
            x=x, y=10 * np.log10(np.maximum(post[mask], 1e-30)),
            mode="lines", name=f"{sname} post", legendgroup=sname,
            line=dict(color=color, width=1.5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=10 * np.log10(np.maximum(gt[mask], 1e-30)),
            mode="lines", name=f"{sname} GT", legendgroup=sname,
            line=dict(color=color, width=1.0, dash="dash"),
        ), row=1, col=1)
        # Bottom: error
        err_db = (10 * np.log10(np.maximum(post[mask], 1e-30))
                  - 10 * np.log10(np.maximum(gt[mask], 1e-30)))
        fig.add_trace(go.Scatter(
            x=x, y=err_db, mode="lines", name=f"{sname} err",
            legendgroup=sname, showlegend=False,
            line=dict(color=color, width=1.5),
        ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"),
                  row=2, col=1)
    fig.update_xaxes(type="log", range=[np.log10(F_MIN_HZ), np.log10(6000)])
    fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="PSD [dB]", row=1, col=1)
    fig.update_yaxes(title_text="error [dB]", row=2, col=1)
    fig.update_layout(
        title=(f"ext_only drone_cone — S0/S1/S2/S3 (nperseg={HIRES_NPERSEG}, "
               f"Δf≈{51200/HIRES_NPERSEG:.2f} Hz)"),
        height=900, width=1100, hovermode="x unified",
    )
    out_html = args.out_dir / "drone_cone_psd_error.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
