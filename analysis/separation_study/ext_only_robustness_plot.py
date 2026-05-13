"""ext_only S0/S1/S2/S3 × {rotor/drone/target}_cone Vergleichsplot.

Fährt für jede (config, scenario)-Kombination einen ext_only-Pipeline-Run und
plottet die target_psd-Kurven aller drei Methoden zusammen mit pre und GT in
einem 4×1 Subplot-Layout (1 row pro scenario).

ext_only = keine Drohne im Signal → reine Validation der spatial filtering.
Pre/Post sollten praktisch deckungsgleich sein (CLEAN-SC entfernt nichts wenn
keine drone-Quelle da ist) und auf GT liegen.
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
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.csm import CsmConfig, build_measurement_csm
from martymicfly.processing.steering import (
    range_compensation_factor, steer_to_psd,
)

log = logging.getLogger("ext_only_robustness")

CONFIGS = {
    "rotor_cone":  "configs/pipeline_external_only_doa_rotor_cone.yaml",
    "drone_disk":  "configs/pipeline_external_only_doa_drone_disk.yaml",
    "target_cone": "configs/pipeline_external_only_doa_target_cone.yaml",
}

METHOD_COLORS = {
    "rotor_cone":  "#d62728",   # red
    "drone_disk":  "#9467bd",   # purple
    "target_cone": "#ff7f0e",   # orange
}


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


def _run_ext_only(method: str, scenario: str, base_cfg: Path, s: dict,
                  out_root: Path) -> Path:
    run_dir = out_root / f"{method}__{scenario}"
    overrides = {
        "input.audio_h5": s["ext_only_audio_h5"],
        "input.ground_truth_h5": s["ext_only_gt_h5"],
        "array_filter.target_point_m": list(s["target_point_m"]),
        "array_filter.doa_grid.focal_radius_m": s["focal_radius_m"],
        "plots.enabled": False,
    }
    log.info("run ext_only %s/%s overrides=%s", method, scenario, overrides)
    return run_pipeline_with_overrides(
        base_config_path=base_cfg, overrides=overrides, output_dir=run_dir,
    )


def _csm_pre_from_audio(audio_h5: Path, csm_cfg: dict, segment_s: float,
                        f_min: float, f_max: float):
    """Compute CSM_pre over middle-segment of audio_h5."""
    d = load_synth_h5(str(audio_h5))
    audio = d["time_data"]   # (N, M)
    fs = float(d["sample_rate"])
    n_seg = int(segment_s * fs)
    if audio.shape[0] > n_seg:
        start = (audio.shape[0] - n_seg) // 2
        audio = audio[start:start + n_seg]
    cfg = CsmConfig(
        nperseg=int(csm_cfg["nperseg"]),
        noverlap=int(csm_cfg["noverlap"]),
        window=csm_cfg.get("window", "hann"),
        diag_loading_rel=float(csm_cfg.get("diag_loading_rel", 0.0)),
        f_min_hz=f_min, f_max_hz=f_max,
    )
    return build_measurement_csm(audio, sample_rate=fs, cfg=cfg)


def _target_psd_curves(audio_h5: Path, gt_h5: Path, run_dir: Path,
                       target_point: tuple, mic_xml: Path, base_cfg: Path):
    cfg = yaml.safe_load(base_cfg.read_text())
    af_stage = next(st for st in cfg["stages"] if st.get("kind") == "array_filter")
    csm_cfg = af_stage["csm"]
    seg = float(cfg.get("segment", {}).get("duration", 10.0))
    f_min = float(csm_cfg.get("f_min_hz", 0.0))
    f_max = float(csm_cfg.get("f_max_hz", 1.0e9))

    csm_pre, freqs_pre = _csm_pre_from_audio(audio_h5, csm_cfg, seg, f_min, f_max)
    with h5py.File(run_dir / "residual_csm.h5", "r") as f:
        freqs_post = f["frequencies"][:]
        csm_post = f["csm_real"][:] + 1j * f["csm_imag"][:]
    assert np.allclose(freqs_pre, freqs_post), "freq mismatch pre/post"
    freqs = freqs_pre

    mic_positions = load_mic_geom_xml(mic_xml)
    cal = range_compensation_factor(mic_positions, target_point)
    psd_pre = steer_to_psd(csm_pre, freqs, mic_positions, target_point) * cal
    psd_post = steer_to_psd(csm_post, freqs, mic_positions, target_point) * cal

    gt = load_ground_truth(str(gt_h5))
    gt_f, gt_psd = welch(
        gt.signal, fs=gt.sample_rate,
        nperseg=int(csm_cfg["nperseg"]), noverlap=int(csm_cfg["noverlap"]),
        scaling="density",
    )
    gt_psd = np.interp(freqs, gt_f, gt_psd)
    return freqs, psd_pre, psd_post, gt_psd


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

    # 1. Ensure synth + run pipelines for all (method, scenario)
    run_dirs: dict = {}
    for sname, s in scenarios.items():
        _ensure_synth(s)
        for method, cfg in CONFIGS.items():
            run_dirs[(method, sname)] = _run_ext_only(
                method, sname, Path(cfg), s, args.out_dir)

    # 2. Compute curves. pre/gt are identical across methods (same input audio
    # + GT), so we read pre/gt once from target_cone and only post per method.
    curves: dict = {}
    for sname, s in scenarios.items():
        for method, cfg in CONFIGS.items():
            run_dir = run_dirs[(method, sname)]
            freqs, pre, post, gt = _target_psd_curves(
                audio_h5=Path(s["ext_only_audio_h5"]),
                gt_h5=Path(s["ext_only_gt_h5"]),
                run_dir=run_dir, target_point=s["target_point_m"],
                mic_xml=Path(s["mic_geom_xml"]), base_cfg=Path(cfg),
            )
            curves[(method, sname)] = (freqs, pre, post, gt)

    # 3. Plot 4 rows × 1 col. Per row: 1 pre (gray), 3 post (color/method), 1 GT (dashed).
    titles = [
        f"{s} target={scenarios[s]['target_point_m']}, r={scenarios[s]['focal_radius_m']}m"
        for s in ("S0", "S1", "S2", "S3")
    ]
    fig = make_subplots(rows=4, cols=1, subplot_titles=titles,
                        shared_xaxes=True, vertical_spacing=0.045)

    for row, sname in enumerate(("S0", "S1", "S2", "S3"), start=1):
        # pre and gt are identical across methods — take from target_cone
        freqs, pre, _, gt = curves[("target_cone", sname)]
        fig.add_trace(go.Scatter(
            x=freqs, y=10 * np.log10(np.maximum(pre, 1e-30)),
            mode="lines", name="pre",
            line=dict(color="#999999", width=1),
            legendgroup="pre", showlegend=(row == 1),
        ), row=row, col=1)
        for method in ("rotor_cone", "drone_disk", "target_cone"):
            _, _, post, _ = curves[(method, sname)]
            fig.add_trace(go.Scatter(
                x=freqs, y=10 * np.log10(np.maximum(post, 1e-30)),
                mode="lines", name=f"post {method}",
                line=dict(color=METHOD_COLORS[method], width=1.5),
                legendgroup=method, showlegend=(row == 1),
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=10 * np.log10(np.maximum(gt, 1e-30)),
            mode="lines", name="ground truth",
            line=dict(color="#2ca02c", dash="dot", width=1.5),
            legendgroup="gt", showlegend=(row == 1),
        ), row=row, col=1)
        fig.update_yaxes(title_text="PSD [dB]", row=row, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=4, col=1)
    fig.update_layout(
        title="ext_only target_psd vs ground truth — S0/S1/S2/S3 × {rotor,drone,target}_cone",
        height=1300, width=1200, hovermode="x unified",
    )
    out_html = args.out_dir / "target_psd_compare.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    log.info("wrote %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
