"""CLI entry point: YAML config → notch pipeline → outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from martymicfly.config import AppConfig
from martymicfly.eval.metrics import compute_metrics
from martymicfly.eval.plots import plot_channel_html
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.io.write_filtered import write_filtered
from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
    linear_r_schedule,
)
from martymicfly.processing.notch import NotchStage, NotchStageParams
from martymicfly.processing.pipeline import PipelineContext, run_pipeline

log = logging.getLogger("martymicfly.run_notch")

# Default config path — repo-root/configs/example_notch.yaml. Resolved from this
# file's location so calling main() from a Python console (without argv) works.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "example_notch.yaml"


def _select_segment(cfg, n_total: int, fs: float) -> tuple[float, int]:
    mode = cfg.segment.mode
    if mode == "explicit":
        start = cfg.segment.start
        end = cfg.segment.end
        n_seg = int(round((end - start) * fs))
        return float(start), n_seg
    duration = cfg.segment.duration
    n_seg = int(round(duration * fs))
    if n_seg > n_total:
        raise ValueError(f"segment duration {duration}s > recording {n_total/fs:.2f}s")
    if mode == "head":
        return 0.0, n_seg
    if mode == "tail":
        return (n_total - n_seg) / fs, n_seg
    # middle (default)
    start_idx = (n_total - n_seg) // 2
    return start_idx / fs, n_seg


def _resolve_pole_radius(cfg, n_harmonics: int, fs: float,
                         per_motor_bpf: np.ndarray):
    pr = cfg.notch.pole_radius
    if pr.mode == "scalar":
        return float(pr.value), f"scalar:{pr.value:.4f}"
    delta_bpf = pr.delta_bpf_hz
    if delta_bpf is None:
        mean_bpf = per_motor_bpf.mean(axis=0)
        delta_bpf = float(mean_bpf.max() - mean_bpf.min())
    sched = linear_r_schedule(
        n_harmonics=n_harmonics, fs=fs,
        delta_bpf_hz=delta_bpf, k_cover=pr.k_cover, margin_hz=pr.margin_hz,
        r_min=pr.r_min, r_max=pr.r_max,
    )
    repr_ = (f"linear:k={pr.k_cover},margin={pr.margin_hz}"
             f",delta_bpf={delta_bpf:.2f}")
    return sched, repr_


def _resolve_channels(cfg, n_total_channels: int) -> list[int]:
    if cfg.channels.selection == "all":
        return list(range(n_total_channels))
    return list(cfg.channels.list)


def _resolve_plot_channels(cfg, channels: list[int]) -> list[int]:
    if cfg.plots.channel_subset is None:
        return channels
    subset = set(cfg.plots.channel_subset)
    return [c for c in channels if c in subset]


def _resolve_fmax(cfg, per_motor_bpf: np.ndarray, n_harmonics: int,
                  fs: float) -> float:
    if cfg.plots.fmax_hz is not None:
        return float(cfg.plots.fmax_hz)
    return min(1.1 * float(per_motor_bpf.max()) * n_harmonics, fs / 2.0)


def _write_metrics(metrics: dict, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fieldnames = [
        "channel", "broadband_pre_db", "broadband_post_db", "broadband_delta_db",
        "tonal_total_pre_db", "tonal_total_post_db", "tonal_reduction_db",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ch in metrics["channels"]:
            w.writerow({k: ch[k] for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run AP2-A stage 1 (notch) pipeline.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"YAML config path. Default: {DEFAULT_CONFIG}")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output.dir from the YAML config.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = AppConfig(**cfg_dict)

    config_hash = cfg.config_hash()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + "_" + config_hash
    out_template = str(args.output_dir) if args.output_dir is not None else cfg.output.dir
    run_dir = Path(out_template.replace("{run_id}", run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run_id=%s, output=%s", run_id, run_dir)

    # 1. Load
    src = load_synth_h5(cfg.input.audio_h5)
    geom = load_mic_geom_xml(cfg.input.mic_geom_xml)
    if geom.shape[0] != src["time_data"].shape[1]:
        raise ValueError(
            f"mic_geom rows ({geom.shape[0]}) != channels ({src['time_data'].shape[1]})"
        )

    # 2. Segment
    seg_start_s, n_seg = _select_segment(cfg, src["time_data"].shape[0], src["sample_rate"])
    seg_start_idx = int(round(seg_start_s * src["sample_rate"]))
    signal = np.ascontiguousarray(
        src["time_data"][seg_start_idx:seg_start_idx + n_seg], dtype=np.float64,
    )

    # 3. Frequencies
    per_motor_bpf = interpolate_per_motor_bpf(
        src["rpm_per_esc"], t_start=seg_start_s,
        n_samples=n_seg, sample_rate=src["sample_rate"],
        n_blades=cfg.rotor.n_blades,
    )
    harm_matrix = build_harmonic_matrix(per_motor_bpf, cfg.rotor.n_harmonics)
    pole_radius, pole_repr = _resolve_pole_radius(
        cfg, cfg.rotor.n_harmonics, src["sample_rate"], per_motor_bpf,
    )

    # 4. Pipeline
    ctx = PipelineContext(
        time_data=signal,
        sample_rate=src["sample_rate"],
        rpm_per_esc=src["rpm_per_esc"],
        mic_positions=geom,
        per_motor_bpf=per_motor_bpf,
        harm_matrix=harm_matrix,
        metadata={},
    )
    notch_stage = NotchStage(NotchStageParams(
        n_blades=cfg.rotor.n_blades,
        n_harmonics=cfg.rotor.n_harmonics,
        pole_radius=pole_radius,
        multichannel=cfg.notch.multichannel,
        block_size=cfg.notch.block_size,
    ))
    ctx = run_pipeline([notch_stage], ctx)

    # 5. Metrics
    channels = _resolve_channels(cfg, signal.shape[1])
    fmax = _resolve_fmax(cfg, per_motor_bpf, cfg.rotor.n_harmonics, src["sample_rate"])
    metrics = compute_metrics(
        pre=ctx.metadata["pre_notch"], post=ctx.time_data,
        fs=src["sample_rate"], per_motor_bpf=per_motor_bpf,
        n_harmonics=cfg.rotor.n_harmonics, pole_radius=pole_radius,
        channels=channels,
        welch_nperseg=cfg.metrics.welch_nperseg,
        welch_noverlap=cfg.metrics.welch_noverlap,
        bandwidth_factor=cfg.metrics.bandwidth_factor,
        broadband_low_hz=cfg.metrics.broadband_low_hz,
        fmax_hz=fmax,
    )
    metrics.update({
        "run_id": run_id,
        "config_hash": config_hash,
        "input_h5": str(cfg.input.audio_h5),
        "segment": {"start_s": seg_start_s, "duration_s": n_seg / src["sample_rate"]},
    })
    _write_metrics(metrics, run_dir / cfg.output.metrics_json,
                   run_dir / cfg.output.metrics_csv)

    # 6. Plots
    if cfg.plots.enabled:
        plot_channels = _resolve_plot_channels(cfg, channels)
        plots_dir = run_dir / cfg.output.plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        for ch in plot_channels:
            plot_channel_html(
                pre=ctx.metadata["pre_notch"], post=ctx.time_data,
                fs=src["sample_rate"], per_motor_bpf=per_motor_bpf,
                channel=ch, outpath=plots_dir / f"ch{ch:02d}.html",
                n_harmonics=cfg.rotor.n_harmonics, fmax_hz=fmax,
                spectrogram_window=cfg.plots.spectrogram_window,
                spectrogram_overlap=cfg.plots.spectrogram_overlap,
            )

    # 7. Filtered HDF5
    write_filtered(
        out_path=run_dir / cfg.output.filtered_h5,
        filtered_time_data=ctx.time_data,
        sample_rate=src["sample_rate"],
        rpm_per_esc=src["rpm_per_esc"],
        attrs={
            "martymicfly_version": "0.1.0",
            "input_h5": str(cfg.input.audio_h5),
            "config_hash": config_hash,
            "segment_start_s": float(seg_start_s),
            "segment_duration_s": float(n_seg / src["sample_rate"]),
            "n_blades": int(cfg.rotor.n_blades),
            "n_harmonics": int(cfg.rotor.n_harmonics),
            "notch_mode": "rpm_external_zerophase",
            "pole_radius_repr": pole_repr,
        },
    )

    # 8. Snapshot config
    if cfg.output.copy_config:
        shutil.copy(args.config, run_dir / "config.yaml")
        (run_dir / "config.hash").write_text(config_hash + "\n", encoding="utf-8")

    log.info("done. outputs in %s", run_dir)
    return 0


if __name__ == "__main__":
    _rc = main()
    # Only raise SystemExit on failure — keeps IPython/PyCharm runfile() console
    # clean on success (rc=0 would otherwise trigger an exit warning).
    if _rc:
        raise SystemExit(_rc)
