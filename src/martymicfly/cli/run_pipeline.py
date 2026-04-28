"""CLI entry point: YAML config → generic stages pipeline → outputs."""

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
)
from martymicfly.processing.pipeline import (
    PipelineContext,
    build_pipeline,
    run_pipeline,
)

log = logging.getLogger("martymicfly.run_pipeline")

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


def _notch_pole_repr(stages_cfg) -> str:
    """Reproduce the pole-radius repr string from the original run_notch CLI.

    Linear-mode is currently rejected by _build_notch_stage (NotImplementedError),
    so we only need to cover the scalar path here.
    """
    for s in stages_cfg:
        if s.kind == "notch":
            pr = s.pole_radius
            if pr.mode == "scalar":
                return f"scalar:{pr.value:.4f}"
            raise AssertionError("unreachable: linear-mode rejected by _build_notch_stage")
    return ""


def _notch_stage_cfg(stages_cfg):
    for s in stages_cfg:
        if s.kind == "notch":
            return s
    return None


def _resolve_run_dir(out_template: str, run_id: str) -> Path:
    """Resolve the per-run output directory.

    If out_template contains '{run_id}', it is substituted in place
    (preserving any prior structure around it). If not, the run is
    auto-nested as out_template / run_id so multiple runs against the same
    static directory don't collide.
    """
    if "{run_id}" in out_template:
        return Path(out_template.replace("{run_id}", run_id))
    return Path(out_template) / run_id


def _emit_notch_outputs(
    *,
    ctx: PipelineContext,
    cfg: AppConfig,
    stages_cfg,
    out_dir: Path,
    run_id: str,
    config_hash: str,
    seg_start_s: float,
    n_seg: int,
) -> None:
    """Write notch-stage outputs (metrics.json, metrics.csv, plots, filtered.h5)
    to out_dir. Idempotent given identical inputs. Reads ctx.metadata['pre_notch']
    set by NotchStage.process()."""
    notch_cfg = _notch_stage_cfg(stages_cfg)
    # NotchStage rejects linear mode at build time → scalar float only here.
    pole_radius = float(notch_cfg.pole_radius.value)
    pole_repr = _notch_pole_repr(stages_cfg)

    fs = ctx.sample_rate
    signal = ctx.time_data
    per_motor_bpf = ctx.per_motor_bpf

    # 5. Metrics
    channels = _resolve_channels(cfg, signal.shape[1])
    fmax = _resolve_fmax(cfg, per_motor_bpf, cfg.rotor.n_harmonics, fs)
    metrics = compute_metrics(
        pre=ctx.metadata["pre_notch"], post=ctx.time_data,
        fs=fs, per_motor_bpf=per_motor_bpf,
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
        "segment": {"start_s": seg_start_s, "duration_s": n_seg / fs},
    })
    _write_metrics(metrics, out_dir / cfg.output.metrics_json,
                   out_dir / cfg.output.metrics_csv)

    # 6. Plots
    if cfg.plots.enabled:
        plot_channels = _resolve_plot_channels(cfg, channels)
        plots_dir = out_dir / cfg.output.plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        for ch in plot_channels:
            plot_channel_html(
                pre=ctx.metadata["pre_notch"], post=ctx.time_data,
                fs=fs, per_motor_bpf=per_motor_bpf,
                channel=ch, outpath=plots_dir / f"ch{ch:02d}.html",
                n_harmonics=cfg.rotor.n_harmonics, fmax_hz=fmax,
                spectrogram_window=cfg.plots.spectrogram_window,
                spectrogram_overlap=cfg.plots.spectrogram_overlap,
            )

    # 7. Filtered HDF5
    write_filtered(
        out_path=out_dir / cfg.output.filtered_h5,
        filtered_time_data=ctx.time_data,
        sample_rate=fs,
        rpm_per_esc=ctx.rpm_per_esc,
        attrs={
            "martymicfly_version": "0.1.0",
            "input_h5": str(cfg.input.audio_h5),
            "config_hash": config_hash,
            "segment_start_s": float(seg_start_s),
            "segment_duration_s": float(n_seg / fs),
            "n_blades": int(cfg.rotor.n_blades),
            "n_harmonics": int(cfg.rotor.n_harmonics),
            "notch_mode": "rpm_external_zerophase",
            "pole_radius_repr": pole_repr,
        },
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run martymicfly pipeline (stages-list YAML).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"YAML config path. Default: {DEFAULT_CONFIG}")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output.dir from config. If the resulting template "
                        "contains {run_id} it is substituted; otherwise the run "
                        "auto-nests under <dir>/<run_id>.")
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
    out_dir = _resolve_run_dir(out_template, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("run_id=%s, output=%s", run_id, out_dir)

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
    stages = build_pipeline(cfg.stages, rotor=cfg.rotor)
    ctx = run_pipeline(stages, ctx)

    # 5–7. Notch-stage outputs (preserved verbatim from run_notch.py)
    if any(s.name == "notch" for s in stages):
        _emit_notch_outputs(
            ctx=ctx,
            cfg=cfg,
            stages_cfg=cfg.stages,
            out_dir=out_dir,
            run_id=run_id,
            config_hash=cfg.config_hash(),
            seg_start_s=seg_start_s,
            n_seg=n_seg,
        )

    # 8. Snapshot config
    if cfg.output.copy_config:
        shutil.copy(args.config, out_dir / "config.yaml")
        (out_dir / "config.hash").write_text(config_hash + "\n", encoding="utf-8")

    log.info("done. outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    _rc = main()
    # Only raise SystemExit on failure — keeps IPython/PyCharm runfile() console
    # clean on success (rc=0 would otherwise trigger an exit warning).
    if _rc:
        raise SystemExit(_rc)
