"""CLI: run CLEAN-SC once, finalize each mask_mode, print combined diagnostic.

Builds the CSM, fits CLEAN-SC and trace-rescales the source-map exactly once,
then for each mask_mode reconstructs the drone CSM, computes the residual and
runs the Stage-2 metrics. Also runs the localization probe on the (mode-
independent) source-map. Output is a single combined report.

Usage:
    python -m martymicfly.cli.compare_modes \
        --config configs/pipeline_external_only_target_box.yaml \
        [--modes target_box drone_box rotor_disc] \
        [--json out.json]

The base YAML's `mask_mode` is irrelevant — all modes are evaluated.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from martymicfly.cli.run_pipeline import _select_segment
from martymicfly.config import AppConfig, ArrayFilterStageConfig
from martymicfly.eval.array_metrics import compute_array_metrics
from martymicfly.eval.localization_probe import (
    format_localization_report,
    localization_stats_from_source_map,
)
from martymicfly.io.ground_truth_h5 import load_ground_truth
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
from martymicfly.processing.algorithms.base import (
    reconstruct_csm,
    rescale_source_map_to_csm_trace,
)
from martymicfly.processing.beamform_grid import (
    build_diagnostic_grid,
    build_rotor_disc_mask,
    build_target_box_mask,
)
from martymicfly.processing.csm import CsmConfig as RuntimeCsmConfig, build_measurement_csm
from martymicfly.processing.steering import steer_to_psd

log = logging.getLogger("martymicfly.compare_modes")
ALL_MODES = ("rotor_disc", "drone_box", "target_box")


def _first_array_filter(cfg: AppConfig) -> ArrayFilterStageConfig:
    for s in cfg.stages:
        if isinstance(s, ArrayFilterStageConfig):
            return s
    raise ValueError("config has no array_filter stage")


def _resolve_z(stage_cfg: ArrayFilterStageConfig, platform: dict) -> tuple[float, float]:
    zmin = stage_cfg.diagnostic_grid.z_min_m
    zmax = stage_cfg.diagnostic_grid.z_max_m
    if zmin is None:
        z = float(np.asarray(platform["rotor_positions"])[2, 0])
        return z, z
    return zmin, zmax


def compare_mask_modes(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    mic_positions: np.ndarray,
    platform: dict,
    stage_cfg: ArrayFilterStageConfig,
    gt_block: dict | None = None,
    modes: tuple[str, ...] = ALL_MODES,
) -> dict:
    """Single CLEAN-SC fit, per-mode finalize. Returns a dict with localization
    stats and per-mode metrics blocks."""
    rcfg = RuntimeCsmConfig(
        nperseg=stage_cfg.csm.nperseg, noverlap=stage_cfg.csm.noverlap,
        window=stage_cfg.csm.window, diag_loading_rel=stage_cfg.csm.diag_loading_rel,
        f_min_hz=stage_cfg.csm.f_min_hz, f_max_hz=stage_cfg.csm.f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, rcfg)

    zmin, zmax = _resolve_z(stage_cfg, platform)
    diag_grid, diag_shape = build_diagnostic_grid(
        stage_cfg.diagnostic_grid.extent_xy_m,
        stage_cfg.diagnostic_grid.increment_m,
        zmin, zmax,
    )

    algo = ALGORITHM_REGISTRY[stage_cfg.algorithm]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None, sample_rate=sample_rate,
        mic_positions=mic_positions, grid_positions=diag_grid,
        params={"damp": stage_cfg.clean_sc.damp,
                "n_iter": stage_cfg.clean_sc.n_iter,
                "r_diag": stage_cfg.clean_sc.r_diag},
    )
    sm, _ = rescale_source_map_to_csm_trace(sm, csm, mic_positions)

    rotor_disc_mask = build_rotor_disc_mask(
        diag_grid,
        np.asarray(platform["rotor_positions"]),
        np.asarray(platform["rotor_radii"]),
        stage_cfg.rotor_z_tolerance_m,
    )
    target_box_mask = build_target_box_mask(
        diag_grid, stage_cfg.target_point_m, stage_cfg.target_box_half_extent_m,
    )
    drone_box_mask = build_target_box_mask(
        diag_grid, stage_cfg.drone_box_center_m, stage_cfg.drone_box_half_extent_m,
    )

    psd_pre = steer_to_psd(csm, freqs, mic_positions, stage_cfg.target_point_m)

    per_mode: dict = {}
    for mode in modes:
        if mode == "rotor_disc":
            sub = rotor_disc_mask
        elif mode == "drone_box":
            sub = drone_box_mask
        elif mode == "target_box":
            sub = ~target_box_mask
        else:
            raise ValueError(f"unknown mask_mode {mode!r}")
        drone_csm = reconstruct_csm(sm.subset(sub), mic_positions)
        residual_csm = csm - drone_csm
        psd_post = steer_to_psd(residual_csm, freqs, mic_positions, stage_cfg.target_point_m)
        m = compute_array_metrics(
            csm_pre=csm, residual_csm=residual_csm,
            frequencies=freqs, psd_pre=psd_pre, psd_post=psd_post,
            source_map_powers=sm.powers, drone_mask=sub,
            bands=stage_cfg.bands, ground_truth=gt_block,
        )
        per_mode[mode] = m

    localization = localization_stats_from_source_map(
        source_map=sm, grid_positions=diag_grid, grid_shape=diag_shape,
        bands=stage_cfg.bands, target_xyz_m=tuple(stage_cfg.target_point_m),
    )
    return {"localization": localization, "per_mode": per_mode}


def _format_comparison(report: dict) -> str:
    lines = [format_localization_report(report["localization"]), ""]
    band_names = list(next(iter(report["per_mode"].values()))["bands"].keys())
    has_gt = any(
        m["bands"][b]["ground_truth"] is not None
        for m in report["per_mode"].values() for b in band_names
    )
    header = (
        f"{'mode':<11}{'band':<6}"
        f"{'csm_red':>9}{'psd_red':>9}{'drone_sh':>10}"
    )
    if has_gt:
        header += f"{'ext_rec':>9}{'mae':>9}"
    lines.append(header)
    for mode, m in report["per_mode"].items():
        for band in band_names:
            b = m["bands"][band]
            row = (
                f"{mode:<11}{band:<6}"
                f"{b['csm_trace_reduction_db']:>9.2f}"
                f"{b['target_psd_reduction_db']:>9.2f}"
                f"{b['drone_power_share_db']:>10.2f}"
            )
            if has_gt and b["ground_truth"] is not None:
                row += (f"{b['ground_truth']['external_recovery_db']:>9.2f}"
                        f"{b['ground_truth']['spectrum_mae_db']:>9.2f}")
            lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare-modes")
    p.add_argument("--config", required=True)
    p.add_argument("--modes", nargs="+", default=list(ALL_MODES),
                   choices=list(ALL_MODES))
    p.add_argument("--json", default=None,
                   help="optional path to write the comparison report as JSON")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg = AppConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
    stage_cfg = _first_array_filter(cfg)

    src = load_synth_h5(cfg.input.audio_h5)
    geom = load_mic_geom_xml(cfg.input.mic_geom_xml)
    fs = src["sample_rate"]
    seg_start_s, n_seg = _select_segment(cfg, src["time_data"].shape[0], fs)
    start_idx = int(round(seg_start_s * fs))
    td = src["time_data"][start_idx : start_idx + n_seg]

    gt_block = None
    if cfg.input.ground_truth_h5:
        from scipy.signal import csd as _csd, welch as _welch
        gt = load_ground_truth(cfg.input.ground_truth_h5)
        gt_f, gt_psd = _welch(
            gt.signal, fs=gt.sample_rate, nperseg=stage_cfg.csm.nperseg,
            noverlap=stage_cfg.csm.noverlap, scaling="density",
        )
        # Match the freq axis that build_measurement_csm uses (band-cropped csd).
        f_full, _ = _csd(td[:, 0], td[:, 0], fs=fs,
                          nperseg=stage_cfg.csm.nperseg,
                          noverlap=stage_cfg.csm.noverlap,
                          window=stage_cfg.csm.window, scaling="density")
        in_band = (f_full >= stage_cfg.csm.f_min_hz) & (f_full <= stage_cfg.csm.f_max_hz)
        freqs = f_full[in_band]
        gt_in_band = (gt_f >= stage_cfg.csm.f_min_hz) & (gt_f <= stage_cfg.csm.f_max_hz)
        gt_psd_interp = np.interp(freqs, gt_f[gt_in_band], gt_psd[gt_in_band])
        gt_block = {"psd_at_target": gt_psd_interp, "frequencies": freqs}

    report = compare_mask_modes(
        time_data=td, sample_rate=fs, mic_positions=geom,
        platform=src["platform"], stage_cfg=stage_cfg,
        gt_block=gt_block, modes=tuple(args.modes),
    )
    print(_format_comparison(report))

    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float),
                                   encoding="utf-8")
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
