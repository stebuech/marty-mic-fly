"""CLI: compare Stage-2 methods on the same input segment.

A single ``--config`` is dispatched by content:
    * Cartesian + CLEAN-SC (default): single fit, all three box mask modes.
    * DOA hemisphere + CLEAN-SC (``cfg.doa_grid`` set): single fit, all
      three cone mask modes.
    * Known-geometry NNLS (``cfg.algorithm == 'known_geometry_lsq'``):
      single run, single subtraction (drone atoms vs target atom).

Pass ``--config`` multiple times to combine methods in one report — useful
to put DOA and NNLS side by side on the same input data.

Usage examples:
    python -m martymicfly.cli.compare_modes \
        --config configs/pipeline_external_only_doa_target_cone.yaml

    python -m martymicfly.cli.compare_modes \
        --config configs/pipeline_external_only_doa_target_cone.yaml \
        --config configs/pipeline_external_only_nnls.yaml \
        [--json combined.json]

Each ``--config``'s own ``mask_mode`` selection is irrelevant — every
applicable mask mode is evaluated for the configured method.
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
    build_doa_cone_mask,
    build_doa_disk_mask,
    build_doa_grid,
    build_rotor_disc_mask,
    build_rotor_doa_cones_mask,
    build_target_box_mask,
)
from martymicfly.processing.csm import CsmConfig as RuntimeCsmConfig, build_measurement_csm
from martymicfly.processing.steering import steer_to_psd

log = logging.getLogger("martymicfly.compare_modes")
ALL_MODES = ("rotor_disc", "drone_box", "target_box")
DOA_MODES = ("rotor_cone", "drone_disk", "target_cone")


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
        per_mode[mode] = {**m, "psd_post": psd_post}

    localization = localization_stats_from_source_map(
        source_map=sm, grid_positions=diag_grid, grid_shape=diag_shape,
        bands=stage_cfg.bands, target_xyz_m=tuple(stage_cfg.target_point_m),
    )
    return {
        "localization": localization, "per_mode": per_mode,
        "frequencies": freqs, "psd_pre": psd_pre,
        "gt_psd": (gt_block["psd_at_target"] if gt_block else None),
    }


def compare_doa_modes(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    mic_positions: np.ndarray,
    platform: dict,
    stage_cfg: ArrayFilterStageConfig,
    gt_block: dict | None = None,
    modes: tuple[str, ...] = DOA_MODES,
) -> dict:
    """DOA hemisphere + CLEAN-SC variant of ``compare_mask_modes``.

    Single CLEAN-SC fit on the focal-sphere search space; per-mode finalize
    builds the appropriate cone mask, reconstructs the drone CSM and runs the
    Stage-2 metrics. Returns the same dict shape as ``compare_mask_modes``.
    """
    if stage_cfg.doa_grid is None:
        raise ValueError("compare_doa_modes requires cfg.doa_grid to be set")
    gcfg = stage_cfg.doa_grid

    rcfg = RuntimeCsmConfig(
        nperseg=stage_cfg.csm.nperseg, noverlap=stage_cfg.csm.noverlap,
        window=stage_cfg.csm.window, diag_loading_rel=stage_cfg.csm.diag_loading_rel,
        f_min_hz=stage_cfg.csm.f_min_hz, f_max_hz=stage_cfg.csm.f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, rcfg)

    diag_grid, (n_az, n_el) = build_doa_grid(
        focal_radius_m=gcfg.focal_radius_m,
        azimuth_step_deg=gcfg.azimuth_step_deg,
        elevation_step_deg=gcfg.elevation_step_deg,
        hemisphere=gcfg.hemisphere,
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

    rotor_positions = np.asarray(platform["rotor_positions"])
    rotor_cone_mask = build_rotor_doa_cones_mask(
        diag_grid, rotor_positions, gcfg.rotor_cone_half_angle_deg,
    )
    target_dir = np.asarray(stage_cfg.target_point_m, dtype=np.float64)
    if float(np.linalg.norm(target_dir)) < 1e-9:
        target_dir = np.array([0.0, 0.0, -1.0])
    target_cone_mask = build_doa_cone_mask(
        diag_grid, target_dir, gcfg.target_cone_half_angle_deg,
    )
    drone_disk_mask = build_doa_disk_mask(
        diag_grid, gcfg.drone_disk_half_width_deg,
    )
    psd_pre = steer_to_psd(csm, freqs, mic_positions, stage_cfg.target_point_m)

    per_mode: dict = {}
    for mode in modes:
        if mode == "rotor_cone":
            sub = rotor_cone_mask
        elif mode == "drone_disk":
            sub = drone_disk_mask
        elif mode == "target_cone":
            sub = ~target_cone_mask
        else:
            raise ValueError(f"unknown DOA mask_mode {mode!r}")
        drone_csm = reconstruct_csm(sm.subset(sub), mic_positions)
        residual_csm = csm - drone_csm
        psd_post = steer_to_psd(residual_csm, freqs, mic_positions, stage_cfg.target_point_m)
        m = compute_array_metrics(
            csm_pre=csm, residual_csm=residual_csm,
            frequencies=freqs, psd_pre=psd_pre, psd_post=psd_post,
            source_map_powers=sm.powers, drone_mask=sub,
            bands=stage_cfg.bands, ground_truth=gt_block,
        )
        per_mode[mode] = {**m, "psd_post": psd_post}

    localization = localization_stats_from_source_map(
        source_map=sm, grid_positions=diag_grid, grid_shape=(n_az, n_el),
        bands=stage_cfg.bands, target_xyz_m=tuple(stage_cfg.target_point_m),
    )
    return {
        "localization": localization, "per_mode": per_mode,
        "frequencies": freqs, "psd_pre": psd_pre,
        "gt_psd": (gt_block["psd_at_target"] if gt_block else None),
    }


def compare_nnls(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    mic_positions: np.ndarray,
    platform: dict,
    stage_cfg: ArrayFilterStageConfig,
    gt_block: dict | None = None,
) -> dict:
    """Single NNLS run on the known-geometry atom set.

    The mask is implicit: drone atoms are subtracted, target atom is
    preserved. The "per_mode" key contains a single 'nnls' entry.
    """
    if stage_cfg.algorithm != "known_geometry_lsq":
        raise ValueError(
            f"compare_nnls requires algorithm='known_geometry_lsq'; "
            f"got {stage_cfg.algorithm!r}"
        )
    # Reuse the actual stage to keep the atom-set and forward-model logic
    # in one place — a separate inline copy here would diverge.
    from martymicfly.processing.array_filter import _array_filter_factory
    from martymicfly.processing.pipeline import PipelineContext
    n_rotors = int(np.asarray(platform["rotor_positions"]).shape[1])
    ctx = PipelineContext(
        time_data=time_data, sample_rate=sample_rate, rpm_per_esc={},
        mic_positions=mic_positions,
        per_motor_bpf=np.zeros((time_data.shape[0], n_rotors)),
        harm_matrix=np.zeros((time_data.shape[0], 4 * n_rotors)),
        metadata={"platform": platform},
    )
    stage = _array_filter_factory(stage_cfg)
    af = stage.process(ctx).metadata["array_filter"]

    metrics = compute_array_metrics(
        csm_pre=af["csm_pre"], residual_csm=af["residual_csm"],
        frequencies=af["frequencies"],
        psd_pre=af["target_psd_pre"], psd_post=af["target_psd_post"],
        source_map_powers=af["source_map"].powers,
        drone_mask=af["drone_mask"],
        bands=stage_cfg.bands, ground_truth=gt_block,
    )
    metrics["psd_post"] = af["target_psd_post"]

    # NNLS "localization" diagnostic: per-band power per atom kind. Extract
    # it from the source-map directly so the printed report has something
    # to put under the 'localization' header for symmetry with the other
    # comparators.
    sm = af["source_map"]
    kinds = stage._build_fit_input(ctx).aux["atom_kinds"]
    band_atom_powers = {}
    for band in stage_cfg.bands:
        fmask = (sm.frequencies >= band.f_min_hz) & (sm.frequencies <= band.f_max_hz)
        band_total = np.maximum(sm.powers[fmask].sum(axis=0), 0.0)
        total = float(band_total.sum())
        if total <= 0.0:
            band_atom_powers[band.name] = {"drone_db": float("-inf"),
                                            "target_db": float("-inf"),
                                            "diffuse_db": float("-inf")}
            continue
        def _share_db(label):
            sel = (kinds == label)
            if not sel.any():
                return float("-inf")
            share = float(band_total[sel].sum() / total)
            return float(10.0 * np.log10(max(share, 1e-30)))
        band_atom_powers[band.name] = {
            "drone_db": _share_db("drone"),
            "target_db": _share_db("target"),
            "diffuse_db": _share_db("diffuse"),
        }

    return {
        "localization": {
            "kind": "nnls_atom_powers",
            "bands": band_atom_powers,
            "target_xyz_m": tuple(float(v) for v in stage_cfg.target_point_m),
        },
        "per_mode": {"nnls": metrics},
        "frequencies": af["frequencies"],
        "psd_pre": af["target_psd_pre"],
        "gt_psd": (gt_block["psd_at_target"] if gt_block else None),
    }


def _classify_method(stage_cfg: ArrayFilterStageConfig) -> str:
    if stage_cfg.doa_grid is not None:
        return "doa"
    if stage_cfg.algorithm == "known_geometry_lsq":
        return "nnls"
    return "cartesian"


def _format_localization(loc: dict) -> str:
    if loc.get("kind") == "nnls_atom_powers":
        lines = [f"NNLS atom-power shares — target=({loc['target_xyz_m'][0]:+.2f},"
                 f"{loc['target_xyz_m'][1]:+.2f},{loc['target_xyz_m'][2]:+.2f}) m"]
        lines.append(f"{'band':<6}{'drone [dB]':>12}{'target [dB]':>13}{'diffuse [dB]':>14}")
        for name, b in loc["bands"].items():
            d = b["drone_db"]; t = b["target_db"]; df = b["diffuse_db"]
            d_s = f"{d:>11.2f}" if np.isfinite(d) else f"{'-inf':>11}"
            t_s = f"{t:>12.2f}" if np.isfinite(t) else f"{'-inf':>12}"
            df_s = f"{df:>13.2f}" if np.isfinite(df) else f"{'n/a':>13}"
            lines.append(f"{name:<6}{d_s} {t_s} {df_s}")
        return "\n".join(lines)
    return format_localization_report(loc)


def _format_comparison(report: dict) -> str:
    lines = [_format_localization(report["localization"]), ""]
    band_names = list(next(iter(report["per_mode"].values()))["bands"].keys())
    has_gt = any(
        m["bands"][b]["ground_truth"] is not None
        for m in report["per_mode"].values() for b in band_names
    )
    header = (
        f"{'mode':<12}{'band':<6}"
        f"{'csm_red':>9}{'psd_red':>9}{'drone_sh':>10}"
    )
    if has_gt:
        header += f"{'ext_rec':>9}{'mae':>9}"
    lines.append(header)
    for mode, m in report["per_mode"].items():
        for band in band_names:
            b = m["bands"][band]
            row = (
                f"{mode:<12}{band:<6}"
                f"{b['csm_trace_reduction_db']:>9.2f}"
                f"{b['target_psd_reduction_db']:>9.2f}"
                f"{b['drone_power_share_db']:>10.2f}"
            )
            if has_gt and b["ground_truth"] is not None:
                row += (f"{b['ground_truth']['external_recovery_db']:>9.2f}"
                        f"{b['ground_truth']['spectrum_mae_db']:>9.2f}")
            lines.append(row)
    return "\n".join(lines)


def _format_combined(by_method: dict) -> str:
    sections = []
    for method, payload in by_method.items():
        sections.append(f"=== method: {method} ===")
        sections.append(_format_comparison(payload["report"]))
        sections.append("")
    return "\n".join(sections).rstrip()


def _load_segment_and_gt(cfg: AppConfig, stage_cfg: ArrayFilterStageConfig):
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
        f_full, _ = _csd(td[:, 0], td[:, 0], fs=fs,
                          nperseg=stage_cfg.csm.nperseg,
                          noverlap=stage_cfg.csm.noverlap,
                          window=stage_cfg.csm.window, scaling="density")
        in_band = (f_full >= stage_cfg.csm.f_min_hz) & (f_full <= stage_cfg.csm.f_max_hz)
        freqs = f_full[in_band]
        gt_in_band = (gt_f >= stage_cfg.csm.f_min_hz) & (gt_f <= stage_cfg.csm.f_max_hz)
        gt_psd_interp = np.interp(freqs, gt_f[gt_in_band], gt_psd[gt_in_band])
        gt_block = {"psd_at_target": gt_psd_interp, "frequencies": freqs}
    return src, geom, td, fs, gt_block


def _run_one_config(cfg_path: str) -> dict:
    cfg = AppConfig.model_validate(yaml.safe_load(Path(cfg_path).read_text()))
    stage_cfg = _first_array_filter(cfg)
    method = _classify_method(stage_cfg)
    src, geom, td, fs, gt_block = _load_segment_and_gt(cfg, stage_cfg)

    common = dict(time_data=td, sample_rate=fs, mic_positions=geom,
                  platform=src["platform"], stage_cfg=stage_cfg,
                  gt_block=gt_block)
    if method == "doa":
        report = compare_doa_modes(**common)
    elif method == "nnls":
        report = compare_nnls(**common)
    else:
        report = compare_mask_modes(**common)
    return {"method": method, "config": cfg_path, "report": report}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare-modes")
    p.add_argument("--config", action="append", required=True,
                   help="pipeline YAML; pass repeatedly to combine methods")
    p.add_argument("--json", default=None,
                   help="optional path to write the combined report as JSON")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    by_method: dict = {}
    for cfg_path in args.config:
        log.info("evaluating %s ...", cfg_path)
        result = _run_one_config(cfg_path)
        # If the same method appears more than once, suffix to disambiguate.
        key = result["method"]
        suffix = 2
        while key in by_method:
            key = f"{result['method']}_{suffix}"
            suffix += 1
        by_method[key] = result

    if len(by_method) == 1:
        # Single-config legacy output: print the comparator's own report.
        print(_format_comparison(next(iter(by_method.values()))["report"]))
    else:
        print(_format_combined(by_method))

    if args.json is not None:
        Path(args.json).write_text(
            json.dumps({"by_method": by_method}, indent=2, default=float),
            encoding="utf-8",
        )
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
