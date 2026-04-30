"""Probe CLEAN-SC source-map localization quality per band.

Used to quantify how well the array deconvolution stage places CLEAN-SC peaks
at the true source position vs spreads them across the diagnostic grid. The
key failure mode to flag is the low-frequency limit of small-aperture arrays
(λ ≳ array diameter): CLEAN-SC mislocalizes the peak away from the truth and
spreads energy over a wide z-range. Any subtraction mask that doesn't enclose
the true source then incurs false-positive subtraction at the external source.

Run via the CLI: `python -m martymicfly.cli.probe_localization --config X.yaml`
or call `probe_clean_sc_localization` directly from analysis code.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from martymicfly.config import ArrayFilterStageConfig, BandConfig
from martymicfly.processing.algorithms.base import SourceMap
from martymicfly.processing.array_filter import ArrayFilterStage
from martymicfly.processing.pipeline import PipelineContext


def _band_iter(bands: Iterable) -> Iterable[BandConfig]:
    for b in bands:
        if isinstance(b, BandConfig):
            yield b
        else:
            yield BandConfig(**b)


def _z_quantile(z_unique: np.ndarray, z_marg: np.ndarray, q: float) -> float:
    cum = np.cumsum(z_marg / z_marg.sum())
    idx = int(np.searchsorted(cum, q))
    return float(z_unique[min(idx, z_unique.size - 1)])


def localization_stats_from_source_map(
    *,
    source_map: SourceMap,
    grid_positions: np.ndarray,        # (G, 3)
    grid_shape: tuple[int, int, int],
    bands: Iterable,
    target_xyz_m: tuple[float, float, float],
    z_tolerance_m: float = 0.10,
    drone_z_tolerance_m: float = 0.10,
) -> dict:
    """Compute the per-band localization report from an already-fit source_map.

    Useful when you want to reuse a single CLEAN-SC fit across multiple
    downstream analyses (e.g. mask-mode comparisons) without re-running the
    expensive deconvolution.
    """
    freqs = source_map.frequencies
    z_unique = np.unique(grid_positions[:, 2])
    target_z = float(target_xyz_m[2])

    out_bands: dict = {}
    for band in _band_iter(bands):
        fmask = (freqs >= band.f_min_hz) & (freqs <= band.f_max_hz)
        if not fmask.any():
            continue
        p = source_map.powers[fmask].sum(axis=0)
        p = np.where(np.isfinite(p), p, 0.0)
        p = np.maximum(p, 0.0)
        total = float(p.sum())
        if total <= 0.0:
            out_bands[band.name] = {
                "peak_xyz_m": (np.nan, np.nan, np.nan),
                "z_span_10_90_m": (np.nan, np.nan),
                "frac_at_target_z": 0.0,
                "frac_at_drone_z": 0.0,
                "total_power": 0.0,
            }
            continue
        ix = int(np.argmax(p))
        peak = grid_positions[ix]
        z_marg = np.array([p[grid_positions[:, 2] == z].sum() for z in z_unique])
        z10 = _z_quantile(z_unique, z_marg, 0.10)
        z90 = _z_quantile(z_unique, z_marg, 0.90)
        near_target = np.abs(z_unique - target_z) <= z_tolerance_m
        near_drone = np.abs(z_unique) <= drone_z_tolerance_m
        out_bands[band.name] = {
            "peak_xyz_m": (float(peak[0]), float(peak[1]), float(peak[2])),
            "z_span_10_90_m": (z10, z90),
            "frac_at_target_z": float(z_marg[near_target].sum() / total),
            "frac_at_drone_z": float(z_marg[near_drone].sum() / total),
            "total_power": total,
        }

    return {
        "grid_shape": tuple(int(s) for s in grid_shape),
        "target_xyz_m": tuple(float(v) for v in target_xyz_m),
        "z_tolerance_m": float(z_tolerance_m),
        "drone_z_tolerance_m": float(drone_z_tolerance_m),
        "bands": out_bands,
    }


def probe_clean_sc_localization(
    *,
    time_data: np.ndarray,           # (N, M)
    sample_rate: float,
    mic_positions: np.ndarray,       # (M, 3)
    platform: dict,                  # at minimum {"rotor_positions","rotor_radii","blade_counts"}
    stage_cfg: ArrayFilterStageConfig,
    target_xyz_m: tuple[float, float, float],
    z_tolerance_m: float = 0.10,
    drone_z_tolerance_m: float = 0.10,
) -> dict:
    """Run the array_filter stage and report per-band localization stats.

    Returns:
        {
            "grid_shape": (nx, ny, nz),
            "target_xyz_m": (x, y, z),
            "bands": {
                band_name: {
                    "peak_xyz_m": (x, y, z),
                    "z_span_10_90_m": (z10, z90),
                    "frac_at_target_z": float,
                    "frac_at_drone_z": float,
                    "total_power": float,
                },
                ...
            },
        }
    """
    n_rotors = int(np.asarray(platform["rotor_positions"]).shape[1])
    ctx = PipelineContext(
        time_data=time_data,
        sample_rate=sample_rate,
        rpm_per_esc={},
        mic_positions=mic_positions,
        per_motor_bpf=np.zeros((time_data.shape[0], n_rotors)),
        harm_matrix=np.zeros((time_data.shape[0], 4 * n_rotors)),
        metadata={"platform": platform},
    )
    af = ArrayFilterStage(stage_cfg).process(ctx).metadata["array_filter"]
    return localization_stats_from_source_map(
        source_map=af["source_map"],
        grid_positions=af["diagnostic_grid"],
        grid_shape=af["diagnostic_grid_shape"],
        bands=stage_cfg.bands,
        target_xyz_m=target_xyz_m,
        z_tolerance_m=z_tolerance_m,
        drone_z_tolerance_m=drone_z_tolerance_m,
    )


def format_localization_report(report: dict) -> str:
    """Pretty-printable table for CLI output."""
    tx, ty, tz = report["target_xyz_m"]
    ztol = report["z_tolerance_m"]
    dtol = report["drone_z_tolerance_m"]
    lines = [
        f"CLEAN-SC localization probe — target=({tx:+.2f},{ty:+.2f},{tz:+.2f}) m, "
        f"grid={report['grid_shape']}",
        f"{'band':<6}{'peak (x,y,z) [m]':<26}{'z 10-90% [m]':<22}"
        f"{'@target±'+f'{ztol}'+'m':<14}{'@drone±'+f'{dtol}'+'m':<14}",
    ]
    for name, b in report["bands"].items():
        px, py, pz = b["peak_xyz_m"]
        z10, z90 = b["z_span_10_90_m"]
        lines.append(
            f"{name:<6}({px:+.2f},{py:+.2f},{pz:+.2f})       "
            f"[{z10:+.2f},{z90:+.2f}]      "
            f"{b['frac_at_target_z']:>9.2%}     {b['frac_at_drone_z']:>9.2%}"
        )
    return "\n".join(lines)
