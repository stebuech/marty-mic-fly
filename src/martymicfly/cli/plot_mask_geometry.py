"""CLI: 3D visualization of Stage-2 DOA cone mask geometry.

Reads platform info (rotor positions, mic positions) from the synth file
referenced in a pipeline YAML, plus DOA mask parameters from the same
yaml's array_filter stage. Produces a single HTML scene with the
hemisphere reference and rotor / drone / target cones.

Usage:
    python -m martymicfly.cli.plot_mask_geometry \\
        --config configs/pipeline_mixed_doa_target_cone.yaml \\
        [--out results/mask_geometry.html]

If the YAML doesn't have ``doa_grid`` set, default DOA values are used.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from martymicfly.cli.compare_modes import _first_array_filter
from martymicfly.config import AppConfig, DoaGridConfig
from martymicfly.eval.array_plots import plot_mask_geometry_3d
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5

log = logging.getLogger("martymicfly.plot_mask_geometry")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plot-mask-geometry")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="results/mask_geometry.html")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg = AppConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
    stage_cfg = _first_array_filter(cfg)
    src = load_synth_h5(cfg.input.audio_h5)
    geom = load_mic_geom_xml(cfg.input.mic_geom_xml)
    plat = src["platform"]

    doa = stage_cfg.doa_grid or DoaGridConfig()  # fall back to defaults

    plot_mask_geometry_3d(
        mic_positions=geom,
        rotor_positions=np.asarray(plat["rotor_positions"]),
        rotor_radii=np.asarray(plat["rotor_radii"]),
        target_point_m=tuple(float(v) for v in stage_cfg.target_point_m),
        doa_focal_radius_m=float(doa.focal_radius_m),
        doa_hemisphere=doa.hemisphere,
        rotor_cone_half_angle_deg=float(doa.rotor_cone_half_angle_deg),
        target_cone_half_angle_deg=float(doa.target_cone_half_angle_deg),
        drone_cone_half_angle_deg=float(doa.drone_cone_half_angle_deg),
        out_path=args.out,
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
