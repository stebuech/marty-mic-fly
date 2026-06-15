"""CLI: read pipeline YAML, run array_filter once on the configured segment,
and print/persist a CLEAN-SC localization report per band.

Usage:
    python -m martymicfly.cli.probe_localization \
        --config configs/pipeline_external_only_doa_target_cone.yaml \
        [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from martymicfly.cli.run_pipeline import _select_segment
from martymicfly.config import AppConfig, ArrayFilterStageConfig
from martymicfly.eval.localization_probe import (
    format_localization_report,
    probe_clean_sc_localization,
)
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5

log = logging.getLogger("martymicfly.probe_localization")


def _first_array_filter(cfg: AppConfig) -> ArrayFilterStageConfig:
    for s in cfg.stages:
        if isinstance(s, ArrayFilterStageConfig):
            return s
    raise ValueError("config has no array_filter stage")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="probe-localization")
    p.add_argument("--config", required=True)
    p.add_argument("--json", default=None,
                   help="optional path to write the report as JSON")
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

    report = probe_clean_sc_localization(
        time_data=td,
        sample_rate=fs,
        mic_positions=geom,
        platform=src["platform"],
        stage_cfg=stage_cfg,
        target_xyz_m=tuple(stage_cfg.target_point_m),
    )
    print(format_localization_report(report))

    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
