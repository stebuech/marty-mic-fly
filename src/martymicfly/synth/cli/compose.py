"""CLI: read compose-config YAML, run compose_external, write outputs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from martymicfly.synth.compose_external import compose_external
from martymicfly.synth.external_source import ExternalSourceSpec


class _ComposeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drone_source_artifact_h5: str
    mic_geom_xml: str


class _ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    synth_h5: str
    ground_truth_h5: str


class _ComposeAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: _ComposeInput
    external: ExternalSourceSpec
    output: _ComposeOutput


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compose")
    p.add_argument("--config", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    payload = yaml.safe_load(Path(args.config).read_text())
    cfg = _ComposeAppConfig.model_validate(payload)
    compose_external(
        cfg.input.drone_source_artifact_h5,
        cfg.input.mic_geom_xml,
        cfg.external,
        cfg.output.synth_h5,
        cfg.output.ground_truth_h5,
    )
    logging.getLogger("martymicfly.compose").info(
        "wrote %s and %s", cfg.output.synth_h5, cfg.output.ground_truth_h5
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
