"""Robustheits-Szenarien S0..S3 als compose-config-Generatoren.

S0: Baseline (existing synth files, kein neuer compose-Lauf nötig)
S1: target = (0.3, 0, -1.5) — laterale Off-axis-Position
S2: source amplitude × 0.3 (≈ -10.46 dB) — niedriges SNR
S3: target = (0, 0, -3.0) — größere Quelldistanz"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


SCENARIO_PARAMS = {
    "S0": None,
    "S1": {"position_m": (0.3, 0.0, -1.5), "amplitude_db": 0.0},
    "S2": {"position_m": (0.0, 0.0, -1.5),
           "amplitude_db": float(20.0 * np.log10(0.3))},
    "S3": {"position_m": (0.0, 0.0, -3.0), "amplitude_db": 0.0},
}


def make_compose_config(
    *,
    scenario: str,
    base_drone_artifact: str,
    mic_geom_xml: str,
    out_synth_h5: str,
    out_gt_h5: str,
    seed: int = 0,
) -> Optional[dict]:
    """Liefert eine compose-CLI-kompatible YAML-config-dict, oder None falls
    Szenario das Baseline ist (existierende Files weiterverwenden)."""
    if scenario not in SCENARIO_PARAMS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    params = SCENARIO_PARAMS[scenario]
    if params is None:
        return None
    return {
        "input": {
            "drone_source_artifact_h5": base_drone_artifact,
            "mic_geom_xml": mic_geom_xml,
        },
        "external": {
            "kind": "noise",
            "position_m": list(params["position_m"]),
            "amplitude_db": float(params["amplitude_db"]),
            "duration_s": None,
            "seed": seed,
        },
        "include_drone": True,
        "output": {
            "synth_h5": str(out_synth_h5),
            "ground_truth_h5": str(out_gt_h5),
        },
    }


def _run_compose(cfg_dict: dict) -> None:
    """Schreibt cfg_dict in temp YAML und ruft martymicfly.synth.cli.compose."""
    import tempfile
    import subprocess
    import yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(cfg_dict, f)
        cfg_path = f.name
    subprocess.run(
        ["uv", "run", "python", "-m", "martymicfly.synth.cli.compose",
         "--config", cfg_path],
        check=True,
    )


def ensure_scenario(
    *,
    scenario: str,
    base_drone_artifact: str,
    mic_geom_xml: str,
    out_synth_h5: Path,
    out_gt_h5: Path,
    seed: int = 0,
) -> None:
    """Generiert Szenario-h5-Dateien falls noch nicht vorhanden (caching).

    Für S0 ein No-Op (Baseline-Files existieren bereits). Für S1..S3 wird
    compose über die CLI gerufen wenn output-h5 fehlen."""
    if Path(out_synth_h5).exists() and Path(out_gt_h5).exists():
        return
    cfg = make_compose_config(
        scenario=scenario,
        base_drone_artifact=base_drone_artifact,
        mic_geom_xml=mic_geom_xml,
        out_synth_h5=str(out_synth_h5),
        out_gt_h5=str(out_gt_h5),
        seed=seed,
    )
    if cfg is None:
        return
    _run_compose(cfg)


def ensure_scenario_pair(
    *,
    scenario: str,
    base_drone_artifact: str,
    mic_geom_xml: str,
    out_mixed_h5: Path,
    out_mixed_gt_h5: Path,
    out_ext_only_h5: Path,
    out_ext_only_gt_h5: Path,
    seed: int = 0,
) -> None:
    """Generiert mixed (drone + ext) und ext_only (ext only) für ein Szenario.

    Beide compose-Läufe nutzen denselben seed → identisches ext-Signal in
    beiden Files → drone_only_at_array = mixed_audio − ext_only_audio gibt
    exakt das drone-Signal zurück."""
    # Mixed (with drone)
    if not (Path(out_mixed_h5).exists() and Path(out_mixed_gt_h5).exists()):
        cfg_m = make_compose_config(
            scenario=scenario,
            base_drone_artifact=base_drone_artifact,
            mic_geom_xml=mic_geom_xml,
            out_synth_h5=str(out_mixed_h5),
            out_gt_h5=str(out_mixed_gt_h5),
            seed=seed,
        )
        if cfg_m is not None:
            _run_compose(cfg_m)

    # Ext-only (no drone)
    if not (Path(out_ext_only_h5).exists() and Path(out_ext_only_gt_h5).exists()):
        cfg_e = make_compose_config(
            scenario=scenario,
            base_drone_artifact=base_drone_artifact,
            mic_geom_xml=mic_geom_xml,
            out_synth_h5=str(out_ext_only_h5),
            out_gt_h5=str(out_ext_only_gt_h5),
            seed=seed,
        )
        if cfg_e is not None:
            cfg_e["include_drone"] = False
            _run_compose(cfg_e)
