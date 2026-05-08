"""Wrapper um martymicfly.cli.run_pipeline + Studien-Metrik-Augmentation.

run_pipeline_with_overrides: schreibt Override-config, ruft die Pipeline,
ergänzt anschließend study_metrics.json + metrics_freq.h5 aus residual_csm.h5
und den GT-h5-Pfaden."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from analysis.separation_study.metric_extensions import compute_run_metrics
from analysis.separation_study.freq_resolved_io import write_freq_resolved
from analysis.separation_study.yaml_override import apply_overrides


def _load_psd_post_from_run(
    run_dir: Path, mic_positions: np.ndarray, target_point: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    from martymicfly.processing.steering import (
        steer_to_psd, range_compensation_factor,
    )
    csm_path = run_dir / "residual_csm.h5"
    with h5py.File(csm_path, "r") as f:
        csm = np.asarray(f["csm"][:], dtype=np.complex128)
        freqs = np.asarray(f["frequencies"][:], dtype=np.float64)
    psd_post = steer_to_psd(csm, freqs, mic_positions, target_point)
    cal = range_compensation_factor(mic_positions, target_point)
    return psd_post * cal, freqs


def augment_metrics(
    *,
    run_dir: Path,
    ext_gt_h5: Path,
    mixed_gt_h5: Path,
    bands: list[dict],
    mic_positions: np.ndarray,
    target_point: tuple[float, float, float],
    welch_nperseg: int,
    welch_noverlap: int,
    window: str,
    welch_floor_db: float,
) -> dict:
    psd_post, freqs = _load_psd_post_from_run(run_dir, mic_positions, target_point)
    out = compute_run_metrics(
        psd_post=psd_post, frequencies=freqs,
        ext_gt_h5=ext_gt_h5, mixed_gt_h5=mixed_gt_h5,
        welch_nperseg=welch_nperseg, welch_noverlap=welch_noverlap, window=window,
        bands=bands, welch_floor_db=welch_floor_db,
    )
    fr = out.pop("frequency_resolved")
    write_freq_resolved(run_dir / "metrics_freq.h5", fr)
    (run_dir / "study_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_pipeline_with_overrides(
    *,
    base_config_path: Path,
    overrides: dict,
    output_dir: Path,
    log_level: str = "INFO",
) -> Path:
    """Patcht base_config mit overrides, schreibt temp-yaml, ruft run_pipeline.
    Liefert das resultierende run_dir Pfad."""
    base_cfg = yaml.safe_load(base_config_path.read_text())
    cfg = apply_overrides(base_cfg, overrides)
    cfg["output"]["dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name
    subprocess.run(
        ["uv", "run", "python", "-m", "martymicfly.cli.run_pipeline",
         "--config", tmp_path, "--output-dir", str(output_dir),
         "--log-level", log_level],
        check=True,
    )
    return output_dir
