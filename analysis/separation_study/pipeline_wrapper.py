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
        csm_real = np.asarray(f["csm_real"][:], dtype=np.float64)
        csm_imag = np.asarray(f["csm_imag"][:], dtype=np.float64)
        csm = (csm_real + 1j * csm_imag).astype(np.complex128)
        freqs = np.asarray(f["frequencies"][:], dtype=np.float64)
    psd_post = steer_to_psd(csm, freqs, mic_positions, target_point)
    cal = range_compensation_factor(mic_positions, target_point)
    return psd_post * cal, freqs


def _drone_only_psd_at_target(
    *,
    ext_only_audio_h5: Path,
    mixed_audio_h5: Path,
    mic_positions: np.ndarray,
    target_point: tuple[float, float, float],
    csm_nperseg: int,
    csm_noverlap: int,
    csm_window: str,
    csm_diag_loading: float,
    f_min_hz: float,
    f_max_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build drone-only mic-array audio via mixed - ext subtraction, then CSM,
    then steer_to_psd at target. Returns (frequencies, psd_d_ref) where
    psd_d_ref includes the range_compensation_factor."""
    from martymicfly.io.synth_h5 import load_synth_h5
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    from martymicfly.processing.steering import (
        steer_to_psd, range_compensation_factor,
    )

    ext_src = load_synth_h5(str(ext_only_audio_h5))
    mix_src = load_synth_h5(str(mixed_audio_h5))
    fs = float(ext_src["sample_rate"])
    if fs != float(mix_src["sample_rate"]):
        raise ValueError("sample_rate mismatch ext_only vs mixed audio")
    drone_audio = (mix_src["time_data"].astype(np.float64)
                   - ext_src["time_data"].astype(np.float64))
    cfg = CsmConfig(
        nperseg=csm_nperseg, noverlap=csm_noverlap, window=csm_window,
        diag_loading_rel=csm_diag_loading,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    drone_csm, freqs = build_measurement_csm(drone_audio, fs, cfg)
    psd_pre = steer_to_psd(drone_csm, freqs, mic_positions, target_point)
    cal = range_compensation_factor(mic_positions, target_point)
    return freqs, psd_pre * cal


def augment_metrics(
    *,
    run_dir: Path,
    ext_gt_h5: Path,
    ext_only_audio_h5: Path,
    mixed_audio_h5: Path,
    bands: list[dict],
    mic_positions: np.ndarray,
    target_point: tuple[float, float, float],
    welch_nperseg: int,
    welch_noverlap: int,
    window: str,
    welch_floor_db: float,
    csm_diag_loading: float = 0.0,
    f_min_hz: float = 0.0,
    f_max_hz: float = 1.0e9,
) -> dict:
    psd_post, freqs = _load_psd_post_from_run(run_dir, mic_positions, target_point)
    freqs_d, psd_d_ref = _drone_only_psd_at_target(
        ext_only_audio_h5=ext_only_audio_h5,
        mixed_audio_h5=mixed_audio_h5,
        mic_positions=mic_positions, target_point=target_point,
        csm_nperseg=welch_nperseg, csm_noverlap=welch_noverlap,
        csm_window=window, csm_diag_loading=csm_diag_loading,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    if not np.allclose(freqs, freqs_d):
        # Different freq grids — interpolate d_ref onto post grid
        psd_d_ref = np.interp(freqs, freqs_d, psd_d_ref)
    out = compute_run_metrics(
        psd_post=psd_post, psd_d_ref=psd_d_ref, frequencies=freqs,
        ext_gt_h5=ext_gt_h5,
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

    Die CLI-Pipeline nestet ihre Outputs unter <output_dir>/<auto_run_id>/.
    Der Wrapper findet das tatsächliche run_dir nach Subprocess-Ende und gibt
    es zurück (das ist der Pfad, in dem residual_csm.h5 etc. liegen).
    """
    base_cfg = yaml.safe_load(base_config_path.read_text())
    cfg = apply_overrides(base_cfg, overrides)
    cfg["output"]["dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {p for p in output_dir.iterdir() if p.is_dir()}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name
    subprocess.run(
        ["uv", "run", "python", "-m", "martymicfly.cli.run_pipeline",
         "--config", tmp_path, "--output-dir", str(output_dir),
         "--log-level", log_level],
        check=True,
    )
    after = {p for p in output_dir.iterdir() if p.is_dir()}
    new_dirs = sorted(after - before)
    if not new_dirs:
        # Pipeline wrote directly into output_dir (no nested run_id dir)
        return output_dir
    if len(new_dirs) > 1:
        raise RuntimeError(
            f"expected single new run dir under {output_dir}, got {new_dirs}"
        )
    return new_dirs[0]
