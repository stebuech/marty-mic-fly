"""Compose a synthetic measurement HDF5 by propagating the drone source
artifact's subsources to the mic array and adding a configurable external
source. Writes a paired ground-truth file with the pure external signal at
its source location (pre-propagation).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from martymicfly.constants import SPEED_OF_SOUND
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.source_artifact import load_source_artifact
from martymicfly.synth.external_source import (
    ExternalSourceSpec,
    generate_external_signal,
)
from martymicfly.synth.propagation import greens_propagate


def _propagate_drone_only(artifact_path: str, mic_geom_path: str) -> np.ndarray:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    return greens_propagate(art.subsource_signals, art.subsource_positions, mic_pos,
                            art.sample_rate)


def _propagate_external_only(
    artifact_path: str, mic_geom_path: str, spec: ExternalSourceSpec
) -> np.ndarray:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    n_samples = art.subsource_signals.shape[1] if spec.duration_s is None \
        else int(round(spec.duration_s * art.sample_rate))
    if n_samples > art.subsource_signals.shape[1]:
        raise ValueError(
            f"duration_s={spec.duration_s}s requires {n_samples} samples, "
            f"but artifact only has {art.subsource_signals.shape[1]}"
        )
    ext = generate_external_signal(spec, art.sample_rate, n_samples)
    pos = np.array([list(spec.position_m)], dtype=np.float64)
    return greens_propagate(ext.reshape(1, -1), pos, mic_pos, art.sample_rate)


def compose_external(
    artifact_path: str,
    mic_geom_path: str,
    ext_spec: ExternalSourceSpec,
    out_synth_path: str,
    out_gt_path: str,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> None:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    n_samples = art.subsource_signals.shape[1] if ext_spec.duration_s is None \
        else int(round(ext_spec.duration_s * art.sample_rate))
    if n_samples > art.subsource_signals.shape[1]:
        raise ValueError(
            f"duration_s={ext_spec.duration_s}s requires {n_samples} samples, "
            f"but artifact only has {art.subsource_signals.shape[1]}"
        )

    drone_at_mics = greens_propagate(
        art.subsource_signals[:, :n_samples],
        art.subsource_positions,
        mic_pos,
        art.sample_rate,
        speed_of_sound,
    )

    ext_signal = generate_external_signal(ext_spec, art.sample_rate, n_samples)
    ext_at_mics = greens_propagate(
        ext_signal.reshape(1, -1),
        np.array([list(ext_spec.position_m)], dtype=np.float64),
        mic_pos,
        art.sample_rate,
        speed_of_sound,
    )

    mix = drone_at_mics + ext_at_mics

    out_synth = Path(out_synth_path)
    out_synth.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_synth, "w") as f:
        f.attrs["composed_by"] = "martymicfly.synth.compose_external"
        td = f.create_dataset("time_data", data=mix)
        td.attrs["sample_freq"] = float(art.sample_rate)
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = int(art.rotor_positions.shape[1])
        plat["rotor_positions"] = art.rotor_positions
        plat["rotor_radii"] = art.rotor_radii
        plat["blade_counts"] = art.blade_counts
        rpm = f.create_group("esc_telemetry")
        for name, data in art.rpm_per_esc.items():
            g = rpm.create_group(name)
            g["rpm"] = data["rpm"]
            g["timestamp"] = data["timestamp"]
        ext_g = f.create_group("external")
        ext_g.attrs["kind"] = ext_spec.kind
        ext_g.attrs["amplitude_db"] = float(ext_spec.amplitude_db)
        ext_g.attrs["position_m"] = np.asarray(ext_spec.position_m, dtype=np.float64)
        ext_g.attrs["seed"] = int(ext_spec.seed) if ext_spec.seed is not None else -1
        ext_g.attrs["speed_of_sound"] = float(speed_of_sound)
        ext_g.attrs["propagation"] = "greens_1_over_r"

    out_gt = Path(out_gt_path)
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_gt, "w") as f:
        f.attrs["kind"] = "external_only"
        td = f.create_dataset("time_data", data=ext_signal.reshape(-1, 1))
        td.attrs["sample_freq"] = float(art.sample_rate)
        ext_g = f.create_group("external")
        ext_g.attrs["kind"] = ext_spec.kind
        ext_g.attrs["amplitude_db"] = float(ext_spec.amplitude_db)
        ext_g.attrs["position_m"] = np.asarray(ext_spec.position_m, dtype=np.float64)
        ext_g.attrs["seed"] = int(ext_spec.seed) if ext_spec.seed is not None else -1
