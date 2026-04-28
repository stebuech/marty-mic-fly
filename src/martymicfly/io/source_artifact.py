"""Reader for drone-source artifacts (gap_tip subsource model).

The artifact is produced upstream by drone_synthdata's reconstruction step
and contains the per-subsource time-domain signal at known geometry plus
platform metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class SourceArtifact:
    sample_rate: float
    subsource_positions: np.ndarray   # (G, 3)
    subsource_signals: np.ndarray     # (G, N)
    rotor_index: np.ndarray           # (G,) int
    rotor_positions: np.ndarray       # (3, R)
    rotor_radii: np.ndarray           # (R,)
    blade_counts: np.ndarray          # (R,)
    rpm_per_esc: dict
    metadata: dict


def load_source_artifact(path: str | Path) -> SourceArtifact:
    p = Path(path)
    with h5py.File(p, "r") as f:
        if "platform" not in f:
            raise ValueError(
                f"{p}: missing /platform group; not a valid drone source artifact"
            )
        sr = float(f.attrs["sample_rate"])
        plat = f["platform"]
        rotor_positions = np.asarray(plat["rotor_positions"][...], dtype=np.float64)
        rotor_radii = np.asarray(plat["rotor_radii"][...], dtype=np.float64)
        blade_counts = np.asarray(plat["blade_counts"][...], dtype=np.int32)
        subsource_positions = np.asarray(f["subsource_positions"][...], dtype=np.float64)
        subsource_signals = np.asarray(f["subsource_signals"][...], dtype=np.float64)
        rotor_index = np.asarray(f["rotor_index"][...], dtype=np.int32)
        rpm_per_esc: dict = {}
        if "rpm_per_esc" in f:
            for name in f["rpm_per_esc"]:
                g = f["rpm_per_esc"][name]
                rpm_per_esc[name] = {
                    "rpm": np.asarray(g["rpm"][...], dtype=np.float64),
                    "timestamp": np.asarray(g["timestamp"][...], dtype=np.float64),
                }
        metadata = dict(f.attrs)
        if "reconstruction" in f:
            metadata["reconstruction"] = dict(f["reconstruction"].attrs)

    return SourceArtifact(
        sample_rate=sr,
        subsource_positions=subsource_positions,
        subsource_signals=subsource_signals,
        rotor_index=rotor_index,
        rotor_positions=rotor_positions,
        rotor_radii=rotor_radii,
        blade_counts=blade_counts,
        rpm_per_esc=rpm_per_esc,
        metadata=metadata,
    )
