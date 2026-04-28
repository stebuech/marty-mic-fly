"""Reader for ground-truth (external-only) HDF5 files written by compose_external."""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class GroundTruth:
    signal: np.ndarray            # (N,) — at the source location, pre-propagation
    sample_rate: float
    position_m: tuple[float, float, float]
    kind: str
    amplitude_db: float


def load_ground_truth(path: str) -> GroundTruth:
    with h5py.File(path, "r") as f:
        sr = float(f.attrs["sample_freq"])
        sig = np.asarray(f["time_data"][...]).reshape(-1)
        ext = f["external"]
        pos = tuple(float(x) for x in np.asarray(ext.attrs["position_m"]))
        return GroundTruth(
            signal=sig,
            sample_rate=sr,
            position_m=pos,
            kind=str(ext.attrs["kind"]),
            amplitude_db=float(ext.attrs.get("amplitude_db", 0.0)),
        )
