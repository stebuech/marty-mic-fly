"""Schreib- und Lese-Helfer für die frequency-resolved h5-Beilage einer
Studien-Run."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from analysis.separation_study.metric_extensions import decompose_residual


def write_freq_resolved(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excess, deficit = decompose_residual(payload["psd_post"], payload["ext_gt"])
    with h5py.File(out_path, "w") as f:
        for k in ("frequencies", "psd_post", "ext_gt", "d_ref"):
            f.create_dataset(k, data=np.asarray(payload[k], dtype=np.float64))
        f.create_dataset("excess", data=excess)
        f.create_dataset("deficit", data=deficit)


def read_freq_resolved(in_path: Path) -> dict:
    with h5py.File(in_path, "r") as f:
        return {k: np.asarray(f[k][:], dtype=np.float64) for k in f.keys()}
