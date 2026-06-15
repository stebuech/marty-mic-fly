"""Writer/reader for the CLEAN-SC source-map HDF5 file.

The source map is the deconvolved (CLEAN-SC) power distribution over the
search grid: ``powers`` has shape (F, G) — one power value per frequency bin
and grid cell. The pipeline keeps it only in memory; this file persists it so
downstream analysis (e.g. map-space ROI integration / "Ansatz 1") can rebuild
spatially-filtered spectra without re-running CLEAN-SC.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_source_map(
    path: str,
    powers: np.ndarray,           # (F, G) Pa^2 per cell, trace-rescaled
    frequencies: np.ndarray,      # (F,)
    positions: np.ndarray,        # (G, 3) grid points
    drone_mask: np.ndarray,       # (G,) bool — cells subtracted as "drone"
    attrs: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v
        f["powers"] = np.asarray(powers, dtype=np.float64)
        f["frequencies"] = np.asarray(frequencies, dtype=np.float64)
        f["positions"] = np.asarray(positions, dtype=np.float64)
        f["drone_mask"] = np.asarray(drone_mask, dtype=bool)


def read_source_map(path: str) -> dict:
    """Return {powers, frequencies, positions, drone_mask} as numpy arrays."""
    with h5py.File(Path(path), "r") as f:
        return {
            "powers": f["powers"][:],
            "frequencies": f["frequencies"][:],
            "positions": f["positions"][:],
            "drone_mask": f["drone_mask"][:].astype(bool),
        }
