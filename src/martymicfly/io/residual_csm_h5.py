"""Writer for residual-CSM HDF5 files."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_residual_csm(
    path: str,
    residual: np.ndarray,         # (F, M, M) complex
    freqs: np.ndarray,            # (F,)
    attrs: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v
        f["csm_real"] = np.real(residual).astype(np.float64)
        f["csm_imag"] = np.imag(residual).astype(np.float64)
        f["frequencies"] = np.asarray(freqs, dtype=np.float64)
