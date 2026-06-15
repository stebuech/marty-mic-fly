"""Write filtered time_data + telemetry pass-through + provenance attrs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_filtered(
    *,
    out_path: str | Path,
    filtered_time_data: np.ndarray,
    sample_rate: float,
    rpm_per_esc: dict[str, dict[str, np.ndarray]],
    attrs: dict[str, str | int | float],
) -> None:
    """Write Acoular-format HDF5 with filtered signal + telemetry + attrs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        td = f.create_dataset(
            "time_data",
            data=np.ascontiguousarray(filtered_time_data, dtype=np.float64),
            dtype="float64",
        )
        td.attrs["sample_freq"] = np.float64(sample_rate)

        grp = f.create_group("esc_telemetry")
        for esc_name, esc in rpm_per_esc.items():
            sub = grp.create_group(esc_name)
            sub.create_dataset("rpm", data=esc["rpm"].astype(np.float64))
            sub.create_dataset("timestamp", data=esc["timestamp"].astype(np.float64))

        for k, v in attrs.items():
            f.attrs[k] = v
