"""Load synthetic mic+telemetry data in Acoular HDF5 layout."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_synth_h5(path: str | Path) -> dict:
    """Load Acoular-format synth HDF5.

    Layout expected::

        /time_data    (N, C) float64  + attr sample_freq (Hz)
        /esc_telemetry/<ESC_NAME>/rpm       (T,) float64
                                  /timestamp (T,) float64 (monoton steigend, in Sekunden)

    Returns
    -------
    dict
        Schlüssel ``time_data``, ``sample_rate``, ``rpm_per_esc``, ``duration``.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "time_data" not in f:
            raise ValueError(f"{path}: missing 'time_data' dataset")
        td = f["time_data"]
        if td.ndim != 2:
            raise ValueError(f"{path}: time_data must be 2D, got {td.ndim}D")
        if "sample_freq" not in td.attrs:
            raise ValueError(f"{path}: time_data is missing 'sample_freq' attribute")

        sample_rate = float(td.attrs["sample_freq"])
        time_data = np.asarray(td[()], dtype=np.float64)

        rpm_per_esc: dict[str, dict[str, np.ndarray]] = {}
        if "esc_telemetry" in f:
            for esc_name, esc_grp in f["esc_telemetry"].items():
                rpm = np.asarray(esc_grp["rpm"][()], dtype=np.float64)
                ts = np.asarray(esc_grp["timestamp"][()], dtype=np.float64)
                if ts.size > 1 and not np.all(np.diff(ts) > 0):
                    raise ValueError(
                        f"{path}: timestamps for {esc_name} not strictly monoton increasing"
                    )
                if rpm.shape != ts.shape:
                    raise ValueError(
                        f"{path}: rpm/timestamp shape mismatch for {esc_name}: "
                        f"{rpm.shape} vs {ts.shape}"
                    )
                rpm_per_esc[esc_name] = {"rpm": rpm, "timestamp": ts}

        platform = None
        if "platform" in f:
            plat = f["platform"]
            platform = {
                "n_rotors": int(plat.attrs["n_rotors"]),
                "rotor_positions": np.asarray(plat["rotor_positions"][...], dtype=np.float64),
                "rotor_radii": np.asarray(plat["rotor_radii"][...], dtype=np.float64),
                "blade_counts": np.asarray(plat["blade_counts"][...], dtype=np.int32),
            }

    duration = time_data.shape[0] / sample_rate
    return {
        "time_data": time_data,
        "sample_rate": sample_rate,
        "rpm_per_esc": rpm_per_esc,
        "duration": duration,
        "platform": platform,
    }
