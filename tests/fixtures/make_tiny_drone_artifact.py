"""Generate a deterministic tiny drone-source artifact for tests."""
from pathlib import Path

import h5py
import numpy as np


def _write_geom(path: Path, mic_positions):
    lines = ['<?xml version="1.0"?>', '<MicArray>']
    for i, p in enumerate(mic_positions):
        lines.append(f'  <pos Name="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>')
    lines.append('</MicArray>')
    path.write_text("\n".join(lines))


def main():
    here = Path(__file__).parent
    art = here / "tiny_drone_artifact.h5"
    geom = here / "tiny_geom_4mic.xml"

    fs = 16000.0
    duration_s = 1.0
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)

    rotor_positions = np.array(
        [[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
    )
    rotor_radii = np.array([0.10, 0.10], dtype=np.float64)
    blade_counts = np.array([2, 2], dtype=np.int32)

    subsource_positions = np.array(
        [
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [-0.10, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotor_index = np.array([0, 0, 1, 1], dtype=np.int32)
    subsource_signals = rng.standard_normal((4, n)).astype(np.float64)

    with h5py.File(art, "w") as f:
        f.attrs["sample_rate"] = fs
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["rotor_positions"] = rotor_positions
        plat["rotor_radii"] = rotor_radii
        plat["blade_counts"] = blade_counts
        f["rotor_index"] = rotor_index
        f["subsource_positions"] = subsource_positions
        f["subsource_signals"] = subsource_signals
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(20, 3000.0)
            g["timestamp"] = np.linspace(0.0, duration_s, 20)

    _write_geom(geom, [(0.3, 0.0, 0.10), (-0.3, 0.0, -0.10),
                       (0.0, 0.3, 0.10), (0.0, -0.3, -0.10)])
    print(f"wrote {art} and {geom}")


if __name__ == "__main__":
    main()
