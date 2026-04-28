"""Generate tests/fixtures/tiny_synth.h5 — deterministic Acoular-format synth."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

FIXTURE = Path(__file__).resolve().parent / "tiny_synth.h5"

SAMPLE_RATE = 16_000.0
DURATION_S = 1.0
N_CHANNELS = 4
N_BLADES = 2  # so BPF = rpm * 2 / 60

ESC_RPM = {"ESC1": 3000.0, "ESC2": 3600.0}  # → BPF 100 Hz / 120 Hz
N_HARMONICS = 5
EXTERNAL_TONE_HZ = 800.0
EXTERNAL_TONE_AMPLITUDE = 0.2
NOISE_RMS = 0.01

ESC_DT_S = 1e-3  # 1 kHz telemetry rate (matches real synth)


def build_signal(rng: np.random.Generator) -> np.ndarray:
    """Return time_data of shape (N, C) float64."""
    n = int(round(SAMPLE_RATE * DURATION_S))
    t = np.arange(n) / SAMPLE_RATE

    base = np.zeros(n, dtype=np.float64)
    for rpm in ESC_RPM.values():
        bpf = rpm * N_BLADES / 60.0
        for h in range(1, N_HARMONICS + 1):
            base += np.sin(2 * np.pi * h * bpf * t) / N_HARMONICS

    base += EXTERNAL_TONE_AMPLITUDE * np.sin(2 * np.pi * EXTERNAL_TONE_HZ * t)

    data = np.empty((n, N_CHANNELS), dtype=np.float64)
    for c in range(N_CHANNELS):
        noise = rng.standard_normal(n) * NOISE_RMS
        data[:, c] = base + noise
    return data


def build_telemetry() -> dict[str, dict[str, np.ndarray]]:
    n_t = int(round(DURATION_S / ESC_DT_S)) + 1
    timestamps = np.arange(n_t) * ESC_DT_S
    return {
        name: {
            "rpm": np.full(n_t, rpm, dtype=np.float64),
            "timestamp": timestamps.astype(np.float64),
        }
        for name, rpm in ESC_RPM.items()
    }


def write_fixture(path: Path = FIXTURE) -> Path:
    rng = np.random.default_rng(seed=42)
    data = build_signal(rng)
    telemetry = build_telemetry()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        td = f.create_dataset("time_data", data=data, dtype="float64")
        td.attrs["sample_freq"] = np.float64(SAMPLE_RATE)
        grp = f.create_group("esc_telemetry")
        for name, esc in telemetry.items():
            sub = grp.create_group(name)
            sub.create_dataset("rpm", data=esc["rpm"], dtype="float64")
            sub.create_dataset("timestamp", data=esc["timestamp"], dtype="float64")
    return path


if __name__ == "__main__":
    out = write_fixture()
    print(f"wrote {out}")
