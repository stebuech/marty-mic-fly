"""External source specification and signal generation for compose_external."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class ExternalSourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["noise", "sweep", "sine", "file"]
    position_m: tuple[float, float, float] = (0.0, 0.0, -1.5)
    amplitude_db: float = 0.0
    duration_s: Optional[float] = None
    sine_freq_hz: Optional[float] = None
    sweep_f_lo_hz: Optional[float] = None
    sweep_f_hi_hz: Optional[float] = None
    file_path: Optional[str] = None
    seed: Optional[int] = 0

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "ExternalSourceSpec":
        if self.kind == "sine" and self.sine_freq_hz is None:
            raise ValueError("kind='sine' requires sine_freq_hz")
        if self.kind == "sweep":
            if self.sweep_f_lo_hz is None or self.sweep_f_hi_hz is None:
                raise ValueError("kind='sweep' requires sweep_f_lo_hz and sweep_f_hi_hz")
            if self.sweep_f_hi_hz <= self.sweep_f_lo_hz:
                raise ValueError("sweep_f_hi_hz must be > sweep_f_lo_hz")
        if self.kind == "file" and not self.file_path:
            raise ValueError("kind='file' requires file_path")
        return self


def generate_external_signal(
    spec: ExternalSourceSpec,
    sample_rate: float,
    n_samples: int,
) -> np.ndarray:
    """Generate the external source signal at the source location (pre-propagation)."""
    n = int(n_samples)
    t = np.arange(n) / sample_rate
    if spec.kind == "noise":
        rng = np.random.default_rng(spec.seed)
        sig = rng.standard_normal(n)
    elif spec.kind == "sine":
        sig = np.sin(2 * np.pi * spec.sine_freq_hz * t)
    elif spec.kind == "sweep":
        from scipy.signal import chirp
        sig = chirp(
            t, f0=spec.sweep_f_lo_hz, t1=t[-1], f1=spec.sweep_f_hi_hz,
            method="logarithmic",
        )
    elif spec.kind == "file":
        import h5py
        with h5py.File(spec.file_path, "r") as f:
            sig = np.asarray(f["time_data"][...], dtype=np.float64).reshape(-1)[:n]
        if sig.shape[0] < n:
            raise ValueError(
                f"file {spec.file_path} has only {sig.shape[0]} samples; need {n}"
            )
    else:
        raise ValueError(f"unsupported kind={spec.kind!r}")

    gain = 10.0 ** (spec.amplitude_db / 20.0)
    return (gain * sig).astype(np.float64)
