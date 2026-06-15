"""Smoke + integration tests for martymicfly.processing.notch."""

import numpy as np
from scipy.signal import welch

from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
)
from martymicfly.processing.notch import NotchStage, NotchStageParams
from martymicfly.processing.pipeline import PipelineContext


def _rpm_per_esc_constant(rpm: float, duration_s: float = 2.0):
    return {
        "ESC1": {
            "rpm": np.array([rpm, rpm], dtype=np.float64),
            "timestamp": np.array([0.0, duration_s], dtype=np.float64),
        }
    }


def _build_ctx(signal: np.ndarray, fs: float, rpm: float, n_blades: int = 2,
               n_harmonics: int = 5):
    rpm_per_esc = _rpm_per_esc_constant(rpm, duration_s=signal.shape[0] / fs)
    per_motor_bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0,
        n_samples=signal.shape[0], sample_rate=fs, n_blades=n_blades,
    )
    harm_matrix = build_harmonic_matrix(per_motor_bpf, n_harmonics)
    return PipelineContext(
        time_data=signal,
        sample_rate=fs,
        rpm_per_esc=rpm_per_esc,
        mic_positions=np.zeros((signal.shape[1], 3), dtype=np.float64),
        per_motor_bpf=per_motor_bpf,
        harm_matrix=harm_matrix,
        metadata={},
    )


def _band_db(signal_1d: np.ndarray, fs: float, f_target: float, half_bw: float = 5.0):
    f, psd = welch(signal_1d, fs=fs, nperseg=4096, noverlap=2048,
                   window="hann", scaling="density")
    band = (f >= f_target - half_bw) & (f <= f_target + half_bw)
    energy = np.trapz(psd[band], f[band])
    return 10.0 * np.log10(max(energy, 1e-30))


def test_notchstage_suppresses_pure_tone():
    fs = 16_000.0
    rpm = 3000.0
    bpf = rpm * 2 / 60.0  # 100 Hz
    n = int(2.0 * fs)
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * bpf * t)[:, None]  # (N, 1)

    ctx = _build_ctx(signal, fs, rpm, n_harmonics=5)
    stage = NotchStage(NotchStageParams(
        n_blades=2, n_harmonics=5, pole_radius=0.998,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)

    pre_db = _band_db(signal[:, 0], fs, bpf)
    post_db = _band_db(out.time_data[:, 0], fs, bpf)
    assert post_db < pre_db - 30.0, f"expected ≥30 dB suppression, got {pre_db - post_db:.1f} dB"


def test_notchstage_preserves_pre_signal_in_metadata():
    fs = 16_000.0
    n = int(0.5 * fs)
    signal = np.random.default_rng(0).standard_normal((n, 1))
    ctx = _build_ctx(signal, fs, rpm=3000.0, n_harmonics=2)
    stage = NotchStage(NotchStageParams(
        n_blades=2, n_harmonics=2, pole_radius=0.99,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)
    np.testing.assert_array_equal(out.metadata["pre_notch"], signal)


def test_notchstage_per_channel_loop_handles_multichannel_input():
    fs = 16_000.0
    n = int(1.0 * fs)
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((n, 3)) * 0.01
    bpf = 100.0
    t = np.arange(n) / fs
    for c in range(3):
        signal[:, c] += np.sin(2 * np.pi * bpf * t)
    ctx = _build_ctx(signal, fs, rpm=3000.0, n_harmonics=3)
    stage = NotchStage(NotchStageParams(
        n_blades=2, n_harmonics=3, pole_radius=0.998,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)
    assert out.time_data.shape == signal.shape
    for c in range(3):
        pre_db = _band_db(signal[:, c], fs, bpf)
        post_db = _band_db(out.time_data[:, c], fs, bpf)
        assert post_db < pre_db - 20.0
