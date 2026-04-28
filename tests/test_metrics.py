"""Tests for martymicfly.eval.metrics."""

import numpy as np
import pytest

from martymicfly.eval.metrics import compute_metrics


def _make_pre_post(fs=16_000.0, dur=1.0, bpf=100.0, n_harm=5,
                   suppression_db=30.0):
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    pre = np.zeros((n, 1), dtype=np.float64)
    for h in range(1, n_harm + 1):
        pre[:, 0] += np.sin(2 * np.pi * h * bpf * t)
    pre[:, 0] += np.random.default_rng(0).standard_normal(n) * 0.005
    post = pre.copy()
    # apply uniform attenuation in tonal bands by adding destructive replica
    factor = 10.0 ** (-suppression_db / 20.0)
    for h in range(1, n_harm + 1):
        post[:, 0] -= np.sin(2 * np.pi * h * bpf * t) * (1.0 - factor)
    return pre, post


def test_compute_metrics_basic_shape():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    assert "channels" in out
    assert len(out["channels"]) == 1
    ch = out["channels"][0]
    assert "tonal_per_harmonic" in ch
    assert len(ch["tonal_per_harmonic"]) == 5
    assert ch["channel"] == 0


def test_compute_metrics_reduction_matches_expected_db():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs, suppression_db=30.0)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    ch = out["channels"][0]
    # ±5 dB Toleranz wegen Welch-Fenster + Random-Noise
    assert ch["tonal_reduction_db"] == pytest.approx(30.0, abs=5.0)


def test_compute_metrics_per_motor_summary_present():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    summary = out["per_motor_bpf_summary"]
    assert summary[0]["bpf_min_hz"] == pytest.approx(100.0)
    assert summary[0]["bpf_max_hz"] == pytest.approx(100.0)
