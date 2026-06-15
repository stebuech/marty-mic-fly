"""Tests for martymicfly.processing.frequencies."""

import numpy as np
import pytest

from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
    linear_r_schedule,
)


def _const_rpm_telemetry(rpm: float, t_end: float = 1.5):
    return {"rpm": np.array([rpm, rpm]), "timestamp": np.array([0.0, t_end])}


def test_interpolate_constant_rpm_yields_constant_bpf():
    rpm_per_esc = {
        "ESC1": _const_rpm_telemetry(3000.0),
        "ESC2": _const_rpm_telemetry(3600.0),
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=1000, sample_rate=1000.0, n_blades=2,
    )
    assert bpf.shape == (1000, 2)
    np.testing.assert_allclose(bpf[:, 0], 100.0)  # ESC1 alphabetical
    np.testing.assert_allclose(bpf[:, 1], 120.0)


def test_interpolate_uses_alphabetical_esc_order():
    rpm_per_esc = {
        "ESC2": _const_rpm_telemetry(3600.0),
        "ESC1": _const_rpm_telemetry(3000.0),
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=10, sample_rate=1000.0, n_blades=2,
    )
    np.testing.assert_allclose(bpf[:, 0], 100.0)  # ESC1 first
    np.testing.assert_allclose(bpf[:, 1], 120.0)


def test_interpolate_linear_ramp():
    rpm_per_esc = {
        "ESC1": {"rpm": np.array([0.0, 6000.0]), "timestamp": np.array([0.0, 1.0])},
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=11, sample_rate=10.0, n_blades=2,
    )
    expected = np.linspace(0.0, 6000.0 * 2 / 60.0, 11)
    np.testing.assert_allclose(bpf[:, 0], expected, rtol=1e-10)


def test_build_harmonic_matrix_shape_and_columns():
    bpf = np.array([[100.0, 120.0], [101.0, 121.0]])  # (N=2, S=2)
    M = build_harmonic_matrix(bpf, n_harmonics=3)
    assert M.shape == (2, 2 * 3)
    # column index s*M + (h-1)
    np.testing.assert_allclose(M[:, 0], [100.0, 101.0])  # s=0, h=1
    np.testing.assert_allclose(M[:, 1], [200.0, 202.0])  # s=0, h=2
    np.testing.assert_allclose(M[:, 2], [300.0, 303.0])  # s=0, h=3
    np.testing.assert_allclose(M[:, 3], [120.0, 121.0])  # s=1, h=1
    np.testing.assert_allclose(M[:, 4], [240.0, 242.0])
    np.testing.assert_allclose(M[:, 5], [360.0, 363.0])


def test_linear_r_schedule_shape_and_clipping():
    r = linear_r_schedule(
        n_harmonics=10, fs=10_000.0, delta_bpf_hz=10.0,
        k_cover=1.5, margin_hz=5.0, r_min=0.90, r_max=0.9995,
    )
    assert r.shape == (10,)
    assert (r >= 0.90).all()
    assert (r <= 0.9995).all()
    # bw monoton wächst → r monoton fällt
    assert (np.diff(r) <= 0).all()


def test_linear_r_schedule_clamps_to_rmax_at_h1():
    # very small bw should hit r_max
    r = linear_r_schedule(
        n_harmonics=5, fs=100_000.0, delta_bpf_hz=0.01,
        k_cover=0.01, margin_hz=0.01, r_min=0.5, r_max=0.999,
    )
    assert r[0] == pytest.approx(0.999)
