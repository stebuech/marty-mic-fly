"""Tests for martymicfly.io.synth_h5."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from martymicfly.io.synth_h5 import load_synth_h5

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_synth.h5"


def test_load_returns_expected_keys():
    out = load_synth_h5(FIXTURE)
    assert set(out.keys()) == {"time_data", "sample_rate", "rpm_per_esc", "duration"}


def test_load_shape_and_dtype():
    out = load_synth_h5(FIXTURE)
    assert out["time_data"].shape == (16_000, 4)
    assert out["time_data"].dtype == np.float64
    assert out["sample_rate"] == pytest.approx(16_000.0)
    assert out["duration"] == pytest.approx(1.0)


def test_load_rpm_per_esc_structure():
    out = load_synth_h5(FIXTURE)
    rpe = out["rpm_per_esc"]
    assert set(rpe.keys()) == {"ESC1", "ESC2"}
    for esc in rpe.values():
        assert set(esc.keys()) == {"rpm", "timestamp"}
        assert esc["rpm"].dtype == np.float64
        assert esc["timestamp"].dtype == np.float64
        assert np.all(np.diff(esc["timestamp"]) > 0)
    assert np.allclose(rpe["ESC1"]["rpm"], 3000.0)
    assert np.allclose(rpe["ESC2"]["rpm"], 3600.0)


def test_load_missing_sample_freq_raises(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("time_data", data=np.zeros((10, 2), dtype=np.float64))
        f.create_group("esc_telemetry")
    with pytest.raises(ValueError, match="sample_freq"):
        load_synth_h5(bad)


def test_load_non_monotonic_timestamps_raises(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        td = f.create_dataset("time_data", data=np.zeros((10, 2), dtype=np.float64))
        td.attrs["sample_freq"] = 1000.0
        grp = f.create_group("esc_telemetry/ESC1")
        grp.create_dataset("rpm", data=np.array([100.0, 100.0]))
        grp.create_dataset("timestamp", data=np.array([0.0, 0.0]))  # not monotonic
    with pytest.raises(ValueError, match="monoton"):
        load_synth_h5(bad)
