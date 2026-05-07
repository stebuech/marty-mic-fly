"""Unit tests for analysis/scaling_ladder.py — forward propagator + 3 rungs."""
from __future__ import annotations

import numpy as np
import pytest

# scaling_ladder.py lives in repo-root analysis/, not on the package path.
# Path-import via importlib so tests don't depend on PYTHONPATH=analysis.
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "analysis" / "scaling_ladder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scaling_ladder", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scaling_ladder"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sl():
    return _load_module()


def test_propagate_white_noise_yields_correct_per_mic_psd(sl):
    """Forward propagator: per-mic PSD = S_q / r_m^2 (free-field 1/r)."""
    rng = np.random.default_rng(42)
    fs = 51_200.0
    duration_s = 4.0
    n_samples = int(fs * duration_s)
    s_q_pa2_per_hz = 1.0  # source PSD at 1 m

    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.0, 0.0, 0.0],   # r = 1.5
        [0.5, 0.0, 0.0],   # r = sqrt(0.25 + 2.25) ≈ 1.581
        [0.0, 0.4, -0.5],  # r = sqrt(0 + 0.16 + 1.0) ≈ 1.077
    ])
    r = np.linalg.norm(mic_positions - source_pos, axis=1)

    time_data = sl.propagate_white_noise(
        n_samples=n_samples,
        sample_rate=fs,
        s_q_pa2_per_hz=s_q_pa2_per_hz,
        source_position=source_pos,
        mic_positions=mic_positions,
        rng=rng,
    )
    assert time_data.shape == (n_samples, mic_positions.shape[0])

    # Welch PSD per mic; compare mean PSD in [200, 6000] Hz to theory.
    from scipy.signal import welch
    f, p = welch(time_data, fs=fs, nperseg=512, noverlap=256,
                 window="hann", scaling="density", axis=0)
    band = (f >= 200.0) & (f <= 6000.0)
    measured = p[band, :].mean(axis=0)              # (M,)
    theoretical = s_q_pa2_per_hz / (r ** 2)          # (M,)

    delta_db = 10.0 * np.log10(measured / theoretical)
    assert np.all(np.abs(delta_db) < 0.5), (
        f"Per-mic PSD off theory by {delta_db} dB"
    )


def test_rung1_mic_psd_matches_theory_within_0p5_db(sl):
    rng = np.random.default_rng(7)
    fs = 51_200.0
    n_samples = int(fs * 4.0)
    s_q = 1e-4

    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung1_mic_psd(
        time_data=time_data,
        sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos,
        mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
    )
    assert "delta_db_per_mic" in result
    assert "delta_db_mean" in result
    assert "frequencies_hz" in result
    assert result["delta_db_per_mic"].shape == (mic_positions.shape[0],)
    assert abs(result["delta_db_mean"]) < 0.5


def test_rung2_csm_diag_matches_theory_within_0p5_db(sl):
    rng = np.random.default_rng(11)
    fs = 51_200.0
    n_samples = int(fs * 4.0)
    s_q = 1e-4
    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung2_csm_diag(
        time_data=time_data, sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
        diag_loading_rel=0.0,   # disable for clean comparison
    )
    assert "csm_shape" in result
    assert "delta_db_per_mic" in result
    assert "csm" in result and "frequencies_hz" in result
    assert abs(result["delta_db_mean"]) < 0.5


def test_rung3_steered_psd_matches_geometry_factor_within_2_db(sl):
    """Rung 3 expectation: S_steered(target) = S_q · <1/r_m>^2 (phase-only DAS).

    Tolerance 2 dB (not 0.5) because phase-only DAS has frequency-dependent
    sidelobe-bleed at small arrays — only the broadband mean of the on-source
    bin should match the geometric factor.
    """
    rng = np.random.default_rng(13)
    fs = 51_200.0
    n_samples = int(fs * 6.0)              # longer for tighter Welch estimate
    s_q = 1e-4
    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
        [0.07, 0.07, 0.0], [-0.07, -0.07, 0.0],
        [0.07, -0.07, 0.0], [-0.07, 0.07, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung3_steered_psd(
        time_data=time_data, sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
        diag_loading_rel=0.0,
    )
    assert "delta_db_band_mean" in result
    assert "theoretical_psd" in result
    assert "steered_psd" in result
    # Geometric factor < 1, so theoretical_psd < S_q. Both reported in dB.
    assert abs(result["delta_db_band_mean"]) < 2.0
