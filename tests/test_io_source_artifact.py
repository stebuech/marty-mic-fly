import h5py
import numpy as np
import pytest


def _write_minimal_artifact(path):
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 16000.0
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["blade_counts"] = np.array([2, 2], dtype=np.int32)
        plat["rotor_positions"] = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.10, 0.10], dtype=np.float64)
        f["rotor_index"] = np.array([0, 0, 1, 1], dtype=np.int32)
        f["subsource_positions"] = np.array(
            [[0.10, 0.0, 0.0], [0.20, 0.0, 0.0], [-0.10, 0.0, 0.0], [-0.20, 0.0, 0.0]],
            dtype=np.float64,
        )
        rng = np.random.default_rng(0)
        f["subsource_signals"] = rng.standard_normal((4, 1600)).astype(np.float64)
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(10, 3000.0)
            g["timestamp"] = np.linspace(0, 0.1, 10)


def test_load_source_artifact_returns_expected_shapes(tmp_path):
    from martymicfly.io.source_artifact import load_source_artifact
    p = tmp_path / "art.h5"
    _write_minimal_artifact(p)
    art = load_source_artifact(str(p))
    assert art.sample_rate == 16000.0
    assert art.subsource_positions.shape == (4, 3)
    assert art.subsource_signals.shape == (4, 1600)
    assert art.rotor_index.shape == (4,)
    assert art.rotor_positions.shape == (3, 2)
    assert art.rotor_radii.tolist() == [0.10, 0.10]
    assert sorted(art.rpm_per_esc.keys()) == ["ESC1", "ESC2"]


def test_load_source_artifact_rejects_missing_platform(tmp_path):
    p = tmp_path / "no_plat.h5"
    with h5py.File(p, "w") as f:
        f.attrs["sample_rate"] = 16000.0
    from martymicfly.io.source_artifact import load_source_artifact
    with pytest.raises(ValueError, match="platform"):
        load_source_artifact(str(p))
