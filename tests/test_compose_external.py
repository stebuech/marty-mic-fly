from pathlib import Path

import h5py
import numpy as np


def _write_artifact(path, fs=16000.0, n=1600, n_subs=4):
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = fs
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["blade_counts"] = np.array([2, 2], dtype=np.int32)
        plat["rotor_positions"] = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.10, 0.10])
        f["rotor_index"] = np.array([0, 0, 1, 1], dtype=np.int32)
        f["subsource_positions"] = np.array(
            [[0.10, 0, 0], [0.20, 0, 0], [-0.10, 0, 0], [-0.20, 0, 0]],
            dtype=np.float64,
        )
        f["subsource_signals"] = rng.standard_normal((n_subs, n)).astype(np.float64)
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(5, 3000.0)
            g["timestamp"] = np.linspace(0.0, 0.1, 5)


def _write_geom_xml(path, mic_positions):
    lines = ['<?xml version="1.0"?>', '<MicArray>']
    for i, p in enumerate(mic_positions):
        lines.append(f'  <pos Name="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>')
    lines.append('</MicArray>')
    Path(path).write_text("\n".join(lines))


def test_compose_external_writes_synth_and_gt(tmp_path):
    from martymicfly.synth.compose_external import compose_external
    from martymicfly.synth.external_source import ExternalSourceSpec
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    _write_artifact(art)
    _write_geom_xml(geom, [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0)])
    spec = ExternalSourceSpec(kind="noise", position_m=(0.5, 0.0, -0.5), seed=0)
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    assert out_synth.exists() and out_gt.exists()
    with h5py.File(out_synth, "r") as f:
        assert f["time_data"].shape == (1600, 4)
        assert "platform" in f
        assert "external" in f
        assert tuple(f["external"].attrs["position_m"]) == (0.5, 0.0, -0.5)
    with h5py.File(out_gt, "r") as f:
        assert f["time_data"].shape == (1600, 1)
        assert f.attrs["kind"] == "external_only"


def test_compose_external_drone_plus_external_equals_mix(tmp_path):
    """Synth mix approx drone-only + external-only propagated separately."""
    from martymicfly.synth.compose_external import (
        compose_external,
        _propagate_drone_only,
        _propagate_external_only,
    )
    from martymicfly.synth.external_source import ExternalSourceSpec
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    _write_artifact(art)
    _write_geom_xml(geom, [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0)])
    spec = ExternalSourceSpec(kind="noise", position_m=(0.5, 0.0, -0.5), seed=0)
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    drone_only = _propagate_drone_only(str(art), str(geom))
    ext_only = _propagate_external_only(str(art), str(geom), spec)
    with h5py.File(out_synth, "r") as f:
        mix = np.asarray(f["time_data"][...])
    np.testing.assert_allclose(mix, drone_only + ext_only, atol=1e-9)
