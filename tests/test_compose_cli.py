from pathlib import Path

import h5py
import numpy as np


def _write_artifact_and_geom(tmp_path):
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    rng = np.random.default_rng(11)
    with h5py.File(art, "w") as f:
        f.attrs["sample_rate"] = 16000.0
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 1
        plat["rotor_positions"] = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.1])
        plat["blade_counts"] = np.array([2], dtype=np.int32)
        f["rotor_index"] = np.array([0, 0], dtype=np.int32)
        f["subsource_positions"] = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
        f["subsource_signals"] = rng.standard_normal((2, 1600))
        rpm = f.create_group("rpm_per_esc")
        g = rpm.create_group("ESC1")
        g["rpm"] = np.full(5, 3000.0)
        g["timestamp"] = np.linspace(0, 0.1, 5)
    geom.write_text(
        '<MicArray>\n'
        '  <pos Name="0" x="0.3" y="0" z="0"/>\n'
        '  <pos Name="1" x="-0.3" y="0" z="0"/>\n'
        '</MicArray>\n'
    )
    return art, geom


def test_compose_cli_runs(tmp_path):
    from martymicfly.synth.cli.compose import main
    art, geom = _write_artifact_and_geom(tmp_path)
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    cfg_yaml = tmp_path / "compose.yaml"
    cfg_yaml.write_text(f"""
input:
  drone_source_artifact_h5: {art}
  mic_geom_xml: {geom}
external:
  kind: noise
  position_m: [0.0, 0.0, -1.0]
  amplitude_db: 0.0
  duration_s: null
  seed: 0
output:
  synth_h5: {out_synth}
  ground_truth_h5: {out_gt}
""")
    rc = main(["--config", str(cfg_yaml)])
    assert rc == 0
    assert out_synth.exists() and out_gt.exists()
