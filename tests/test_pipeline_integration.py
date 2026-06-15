"""End-to-end: tiny fixture → CLI → outputs."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO / "tests" / "fixtures" / "tiny_synth.h5"
FIXTURE_XML = REPO / "tests" / "fixtures" / "tiny_geom.xml"


def _config_for_fixture(out_dir: Path) -> dict:
    return {
        "input": {
            "audio_h5": str(FIXTURE_H5),
            "mic_geom_xml": str(FIXTURE_XML),
        },
        "segment": {"mode": "middle", "duration": 0.5},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 5},
        "stages": [
            {
                "kind": "notch",
                "pole_radius": {"mode": "scalar", "value": 0.998},
                "multichannel": False,
                "block_size": 4096,
            },
        ],
        "metrics": {
            "welch_nperseg": 2048,
            "welch_noverlap": 1024,
            "bandwidth_factor": 1.0,
            "broadband_low_hz": None,
        },
        "plots": {
            "enabled": True,
            "fmax_hz": 2000.0,
            "spectrogram_window": 1024,
            "spectrogram_overlap": 512,
            "channel_subset": [0, 1],
        },
        "output": {
            "dir": str(out_dir / "{run_id}"),
            "filtered_h5": "filtered.h5",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "plots_subdir": "plots",
            "copy_config": True,
        },
    }


def test_end_to_end_on_tiny_fixture(tmp_path: Path):
    cfg = _config_for_fixture(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "martymicfly.cli.run_notch",
         "--config", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    runs = list(tmp_path.glob("*"))
    run_dirs = [p for p in runs if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "filtered.h5").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "stage1_metrics.csv").exists()
    assert (run_dir / "config.yaml").exists()
    plots = list((run_dir / "plots").glob("ch*.html"))
    assert len(plots) == 2  # channel_subset

    metrics = json.loads((run_dir / "metrics.json").read_text())
    notch = metrics["stage1_notch"]
    assert notch["n_motors"] == 2
    assert notch["n_harmonics"] == 5
    # Tonal reduction at the seeded harmonics should be substantial
    for ch in notch["channels"]:
        assert ch["tonal_reduction_db"] > 15.0, ch


def test_run_pipeline_cli_with_stages_list(tmp_path):
    """The new CLI accepts the stages-list YAML and runs notch only."""
    from pathlib import Path
    fixtures = Path("tests/fixtures")
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(f"""
input:
  audio_h5: {fixtures / 'tiny_synth.h5'}
  mic_geom_xml: {fixtures / 'tiny_geom.xml'}
segment: {{mode: head, duration: 0.5}}
channels: {{selection: all}}
rotor: {{n_blades: 2, n_harmonics: 4}}
stages:
  - kind: notch
    pole_radius: {{mode: scalar, value: 0.99}}
    multichannel: false
    block_size: 1024
metrics: {{welch_nperseg: 1024, welch_noverlap: 512}}
plots: {{enabled: false, spectrogram_window: 512, spectrogram_overlap: 256}}
output: {{dir: {tmp_path / 'out'}}}
""")
    from martymicfly.cli.run_pipeline import main
    rc = main(["--config", str(cfg_yaml)])
    assert rc == 0
    out_dirs = list((tmp_path / "out").glob("*"))
    assert len(out_dirs) == 1
    assert (out_dirs[0] / "filtered.h5").exists()
    assert (out_dirs[0] / "metrics.json").exists()
