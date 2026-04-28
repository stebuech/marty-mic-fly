"""Tests for martymicfly.config."""

import pytest
from pydantic import ValidationError

from martymicfly.config import AppConfig, NotchConfig, PoleRadiusConfig


def _minimal_config_dict():
    return {
        "input": {"audio_h5": "/tmp/x.h5", "mic_geom_xml": "/tmp/m.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 20},
        "stages": [
            {
                "kind": "notch",
                "pole_radius": {"mode": "scalar", "value": 0.9994},
                "multichannel": False,
                "block_size": 4096,
            }
        ],
        "metrics": {
            "welch_nperseg": 8192,
            "welch_noverlap": 4096,
            "bandwidth_factor": 1.0,
            "broadband_low_hz": None,
        },
        "plots": {
            "enabled": True,
            "fmax_hz": None,
            "spectrogram_window": 4096,
            "spectrogram_overlap": 2048,
            "channel_subset": None,
        },
        "output": {
            "dir": "results/notch/{run_id}",
            "filtered_h5": "filtered.h5",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "plots_subdir": "plots",
            "copy_config": True,
        },
    }


def test_minimal_config_validates():
    cfg = AppConfig(**_minimal_config_dict())
    assert cfg.rotor.n_blades == 2
    assert cfg.stages[0].kind == "notch"
    assert cfg.stages[0].pole_radius.mode == "scalar"
    assert cfg.stages[0].pole_radius.value == pytest.approx(0.9994)


def test_segment_explicit_requires_start_end():
    d = _minimal_config_dict()
    d["segment"] = {"mode": "explicit"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_requires_k_cover_and_margin():
    d = _minimal_config_dict()
    d["stages"][0]["pole_radius"] = {"mode": "linear"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_full():
    d = _minimal_config_dict()
    d["stages"][0]["pole_radius"] = {
        "mode": "linear",
        "k_cover": 1.5,
        "margin_hz": 5.0,
        "delta_bpf_hz": None,
        "r_min": 0.90,
        "r_max": 0.9995,
    }
    cfg = AppConfig(**d)
    assert cfg.stages[0].pole_radius.k_cover == pytest.approx(1.5)


def test_channels_list_requires_list_field():
    d = _minimal_config_dict()
    d["channels"] = {"selection": "list"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_config_hash_is_deterministic():
    d = _minimal_config_dict()
    cfg1 = AppConfig(**d)
    cfg2 = AppConfig(**d)
    assert cfg1.config_hash() == cfg2.config_hash()
    assert len(cfg1.config_hash()) == 8


def test_app_config_with_notch_stage():
    from martymicfly.config import AppConfig
    payload = {
        "input": {"audio_h5": "/x.h5", "mic_geom_xml": "/x.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 13},
        "stages": [
            {
                "kind": "notch",
                "pole_radius": {"mode": "scalar", "value": 0.9994},
                "multichannel": False,
                "block_size": 4096,
            }
        ],
        "metrics": {"welch_nperseg": 4096, "welch_noverlap": 2048},
        "plots": {"enabled": True, "spectrogram_window": 4096, "spectrogram_overlap": 2048},
        "output": {"dir": "results/notch/{run_id}"},
    }
    cfg = AppConfig.model_validate(payload)
    assert cfg.stages[0].kind == "notch"
    assert cfg.stages[0].pole_radius.value == 0.9994


def test_app_config_rejects_old_top_level_notch():
    import pytest
    from pydantic import ValidationError
    from martymicfly.config import AppConfig
    payload = {
        "input": {"audio_h5": "/x.h5", "mic_geom_xml": "/x.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 13},
        "stages": [],
        "notch": {"pole_radius": {"mode": "scalar", "value": 0.9994}},
        "metrics": {"welch_nperseg": 4096, "welch_noverlap": 2048},
        "plots": {"enabled": True, "spectrogram_window": 4096, "spectrogram_overlap": 2048},
        "output": {"dir": "results/notch/{run_id}"},
    }
    with pytest.raises(ValidationError) as exc:
        AppConfig.model_validate(payload)
    assert "notch" in str(exc.value).lower()


def test_array_filter_stage_config_full_payload():
    from martymicfly.config import ArrayFilterStageConfig
    payload = {
        "kind": "array_filter",
        "algorithm": "clean_sc",
        "csm": {"nperseg": 512, "noverlap": 256, "window": "hann",
                "diag_loading_rel": 1e-6, "f_min_hz": 200.0, "f_max_hz": 6000.0},
        "diagnostic_grid": {"extent_xy_m": 0.5, "increment_m": 0.02, "z_m": None},
        "bands": [
            {"name": "low", "f_min_hz": 200.0, "f_max_hz": 500.0},
            {"name": "mid", "f_min_hz": 500.0, "f_max_hz": 2000.0},
        ],
        "target_point_m": [0.0, 0.0, -1.5],
        "rotor_z_tolerance_m": 0.05,
        "clean_sc": {"damp": 0.6, "n_iter": 100},
    }
    cfg = ArrayFilterStageConfig.model_validate(payload)
    assert cfg.kind == "array_filter"
    assert cfg.algorithm == "clean_sc"
    assert cfg.target_point_m == (0.0, 0.0, -1.5)
    assert cfg.bands[0].name == "low"


def test_array_filter_stage_config_unknown_algorithm_rejected():
    import pytest
    from pydantic import ValidationError
    from martymicfly.config import ArrayFilterStageConfig
    with pytest.raises(ValidationError):
        ArrayFilterStageConfig.model_validate({"kind": "array_filter", "algorithm": "nope"})
