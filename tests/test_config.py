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
        "notch": {
            "pole_radius": {"mode": "scalar", "value": 0.9994},
            "multichannel": False,
            "block_size": 4096,
        },
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
    assert cfg.notch.pole_radius.mode == "scalar"
    assert cfg.notch.pole_radius.value == pytest.approx(0.9994)


def test_segment_explicit_requires_start_end():
    d = _minimal_config_dict()
    d["segment"] = {"mode": "explicit"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_requires_k_cover_and_margin():
    d = _minimal_config_dict()
    d["notch"]["pole_radius"] = {"mode": "linear"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_full():
    d = _minimal_config_dict()
    d["notch"]["pole_radius"] = {
        "mode": "linear",
        "k_cover": 1.5,
        "margin_hz": 5.0,
        "delta_bpf_hz": None,
        "r_min": 0.90,
        "r_max": 0.9995,
    }
    cfg = AppConfig(**d)
    assert cfg.notch.pole_radius.k_cover == pytest.approx(1.5)


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
