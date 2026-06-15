"""Tests for martymicfly.io.write_filtered."""

from pathlib import Path

import h5py
import numpy as np

from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.io.write_filtered import write_filtered


def test_roundtrip_preserves_telemetry_and_writes_attrs(tmp_path: Path):
    src = Path(__file__).parent / "fixtures" / "tiny_synth.h5"
    src_data = load_synth_h5(src)

    filtered = src_data["time_data"] * 0.5  # bogus filter
    out_path = tmp_path / "filtered.h5"
    write_filtered(
        out_path=out_path,
        filtered_time_data=filtered,
        sample_rate=src_data["sample_rate"],
        rpm_per_esc=src_data["rpm_per_esc"],
        attrs={
            "martymicfly_version": "0.1.0",
            "input_h5": str(src),
            "config_hash": "deadbeef",
            "segment_start_s": 0.0,
            "segment_duration_s": 1.0,
            "n_blades": 2,
            "n_harmonics": 5,
            "notch_mode": "rpm_external_zerophase",
            "pole_radius_repr": "scalar:0.998",
        },
    )

    out_data = load_synth_h5(out_path)
    np.testing.assert_allclose(out_data["time_data"], filtered)
    assert set(out_data["rpm_per_esc"].keys()) == set(src_data["rpm_per_esc"].keys())
    np.testing.assert_array_equal(
        out_data["rpm_per_esc"]["ESC1"]["rpm"],
        src_data["rpm_per_esc"]["ESC1"]["rpm"],
    )

    with h5py.File(out_path, "r") as f:
        v = f.attrs["config_hash"]
        v_str = v.decode() if isinstance(v, bytes) else v
        assert v_str == "deadbeef"
        v = f.attrs["notch_mode"]
        v_str = v.decode() if isinstance(v, bytes) else v
        assert v_str == "rpm_external_zerophase"
