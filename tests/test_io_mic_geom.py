"""Tests for martymicfly.io.mic_geom."""

from pathlib import Path

import numpy as np
import pytest

from martymicfly.io.mic_geom import load_mic_geom_xml

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_geom.xml"


def test_load_returns_correct_shape_and_order():
    pos = load_mic_geom_xml(FIXTURE)
    assert pos.shape == (4, 3)
    assert pos.dtype == np.float64
    np.testing.assert_allclose(pos[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(pos[1], [0.0, 0.1, 0.0])
    np.testing.assert_allclose(pos[2], [-0.1, 0.0, 0.0])
    np.testing.assert_allclose(pos[3], [0.0, -0.1, 0.0])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mic_geom_xml(tmp_path / "nope.xml")


def test_load_malformed_xml_raises(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<MicArray><pos x='1' y='2'/></MicArray>")  # missing z
    with pytest.raises(ValueError, match="z"):
        load_mic_geom_xml(bad)
