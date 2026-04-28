"""Tests for martymicfly.processing.sources."""

import numpy as np

from martymicfly.processing.sources import ArrayFreqSource, ArraySamplesGenerator


def test_samples_generator_traits():
    data = np.arange(2000, dtype=np.float64).reshape(1000, 2)
    src = ArraySamplesGenerator(data, sample_freq=4096.0)
    assert src.sample_freq == 4096.0
    assert src.num_samples == 1000
    assert src.num_channels == 2


def test_samples_generator_yields_full_blocks_then_remainder():
    data = np.arange(35, dtype=np.float64).reshape(35, 1)
    src = ArraySamplesGenerator(data, sample_freq=1.0)
    blocks = list(src.result(10))
    assert [b.shape for b in blocks] == [(10, 1), (10, 1), (10, 1), (5, 1)]
    np.testing.assert_array_equal(np.vstack(blocks), data)


def test_freq_source_yields_2d_blocks():
    matrix = np.arange(60, dtype=np.float64).reshape(20, 3)
    src = ArrayFreqSource(matrix)
    blocks = list(src.result(7))
    assert [b.shape for b in blocks] == [(7, 3), (7, 3), (6, 3)]
    np.testing.assert_array_equal(np.vstack(blocks), matrix)


def test_freq_source_accepts_1d_input():
    arr = np.arange(15, dtype=np.float64)
    src = ArrayFreqSource(arr)
    blocks = list(src.result(5))
    assert [b.shape for b in blocks] == [(5,), (5,), (5,)]
    np.testing.assert_array_equal(np.concatenate(blocks), arr)
