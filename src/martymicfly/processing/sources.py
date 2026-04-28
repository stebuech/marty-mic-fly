"""Acoular-compatible in-memory sources for notch filtering.

Mirrors the pattern of MockSamplesGenerator/MockFreqSource from
NotchFilter/validate/run_filter_comparison.py — but lives here so we can
build the pipeline without importing from NotchFilter's validate/ tree.
"""

from __future__ import annotations

import numpy as np
from acoular.base import SamplesGenerator
from traits.api import Array, HasTraits


class ArraySamplesGenerator(SamplesGenerator):
    """Wrap an ``(N, K)`` ndarray as a streaming SamplesGenerator."""

    _signal_data = Array(dtype=np.float64, shape=(None, None))

    def __init__(self, data: np.ndarray, sample_freq: float, **kwargs) -> None:
        arr = np.ascontiguousarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected 2D (N, K) array, got {arr.ndim}D")
        super().__init__(
            sample_freq=float(sample_freq),
            num_samples=int(arr.shape[0]),
            num_channels=int(arr.shape[1]),
            **kwargs,
        )
        self._signal_data = arr

    def result(self, num: int):
        n = self._signal_data.shape[0]
        for i in range(0, n, num):
            yield self._signal_data[i:i + num]


class ArrayFreqSource(HasTraits):
    """Wrap a 1D ``(N,)`` or 2D ``(N, M)`` array as a streaming freq source.

    ``result(num)`` yields row-wise slices, matching the contract expected by
    ``CascadeNotchFilter`` (mode='external') and the underlying
    ``AdaptiveNotchFilter._result_external``.
    """

    _data = Array(dtype=np.float64)

    def __init__(self, data: np.ndarray, **kwargs) -> None:
        arr = np.asarray(data, dtype=np.float64)
        super().__init__(**kwargs)
        self._data = arr

    def result(self, num: int):
        n = self._data.shape[0]
        pos = 0
        while pos < n:
            end = min(pos + num, n)
            yield self._data[pos:end]
            pos = end
