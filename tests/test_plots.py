"""Tests for martymicfly.eval.plots."""

import json
from pathlib import Path

import numpy as np

from martymicfly.eval.plots import plot_channel_html


def test_plot_channel_html_writes_valid_file(tmp_path: Path):
    fs = 16_000.0
    n = int(0.5 * fs)
    t = np.arange(n) / fs
    pre = np.sin(2 * np.pi * 100 * t)[:, None]
    post = pre * 0.1
    bpf = np.full((n, 1), 100.0)
    out = tmp_path / "ch00.html"
    plot_channel_html(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf,
        channel=0, outpath=out, n_harmonics=5,
        fmax_hz=1000.0, spectrogram_window=2048, spectrogram_overlap=1024,
    )
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "Plotly" in content
    # Trace data presence (Plotly embeds JSON):
    assert "100" in content  # the BPF marker should appear
