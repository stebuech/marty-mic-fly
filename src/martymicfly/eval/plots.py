"""Plotly HTML output: per-channel spectrum + spectrogram before/after."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

_MOTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_channel_html(
    *,
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    per_motor_bpf: np.ndarray,
    channel: int,
    outpath: Path,
    n_harmonics: int,
    fmax_hz: float,
    spectrogram_window: int,
    spectrogram_overlap: int,
) -> None:
    """Write a standalone HTML with two stacked subplots for one channel."""
    pre_1d = np.asarray(pre[:, channel], dtype=np.float64)
    post_1d = np.asarray(post[:, channel], dtype=np.float64)
    S = per_motor_bpf.shape[1]

    f_pre, psd_pre = welch(pre_1d, fs=fs, window="hann",
                           nperseg=min(8192, pre_1d.size),
                           noverlap=min(4096, pre_1d.size // 2),
                           scaling="density")
    f_post, psd_post = welch(post_1d, fs=fs, window="hann",
                             nperseg=min(8192, post_1d.size),
                             noverlap=min(4096, post_1d.size // 2),
                             scaling="density")
    fmask = f_pre <= fmax_hz

    f_spec, t_spec, sxx = spectrogram(
        post_1d, fs=fs, window="hann",
        nperseg=spectrogram_window, noverlap=spectrogram_overlap,
        scaling="density",
    )
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-20))
    spec_mask = f_spec <= fmax_hz

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.10,
                        subplot_titles=("Welch PSD (pre vs post)",
                                        "Spectrogram (post)"))

    fig.add_trace(go.Scatter(
        x=f_pre[fmask], y=10.0 * np.log10(np.maximum(psd_pre[fmask], 1e-20)),
        name="pre", line=dict(color="#444", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f_post[fmask], y=10.0 * np.log10(np.maximum(psd_post[fmask], 1e-20)),
        name="post", line=dict(color="#d62728", width=1.5),
    ), row=1, col=1)

    for s in range(S):
        color = _MOTOR_COLORS[s % len(_MOTOR_COLORS)]
        bpf_mean = float(per_motor_bpf[:, s].mean())
        for h in range(1, n_harmonics + 1):
            f_h = bpf_mean * h
            if f_h > fmax_hz:
                break
            fig.add_vline(x=f_h, line=dict(color=color, dash="dot", width=1),
                          row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=t_spec, y=f_spec[spec_mask], z=sxx_db[spec_mask, :],
        colorscale="Viridis", colorbar=dict(title="dB", len=0.45, y=0.22),
        showscale=True,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="PSD [dB / Hz, ref=1]", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_layout(
        title=f"Channel {channel:02d} — pre/post notch",
        height=900, showlegend=True,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath, include_plotlyjs="cdn", full_html=True)
