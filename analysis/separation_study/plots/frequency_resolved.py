"""Frequency-resolved Plot: psd_post + ext_GT + D_ref + excess/deficit-Subplot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from analysis.separation_study.freq_resolved_io import read_freq_resolved


def write_frequency_resolved(
    h5_path: Path, *, out_path: Path, bands: list[dict],
) -> None:
    payload = read_freq_resolved(h5_path)
    f = payload["frequencies"]
    fig = sp.make_subplots(
        rows=2, cols=1, vertical_spacing=0.10,
        subplot_titles=("PSD am Target [dB re 1 Pa²/Hz]",
                       "Per-Bin excess (rot) / deficit (blau) [linear power]"),
    )
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
    for label, color, key in [
        ("psd_post", "#1f77b4", "psd_post"),
        ("ext_gt",   "#2ca02c", "ext_gt"),
        ("d_ref",    "#d62728", "d_ref"),
    ]:
        fig.add_trace(go.Scatter(
            x=f, y=db(payload[key]), mode="lines", name=label,
            line={"color": color, "width": 2},
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=f, y=payload["excess"], marker_color="#d62728",
        name="excess", opacity=0.6,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=f, y=payload["deficit"], marker_color="#1f77b4",
        name="deficit", opacity=0.6,
    ), row=2, col=1)
    for b in bands:
        for r in (1, 2):
            fig.add_vline(x=b["f_min_hz"],
                          line={"color": "#888", "width": 1, "dash": "dot"},
                          row=r, col=1)
            fig.add_vline(x=b["f_max_hz"],
                          line={"color": "#888", "width": 1, "dash": "dot"},
                          row=r, col=1)
    fig.update_layout(template="plotly_white", height=800,
                      title=str(h5_path.parent.name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
