"""Heatmap: Achsen × Metriken pro Methode, Farbe = sensitivity_db."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp


def write_heatmap(sens_df: pd.DataFrame, *, out_path: Path, band: str) -> None:
    band_df = sens_df[sens_df["band"] == band]
    methods = sorted(band_df["method"].unique())
    fig = sp.make_subplots(
        rows=len(methods), cols=1,
        subplot_titles=[f"{m} — band={band}" for m in methods],
        vertical_spacing=0.05,
    )
    for i, method in enumerate(methods, start=1):
        m_df = band_df[band_df["method"] == method]
        pivot = m_df.pivot_table(
            index="axis", columns="metric", values="sensitivity_db",
            aggfunc="first",
        )
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="Viridis", colorbar={"title": "Δ [dB]"},
            zmin=0.0, showscale=(i == 1),
        ), row=i, col=1)
    fig.update_layout(
        title=f"Phase-1 Sensitivity Heatmap (band={band})",
        height=300 * len(methods) + 100,
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
