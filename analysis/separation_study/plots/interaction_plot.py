"""Wechselwirkungs-Heatmap: 2D-Grid von Knob_a × Knob_b mit Metrik als Farbe."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def write_interaction(
    df: pd.DataFrame, *, axis_a: str, axis_b: str, metric: str, out_path: Path,
) -> None:
    pivot = df.pivot_table(
        index="axis_a_value", columns="axis_b_value", values=metric, aggfunc="first",
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdBu_r", colorbar={"title": metric},
    ))
    fig.update_layout(
        title=f"{metric}: {axis_a} × {axis_b}",
        xaxis_title=axis_b, yaxis_title=axis_a,
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
