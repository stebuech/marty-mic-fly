"""Pareto-Scatter: over_subtraction_db × drone_leakage_db_def2,
Punktfarbe = Knob-Wert."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def write_pareto(
    df: pd.DataFrame, *,
    axis_label: str,
    out_path: Path,
    welch_floor_db: float,
    band: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["over_subtraction_db"], y=df["drone_leakage_db_def2"],
        mode="markers+text",
        marker={"size": 12, "color": df["axis_value"], "colorscale": "Viridis",
                "showscale": True, "colorbar": {"title": axis_label}},
        text=[f"{v:g}" for v in df["axis_value"]], textposition="top center",
    ))
    fig.add_shape(type="rect",
                  x0=-100, x1=welch_floor_db, y0=-100, y1=welch_floor_db,
                  fillcolor="rgba(200,200,200,0.3)", line={"width": 0})
    fig.update_layout(
        title=f"Pareto — {axis_label} (band={band})",
        xaxis_title="over_subtraction_db (gepeelt)",
        yaxis_title="drone_leakage_db_def2 (Drohnen-Leck)",
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
