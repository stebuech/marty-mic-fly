"""Stage-2 plotting: 2x3 beam-map subplots + 1x1 target-PSD plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _add_overlays(fig, row, col, extent, rotor_pos, rotor_radii, mic_pos, target_xy):
    theta = np.linspace(0, 2 * np.pi, 64)
    for i in range(rotor_pos.shape[1]):
        cx, cy, _ = rotor_pos[:, i]
        rr = rotor_radii[i]
        fig.add_trace(go.Scatter(
            x=cx + rr * np.cos(theta), y=cy + rr * np.sin(theta),
            mode="lines", line=dict(color="cyan", width=1, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=mic_pos[:, 0], y=mic_pos[:, 1],
        mode="markers", marker=dict(color="white", size=4, opacity=0.6),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=[target_xy[0]], y=[target_xy[1]],
        mode="markers", marker=dict(color="red", size=10, symbol="x"),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)
    fig.update_xaxes(range=[-extent, extent], row=row, col=col, title_text="X [m]")
    fig.update_yaxes(range=[-extent, extent], row=row, col=col,
                     scaleanchor=f"x{col}", scaleratio=1)


def _to_xy_max(arr: np.ndarray) -> np.ndarray:
    """Return a 2D (nx, ny) max-projection of an (nx, ny) or (nx, ny, nz) map."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.max(axis=2)
    raise ValueError(f"beam map must be 2D or 3D, got ndim={arr.ndim}")


def plot_beam_maps(
    pre_maps: dict[str, np.ndarray],
    post_maps: dict[str, np.ndarray],
    extent_xy_m: float,
    rotor_positions: np.ndarray,
    rotor_radii: np.ndarray,
    mic_positions: np.ndarray,
    target_xy_m: tuple[float, float],
    out_path: str,
    db_dyn_range: float = 15.0,
) -> None:
    band_names = list(pre_maps.keys())

    # Detect dimensionality. If any band is 3D with nz > 1, label "max over z".
    sample = next(iter(pre_maps.values()))
    is_3d_multi = sample.ndim == 3 and sample.shape[2] > 1
    suffix = " (max over z)" if is_3d_multi else ""
    titles = (
        [f"pre — {n}{suffix}" for n in band_names]
        + [f"post — {n}{suffix}" for n in band_names]
    )
    fig = make_subplots(rows=2, cols=len(band_names), subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.10)

    proj0 = _to_xy_max(sample)
    nx, ny = proj0.shape
    gx = np.linspace(-extent_xy_m, +extent_xy_m, nx)
    gy = np.linspace(-extent_xy_m, +extent_xy_m, ny)

    for col, name in enumerate(band_names, start=1):
        for row, source in enumerate((pre_maps, post_maps), start=1):
            m2d = np.maximum(_to_xy_max(source[name]), 1e-30)
            peak = float(m2d.max())
            rel_db = 10.0 * np.log10(m2d / peak)
            fig.add_trace(go.Heatmap(
                x=gx, y=gy, z=rel_db.T,
                zmin=-db_dyn_range, zmax=0, colorscale="Inferno",
                showscale=(col == len(band_names) and row == 1),
                colorbar=dict(title="dB<br>(rel.<br>peak)", len=0.95, y=0.5, x=1.01),
            ), row=row, col=col)
            _add_overlays(fig, row, col, extent_xy_m, rotor_positions, rotor_radii,
                          mic_positions, target_xy_m)

    fig.update_layout(title="Stage-2 beam maps (CLEAN-SC, pre vs post)",
                      height=900, width=1500)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def _per_z_max(arr3d: np.ndarray, xy_mask_3d: np.ndarray | None = None) -> np.ndarray:
    """Per-z-slice maximum over xy, optionally restricted to xy-cells where
    xy_mask_3d is True. Returns shape (nz,). Empty mask cells -> 1e-30."""
    nx, ny, nz = arr3d.shape
    if xy_mask_3d is None:
        return arr3d.reshape(nx * ny, nz).max(axis=0)
    masked = np.where(xy_mask_3d, arr3d, -np.inf)
    out = masked.reshape(nx * ny, nz).max(axis=0)
    return np.where(np.isfinite(out), out, 1e-30)


def plot_z_marginal(
    pre_maps: dict[str, np.ndarray],
    post_maps: dict[str, np.ndarray],
    z_values: np.ndarray,
    bpfs: list[float],
    out_path: str,
    rotor_z_m: float = 0.0,
    target_z_m: float | None = None,
    preserved_mask_3d: np.ndarray | None = None,
) -> None:
    """Per-z marginal of CLEAN-SC source map.

    Two display modes:

    * If ``preserved_mask_3d`` is None: classic pre/post per band.

    * If ``preserved_mask_3d`` is given (shape (nx, ny, nz) bool, True =
      cell preserved, ~ = cell subtracted): per band, plot two lines:
        - **kept**: max over xy where preserved_mask_3d is True
        - **subtracted**: max over xy where preserved_mask_3d is False
      This makes the subtraction effect transparent at every z-slice. The
      z-range fully covered by preserved cells is shaded green; the range
      fully subtracted is shaded red.

    pre_maps / post_maps: each value (nx, ny, nz). z_values: (nz,).
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    nz = z_values.size

    fig = go.Figure()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    split_mode = preserved_mask_3d is not None
    for idx, name in enumerate(pre_maps.keys()):
        color = palette[idx % len(palette)]
        pre = pre_maps[name]
        post = post_maps[name]
        if pre.ndim == 2:
            pre = pre[..., None]
            post = post[..., None]

        if split_mode:
            preserved = preserved_mask_3d
            kept = np.maximum(_per_z_max(pre, preserved), 1e-30)
            subtracted = np.maximum(_per_z_max(pre, ~preserved), 1e-30)
            fig.add_trace(go.Scatter(
                x=z_values, y=10.0 * np.log10(kept),
                mode="lines+markers", name=f"kept — {name}",
                line=dict(color=color, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=z_values, y=10.0 * np.log10(subtracted),
                mode="lines+markers", name=f"subtracted — {name}",
                line=dict(color=color, dash="dot", width=1),
            ))
        else:
            pre_maxxy = np.maximum(_per_z_max(pre), 1e-30)
            post_maxxy = np.maximum(_per_z_max(post), 1e-30)
            fig.add_trace(go.Scatter(
                x=z_values, y=10.0 * np.log10(pre_maxxy),
                mode="lines+markers", name=f"pre — {name}",
                line=dict(color=color),
            ))
            fig.add_trace(go.Scatter(
                x=z_values, y=10.0 * np.log10(post_maxxy),
                mode="lines+markers", name=f"post — {name}",
                line=dict(color=color, dash="dot"),
            ))

    if nz > 1:
        # Shade preservation / subtraction zones along z.
        if split_mode:
            preserved_per_z = preserved_mask_3d.reshape(-1, nz).any(axis=0)
            subtracted_per_z = (~preserved_mask_3d).reshape(-1, nz).any(axis=0)
            fully_preserved = preserved_per_z & ~subtracted_per_z
            fully_subtracted = subtracted_per_z & ~preserved_per_z
            dz = float(z_values[1] - z_values[0]) if nz > 1 else 0.0
            for k in np.where(fully_preserved)[0]:
                fig.add_vrect(
                    x0=z_values[k] - dz / 2, x1=z_values[k] + dz / 2,
                    fillcolor="lightgreen", opacity=0.10,
                    layer="below", line_width=0,
                )
            for k in np.where(fully_subtracted)[0]:
                fig.add_vrect(
                    x0=z_values[k] - dz / 2, x1=z_values[k] + dz / 2,
                    fillcolor="lightcoral", opacity=0.10,
                    layer="below", line_width=0,
                )
        fig.add_vline(x=rotor_z_m, line=dict(color="cyan", dash="dash", width=1),
                      annotation_text="rotor z")
        if target_z_m is not None:
            fig.add_vline(x=target_z_m, line=dict(color="red", dash="dash", width=1),
                          annotation_text="target z")
    for f in bpfs:
        # bpfs aren't on the z-axis, but the parameter is in the API for future use
        # (e.g., overlaid frequency lines on a separate axis). Currently unused.
        _ = f

    fig.update_xaxes(title_text="z [m]")
    ylabel = (
        "10·log10( max over xy in subset ) [dB]"
        if split_mode else "10·log10( max over xy ) [dB]"
    )
    fig.update_yaxes(title_text=ylabel)
    suffix = " (kept vs subtracted)" if split_mode else " (pre vs post)"
    title = f"Stage-2 z-marginal (CLEAN-SC,{suffix})"
    if nz == 1:
        title += " — single z-slice (degenerate)"
    fig.update_layout(title=title, height=500, width=1000)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_target_psd(
    frequencies: np.ndarray,
    psd_pre: np.ndarray,
    psd_post: np.ndarray,
    gt_psd: np.ndarray | None,
    bpfs: list[float],
    out_path: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(psd_pre, 1e-30)),
                             mode="lines", name="pre"))
    fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(psd_post, 1e-30)),
                             mode="lines", name="post"))
    if gt_psd is not None:
        fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(gt_psd, 1e-30)),
                                 mode="lines", name="ground truth", line=dict(dash="dot")))
    for f in bpfs:
        fig.add_vline(x=f, line=dict(color="gray", dash="dot", width=1))
    fig.update_xaxes(title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="PSD [dB rel. unit²/Hz]")
    fig.update_layout(title="Pseudo-target PSD (pre/post)", height=500, width=1000)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
