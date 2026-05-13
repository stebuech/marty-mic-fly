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


def plot_doa_heatmaps(
    pre_maps: dict[str, np.ndarray],          # band -> (n_az, n_el, 1)
    post_maps: dict[str, np.ndarray],         # band -> (n_az, n_el, 1)
    n_az: int,
    n_el: int,
    hemisphere: str,                          # "lower" | "upper" | "full"
    rotor_positions: np.ndarray | None,       # (3, R) — for DOA markers
    target_xyz_m: tuple[float, float, float], # target source point — for DOA marker
    out_path: str,
    db_dyn_range: float = 20.0,
) -> None:
    """Per-band azimuth/elevation heatmaps for the DOA hemisphere stage.

    The grid layout follows ``build_doa_grid``: meshgrid(indexing='ij') with
    ravel index = i_az * n_el + i_el. Pre/post sit side by side per band.
    Rotor and target directions are overlaid as markers (DOAs from the
    array center).
    """
    band_names = list(pre_maps.keys())
    if hemisphere == "lower":
        el_min, el_max = -90.0, 0.0
    elif hemisphere == "upper":
        el_min, el_max = 0.0, 90.0
    else:
        el_min, el_max = -90.0, 90.0
    az_axis = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el_axis = np.linspace(el_min, el_max, n_el)

    def _direction_to_az_el(xyz: np.ndarray) -> tuple[float, float] | None:
        v = np.asarray(xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return None
        v = v / norm
        el = float(np.degrees(np.arcsin(v[2])))
        az = float(np.degrees(np.arctan2(v[1], v[0])))
        if az < 0.0:
            az += 360.0
        return az, el

    rotor_doas: list[tuple[float, float]] = []
    if rotor_positions is not None:
        for i in range(rotor_positions.shape[1]):
            d = _direction_to_az_el(rotor_positions[:, i])
            if d is not None:
                rotor_doas.append(d)
    target_doa = _direction_to_az_el(np.asarray(target_xyz_m))

    titles = (
        [f"pre — {n}" for n in band_names]
        + [f"post — {n}" for n in band_names]
    )
    fig = make_subplots(rows=2, cols=len(band_names), subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.10)

    def _to_2d(arr: np.ndarray) -> np.ndarray:
        # incoming shape is (n_az, n_el, 1) per the DOA reshape_hint convention
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], arr.shape[1])
        return arr

    for col, name in enumerate(band_names, start=1):
        for row, source in enumerate((pre_maps, post_maps), start=1):
            m2d = np.maximum(_to_2d(source[name]), 1e-30)
            peak = float(m2d.max())
            rel_db = 10.0 * np.log10(m2d / peak)
            fig.add_trace(go.Heatmap(
                x=az_axis, y=el_axis, z=rel_db.T,   # transpose so y=el rows
                zmin=-db_dyn_range, zmax=0.0, colorscale="Inferno",
                showscale=(col == len(band_names) and row == 1),
                colorbar=dict(title="dB<br>(rel.<br>peak)", len=0.95, y=0.5, x=1.01),
                hovertemplate="az=%{x:.0f}°<br>el=%{y:.0f}°<br>rel=%{z:.1f} dB<extra></extra>",
            ), row=row, col=col)
            if rotor_doas:
                fig.add_trace(go.Scatter(
                    x=[d[0] for d in rotor_doas], y=[d[1] for d in rotor_doas],
                    mode="markers",
                    marker=dict(color="cyan", size=10, symbol="circle-open",
                                line=dict(width=2)),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)
            if target_doa is not None:
                fig.add_trace(go.Scatter(
                    x=[target_doa[0]], y=[target_doa[1]],
                    mode="markers", marker=dict(color="red", size=12, symbol="x"),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)
            fig.update_xaxes(range=[0, 360], title_text="Azimuth [°]",
                             row=row, col=col)
            fig.update_yaxes(range=[el_min, el_max], title_text="Elevation [°]",
                             row=row, col=col)

    fig.update_layout(
        title=f"Stage-2 DOA heatmaps ({hemisphere} hemisphere, CLEAN-SC, pre vs post)",
        height=900, width=1500,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_doa_polar_skymap(
    pre_maps: dict[str, np.ndarray],          # band -> (n_az, n_el, 1)
    n_az: int,
    n_el: int,
    hemisphere: str,
    rotor_positions: np.ndarray | None,
    target_xyz_m: tuple[float, float, float],
    out_path: str,
    db_dyn_range: float = 20.0,
) -> None:
    """Polar skymap projection: looking 'down' onto the lower hemisphere
    (or 'up' onto the upper one). Radius = cos(|elevation|), so the nadir
    or zenith sits at the center; the equator at the rim.

    One panel per band (pre only — post tends to look similar after
    subtraction and is busier than informative in skymap form).
    """
    if hemisphere == "lower":
        el_min, el_max = -90.0, 0.0
    elif hemisphere == "upper":
        el_min, el_max = 0.0, 90.0
    else:
        el_min, el_max = -90.0, 90.0
    az_axis = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el_axis = np.linspace(el_min, el_max, n_el)

    band_names = list(pre_maps.keys())
    fig = make_subplots(
        rows=1, cols=len(band_names),
        specs=[[{"type": "polar"}] * len(band_names)],
        subplot_titles=[f"pre — {n}" for n in band_names],
        horizontal_spacing=0.10,
    )

    # Build per-cell az (degrees) and r (= |sin(el)| → 0 at pole, 1 at equator
    # for lower; for full, 1 at the equator and 0 at both poles).
    AZ, EL = np.meshgrid(az_axis, el_axis, indexing="ij")
    r_grid = np.cos(np.deg2rad(EL))   # 0 at pole, 1 at equator

    def _map_dir_to_polar(xyz: np.ndarray) -> tuple[float, float] | None:
        v = np.asarray(xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return None
        v = v / norm
        el = float(np.degrees(np.arcsin(v[2])))
        az = float(np.degrees(np.arctan2(v[1], v[0])))
        if az < 0.0:
            az += 360.0
        if hemisphere == "lower" and el > 0:
            return None
        if hemisphere == "upper" and el < 0:
            return None
        return az, float(np.cos(np.deg2rad(el)))

    rotor_polar: list[tuple[float, float]] = []
    if rotor_positions is not None:
        for i in range(rotor_positions.shape[1]):
            p = _map_dir_to_polar(rotor_positions[:, i])
            if p is not None:
                rotor_polar.append(p)
    target_polar = _map_dir_to_polar(np.asarray(target_xyz_m))

    for col, name in enumerate(band_names, start=1):
        m = pre_maps[name]
        if m.ndim == 3:
            m = m.reshape(m.shape[0], m.shape[1])
        m = np.maximum(m, 1e-30)
        peak = float(m.max())
        rel_db = 10.0 * np.log10(m / peak)
        rel_db = np.maximum(rel_db, -db_dyn_range)
        # Render as scattered markers — plotly polar heatmap support is
        # limited; markers sized & colored by power give a passable skymap
        # without bringing in shapely or matplotlib.
        fig.add_trace(go.Scatterpolar(
            r=r_grid.ravel(), theta=AZ.ravel(),
            mode="markers",
            marker=dict(
                color=rel_db.ravel(), colorscale="Inferno",
                cmin=-db_dyn_range, cmax=0.0,
                size=8, opacity=0.8,
                showscale=(col == len(band_names)),
                colorbar=dict(title="dB<br>(rel.<br>peak)", len=0.85, y=0.5,
                              x=1.02) if col == len(band_names) else None,
            ),
            hovertemplate=("az=%{theta:.0f}°<br>r=%{r:.2f} (cos|el|)"
                           "<br>rel=%{marker.color:.1f} dB<extra></extra>"),
            showlegend=False,
        ), row=1, col=col)
        if rotor_polar:
            fig.add_trace(go.Scatterpolar(
                r=[p[1] for p in rotor_polar], theta=[p[0] for p in rotor_polar],
                mode="markers",
                marker=dict(color="cyan", size=14, symbol="circle-open",
                            line=dict(width=2)),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col)
        if target_polar is not None:
            fig.add_trace(go.Scatterpolar(
                r=[target_polar[1]], theta=[target_polar[0]],
                mode="markers", marker=dict(color="red", size=16, symbol="x"),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col)

    fig.update_layout(
        title=f"Stage-2 DOA skymap ({hemisphere}; "
              "center=pole, rim=equator; cyan=rotor DOA, red=target DOA)",
        height=600, width=1500,
    )
    fig.update_polars(
        radialaxis=dict(range=[0, 1.05], showticklabels=False),
        angularaxis=dict(direction="counterclockwise", rotation=0),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def _cone_surface(
    direction_xyz: np.ndarray,
    half_angle_deg: float,
    height: float,
    color: str, opacity: float,
    name: str,
    n_r: int = 12, n_theta: int = 36,
) -> go.Surface:
    """Cone from origin along ``direction_xyz`` with given half-angle, drawn
    as a parametric surface up to ``height``. The cone's axis vector is
    normalized internally; an orthonormal frame is built from it."""
    d = np.asarray(direction_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError("direction_xyz must be non-zero")
    d = d / norm
    # Build orthonormal frame (d, u, v)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    alpha = np.deg2rad(half_angle_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    # Parametrization: r ∈ [0, height/cos_a], theta ∈ [0, 2π]
    # Point on the cone at axial distance s = r*cos_a, ring radius r*sin_a:
    rs = np.linspace(0.0, height / max(cos_a, 1e-9), n_r)
    ths = np.linspace(0.0, 2.0 * np.pi, n_theta)
    R, TH = np.meshgrid(rs, ths, indexing="ij")
    axis = R * cos_a   # (n_r, n_theta)
    radial = R * sin_a # (n_r, n_theta)
    cos_t = np.cos(TH); sin_t = np.sin(TH)
    X = axis * d[0] + radial * (cos_t * u[0] + sin_t * v[0])
    Y = axis * d[1] + radial * (cos_t * u[1] + sin_t * v[1])
    Z = axis * d[2] + radial * (cos_t * u[2] + sin_t * v[2])
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=np.zeros_like(X),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=opacity, name=name,
        showlegend=True, legendgroup=name,
        hovertemplate=f"{name}<extra></extra>",
    )


def _disk_band_surface(
    radius: float, half_width_deg: float, color: str, opacity: float,
    name: str, n_az: int = 72, n_el: int = 11,
) -> go.Surface:
    """Equatorial band on the focal sphere between elevation ±half_width_deg
    — visual analogue of build_doa_disk_mask."""
    el = np.linspace(-half_width_deg, half_width_deg, n_el)
    az = np.linspace(0.0, 360.0, n_az)
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    az_r = np.deg2rad(AZ); el_r = np.deg2rad(EL)
    X = radius * np.cos(el_r) * np.cos(az_r)
    Y = radius * np.cos(el_r) * np.sin(az_r)
    Z = radius * np.sin(el_r)
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=np.zeros_like(X),
        colorscale=[[0, color], [1, color]],
        showscale=False, opacity=opacity, name=name,
        showlegend=True, legendgroup=name,
        hovertemplate=f"{name}<extra></extra>",
    )


def _hemisphere_surface(
    radius: float, hemisphere: str, color: str, opacity: float,
    name: str, n_az: int = 72, n_el: int = 19,
) -> go.Surface:
    """Wireframe-ish hemisphere as a faint reference for the DOA scene."""
    if hemisphere == "lower":
        el_min, el_max = -90.0, 0.0
    elif hemisphere == "upper":
        el_min, el_max = 0.0, 90.0
    else:
        el_min, el_max = -90.0, 90.0
    az = np.linspace(0.0, 360.0, n_az)
    el = np.linspace(el_min, el_max, n_el)
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    az_r = np.deg2rad(AZ); el_r = np.deg2rad(EL)
    X = radius * np.cos(el_r) * np.cos(az_r)
    Y = radius * np.cos(el_r) * np.sin(az_r)
    Z = radius * np.sin(el_r)
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=np.zeros_like(X),
        colorscale=[[0, color], [1, color]],
        showscale=False, opacity=opacity, name=name,
        showlegend=True, legendgroup=name,
        hovertemplate=f"{name}<extra></extra>",
    )


def build_mask_geometry_3d_fig(
    *,
    mic_positions: np.ndarray,                              # (M, 3)
    rotor_positions: np.ndarray,                            # (3, R) platform convention
    rotor_radii: np.ndarray,                                # (R,) — accepted for API stability, unused
    target_point_m: tuple[float, float, float],
    doa_focal_radius_m: float,
    doa_hemisphere: str,
    rotor_cone_half_angle_deg: float,
    target_cone_half_angle_deg: float,
    drone_disk_half_width_deg: float,
) -> go.Figure:
    """3D visualization of the DOA cone mask geometry.

    Single 3D scene: hemisphere reference + rotor / drone / target cones,
    overlaid on the array fixtures (mic positions, rotor positions,
    target point).
    """
    del rotor_radii  # kept in signature for caller compatibility
    fig = go.Figure()

    R = rotor_positions.shape[1]
    rotor_xs = rotor_positions[0]
    rotor_ys = rotor_positions[1]
    rotor_zs = rotor_positions[2]
    target = np.asarray(target_point_m, dtype=np.float64)

    fig.add_trace(go.Scatter3d(
        x=mic_positions[:, 0], y=mic_positions[:, 1], z=mic_positions[:, 2],
        mode="markers", marker=dict(color="black", size=3), name="mics",
    ))
    fig.add_trace(go.Scatter3d(
        x=rotor_xs, y=rotor_ys, z=rotor_zs,
        mode="markers", marker=dict(color="cyan", size=6, symbol="diamond"),
        name="rotor centers",
    ))
    fig.add_trace(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]],
        mode="markers", marker=dict(color="red", size=8, symbol="x"),
        name="target",
    ))

    fig.add_trace(_hemisphere_surface(
        radius=doa_focal_radius_m, hemisphere=doa_hemisphere,
        color="#1f77b4", opacity=0.10,
        name=f"focal sphere ({doa_hemisphere}, r={doa_focal_radius_m:.2f}m)",
    ))
    # rotor_cone axes — inter-rotor midpoints — which is what the
    # DoaArrayFilterStage actually computes with.
    from martymicfly.processing.beamform_grid import inter_rotor_midpoints_3xR
    cone_axes = inter_rotor_midpoints_3xR(rotor_positions)
    M = cone_axes.shape[1]
    fig.add_trace(go.Scatter3d(
        x=cone_axes[0], y=cone_axes[1], z=cone_axes[2],
        mode="markers", marker=dict(color="magenta", size=5, symbol="cross"),
        name="inter-rotor midpoints (cone axes)",
    ))
    for i in range(M):
        d = cone_axes[:, i]
        if float(np.linalg.norm(d)) < 1e-9:
            continue
        fig.add_trace(_cone_surface(
            direction_xyz=d,
            half_angle_deg=rotor_cone_half_angle_deg,
            height=doa_focal_radius_m,
            color="#ff7f0e", opacity=0.30,
            name=f"rotor_cone[{i}]" if M > 1 else "rotor_cone",
        ))
    # drone_disk is the equatorial belt around the rotor plane — visualize
    # both the disk band (kept) and the two ±z exclusion cones (rejected).
    cone_half = max(0.0, 90.0 - float(drone_disk_half_width_deg))
    for sign, label in ((1.0, "+z"), (-1.0, "-z")):
        fig.add_trace(_cone_surface(
            direction_xyz=np.array([0.0, 0.0, sign]),
            half_angle_deg=cone_half,
            height=doa_focal_radius_m,
            color="#7f7f7f", opacity=0.10,
            name=f"drone_disk excl ({label})",
        ))
    fig.add_trace(_disk_band_surface(
        radius=doa_focal_radius_m,
        half_width_deg=float(drone_disk_half_width_deg),
        color="#d62728", opacity=0.25,
        name="drone_disk",
    ))
    target_dir = target if float(np.linalg.norm(target)) > 1e-9 else np.array([0.0, 0.0, -1.0])
    fig.add_trace(_cone_surface(
        direction_xyz=target_dir,
        half_angle_deg=target_cone_half_angle_deg,
        height=doa_focal_radius_m,
        color="#2ca02c", opacity=0.30,
        name="target_cone",
    ))

    extent = max(
        float(doa_focal_radius_m) * 1.05,
        float(np.max(np.abs(rotor_positions))) * 1.5,
        max(abs(target[i]) for i in range(3)) * 1.05,
    )
    fig.update_layout(
        title="Stage-2 DOA mask geometry",
        height=800, width=900,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x [m]", range=[-extent, extent]),
            yaxis=dict(title="y [m]", range=[-extent, extent]),
            zaxis=dict(title="z [m]", range=[-extent, extent]),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        legend=dict(orientation="v", x=1.02, y=1.0),
    )
    return fig


def plot_mask_geometry_3d(out_path: str, **kwargs) -> None:
    """Build the mask-geometry 3D figure and write it as standalone HTML.

    All keyword arguments are forwarded to ``build_mask_geometry_3d_fig``.
    """
    fig = build_mask_geometry_3d_fig(**kwargs)
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


def plot_target_box_slices(
    pre_maps: dict[str, np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    box_center_m: tuple[float, float, float],
    box_half_extent_m: tuple[float, float, float],
    out_path: str,
    rotor_positions: np.ndarray | None = None,
    db_dyn_range: float = 20.0,
) -> None:
    """Three orthogonal slices through the target-box center, per band.

    Layout: rows = bands, cols = (xy, xz, yz). Each slice is clipped to the
    box extent so only the preserved region is shown. Useful for inspecting
    where CLEAN-SC actually places the external-target energy and how it
    distributes inside the preservation volume.

    pre_maps: dict[band_name → (nx, ny, nz) ndarray].
    grid_x, grid_y, grid_z: 1-D coordinate axes used to build the 3D grid.
    box_center_m / box_half_extent_m: target_box geometry.
    rotor_positions: optional (3, R) for overlay on the xy slice.
    """
    band_names = list(pre_maps.keys())
    n_bands = len(band_names)
    cx, cy, cz = box_center_m
    hx, hy, hz = box_half_extent_m

    ix = int(np.argmin(np.abs(grid_x - cx)))
    iy = int(np.argmin(np.abs(grid_y - cy)))
    iz = int(np.argmin(np.abs(grid_z - cz)))

    x_in = (grid_x >= cx - hx) & (grid_x <= cx + hx)
    y_in = (grid_y >= cy - hy) & (grid_y <= cy + hy)
    z_in = (grid_z >= cz - hz) & (grid_z <= cz + hz)

    titles = []
    for n in band_names:
        titles.extend([
            f"{n} — xy @ z={grid_z[iz]:.2f}",
            f"{n} — xz @ y={grid_y[iy]:.2f}",
            f"{n} — yz @ x={grid_x[ix]:.2f}",
        ])
    fig = make_subplots(rows=n_bands, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    for ri, name in enumerate(band_names, start=1):
        m3d = np.maximum(pre_maps[name], 1e-30)
        peak = float(m3d.max())

        # xy slice
        xy = m3d[np.ix_(x_in, y_in, [iz])][..., 0]
        rel_db = 10.0 * np.log10(xy / peak)
        fig.add_trace(go.Heatmap(
            x=grid_x[x_in], y=grid_y[y_in], z=rel_db.T,
            zmin=-db_dyn_range, zmax=0, colorscale="Inferno",
            showscale=(ri == 1), colorbar=dict(
                title="dB<br>(rel.<br>peak)", len=0.95, y=0.5, x=1.02),
        ), row=ri, col=1)
        fig.update_xaxes(title_text="x [m]", row=ri, col=1)
        fig.update_yaxes(title_text="y [m]", row=ri, col=1,
                         scaleanchor=f"x{(ri-1)*3+1}", scaleratio=1)
        if rotor_positions is not None:
            in_z = abs(grid_z[iz] - rotor_positions[2, :]) < 0.01
            if in_z.any():
                fig.add_trace(go.Scatter(
                    x=rotor_positions[0, in_z], y=rotor_positions[1, in_z],
                    mode="markers", marker=dict(color="cyan", size=8, symbol="circle-open"),
                    hoverinfo="skip", showlegend=False,
                ), row=ri, col=1)
        # target marker on xy
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            hoverinfo="skip", showlegend=False,
        ), row=ri, col=1)

        # xz slice
        xz = m3d[np.ix_(x_in, [iy], z_in)][:, 0, :]
        rel_db = 10.0 * np.log10(xz / peak)
        fig.add_trace(go.Heatmap(
            x=grid_x[x_in], y=grid_z[z_in], z=rel_db.T,
            zmin=-db_dyn_range, zmax=0, colorscale="Inferno",
            showscale=False,
        ), row=ri, col=2)
        fig.update_xaxes(title_text="x [m]", row=ri, col=2)
        fig.update_yaxes(title_text="z [m]", row=ri, col=2)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cz], mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            hoverinfo="skip", showlegend=False,
        ), row=ri, col=2)

        # yz slice
        yz = m3d[np.ix_([ix], y_in, z_in)][0, :, :]
        rel_db = 10.0 * np.log10(yz / peak)
        fig.add_trace(go.Heatmap(
            x=grid_y[y_in], y=grid_z[z_in], z=rel_db.T,
            zmin=-db_dyn_range, zmax=0, colorscale="Inferno",
            showscale=False,
        ), row=ri, col=3)
        fig.update_xaxes(title_text="y [m]", row=ri, col=3)
        fig.update_yaxes(title_text="z [m]", row=ri, col=3)
        fig.add_trace(go.Scatter(
            x=[cy], y=[cz], mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            hoverinfo="skip", showlegend=False,
        ), row=ri, col=3)

    fig.update_layout(
        title=(f"Stage-2 target-box slices (center=({cx:.2f}, {cy:.2f}, {cz:.2f}), "
               f"half-extent=({hx:.2f}, {hy:.2f}, {hz:.2f}))"),
        height=300 * n_bands + 80, width=1500,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
