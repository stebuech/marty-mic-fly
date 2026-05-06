"""Diagnostic-grid builders and rotor-disc spatial masks for Stage 2."""
from __future__ import annotations

from typing import Literal

import numpy as np


def build_diagnostic_grid(
    extent_xy_m: float,
    increment_m: float,
    z_min_m: float,
    z_max_m: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """3D rectangular grid. Returns (G, 3) positions and (nx, ny, nz) shape.

    When z_min_m == z_max_m, nz = 1 (degenerate single-slice case).

    Order: meshgrid with ``indexing="ij"``; ravel order so (i, j, k) maps to
    row ``i*ny*nz + j*nz + k``.
    """
    if extent_xy_m <= 0:
        raise ValueError("extent_xy_m must be > 0")
    if increment_m <= 0:
        raise ValueError("increment_m must be > 0")
    if z_max_m < z_min_m:
        raise ValueError("z_max_m must be >= z_min_m")

    nx = int(round(2 * extent_xy_m / increment_m)) + 1
    ny = nx
    if z_max_m == z_min_m:
        nz = 1
        zs = np.array([z_min_m], dtype=np.float64)
    else:
        nz = int(round((z_max_m - z_min_m) / increment_m)) + 1
        if nz < 1:
            nz = 1
        zs = np.linspace(z_min_m, z_max_m, nz)

    xs = np.linspace(-extent_xy_m, +extent_xy_m, nx)
    ys = np.linspace(-extent_xy_m, +extent_xy_m, ny)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    return points, (nx, ny, nz)


def build_rotor_disc_mask(
    grid_positions: np.ndarray,    # (G, 3)
    rotor_positions: np.ndarray,   # (3, R)
    rotor_radii: np.ndarray,       # (R,)
    z_tol_m: float = 0.05,
) -> np.ndarray:
    g_xy = grid_positions[:, :2]
    g_z = grid_positions[:, 2]
    r_xy = rotor_positions[:2, :].T   # (R, 2)
    r_z = rotor_positions[2, :]        # (R,)
    mask = np.zeros(g_xy.shape[0], dtype=bool)
    for i in range(r_xy.shape[0]):
        d_xy = np.linalg.norm(g_xy - r_xy[i], axis=1)
        z_ok = np.abs(g_z - r_z[i]) <= z_tol_m
        mask |= (d_xy <= rotor_radii[i]) & z_ok
    return mask


def build_target_box_mask(
    grid_positions: np.ndarray,                 # (G, 3)
    target_point_m: tuple[float, float, float],
    half_extent_m: tuple[float, float, float],
) -> np.ndarray:
    """Boolean mask: True for cells inside an axis-aligned box around the target.

    Used by ArrayFilterStage with mask_mode='target_box': cells *outside* the
    box are subtracted (~drone), cells *inside* are preserved (~target).
    """
    target = np.asarray(target_point_m, dtype=np.float64)
    half = np.asarray(half_extent_m, dtype=np.float64)
    delta = np.abs(grid_positions - target[None, :])   # (G, 3)
    return np.all(delta <= half[None, :], axis=1)


# ---------------------------------------------------------------- DOA grid (A)

def build_doa_grid(
    focal_radius_m: float,
    azimuth_step_deg: float,
    elevation_step_deg: float,
    hemisphere: Literal["lower", "upper", "full"] = "lower",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Sample a focal-radius sphere in (azimuth, elevation).

    Returns (G, 3) Cartesian points on the sphere of radius ``focal_radius_m``,
    plus the (n_az, n_el) shape for reshape. Convention:
        x = r · cos(el) · cos(az)
        y = r · cos(el) · sin(az)
        z = r · sin(el)
    Azimuth in [0, 360) (endpoint=False so 0° and 360° aren't duplicated).
    Elevation in degrees, range determined by ``hemisphere``:
        - 'lower': [-90, 0]   — appropriate for the AP2-A setup where rotors
                                sit at z≈0 and the external source at z<0.
        - 'upper': [0, +90]
        - 'full' : [-90, +90]
    Order: meshgrid(indexing="ij"), ravel index = i_az * n_el + i_el.
    """
    if focal_radius_m <= 0:
        raise ValueError("focal_radius_m must be > 0")
    if azimuth_step_deg <= 0 or elevation_step_deg <= 0:
        raise ValueError("step_deg values must be > 0")

    if hemisphere == "lower":
        el_min, el_max = -90.0, 0.0
    elif hemisphere == "upper":
        el_min, el_max = 0.0, 90.0
    elif hemisphere == "full":
        el_min, el_max = -90.0, 90.0
    else:
        raise ValueError(f"unknown hemisphere {hemisphere!r}")

    n_az = max(1, int(round(360.0 / azimuth_step_deg)))
    n_el = max(1, int(round((el_max - el_min) / elevation_step_deg)) + 1)
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el = np.linspace(el_min, el_max, n_el)

    AZ, EL = np.meshgrid(az, el, indexing="ij")
    az_rad = np.deg2rad(AZ.ravel())
    el_rad = np.deg2rad(EL.ravel())
    r = float(focal_radius_m)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    points = np.stack([x, y, z], axis=1)
    return points, (n_az, n_el)


def build_doa_cone_mask(
    grid_positions: np.ndarray,
    direction_xyz: tuple[float, float, float] | np.ndarray,
    cone_half_angle_deg: float,
) -> np.ndarray:
    """Boolean mask: True for grid cells whose unit direction (from origin) is
    within ``cone_half_angle_deg`` of ``direction_xyz`` (which is normalized
    internally and treated as a direction, not a point).

    Caller's responsibility: pass a non-zero vector. Grid points at the origin
    (norm < 1e-12) are unconditionally excluded.
    """
    target = np.asarray(direction_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(target))
    if norm < 1e-12:
        raise ValueError("direction_xyz must be non-zero")
    target_unit = target / norm

    grid_norms = np.linalg.norm(grid_positions, axis=1)
    valid = grid_norms > 1e-12
    grid_units = np.zeros_like(grid_positions)
    grid_units[valid] = grid_positions[valid] / grid_norms[valid, None]

    cos_threshold = float(np.cos(np.deg2rad(cone_half_angle_deg)))
    dots = grid_units @ target_unit
    return valid & (dots >= cos_threshold)


def build_rotor_doa_cones_mask(
    grid_positions: np.ndarray,
    rotor_positions: np.ndarray,            # (3, R) — same convention as platform/rotor_positions
    cone_half_angle_deg: float,
) -> np.ndarray:
    """Union of cones around each rotor's direction-from-origin. Rotors at
    origin (|pos| < 1e-9) are skipped silently."""
    if rotor_positions.shape[0] != 3:
        raise ValueError(
            f"rotor_positions must be (3, R); got {rotor_positions.shape}"
        )
    mask = np.zeros(grid_positions.shape[0], dtype=bool)
    for i in range(rotor_positions.shape[1]):
        rotor_xyz = rotor_positions[:, i]
        if float(np.linalg.norm(rotor_xyz)) < 1e-9:
            continue
        mask |= build_doa_cone_mask(grid_positions, rotor_xyz, cone_half_angle_deg)
    return mask


def inter_rotor_midpoints_3xR(rotor_positions: np.ndarray, tol: float = 0.10) -> np.ndarray:
    """Midpoints between nearest-neighbor rotor pairs, in (3, M) layout.

    Pairwise distances; identify d_min; keep every unordered pair (i,j) with
    |d_ij − d_min| ≤ tol·d_min; return their midpoints. For a quadcopter
    (4 rotors at square corners) this returns the 4 side midpoints — i.e.
    the points lying *between* adjacent rotors in the rotor plane.
    """
    if rotor_positions.shape[0] != 3:
        raise ValueError(
            f"rotor_positions must be (3, R); got {rotor_positions.shape}"
        )
    rp = rotor_positions.T  # (R, 3) for the pair math
    n = rp.shape[0]
    if n < 2:
        return rotor_positions.copy()
    diffs = rp[:, None, :] - rp[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    pair_d = dists[iu, ju]
    d_min = float(pair_d.min())
    keep = np.abs(pair_d - d_min) <= tol * d_min
    pairs_i = iu[keep]
    pairs_j = ju[keep]
    mids = 0.5 * (rp[pairs_i] + rp[pairs_j])  # (M, 3)
    return mids.T  # (3, M)
