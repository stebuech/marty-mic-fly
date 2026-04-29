"""Diagnostic-grid builders and rotor-disc spatial masks for Stage 2."""
from __future__ import annotations

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
