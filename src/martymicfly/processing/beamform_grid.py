"""Diagnostic-grid builders and rotor-disc spatial masks for Stage 2."""
from __future__ import annotations

import numpy as np


def build_diagnostic_grid(
    extent_xy_m: float,
    increment_m: float,
    z_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    nx = int(round(2 * extent_xy_m / increment_m)) + 1
    ny = nx
    xs = np.linspace(-extent_xy_m, +extent_xy_m, nx)
    ys = np.linspace(-extent_xy_m, +extent_xy_m, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z_m)], axis=1)
    return points, (nx, ny)


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
