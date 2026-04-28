import numpy as np


def test_build_diagnostic_grid_shape_and_z():
    from martymicfly.processing.beamform_grid import build_diagnostic_grid
    grid, shape = build_diagnostic_grid(extent_xy_m=0.5, increment_m=0.05, z_m=0.0)
    nx, ny = shape
    assert nx == ny == 21  # round(2*0.5/0.05)+1
    assert grid.shape == (nx * ny, 3)
    assert np.all(grid[:, 2] == 0.0)


def test_build_rotor_disc_mask_marks_inside_only():
    from martymicfly.processing.beamform_grid import (
        build_diagnostic_grid,
        build_rotor_disc_mask,
    )
    grid, _ = build_diagnostic_grid(0.5, 0.02, 0.0)
    rotor_pos = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]])
    rotor_radii = np.array([0.10, 0.10])
    mask = build_rotor_disc_mask(grid, rotor_pos, rotor_radii, z_tol_m=0.02)
    assert mask.sum() > 0
    inside = grid[mask]
    for p in inside:
        d0 = np.linalg.norm(p[:2] - np.array([0.15, 0.0]))
        d1 = np.linalg.norm(p[:2] - np.array([-0.15, 0.0]))
        assert d0 <= 0.10 + 1e-9 or d1 <= 0.10 + 1e-9
