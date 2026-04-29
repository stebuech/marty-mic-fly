import numpy as np
import pytest


def test_build_diagnostic_grid_2d_slice():
    from martymicfly.processing.beamform_grid import build_diagnostic_grid
    grid, shape = build_diagnostic_grid(
        extent_xy_m=0.5, increment_m=0.05, z_min_m=0.0, z_max_m=0.0,
    )
    nx, ny, nz = shape
    assert nx == ny == 21  # round(2*0.5/0.05)+1
    assert nz == 1
    assert grid.shape == (nx * ny * nz, 3)
    assert np.all(grid[:, 2] == 0.0)


def test_build_diagnostic_grid_3d_box():
    from martymicfly.processing.beamform_grid import build_diagnostic_grid
    grid, shape = build_diagnostic_grid(
        extent_xy_m=0.1, increment_m=0.05, z_min_m=-0.1, z_max_m=0.1,
    )
    nx, ny, nz = shape
    assert nx == ny == 5  # round(2*0.1/0.05)+1
    assert nz == 5  # round((0.1 - -0.1)/0.05)+1
    assert grid.shape == (nx * ny * nz, 3)
    # z values span correctly
    z_unique = np.unique(grid[:, 2])
    np.testing.assert_allclose(z_unique, np.linspace(-0.1, 0.1, 5))
    # Order: row i*ny*nz + j*nz + k
    # Cell (0, 0, 0) is first row
    np.testing.assert_allclose(grid[0], [-0.1, -0.1, -0.1])
    # Cell (0, 0, 1) is second row
    np.testing.assert_allclose(grid[1], [-0.1, -0.1, -0.05])
    # Cell (0, 1, 0)
    np.testing.assert_allclose(grid[nz], [-0.1, -0.05, -0.1])


def test_build_diagnostic_grid_validates_inputs():
    from martymicfly.processing.beamform_grid import build_diagnostic_grid
    with pytest.raises(ValueError):
        build_diagnostic_grid(extent_xy_m=0.0, increment_m=0.05, z_min_m=0.0, z_max_m=0.0)
    with pytest.raises(ValueError):
        build_diagnostic_grid(extent_xy_m=0.5, increment_m=0.0, z_min_m=0.0, z_max_m=0.0)
    with pytest.raises(ValueError):
        build_diagnostic_grid(extent_xy_m=0.5, increment_m=0.05, z_min_m=0.1, z_max_m=-0.1)


def test_build_rotor_disc_mask_marks_inside_only():
    from martymicfly.processing.beamform_grid import (
        build_diagnostic_grid,
        build_rotor_disc_mask,
    )
    grid, _ = build_diagnostic_grid(0.5, 0.02, 0.0, 0.0)
    rotor_pos = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]])
    rotor_radii = np.array([0.10, 0.10])
    mask = build_rotor_disc_mask(grid, rotor_pos, rotor_radii, z_tol_m=0.02)
    assert mask.sum() > 0
    inside = grid[mask]
    for p in inside:
        d0 = np.linalg.norm(p[:2] - np.array([0.15, 0.0]))
        d1 = np.linalg.norm(p[:2] - np.array([-0.15, 0.0]))
        assert d0 <= 0.10 + 1e-9 or d1 <= 0.10 + 1e-9
