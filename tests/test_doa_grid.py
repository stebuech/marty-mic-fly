import numpy as np
import pytest


def test_build_doa_grid_lower_hemisphere_shape_and_radius():
    from martymicfly.processing.beamform_grid import build_doa_grid
    points, shape = build_doa_grid(
        focal_radius_m=1.5, azimuth_step_deg=30.0, elevation_step_deg=30.0,
        hemisphere="lower",
    )
    n_az, n_el = shape
    assert n_az == 12              # 360 / 30
    assert n_el == 4               # (-90, -60, -30, 0)
    assert points.shape == (n_az * n_el, 3)
    # All points on the focal sphere
    radii = np.linalg.norm(points, axis=1)
    np.testing.assert_allclose(radii, 1.5, atol=1e-9)
    # All elevations <= 0 for lower hemisphere
    assert np.all(points[:, 2] <= 1e-9)


def test_build_doa_grid_full_hemisphere():
    from martymicfly.processing.beamform_grid import build_doa_grid
    points, (n_az, n_el) = build_doa_grid(
        focal_radius_m=2.0, azimuth_step_deg=60.0, elevation_step_deg=45.0,
        hemisphere="full",
    )
    assert n_az == 6
    assert n_el == 5               # (-90, -45, 0, 45, 90)
    assert np.allclose(np.linalg.norm(points, axis=1), 2.0)


def test_build_doa_grid_rejects_bad_inputs():
    from martymicfly.processing.beamform_grid import build_doa_grid
    with pytest.raises(ValueError):
        build_doa_grid(0.0, 10.0, 10.0)
    with pytest.raises(ValueError):
        build_doa_grid(1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        build_doa_grid(1.0, 10.0, 10.0, hemisphere="diagonal")  # type: ignore


def test_build_doa_cone_mask_picks_only_aligned_cells():
    from martymicfly.processing.beamform_grid import build_doa_cone_mask, build_doa_grid
    points, _ = build_doa_grid(1.5, 15.0, 15.0, "lower")
    # Cone around (0, 0, -1) with 15° half-angle picks only points near south pole
    mask = build_doa_cone_mask(points, (0.0, 0.0, -1.0), 15.0)
    selected = points[mask]
    # Selected points must all have z very close to -1.5 (south pole region)
    assert selected.shape[0] >= 1
    assert np.all(selected[:, 2] <= -1.5 * np.cos(np.deg2rad(15.0)) + 1e-9)


def test_build_doa_cone_mask_rejects_zero_direction():
    import pytest
    from martymicfly.processing.beamform_grid import build_doa_cone_mask
    points = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        build_doa_cone_mask(points, (0.0, 0.0, 0.0), 30.0)


def test_build_rotor_doa_cones_mask_unions_per_rotor():
    from martymicfly.processing.beamform_grid import (
        build_doa_grid, build_rotor_doa_cones_mask,
    )
    points, _ = build_doa_grid(1.0, 10.0, 10.0, "full")
    # Two rotors at opposite equator points; their DOAs are +x and -x.
    rotors = np.array([[0.23, -0.23], [0.0, 0.0], [0.0, 0.0]])  # (3, 2)
    mask = build_rotor_doa_cones_mask(points, rotors, 30.0)
    # Mask should cover both +x and -x cones; spot-check by adding a single
    # rotor mask on the same grid and verifying it is a strict subset.
    from martymicfly.processing.beamform_grid import build_doa_cone_mask
    only_plus = build_doa_cone_mask(points, (1.0, 0.0, 0.0), 30.0)
    only_minus = build_doa_cone_mask(points, (-1.0, 0.0, 0.0), 30.0)
    np.testing.assert_array_equal(mask, only_plus | only_minus)


def test_build_rotor_doa_cones_mask_skips_origin_rotor():
    from martymicfly.processing.beamform_grid import (
        build_doa_grid, build_rotor_doa_cones_mask,
    )
    points, _ = build_doa_grid(1.0, 10.0, 10.0, "full")
    # First rotor at origin (degenerate), second rotor at +x
    rotors = np.array([[0.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
    mask = build_rotor_doa_cones_mask(points, rotors, 30.0)
    # Should equal the +x cone alone
    from martymicfly.processing.beamform_grid import build_doa_cone_mask
    plus_x = build_doa_cone_mask(points, (0.5, 0.0, 0.0), 30.0)
    np.testing.assert_array_equal(mask, plus_x)
