import numpy as np


def test_source_map_subset_drops_columns():
    from martymicfly.processing.algorithms.base import SourceMap
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    powers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sm = SourceMap(
        positions=positions, powers=powers,
        frequencies=np.array([100.0, 200.0]), grid_shape=None, metadata={},
    )
    keep = np.array([True, False, True])
    sub = sm.subset(keep)
    assert sub.positions.shape == (2, 3)
    assert sub.powers.shape == (2, 2)
    np.testing.assert_array_equal(sub.powers[:, 0], np.array([1.0, 4.0]))


def test_reconstruct_csm_single_source_round_trip():
    """One unit-power source at (1, 0, 0); reconstruct_csm must be hermitian
    and have non-negative diagonal."""
    from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
    positions = np.array([[1.0, 0.0, 0.0]])
    freqs = np.array([500.0])
    powers = np.array([[1.0]])
    sm = SourceMap(positions=positions, powers=powers, frequencies=freqs,
                   grid_shape=None, metadata={})
    mics = np.array([[0.0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
    csm = reconstruct_csm(sm, mics)
    assert csm.shape == (1, 3, 3)
    np.testing.assert_allclose(csm, csm.conj().transpose(0, 2, 1), atol=1e-12)
    assert (np.real(np.diagonal(csm, axis1=1, axis2=2)) >= 0).all()


def test_rescale_source_map_to_csm_trace_matches_observed_trace():
    """Trace-rescaling makes reconstruct_csm.trace match the observed CSM
    trace per frequency, regardless of the input source_map's absolute
    scaling. Models the Acoular `classic`-steer power-convention fix."""
    from martymicfly.processing.algorithms.base import (
        SourceMap, reconstruct_csm, rescale_source_map_to_csm_trace,
    )
    positions = np.array([[0.1, 0.0, -0.5]])
    mics = np.array([[0.4, 0, 0], [-0.4, 0, 0],
                     [0, 0.4, 0], [0, -0.4, 0]])
    freqs = np.array([1000.0, 2000.0, 3000.0])
    # Truth source: 1.0 Pa² per frequency
    sm_true = SourceMap(positions=positions, powers=np.full((3, 1), 1.0),
                        frequencies=freqs, grid_shape=None, metadata={})
    csm_obs = reconstruct_csm(sm_true, mics)
    # Mock CLEAN-SC output: same map shape, but powers scaled by some constant
    # (CLEAN-SC's `classic` steer convention does this)
    sm_uncal = SourceMap(positions=positions, powers=np.full((3, 1), 0.0634),
                         frequencies=freqs, grid_shape=None, metadata={})
    sm_cal, factors = rescale_source_map_to_csm_trace(sm_uncal, csm_obs, mics)
    # Calibrated trace must match observed trace per frequency
    csm_rebuilt = reconstruct_csm(sm_cal, mics)
    np.testing.assert_allclose(
        np.real(np.trace(csm_rebuilt, axis1=1, axis2=2)),
        np.real(np.trace(csm_obs, axis1=1, axis2=2)),
        rtol=1e-10,
    )
    # Calibrated power should be ~1.0 since the truth was 1.0
    np.testing.assert_allclose(sm_cal.powers, 1.0, rtol=1e-3)
    # Factor is ~1/0.0634 ≈ 15.77
    assert factors.shape == (3,)
    assert (factors > 10).all() and (factors < 20).all()
