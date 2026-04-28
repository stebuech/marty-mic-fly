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
