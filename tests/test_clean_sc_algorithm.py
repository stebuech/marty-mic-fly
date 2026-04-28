import numpy as np


def test_clean_sc_recovers_single_source_on_diagonal_grid():
    """Synthesize a CSM with one known source via reconstruct_csm; CLEAN-SC
    on a 2D RectGrid that includes the true position must recover >= 70 %
    of the total power at the corresponding grid point."""
    from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
    from martymicfly.processing.algorithms.clean_sc import CleanScAlgorithm
    from martymicfly.processing.beamform_grid import build_diagnostic_grid

    grid, shape = build_diagnostic_grid(0.5, 0.05, 0.0)
    true_pos = np.array([[0.20, 0.0, 0.0]])
    freqs = np.array([1000.0, 1500.0, 2000.0])
    sm_true = SourceMap(
        positions=true_pos,
        powers=np.ones((3, 1)),
        frequencies=freqs,
        grid_shape=None,
        metadata={},
    )
    mic_positions = np.array([
        [0.4, 0.0, 0.05], [-0.4, 0.0, -0.05],
        [0.0, 0.4, 0.05], [0.0, -0.4, -0.05],
    ])
    csm = reconstruct_csm(sm_true, mic_positions)

    algo = CleanScAlgorithm()
    sm_est = algo.fit(
        csm=csm, frequencies=freqs, time_data=None, sample_rate=16000.0,
        mic_positions=mic_positions, grid_positions=grid,
        params={"damp": 0.6, "n_iter": 50},
    )
    total = sm_est.powers.sum()
    nearest_idx = int(np.argmin(np.linalg.norm(grid[:, :2] - true_pos[:, :2], axis=1)))
    near_neighbors = np.linalg.norm(grid[:, :2] - true_pos[:, :2], axis=1) <= 0.07
    near_power = sm_est.powers[:, near_neighbors].sum()
    assert near_power / total > 0.70, f"only {near_power/total:.2f} of power near true source"
