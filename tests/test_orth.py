"""Tests for the orthogonal-deconvolution algorithm wrapper."""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def _build_atom_csm(positions: np.ndarray, mic_positions: np.ndarray,
                    powers_per_freq: np.ndarray, frequencies: np.ndarray,
                    c: float = SPEED_OF_SOUND) -> np.ndarray:
    """C[f] = sum_g p[f,g] h_g h_g^H with h_g = 1/(4πr) exp(-j2πf r/c)."""
    diff = mic_positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r < 1e-9, 1e-9, r)
    inv_r = 1.0 / (4.0 * np.pi * r)
    csm = np.zeros((frequencies.size, mic_positions.shape[0],
                    mic_positions.shape[0]), dtype=np.complex128)
    for fi, f in enumerate(frequencies):
        h = inv_r * np.exp(-2j * np.pi * f * r / c)
        weighted = h * powers_per_freq[fi][None, :]
        csm[fi] = weighted @ h.conj().T
    return csm


def _mic_geom() -> np.ndarray:
    """16-mic planar array, ~0.6 m aperture — matches the AP2-A platform size."""
    xs = np.linspace(-0.3, 0.3, 4)
    return np.array([[x, y, 0.0] for x in xs for y in xs])


def _focal_grid() -> np.ndarray:
    """5x5 candidate grid on the z=-1.5 m focal plane below the array."""
    g = np.linspace(-0.6, 0.6, 5)
    return np.array([[x, y, -1.5] for x in g for y in g])


def test_algorithm_registered():
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
    import martymicfly.processing.algorithms.orth  # noqa: F401
    assert "orth" in ALGORITHM_REGISTRY


def test_source_map_shape():
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
    import martymicfly.processing.algorithms.orth  # noqa: F401

    mic = _mic_geom()
    grid = _focal_grid()
    freqs = np.array([800.0, 2000.0])
    P = np.zeros((freqs.size, grid.shape[0]))
    P[:, 12] = [4.0, 9.0]
    csm = _build_atom_csm(grid, mic, P, freqs)

    sm = ALGORITHM_REGISTRY["orth"]().fit(
        csm=csm, frequencies=freqs, time_data=None, sample_rate=48000.0,
        mic_positions=mic, grid_positions=grid, params={"n": 4},
    )
    assert sm.powers.shape == (freqs.size, grid.shape[0])
    assert np.all(sm.powers >= 0.0)
    assert np.all(np.isfinite(sm.powers))


def test_single_source_localization():
    """Orthogonal deconvolution must place the strongest power at the true cell."""
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
    import martymicfly.processing.algorithms.orth  # noqa: F401

    mic = _mic_geom()
    grid = _focal_grid()
    src_cell = 7
    freqs = np.array([1500.0, 3000.0])
    P = np.zeros((freqs.size, grid.shape[0]))
    P[:, src_cell] = [5.0, 5.0]
    csm = _build_atom_csm(grid, mic, P, freqs)

    sm = ALGORITHM_REGISTRY["orth"]().fit(
        csm=csm, frequencies=freqs, time_data=None, sample_rate=48000.0,
        mic_positions=mic, grid_positions=grid, params={"n": 1},
    )
    for fi in range(freqs.size):
        assert int(np.argmax(sm.powers[fi])) == src_cell
