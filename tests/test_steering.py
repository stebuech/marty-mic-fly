import numpy as np


def test_steer_to_psd_shapes_and_units():
    """Steering an identity-only CSM (white noise per mic) at a random target
    yields a positive-real PSD."""
    from martymicfly.processing.steering import steer_to_psd
    n_f, n_m = 8, 4
    csm = np.tile(np.eye(n_m, dtype=np.complex128), (n_f, 1, 1))
    freqs = np.linspace(200.0, 1000.0, n_f)
    mic_positions = np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
    target = (0.0, 0.0, -0.5)
    psd = steer_to_psd(csm, freqs, mic_positions, target)
    assert psd.shape == (n_f,)
    assert (psd > 0).all()
    assert np.iscomplexobj(psd) is False


def test_integrate_band_maps_produces_band_dict():
    from martymicfly.processing.algorithms.base import SourceMap
    from martymicfly.processing.array_filter import integrate_band_maps, BandConfig
    n_f, n_g = 5, 9  # 3x3 grid
    powers = np.ones((n_f, n_g))
    sm = SourceMap(
        positions=np.zeros((n_g, 3)),
        powers=powers,
        frequencies=np.linspace(200.0, 1000.0, n_f),
        grid_shape=(3, 3),
        metadata={},
    )
    bands = [
        BandConfig(name="lo", f_min_hz=0.0, f_max_hz=500.0),
        BandConfig(name="hi", f_min_hz=500.0, f_max_hz=10_000.0),
    ]
    maps = integrate_band_maps(sm, bands, (3, 3))
    assert set(maps.keys()) == {"lo", "hi"}
    assert maps["lo"].shape == (3, 3)
    assert maps["hi"].shape == (3, 3)
