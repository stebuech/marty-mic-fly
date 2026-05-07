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


def test_range_compensation_factor_matches_4pi_over_mean_inv_r_squared():
    """For a 1/(4π·r) Greens propagator + phase-only DAS with 1/M² norm,
    steer_to_psd at the source returns S_q · (⟨1/r⟩/(4π))². The compensation
    factor is the inverse: (4π/⟨1/r⟩)²."""
    from martymicfly.processing.steering import range_compensation_factor
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
    ])
    target = (0.0, 0.0, -1.5)
    r = np.linalg.norm(mic_positions - np.array(target), axis=1)
    expected = (4.0 * np.pi / (1.0 / r).mean()) ** 2
    factor = range_compensation_factor(mic_positions, target)
    assert abs(factor - expected) / expected < 1e-12


def test_range_compensation_factor_inverts_steered_psd_on_synth_signal():
    """End-to-end check: with project's greens_propagate (1/4πr) → CSM → steer_to_psd,
    multiplying by the compensation factor recovers the source PSD within 1 dB."""
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    from martymicfly.processing.steering import (
        range_compensation_factor, steer_to_psd,
    )
    from martymicfly.synth.propagation import greens_propagate

    rng = np.random.default_rng(99)
    fs = 51_200.0
    n_samples = int(fs * 6.0)
    s_q = 1e-4
    sigma = float(np.sqrt(s_q * fs / 2.0))
    src_signal = rng.normal(0.0, sigma, size=n_samples)[None, :]   # (1, N)
    src_pos = np.array([[0.0, 0.0, -1.5]])
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
        [0.07, 0.07, 0.0], [-0.07, -0.07, 0.0],
        [0.07, -0.07, 0.0], [-0.07, 0.07, 0.0],
    ])
    time_data = greens_propagate(src_signal, src_pos, mic_positions, fs)

    cfg = CsmConfig(nperseg=512, noverlap=256, window="hann",
                    diag_loading_rel=0.0, f_min_hz=200.0, f_max_hz=6000.0)
    csm, freqs = build_measurement_csm(time_data, fs, cfg)
    psd_raw = steer_to_psd(csm, freqs, mic_positions, tuple(src_pos[0].tolist()))
    factor = range_compensation_factor(mic_positions, tuple(src_pos[0].tolist()))
    psd_corrected = psd_raw * factor
    delta_db = 10.0 * np.log10(psd_corrected.mean() / s_q)
    assert abs(delta_db) < 1.0, f"recovered S_q off by {delta_db:+.2f} dB"


def test_integrate_band_maps_produces_band_dict():
    from martymicfly.processing.algorithms.base import SourceMap
    from martymicfly.processing.array_filter import integrate_band_maps, BandConfig
    n_f, n_g = 5, 9  # 3x3 grid
    powers = np.ones((n_f, n_g))
    sm = SourceMap(
        positions=np.zeros((n_g, 3)),
        powers=powers,
        frequencies=np.linspace(200.0, 1000.0, n_f),
        grid_shape=(3, 3, 1),
        metadata={},
    )
    bands = [
        BandConfig(name="lo", f_min_hz=0.0, f_max_hz=500.0),
        BandConfig(name="hi", f_min_hz=500.0, f_max_hz=10_000.0),
    ]
    maps = integrate_band_maps(sm, bands, (3, 3, 1))
    assert set(maps.keys()) == {"lo", "hi"}
    assert maps["lo"].shape == (3, 3, 1)
    assert maps["hi"].shape == (3, 3, 1)
