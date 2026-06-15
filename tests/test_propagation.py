import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def test_greens_propagate_one_source_one_mic_attenuation_and_delay():
    from martymicfly.synth.propagation import greens_propagate
    fs = 16000.0
    n = 4096
    t = np.arange(n) / fs
    src = np.sin(2 * np.pi * 500.0 * t).astype(np.float64)
    src_pos = np.array([[0.0, 0.0, 0.0]])
    mic_pos = np.array([[1.5, 0.0, 0.0]])
    out = greens_propagate(src.reshape(1, -1), src_pos, mic_pos, fs)
    assert out.shape == (n, 1)
    r = 1.5
    expected_amp = 1.0 / (4 * np.pi * r)
    delay_samples = r / SPEED_OF_SOUND * fs
    n_lo = int(np.floor(delay_samples)) + 200
    n_hi = n - 200
    rms_out = np.sqrt(np.mean(out[n_lo:n_hi, 0] ** 2))
    rms_src = np.sqrt(np.mean(src[: n_hi - n_lo] ** 2))
    assert abs(rms_out / rms_src - expected_amp) / expected_amp < 0.05


def test_greens_propagate_sums_two_sources():
    from martymicfly.synth.propagation import greens_propagate
    fs = 16000.0
    n = 4096
    rng = np.random.default_rng(1)
    s1 = rng.standard_normal(n)
    s2 = rng.standard_normal(n)
    src_pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mic_pos = np.array([[0.0, 0.0, 0.0]])
    sig = np.stack([s1, s2])
    out_both = greens_propagate(sig, src_pos, mic_pos, fs)
    out_1 = greens_propagate(sig[:1], src_pos[:1], mic_pos, fs)
    out_2 = greens_propagate(sig[1:], src_pos[1:], mic_pos, fs)
    np.testing.assert_allclose(out_both, out_1 + out_2, atol=1e-9)
