import numpy as np


def test_build_measurement_csm_shape_and_hermitian():
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    fs = 16000.0
    n = 8192
    rng = np.random.default_rng(0)
    time_data = rng.standard_normal((n, 4))
    cfg = CsmConfig(nperseg=512, noverlap=256, f_min_hz=200.0, f_max_hz=4000.0,
                    diag_loading_rel=0.0)
    csm, freqs = build_measurement_csm(time_data, fs, cfg)
    assert csm.shape == (freqs.shape[0], 4, 4)
    assert csm.dtype == np.complex128
    np.testing.assert_allclose(csm, csm.conj().transpose(0, 2, 1), atol=1e-9)
    assert freqs.min() >= 200.0
    assert freqs.max() <= 4000.0


def test_build_measurement_csm_diag_loading_increases_diagonal():
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    fs = 16000.0
    rng = np.random.default_rng(1)
    time_data = rng.standard_normal((4096, 3))
    base = CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0,
                     diag_loading_rel=0.0)
    loaded = CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0,
                       diag_loading_rel=1e-2)
    csm_b, _ = build_measurement_csm(time_data, fs, base)
    csm_l, _ = build_measurement_csm(time_data, fs, loaded)
    diag_b = np.abs(np.diagonal(csm_b, axis1=1, axis2=2)).mean()
    diag_l = np.abs(np.diagonal(csm_l, axis1=1, axis2=2)).mean()
    assert diag_l > diag_b
