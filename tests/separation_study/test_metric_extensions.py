import numpy as np


def test_decompose_excess_deficit_per_bin():
    """excess(f) = max(post - gt, 0); deficit(f) = max(gt - post, 0); je f nur einer."""
    from analysis.separation_study.metric_extensions import decompose_residual
    psd_post = np.array([1.0, 2.0, 0.5, 3.0])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    excess, deficit = decompose_residual(psd_post, ext_gt)
    np.testing.assert_array_equal(excess,  [0.0, 1.0, 0.0, 2.0])
    np.testing.assert_array_equal(deficit, [0.0, 0.0, 0.5, 0.0])


def test_spectrum_l1_db_no_compensation():
    """Σexcess = Σdeficit konstruiert → recovery_signed = 0, l1 ≫ 0."""
    from analysis.separation_study.metric_extensions import (
        decompose_residual, band_metrics,
    )
    psd_post = np.array([2.0, 0.5, 2.0, 0.5])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    delta_f = 1.0
    m = band_metrics(psd_post, ext_gt, d_ref=np.ones(4), delta_f=delta_f)
    p_post = psd_post.sum() * delta_f
    e_gt = ext_gt.sum() * delta_f
    expected_signed = 10 * np.log10(p_post / e_gt)
    assert abs(m["recovery_db_signed"] - expected_signed) < 1e-9
    assert m["spectrum_l1_db"] > -1.0
    assert m["compensation_flag"] is True


def test_spectrum_l1_db_perfect_match():
    """psd_post = ext_GT identisch → l1 → -inf, compensation_flag=False."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([1.0, 2.0, 3.0])
    m = band_metrics(psd_post, psd_post, d_ref=np.ones(3), delta_f=1.0)
    assert m["spectrum_l1_db"] == float("-inf")
    assert m["over_subtraction_db"] == float("-inf")
    assert m["drone_leakage_db_def2"] == float("-inf")
    assert m["compensation_flag"] is False


def test_over_subtraction_db_no_excess():
    """psd_post = α·ext_GT, α<1 ⇒ over_subtraction_db = 10·log10(1-α)."""
    from analysis.separation_study.metric_extensions import band_metrics
    alpha = 0.4
    ext_gt = np.array([2.0, 4.0, 6.0])
    m = band_metrics(alpha * ext_gt, ext_gt, d_ref=np.ones(3), delta_f=1.0)
    expected = 10 * np.log10(1.0 - alpha)
    assert abs(m["over_subtraction_db"] - expected) < 1e-9


def test_drone_leakage_def1_at_perfect_filter_equals_neg_snr():
    """psd_post = ext_GT, def1 = 10·log10(E_GT / D_unflt) = -SNR."""
    from analysis.separation_study.metric_extensions import band_metrics
    ext_gt = np.array([1.0, 2.0, 3.0])
    d_ref  = np.array([10.0, 20.0, 30.0])
    m = band_metrics(ext_gt.copy(), ext_gt, d_ref=d_ref, delta_f=1.0)
    expected = 10 * np.log10(ext_gt.sum() / d_ref.sum())
    assert abs(m["drone_leakage_db_def1"] - expected) < 1e-9


def test_recovery_minus_leakage_def1_eq_neg_snr_sanity():
    """Sanity: recovery_signed - leakage_def1 = -SNR_band, unabhängig vom Filter."""
    from analysis.separation_study.metric_extensions import band_metrics
    rng = np.random.default_rng(42)
    psd_post = rng.uniform(0.5, 5.0, size=10)
    ext_gt   = rng.uniform(0.1, 1.0, size=10)
    d_ref    = rng.uniform(2.0, 10.0, size=10)
    m = band_metrics(psd_post, ext_gt, d_ref=d_ref, delta_f=1.0)
    snr_band = 10 * np.log10(ext_gt.sum() / d_ref.sum())
    sanity = m["recovery_db_signed"] - m["drone_leakage_db_def1"] + snr_band
    assert abs(sanity) < 1e-9


def test_spectrum_rms_db_uniform_offset():
    """psd_post = β·ext_GT konstant ⇒ rms_db = |10·log10(β)|."""
    from analysis.separation_study.metric_extensions import band_metrics
    beta = 0.5
    ext_gt = np.array([1.0, 2.0, 3.0])
    m = band_metrics(beta * ext_gt, ext_gt, d_ref=np.ones(3), delta_f=1.0)
    expected = abs(10 * np.log10(beta))
    assert abs(m["spectrum_rms_db"] - expected) < 1e-9


def test_compensation_flag_above_floor():
    """Beide Komponenten oberhalb welch_floor + 3 dB → flag=true."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([2.0, 0.5, 2.0, 0.5])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    m = band_metrics(
        psd_post, ext_gt, d_ref=np.ones(4), delta_f=1.0, welch_floor_db=-30.0,
    )
    assert m["compensation_flag"] is True


def test_compensation_flag_below_floor():
    """Excess winzig (Welch-Streuung) → flag=false."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([1.001, 0.999, 1.0])
    ext_gt   = np.array([1.0,   1.0,   1.0])
    m = band_metrics(
        psd_post, ext_gt, d_ref=np.ones(3), delta_f=1.0, welch_floor_db=-10.0,
    )
    assert m["compensation_flag"] is False
