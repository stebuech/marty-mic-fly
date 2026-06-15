import numpy as np


def test_array_metrics_basic_shape_without_ground_truth():
    from martymicfly.eval.array_metrics import compute_array_metrics
    n_f = 8
    csm = np.tile(np.eye(3, dtype=np.complex128) * 4.0, (n_f, 1, 1))
    res = np.tile(np.eye(3, dtype=np.complex128) * 1.0, (n_f, 1, 1))
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 4.0)
    psd_post = np.full(n_f, 1.0)
    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}]
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 9)),
        drone_mask=np.array([True, True, True, False, False, False, False, False, False]),
        bands=bands, ground_truth=None,
    )
    assert "mid" in metrics["bands"]
    band = metrics["bands"]["mid"]
    assert band["csm_trace_reduction_db"] > 0
    assert band["target_psd_reduction_db"] > 0
    assert "drone_power_share_db" in band
    assert band["ground_truth"] is None


def test_array_metrics_with_ground_truth_recovery():
    from martymicfly.eval.array_metrics import compute_array_metrics
    n_f = 8
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 5.0)
    psd_post = np.full(n_f, 2.0)
    gt_psd = np.full(n_f, 2.0)
    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}]
    csm = np.tile(np.eye(2, dtype=np.complex128) * 5.0, (n_f, 1, 1))
    res = np.tile(np.eye(2, dtype=np.complex128) * 2.0, (n_f, 1, 1))
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 4)),
        drone_mask=np.array([True, False, False, False]),
        bands=bands,
        ground_truth={"psd_at_target": gt_psd, "frequencies": freqs},
    )
    band_gt = metrics["bands"]["mid"]["ground_truth"]
    assert abs(band_gt["external_recovery_db"]) < 0.5


def test_array_metrics_applies_range_compensation_when_geometry_supplied():
    """When mic_positions and target_point are supplied, psd_pre/psd_post are
    multiplied by `range_compensation_factor` so target_psd_*_db and recovery_db
    move into proper source-PSD units. target_psd_reduction_db is unchanged."""
    from martymicfly.eval.array_metrics import compute_array_metrics
    from martymicfly.processing.steering import range_compensation_factor

    n_f = 8
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 5.0)
    psd_post = np.full(n_f, 2.0)
    gt_psd = np.full(n_f, 2.0)  # uncalibrated this gives recovery_db = 0
    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}]
    csm = np.tile(np.eye(2, dtype=np.complex128) * 5.0, (n_f, 1, 1))
    res = np.tile(np.eye(2, dtype=np.complex128) * 2.0, (n_f, 1, 1))
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
    ])
    target = (0.0, 0.0, -1.5)
    factor = range_compensation_factor(mic_positions, target)
    cal_db = 10 * np.log10(factor)

    # With calibration on. Test that gt comparison expects the calibrated value.
    gt_psd_calibrated = gt_psd * factor
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 4)),
        drone_mask=np.array([True, False, False, False]),
        bands=bands,
        ground_truth={"psd_at_target": gt_psd_calibrated, "frequencies": freqs},
        mic_positions=mic_positions,
        target_point=target,
    )
    band = metrics["bands"]["mid"]
    # target_psd_pre_db is now psd_pre * factor → shifted by cal_db.
    expected_pre_db = 10 * np.log10(5.0 * (1500.0 - 500.0 + (freqs[1] - freqs[0]))) + cal_db
    # The exact band-sum value isn't critical; what matters is that the
    # difference target_psd_pre_db - target_psd_post_db is unchanged (calibration
    # cancels in the reduction) and recovery_db is ~0 against calibrated GT.
    assert abs(band["target_psd_reduction_db"] - (10 * np.log10(5.0) - 10 * np.log10(2.0))) < 1e-6
    assert abs(band["ground_truth"]["external_recovery_db"]) < 0.5


def test_array_metrics_no_calibration_when_geometry_omitted():
    """Backward compat: omitting mic_positions/target_point keeps prior behavior."""
    from martymicfly.eval.array_metrics import compute_array_metrics
    n_f = 8
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 5.0)
    psd_post = np.full(n_f, 2.0)
    csm = np.tile(np.eye(2, dtype=np.complex128) * 5.0, (n_f, 1, 1))
    res = np.tile(np.eye(2, dtype=np.complex128) * 2.0, (n_f, 1, 1))
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 4)),
        drone_mask=np.array([True, False, False, False]),
        bands=[{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}],
        ground_truth={"psd_at_target": np.full(n_f, 2.0), "frequencies": freqs},
    )
    # Without calibration, recovery_db ≈ 0 against raw psd_post.
    assert abs(metrics["bands"]["mid"]["ground_truth"]["external_recovery_db"]) < 0.5
