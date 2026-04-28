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
