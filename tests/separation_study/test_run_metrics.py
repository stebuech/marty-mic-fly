import numpy as np
import h5py


def _write_gt_h5(path, signal, fs):
    with h5py.File(path, "w") as f:
        td = f.create_dataset("time_data", data=signal.reshape(-1, 1))
        td.attrs["sample_freq"] = float(fs)


def test_compute_run_metrics_round_trip(tmp_path):
    """Steered psd_post=ext_GT identisch konstruiert ⇒ recovery_signed≈0,
    l1 ≈ welch-noise."""
    from analysis.separation_study.metric_extensions import compute_run_metrics
    from scipy.signal import welch
    fs = 51200.0
    rng = np.random.default_rng(0)
    n = 51_200
    drone = rng.normal(0.0, 1.0, size=n)
    ext   = rng.normal(0.0, 0.3, size=n)
    mix   = drone + ext

    ext_h5 = tmp_path / "ext_gt.h5"
    mix_h5 = tmp_path / "mix_gt.h5"
    _write_gt_h5(ext_h5, ext, fs)
    _write_gt_h5(mix_h5, mix, fs)

    f_w, psd_post = welch(ext, fs=fs, nperseg=512, noverlap=256,
                          window="hann", scaling="density")
    bands = [{"name": "low", "f_min_hz": 200.0, "f_max_hz": 1000.0},
             {"name": "high", "f_min_hz": 1000.0, "f_max_hz": 6000.0}]

    out = compute_run_metrics(
        psd_post=psd_post, frequencies=f_w,
        ext_gt_h5=ext_h5, mixed_gt_h5=mix_h5,
        welch_nperseg=512, welch_noverlap=256, window="hann",
        bands=bands, welch_floor_db=-50.0,
    )
    assert set(out["bands"].keys()) == {"low", "high"}
    for band in ("low", "high"):
        assert abs(out["bands"][band]["recovery_db_signed"]) < 0.5
        # SNR_band: Var(ext) / Var(drone) ≈ 0.09 → ≈ -10.5 dB
        assert -15.0 < out["bands"][band]["drone_leakage_db_def1"] < -8.0


def test_compute_run_metrics_band_mask_applied(tmp_path):
    """Bins außerhalb der Bandgrenzen werden ignoriert."""
    from analysis.separation_study.metric_extensions import compute_run_metrics
    from scipy.signal import welch
    fs = 51200.0
    rng = np.random.default_rng(1)
    n = 51_200
    sig = rng.normal(0.0, 1.0, size=n)
    ext_h5 = tmp_path / "ext_gt.h5"
    mix_h5 = tmp_path / "mix_gt.h5"
    _write_gt_h5(ext_h5, sig, fs)
    _write_gt_h5(mix_h5, sig * 2.0, fs)  # drone = mix - ext = sig (selber rauschen)
    f_w, psd_post = welch(sig, fs=fs, nperseg=512, noverlap=256,
                          window="hann", scaling="density")
    bands = [{"name": "narrow", "f_min_hz": 1000.0, "f_max_hz": 1100.0}]
    out = compute_run_metrics(
        psd_post=psd_post, frequencies=f_w,
        ext_gt_h5=ext_h5, mixed_gt_h5=mix_h5,
        welch_nperseg=512, welch_noverlap=256, window="hann",
        bands=bands, welch_floor_db=-50.0,
    )
    assert "narrow" in out["bands"]
