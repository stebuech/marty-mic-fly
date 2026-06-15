import numpy as np
import h5py


def test_subtraction_recovers_drone_audio(tmp_path):
    """drone_only_at_target = mixed_gt - ext_gt sample-genau."""
    from analysis.separation_study.drone_only_helper import (
        drone_only_at_target_from_files,
    )
    fs = 51200.0
    rng = np.random.default_rng(0)
    drone = rng.normal(0.0, 1.0, size=10_000).astype(np.float64)
    ext = rng.normal(0.0, 0.3, size=10_000).astype(np.float64)
    mix = drone + ext

    ext_h5 = tmp_path / "ext_gt.h5"
    mix_h5 = tmp_path / "mix_gt.h5"
    for path, sig in [(ext_h5, ext), (mix_h5, mix)]:
        with h5py.File(path, "w") as f:
            td = f.create_dataset("time_data", data=sig.reshape(-1, 1))
            td.attrs["sample_freq"] = float(fs)

    drone_recovered, fs_out = drone_only_at_target_from_files(mix_h5, ext_h5)
    assert fs_out == fs
    # float64 (a+b)-b != a in general (ULP-level differences); use allclose.
    np.testing.assert_allclose(drone_recovered.flatten(), drone, rtol=1e-13, atol=0)


def test_d_ref_matches_direct_welch(tmp_path):
    """welch(D_ref) ≈ welch(drone direkt) bei bekanntem drone-Signal."""
    from analysis.separation_study.drone_only_helper import (
        drone_only_at_target_from_files, welch_psd_at_target,
    )
    from scipy.signal import welch
    fs = 51200.0
    rng = np.random.default_rng(1)
    drone = rng.normal(0.0, 1.0, size=51_200).astype(np.float64)
    ext = rng.normal(0.0, 0.3, size=51_200).astype(np.float64)
    mix = drone + ext

    ext_h5 = tmp_path / "ext_gt.h5"
    mix_h5 = tmp_path / "mix_gt.h5"
    for path, sig in [(ext_h5, ext), (mix_h5, mix)]:
        with h5py.File(path, "w") as f:
            td = f.create_dataset("time_data", data=sig.reshape(-1, 1))
            td.attrs["sample_freq"] = float(fs)

    f_d, psd_d = welch_psd_at_target(
        mix_h5, ext_h5, nperseg=512, noverlap=256, window="hann",
    )
    f_ref, psd_ref = welch(drone, fs=fs, nperseg=512, noverlap=256,
                           window="hann", scaling="density")
    np.testing.assert_array_equal(f_d, f_ref)
    np.testing.assert_allclose(psd_d, psd_ref, rtol=1e-12)
