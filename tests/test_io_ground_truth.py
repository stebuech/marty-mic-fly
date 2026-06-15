import h5py
import numpy as np


def test_load_ground_truth_returns_signal_and_metadata(tmp_path):
    from martymicfly.io.ground_truth_h5 import load_ground_truth
    p = tmp_path / "gt.h5"
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1024).reshape(-1, 1)
    with h5py.File(p, "w") as f:
        f.attrs["sample_freq"] = 16000.0
        f.attrs["kind"] = "external_only"
        f["time_data"] = sig
        ext = f.create_group("external")
        ext.attrs["kind"] = "noise"
        ext.attrs["position_m"] = np.array([0.5, 0.0, -0.5])
    res = load_ground_truth(str(p))
    np.testing.assert_array_equal(res.signal, sig.ravel())
    assert res.sample_rate == 16000.0
    assert res.position_m == (0.5, 0.0, -0.5)
    assert res.kind == "noise"
