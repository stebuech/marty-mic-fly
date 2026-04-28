import h5py
import numpy as np


def test_write_and_round_trip_residual_csm(tmp_path):
    from martymicfly.io.residual_csm_h5 import write_residual_csm
    csm = np.zeros((5, 3, 3), dtype=np.complex128)
    csm[:] = np.eye(3) * (1.0 + 2.0j)
    freqs = np.linspace(200.0, 1000.0, 5)
    p = tmp_path / "r.h5"
    write_residual_csm(str(p), csm, freqs, attrs={"algorithm": "clean_sc"})
    with h5py.File(p, "r") as f:
        assert f.attrs["algorithm"] == "clean_sc"
        re = np.asarray(f["csm_real"][...])
        im = np.asarray(f["csm_imag"][...])
        fr = np.asarray(f["frequencies"][...])
    np.testing.assert_allclose(re + 1j * im, csm)
    np.testing.assert_allclose(fr, freqs)
