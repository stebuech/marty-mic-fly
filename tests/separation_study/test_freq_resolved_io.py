import numpy as np
import h5py


def test_write_and_read_freq_resolved(tmp_path):
    from analysis.separation_study.freq_resolved_io import (
        write_freq_resolved, read_freq_resolved,
    )
    f = np.linspace(0.0, 6000.0, 100)
    payload = {
        "frequencies": f,
        "psd_post": np.random.default_rng(0).uniform(1e-6, 1e-3, size=100),
        "ext_gt":   np.random.default_rng(1).uniform(1e-6, 1e-3, size=100),
        "d_ref":    np.random.default_rng(2).uniform(1e-6, 1e-3, size=100),
    }
    out = tmp_path / "metrics_freq.h5"
    write_freq_resolved(out, payload)
    rt = read_freq_resolved(out)
    for k, v in payload.items():
        np.testing.assert_array_equal(rt[k], v)
    # excess + deficit werden beim Schreiben automatisch ergänzt
    assert "excess" in rt
    assert "deficit" in rt
