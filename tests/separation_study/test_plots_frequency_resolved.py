import numpy as np
from pathlib import Path


def test_frequency_resolved_writes_html(tmp_path):
    from analysis.separation_study.plots.frequency_resolved import (
        write_frequency_resolved,
    )
    from analysis.separation_study.freq_resolved_io import write_freq_resolved
    f = np.linspace(200.0, 6000.0, 100)
    payload = {
        "frequencies": f, "psd_post": np.ones(100) * 1e-5,
        "ext_gt": np.ones(100) * 1e-5, "d_ref": np.ones(100) * 1e-3,
    }
    h5_path = tmp_path / "metrics_freq.h5"
    write_freq_resolved(h5_path, payload)
    out = tmp_path / "fr.html"
    write_frequency_resolved(h5_path, out_path=out,
                             bands=[{"name":"mid","f_min_hz":500.0,"f_max_hz":2000.0}])
    assert out.exists()
