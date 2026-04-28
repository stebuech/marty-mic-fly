import numpy as np


def test_plot_beam_maps_writes_html(tmp_path):
    from martymicfly.eval.array_plots import plot_beam_maps
    nx, ny = 11, 11
    pre = {"low": np.random.default_rng(0).random((nx, ny)),
           "mid": np.random.default_rng(1).random((nx, ny)),
           "high": np.random.default_rng(2).random((nx, ny))}
    post = {k: v * 0.1 for k, v in pre.items()}
    extent = 0.5
    rotor_pos = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]])
    rotor_radii = np.array([0.10, 0.10])
    mic_pos = np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
    target_xy = (0.5, 0.0)
    out = tmp_path / "beam.html"
    plot_beam_maps(pre, post, extent, rotor_pos, rotor_radii, mic_pos, target_xy, str(out))
    assert out.exists()
    text = out.read_text()
    assert "Plotly" in text or "plotly" in text
    assert text.startswith("<")


def test_plot_target_psd_writes_html(tmp_path):
    from martymicfly.eval.array_plots import plot_target_psd
    f = np.linspace(200, 4000, 200)
    pre = 10 ** (f / 4000)
    post = pre * 0.5
    out = tmp_path / "psd.html"
    plot_target_psd(f, pre, post, gt_psd=None, bpfs=[100.0, 200.0, 300.0], out_path=str(out))
    assert out.exists()
