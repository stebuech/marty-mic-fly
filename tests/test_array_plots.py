import numpy as np


def test_plot_beam_maps_writes_html_2d_slice(tmp_path):
    from martymicfly.eval.array_plots import plot_beam_maps
    nx, ny, nz = 11, 11, 1
    pre = {"low": np.random.default_rng(0).random((nx, ny, nz)),
           "mid": np.random.default_rng(1).random((nx, ny, nz)),
           "high": np.random.default_rng(2).random((nx, ny, nz))}
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


def test_plot_beam_maps_writes_html_3d_box(tmp_path):
    from martymicfly.eval.array_plots import plot_beam_maps
    nx, ny, nz = 9, 9, 3
    rng = np.random.default_rng(0)
    pre = {"mid": rng.random((nx, ny, nz))}
    post = {"mid": rng.random((nx, ny, nz)) * 0.1}
    extent = 0.4
    rotor_pos = np.array([[0.0], [0.0], [0.0]])
    rotor_radii = np.array([0.10])
    mic_pos = np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
    target_xy = (0.0, 0.0)
    out = tmp_path / "beam3d.html"
    plot_beam_maps(pre, post, extent, rotor_pos, rotor_radii, mic_pos, target_xy, str(out))
    assert out.exists()
    text = out.read_text()
    # When nz > 1 we annotate "max over z" in the subplot titles.
    assert "max over z" in text


def test_plot_target_psd_writes_html(tmp_path):
    from martymicfly.eval.array_plots import plot_target_psd
    f = np.linspace(200, 4000, 200)
    pre = 10 ** (f / 4000)
    post = pre * 0.5
    out = tmp_path / "psd.html"
    plot_target_psd(f, pre, post, gt_psd=None, bpfs=[100.0, 200.0, 300.0], out_path=str(out))
    assert out.exists()


def test_plot_z_marginal_writes_html(tmp_path):
    from martymicfly.eval.array_plots import plot_z_marginal
    nx, ny, nz = 5, 5, 7
    rng = np.random.default_rng(0)
    pre = {"low": rng.random((nx, ny, nz)),
           "mid": rng.random((nx, ny, nz))}
    post = {k: v * 0.5 for k, v in pre.items()}
    z_values = np.linspace(-1.5, 0.1, nz)
    out = tmp_path / "zmarg.html"
    plot_z_marginal(pre, post, z_values, bpfs=[], out_path=str(out),
                    rotor_z_m=0.0, target_z_m=-1.5)
    assert out.exists()
    text = out.read_text()
    assert "z [m]" in text


def test_plot_z_marginal_handles_nz1(tmp_path):
    from martymicfly.eval.array_plots import plot_z_marginal
    nx, ny, nz = 5, 5, 1
    rng = np.random.default_rng(0)
    pre = {"mid": rng.random((nx, ny, nz))}
    post = {"mid": rng.random((nx, ny, nz)) * 0.1}
    z_values = np.array([0.0])
    out = tmp_path / "zmarg_nz1.html"
    plot_z_marginal(pre, post, z_values, bpfs=[], out_path=str(out))
    assert out.exists()


def test_plot_z_marginal_split_mode(tmp_path):
    """preserved_mask_3d switches to kept-vs-subtracted view."""
    from martymicfly.eval.array_plots import plot_z_marginal
    nx, ny, nz = 5, 5, 7
    rng = np.random.default_rng(0)
    pre = {"mid": rng.random((nx, ny, nz))}
    post = {"mid": rng.random((nx, ny, nz)) * 0.1}
    z_values = np.linspace(-1.0, 0.5, nz)
    # Preserved zone: z in [-0.5, 0.0], xy <= |0.1|
    preserved = np.zeros((nx, ny, nz), dtype=bool)
    z_keep = (z_values >= -0.5) & (z_values <= 0.0)
    xy_keep = np.zeros((nx, ny), dtype=bool)
    xy_keep[1:4, 1:4] = True
    preserved[..., z_keep] = xy_keep[..., None]
    out = tmp_path / "zmarg_split.html"
    plot_z_marginal(pre, post, z_values, bpfs=[], out_path=str(out),
                    rotor_z_m=0.0, target_z_m=-0.3,
                    preserved_mask_3d=preserved)
    assert out.exists()
    text = out.read_text()
    assert "kept" in text
    assert "subtracted" in text
    assert "kept vs subtracted" in text
