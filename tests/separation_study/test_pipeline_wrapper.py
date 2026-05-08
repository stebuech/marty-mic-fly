"""Tests for analysis.separation_study.pipeline_wrapper."""
from __future__ import annotations

import json

import h5py
import numpy as np


def test_augment_metrics_with_separation(tmp_path, monkeypatch):
    """Nach Pipeline-Run wird study_metrics.json mit Separation-Metriken
    angereichert. _drone_only_psd_at_target und _load_psd_post_from_run
    werden gemockt — wir testen nur die Augmentation-Pipe."""
    from analysis.separation_study import pipeline_wrapper as pw

    fs = 51200.0
    n = 51_200
    rng = np.random.default_rng(0)
    ext = rng.normal(0, 0.3, size=n)
    ext_h5 = tmp_path / "ext_gt.h5"
    with h5py.File(ext_h5, "w") as f:
        td = f.create_dataset("time_data", data=ext.reshape(-1, 1))
        td.attrs["sample_freq"] = float(fs)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({"bands": {}}))

    fake_freqs = np.linspace(200.0, 6000.0, 50)
    monkeypatch.setattr(
        pw, "_load_psd_post_from_run",
        lambda rd, mp, tp: (np.ones(50) * 1e-5, fake_freqs),
    )
    monkeypatch.setattr(
        pw, "_drone_only_psd_at_target",
        lambda **kw: (fake_freqs, np.ones(50) * 1e-3),
    )

    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 2000.0}]
    pw.augment_metrics(
        run_dir=run_dir, ext_gt_h5=ext_h5,
        ext_only_audio_h5=tmp_path / "ext_audio.h5",
        mixed_audio_h5=tmp_path / "mix_audio.h5",
        bands=bands, mic_positions=np.zeros((4, 3)), target_point=(0, 0, -1.5),
        welch_nperseg=512, welch_noverlap=256, window="hann",
        welch_floor_db=-50.0,
    )
    out = json.loads((run_dir / "study_metrics.json").read_text())
    assert "mid" in out["bands"]
    assert "spectrum_l1_db" in out["bands"]["mid"]
    assert (run_dir / "metrics_freq.h5").exists()
