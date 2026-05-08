"""Tests for analysis.separation_study.pipeline_wrapper."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest


def test_augment_metrics_with_separation(tmp_path, monkeypatch):
    """Nach Pipeline-Run wird study_metrics.json mit Separation-Metriken
    angereichert (psd_post + ext/mix-GT-Pfade aus config)."""
    from analysis.separation_study import pipeline_wrapper as pw

    fs = 51200.0
    n = 51_200
    rng = np.random.default_rng(0)
    drone = rng.normal(0, 1.0, size=n)
    ext = rng.normal(0, 0.3, size=n)
    mix = drone + ext
    ext_h5 = tmp_path / "ext_gt.h5"
    mix_h5 = tmp_path / "mix_gt.h5"
    for path, sig in [(ext_h5, ext), (mix_h5, mix)]:
        with h5py.File(path, "w") as f:
            td = f.create_dataset("time_data", data=sig.reshape(-1, 1))
            td.attrs["sample_freq"] = float(fs)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({"bands": {}}))
    monkeypatch.setattr(pw, "_load_psd_post_from_run", lambda rd, mp, tp: (
        np.ones(50) * 1e-5,
        np.linspace(200.0, 6000.0, 50),
    ))

    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 2000.0}]
    pw.augment_metrics(
        run_dir=run_dir, ext_gt_h5=ext_h5, mixed_gt_h5=mix_h5,
        bands=bands, mic_positions=np.zeros((4, 3)), target_point=(0, 0, -1.5),
        welch_nperseg=512, welch_noverlap=256, window="hann", welch_floor_db=-50.0,
    )
    out = json.loads((run_dir / "study_metrics.json").read_text())
    assert "mid" in out["bands"]
    assert "spectrum_l1_db" in out["bands"]["mid"]
    assert (run_dir / "metrics_freq.h5").exists()
