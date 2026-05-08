# Mixed-Separation-Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the mixed-separation parameter study from `docs/superpowers/specs/2026-05-08-mixed-separation-study-design.md`: kompensationsfreie Metriken, Synth-Robustheits-Szenarien, study-runner mit Caching, Aggregation, Plots und phasenweise Studienausführung (Baseline → Phase 1 → Phase 2 → Empfehlung).

**Architecture:** Neues Modul `analysis/separation_study/` als Python-Package mit Hilfsmodulen (`metric_extensions`, `drone_only_helper`, `synth_scenarios`), CLI-Skripten (`study_runner.py`, `aggregate_results.py`) und Plot-Skripten. Bestehende Pipeline (`src/martymicfly/`) wird **nicht** modifiziert; Studien-Metriken laufen als Decorator über `compute_array_metrics`. Alle Studien-Runs landen unter `results/separation_study/`. Phasen werden durch separate YAML-Studienkonfigurationen unter `analysis/separation_study/studies/` getriggert.

**Tech Stack:** Python 3.13, numpy, scipy.signal.welch, h5py, pyyaml, plotly, pyarrow (für parquet), pydantic (Schema), pytest (TDD).

---

## Phase A: Foundation (Code + Unit-Tests)

### Task 1: Drone-only-Helper

**Files:**
- Create: `analysis/separation_study/__init__.py` (leer)
- Create: `analysis/separation_study/drone_only_helper.py`
- Create: `tests/separation_study/__init__.py` (leer)
- Test: `tests/separation_study/test_drone_only_helper.py`

- [ ] **Step 1: Write failing test for time-domain subtraction**

`tests/separation_study/test_drone_only_helper.py`:
```python
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
    np.testing.assert_array_equal(drone_recovered.flatten(), drone)


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
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/separation_study/test_drone_only_helper.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement drone_only_helper**

`analysis/separation_study/drone_only_helper.py`:
```python
"""Time-Domain-Subtraktion mixed - ext_only liefert drone-only-Signal,
ohne separate drone-only synth-Datei zu brauchen (Synth ist linear)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch


def drone_only_at_target_from_files(
    mixed_gt_h5: Path, ext_only_gt_h5: Path,
) -> tuple[np.ndarray, float]:
    """drone_only(t) = mixed_gt(t) - ext_only_gt(t).

    Beide Dateien müssen identische Sample-Rate, Länge und Anzahl Kanäle haben.
    Liefert (signal_(N, C), sample_rate).
    """
    with h5py.File(mixed_gt_h5, "r") as fm, h5py.File(ext_only_gt_h5, "r") as fe:
        mix = np.asarray(fm["time_data"][:], dtype=np.float64)
        ext = np.asarray(fe["time_data"][:], dtype=np.float64)
        fs_m = float(fm["time_data"].attrs["sample_freq"])
        fs_e = float(fe["time_data"].attrs["sample_freq"])
    if fs_m != fs_e:
        raise ValueError(f"sample_freq mismatch: mixed={fs_m}, ext={fs_e}")
    if mix.shape != ext.shape:
        raise ValueError(f"shape mismatch: mixed={mix.shape}, ext={ext.shape}")
    return mix - ext, fs_m


def welch_psd_at_target(
    mixed_gt_h5: Path, ext_only_gt_h5: Path,
    *, nperseg: int, noverlap: int, window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch-PSD des drone-only-Signals am Target. Channel 0 used."""
    drone, fs = drone_only_at_target_from_files(mixed_gt_h5, ext_only_gt_h5)
    f, psd = welch(
        drone[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        scaling="density",
    )
    return f, psd
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/separation_study/test_drone_only_helper.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/__init__.py \
        analysis/separation_study/drone_only_helper.py \
        tests/separation_study/__init__.py \
        tests/separation_study/test_drone_only_helper.py
git commit -m "feat(separation_study): drone-only helper via mixed - ext subtraction"
```

---

### Task 2: Per-Bin Excess/Deficit Decomposition

**Files:**
- Create: `analysis/separation_study/metric_extensions.py`
- Test: `tests/separation_study/test_metric_extensions.py`

- [ ] **Step 1: Write failing tests for excess/deficit + spectrum_l1_db + over_subtraction_db**

`tests/separation_study/test_metric_extensions.py`:
```python
import numpy as np


def test_decompose_excess_deficit_per_bin():
    """excess(f) = max(post - gt, 0); deficit(f) = max(gt - post, 0); je f nur einer."""
    from analysis.separation_study.metric_extensions import decompose_residual
    psd_post = np.array([1.0, 2.0, 0.5, 3.0])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    excess, deficit = decompose_residual(psd_post, ext_gt)
    np.testing.assert_array_equal(excess,  [0.0, 1.0, 0.0, 2.0])
    np.testing.assert_array_equal(deficit, [0.0, 0.0, 0.5, 0.0])


def test_spectrum_l1_db_no_compensation():
    """Σexcess = Σdeficit konstruiert → recovery_signed = 0, l1 ≫ 0."""
    from analysis.separation_study.metric_extensions import (
        decompose_residual, band_metrics,
    )
    psd_post = np.array([2.0, 0.5, 2.0, 0.5])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    delta_f = 1.0
    m = band_metrics(psd_post, ext_gt, d_ref=np.ones(4), delta_f=delta_f)
    p_post = psd_post.sum() * delta_f
    e_gt = ext_gt.sum() * delta_f
    expected_signed = 10 * np.log10(p_post / e_gt)
    assert abs(m["recovery_db_signed"] - expected_signed) < 1e-9
    assert m["spectrum_l1_db"] > -1.0
    assert m["compensation_flag"] is True


def test_spectrum_l1_db_perfect_match():
    """psd_post = ext_GT identisch → l1 → -inf, compensation_flag=False."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([1.0, 2.0, 3.0])
    m = band_metrics(psd_post, psd_post, d_ref=np.ones(3), delta_f=1.0)
    assert m["spectrum_l1_db"] == float("-inf")
    assert m["over_subtraction_db"] == float("-inf")
    assert m["drone_leakage_db_def2"] == float("-inf")
    assert m["compensation_flag"] is False


def test_over_subtraction_db_no_excess():
    """psd_post = α·ext_GT, α<1 ⇒ over_subtraction_db = 10·log10(1-α)."""
    from analysis.separation_study.metric_extensions import band_metrics
    alpha = 0.4
    ext_gt = np.array([2.0, 4.0, 6.0])
    m = band_metrics(alpha * ext_gt, ext_gt, d_ref=np.ones(3), delta_f=1.0)
    expected = 10 * np.log10(1.0 - alpha)
    assert abs(m["over_subtraction_db"] - expected) < 1e-9


def test_drone_leakage_def1_at_perfect_filter_equals_neg_snr():
    """psd_post = ext_GT, def1 = 10·log10(E_GT / D_unflt) = -SNR."""
    from analysis.separation_study.metric_extensions import band_metrics
    ext_gt = np.array([1.0, 2.0, 3.0])
    d_ref  = np.array([10.0, 20.0, 30.0])
    m = band_metrics(ext_gt.copy(), ext_gt, d_ref=d_ref, delta_f=1.0)
    expected = 10 * np.log10(ext_gt.sum() / d_ref.sum())
    assert abs(m["drone_leakage_db_def1"] - expected) < 1e-9


def test_recovery_minus_leakage_def1_eq_neg_snr_sanity():
    """Sanity: recovery_signed - leakage_def1 = -SNR_band, unabhängig vom Filter."""
    from analysis.separation_study.metric_extensions import band_metrics
    rng = np.random.default_rng(42)
    psd_post = rng.uniform(0.5, 5.0, size=10)
    ext_gt   = rng.uniform(0.1, 1.0, size=10)
    d_ref    = rng.uniform(2.0, 10.0, size=10)
    m = band_metrics(psd_post, ext_gt, d_ref=d_ref, delta_f=1.0)
    snr_band = 10 * np.log10(ext_gt.sum() / d_ref.sum())
    sanity = m["recovery_db_signed"] - m["drone_leakage_db_def1"] + snr_band
    assert abs(sanity) < 1e-9


def test_spectrum_rms_db_uniform_offset():
    """psd_post = β·ext_GT konstant ⇒ rms_db = |10·log10(β)|."""
    from analysis.separation_study.metric_extensions import band_metrics
    beta = 0.5
    ext_gt = np.array([1.0, 2.0, 3.0])
    m = band_metrics(beta * ext_gt, ext_gt, d_ref=np.ones(3), delta_f=1.0)
    expected = abs(10 * np.log10(beta))
    assert abs(m["spectrum_rms_db"] - expected) < 1e-9


def test_compensation_flag_above_floor():
    """Beide Komponenten oberhalb welch_floor + 3 dB → flag=true."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([2.0, 0.5, 2.0, 0.5])
    ext_gt   = np.array([1.0, 1.0, 1.0, 1.0])
    m = band_metrics(
        psd_post, ext_gt, d_ref=np.ones(4), delta_f=1.0, welch_floor_db=-30.0,
    )
    assert m["compensation_flag"] is True


def test_compensation_flag_below_floor():
    """Excess winzig (Welch-Streuung) → flag=false."""
    from analysis.separation_study.metric_extensions import band_metrics
    psd_post = np.array([1.001, 0.999, 1.0])
    ext_gt   = np.array([1.0,   1.0,   1.0])
    m = band_metrics(
        psd_post, ext_gt, d_ref=np.ones(3), delta_f=1.0, welch_floor_db=-10.0,
    )
    assert m["compensation_flag"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/separation_study/test_metric_extensions.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement metric_extensions**

`analysis/separation_study/metric_extensions.py`:
```python
"""Kompensationsfreie Separation-Metriken für mixed-Studie.

Per-Bin-Zerlegung: excess(f) = max(psd_post - ext_GT, 0),
                   deficit(f) = max(ext_GT - psd_post, 0).
Bandintegrale dieser Komponenten verhindern, dass Over- und Undersubtraktion
in benachbarten Frequenzen sich gegenseitig auslöschen."""
from __future__ import annotations

import numpy as np


def decompose_residual(
    psd_post: np.ndarray, ext_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-Bin Zerlegung in (excess, deficit). Beide ≥ 0, an jedem f genau einer ≠ 0."""
    diff = psd_post - ext_gt
    excess = np.maximum(diff, 0.0)
    deficit = np.maximum(-diff, 0.0)
    return excess, deficit


def _db(x: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(x)


def band_metrics(
    psd_post: np.ndarray,
    ext_gt: np.ndarray,
    *,
    d_ref: np.ndarray,
    delta_f: float,
    welch_floor_db: float = -50.0,
) -> dict:
    """Alle 6 Studien-Metriken + compensation_flag für ein bereits gemaskeds Band.

    Argumente sind 1D-Arrays gleicher Länge (alle nach band-mask gefiltert).
    `delta_f` ist der Welch-Bin-Abstand (für Σ·Δf-Integration).
    `welch_floor_db` setzt die Schwelle für `compensation_flag`.
    """
    assert psd_post.shape == ext_gt.shape == d_ref.shape, "shape mismatch"
    excess, deficit = decompose_residual(psd_post, ext_gt)
    e_excess = float(excess.sum() * delta_f)
    e_deficit = float(deficit.sum() * delta_f)
    e_gt = float(ext_gt.sum() * delta_f)
    p_post = float(psd_post.sum() * delta_f)
    d_unflt = float(d_ref.sum() * delta_f)

    spectrum_l1_db = _db((e_excess + e_deficit) / e_gt) if e_gt > 0 else float("-inf")
    over_subtraction_db = _db(e_deficit / e_gt) if e_gt > 0 else float("-inf")
    drone_leakage_db_def1 = _db(p_post / d_unflt) if d_unflt > 0 else float("inf")
    drone_leakage_db_def2 = _db(e_excess / d_unflt) if d_unflt > 0 else float("inf")
    recovery_db_signed = _db(p_post / e_gt) if e_gt > 0 else float("-inf")

    safe_post = np.maximum(psd_post, 1e-30)
    safe_gt = np.maximum(ext_gt, 1e-30)
    log_ratio_per_bin = 10.0 * np.log10(safe_post / safe_gt)
    spectrum_rms_db = float(np.sqrt(np.mean(log_ratio_per_bin ** 2)))

    excess_db = _db(e_excess / e_gt) if e_gt > 0 else float("-inf")
    compensation_flag = (
        over_subtraction_db > welch_floor_db + 3.0
        and excess_db > welch_floor_db + 3.0
    )

    return {
        "spectrum_l1_db": spectrum_l1_db,
        "over_subtraction_db": over_subtraction_db,
        "drone_leakage_db_def1": drone_leakage_db_def1,
        "drone_leakage_db_def2": drone_leakage_db_def2,
        "spectrum_rms_db": spectrum_rms_db,
        "recovery_db_signed": recovery_db_signed,
        "compensation_flag": bool(compensation_flag),
        "e_excess": e_excess,
        "e_deficit": e_deficit,
        "e_gt": e_gt,
        "d_unflt": d_unflt,
        "p_post": p_post,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/separation_study/test_metric_extensions.py -v
```
Expected: 9 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/metric_extensions.py \
        tests/separation_study/test_metric_extensions.py
git commit -m "feat(separation_study): per-bin excess/deficit metric framework"
```

---

### Task 3: Run-Level Metric Computation (steered psd_post + GT-Welch)

**Files:**
- Modify: `analysis/separation_study/metric_extensions.py` (add `compute_run_metrics`)
- Test: `tests/separation_study/test_run_metrics.py`

- [ ] **Step 1: Write failing test for compute_run_metrics**

`tests/separation_study/test_run_metrics.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/separation_study/test_run_metrics.py -v
```
Expected: FAIL — `compute_run_metrics` not found.

- [ ] **Step 3: Implement compute_run_metrics**

Append to `analysis/separation_study/metric_extensions.py`:
```python
import h5py
from pathlib import Path
from scipy.signal import welch as _scipy_welch


def _welch_signal_h5(path: Path, *, nperseg: int, noverlap: int, window: str,
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as f:
        td = np.asarray(f["time_data"][:], dtype=np.float64)
        fs = float(f["time_data"].attrs["sample_freq"])
    sig = td[:, 0] if td.ndim == 2 else td
    f_w, psd = _scipy_welch(sig, fs=fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, scaling="density")
    return f_w, psd, fs


def compute_run_metrics(
    *,
    psd_post: np.ndarray,
    frequencies: np.ndarray,
    ext_gt_h5: Path,
    mixed_gt_h5: Path,
    welch_nperseg: int,
    welch_noverlap: int,
    window: str,
    bands: list[dict],
    welch_floor_db: float,
) -> dict:
    """Erzeuge alle Studien-Metriken pro Band aus psd_post + GT-h5-Pfaden.

    `psd_post` und `frequencies` müssen konsistent sein (gleiche Welch-Parameter
    wie für ext_GT/D_ref). Aufrufer ist verantwortlich für `range_compensation_factor`-
    Anwendung auf psd_post."""
    f_ext, psd_ext_gt, fs_ext = _welch_signal_h5(
        ext_gt_h5, nperseg=welch_nperseg, noverlap=welch_noverlap, window=window,
    )
    f_mix, psd_mix_gt, fs_mix = _welch_signal_h5(
        mixed_gt_h5, nperseg=welch_nperseg, noverlap=welch_noverlap, window=window,
    )
    if not np.allclose(f_ext, f_mix):
        raise ValueError("freq grid mismatch ext_gt vs mixed_gt")
    psd_d_ref = psd_mix_gt - psd_ext_gt   # Linearität: PSD(mix) - PSD(ext) ≈ PSD(drone)
    # Für negative Werte aus Welch-Streuung clamped auf 0
    psd_d_ref = np.maximum(psd_d_ref, 0.0)

    # psd_post ist auf einem (i.d.R. groberen) Frequenzgitter — interpoliere
    # ext/d_ref auf das psd_post-Gitter.
    psd_ext_on_post = np.interp(frequencies, f_ext, psd_ext_gt)
    psd_d_on_post = np.interp(frequencies, f_ext, psd_d_ref)

    delta_f_post = float(frequencies[1] - frequencies[0])

    out: dict = {"bands": {}, "frequency_resolved": {
        "frequencies": frequencies,
        "psd_post": psd_post,
        "ext_gt": psd_ext_on_post,
        "d_ref": psd_d_on_post,
    }}
    for band in bands:
        name = band["name"]
        f_lo = float(band["f_min_hz"])
        f_hi = float(band["f_max_hz"])
        mask = (frequencies >= f_lo) & (frequencies <= f_hi)
        if not mask.any():
            continue
        m = band_metrics(
            psd_post=psd_post[mask],
            ext_gt=psd_ext_on_post[mask],
            d_ref=psd_d_on_post[mask],
            delta_f=delta_f_post,
            welch_floor_db=welch_floor_db,
        )
        out["bands"][name] = m
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/separation_study/test_run_metrics.py -v
uv run pytest tests/separation_study/ -v
```
Expected: alle PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/metric_extensions.py \
        tests/separation_study/test_run_metrics.py
git commit -m "feat(separation_study): compute_run_metrics with Welch-PSD interpolation"
```

---

### Task 4: Frequency-Resolved h5 Writer

**Files:**
- Create: `analysis/separation_study/freq_resolved_io.py`
- Test: `tests/separation_study/test_freq_resolved_io.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_freq_resolved_io.py`:
```python
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
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_freq_resolved_io.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement freq_resolved_io**

`analysis/separation_study/freq_resolved_io.py`:
```python
"""Schreib- und Lese-Helfer für die frequency-resolved h5-Beilage einer
Studien-Run."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from analysis.separation_study.metric_extensions import decompose_residual


def write_freq_resolved(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excess, deficit = decompose_residual(payload["psd_post"], payload["ext_gt"])
    with h5py.File(out_path, "w") as f:
        for k in ("frequencies", "psd_post", "ext_gt", "d_ref"):
            f.create_dataset(k, data=np.asarray(payload[k], dtype=np.float64))
        f.create_dataset("excess", data=excess)
        f.create_dataset("deficit", data=deficit)


def read_freq_resolved(in_path: Path) -> dict:
    with h5py.File(in_path, "r") as f:
        return {k: np.asarray(f[k][:], dtype=np.float64) for k in f.keys()}
```

- [ ] **Step 4: Run test to verify pass**

```
uv run pytest tests/separation_study/test_freq_resolved_io.py -v
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/freq_resolved_io.py \
        tests/separation_study/test_freq_resolved_io.py
git commit -m "feat(separation_study): freq-resolved h5 IO helper"
```

---

## Phase B: Synth-Szenarien

### Task 5: Compose-Config Generator für S0-S3

**Files:**
- Create: `analysis/separation_study/synth_scenarios.py`
- Test: `tests/separation_study/test_synth_scenarios.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_synth_scenarios.py`:
```python
from pathlib import Path

import yaml


def test_make_compose_config_S0_uses_existing_paths(tmp_path):
    """S0 ist die existierende Default-Synth-Datei — Generator gibt None zurück
    (kein neuer compose-Lauf)."""
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S0", base_drone_artifact="/dev/null",
        mic_geom_xml="/dev/null", out_synth_h5="...", out_gt_h5="...",
    )
    assert cfg is None


def test_make_compose_config_S1_offaxis_target():
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S1", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    assert tuple(cfg["external"]["position_m"]) == (0.3, 0.0, -1.5)
    assert cfg["external"]["amplitude_db"] == 0.0
    assert cfg["include_drone"] is True


def test_make_compose_config_S2_low_snr_amplitude():
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S2", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    # amp×0.3 entspricht 20·log10(0.3) ≈ -10.46 dB
    assert abs(cfg["external"]["amplitude_db"] - (20.0 * 0.3010299956)) < 1e-6 \
        or abs(cfg["external"]["amplitude_db"] - (-10.4576)) < 0.01


def test_make_compose_config_S3_far_distance():
    from analysis.separation_study.synth_scenarios import make_compose_config
    cfg = make_compose_config(
        scenario="S3", base_drone_artifact="/a.h5",
        mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
    )
    assert tuple(cfg["external"]["position_m"]) == (0.0, 0.0, -3.0)


def test_make_compose_config_unknown_scenario_raises():
    from analysis.separation_study.synth_scenarios import make_compose_config
    import pytest
    with pytest.raises(ValueError, match="unknown scenario"):
        make_compose_config(
            scenario="S99", base_drone_artifact="/a.h5",
            mic_geom_xml="/m.xml", out_synth_h5="/o.h5", out_gt_h5="/o_gt.h5",
        )


def test_ensure_scenarios_caches(tmp_path, monkeypatch):
    """Bestehende h5 werden nicht neu erzeugt."""
    from analysis.separation_study import synth_scenarios as ss
    out_synth = tmp_path / "synth.h5"
    out_gt = tmp_path / "gt.h5"
    out_synth.write_bytes(b"0")
    out_gt.write_bytes(b"0")
    called = {"compose": False}
    def fake_compose(*a, **kw):
        called["compose"] = True
    monkeypatch.setattr(ss, "_run_compose", fake_compose)
    ss.ensure_scenario(
        scenario="S1", base_drone_artifact="/a.h5", mic_geom_xml="/m.xml",
        out_synth_h5=out_synth, out_gt_h5=out_gt,
    )
    assert called["compose"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/separation_study/test_synth_scenarios.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement synth_scenarios**

`analysis/separation_study/synth_scenarios.py`:
```python
"""Robustheits-Szenarien S0..S3 als compose-config-Generatoren.

S0: Baseline (existing synth files, kein neuer compose-Lauf nötig)
S1: target = (0.3, 0, -1.5) — laterale Off-axis-Position
S2: source amplitude × 0.3 (≈ -10.46 dB) — niedriges SNR
S3: target = (0, 0, -3.0) — größere Quelldistanz"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


SCENARIO_PARAMS = {
    "S0": None,  # Baseline = existing files
    "S1": {"position_m": (0.3, 0.0, -1.5), "amplitude_db": 0.0},
    "S2": {"position_m": (0.0, 0.0, -1.5),
           "amplitude_db": float(20.0 * np.log10(0.3))},  # ~ -10.46 dB
    "S3": {"position_m": (0.0, 0.0, -3.0), "amplitude_db": 0.0},
}


def make_compose_config(
    *,
    scenario: str,
    base_drone_artifact: str,
    mic_geom_xml: str,
    out_synth_h5: str,
    out_gt_h5: str,
    seed: int = 0,
) -> Optional[dict]:
    """Liefert eine compose-CLI-kompatible YAML-config-dict, oder None falls
    Szenario das Baseline ist (existierende Files weiterverwenden)."""
    if scenario not in SCENARIO_PARAMS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    params = SCENARIO_PARAMS[scenario]
    if params is None:
        return None
    return {
        "input": {
            "drone_source_artifact_h5": base_drone_artifact,
            "mic_geom_xml": mic_geom_xml,
        },
        "external": {
            "kind": "noise",
            "position_m": list(params["position_m"]),
            "amplitude_db": float(params["amplitude_db"]),
            "duration_s": None,
            "seed": seed,
        },
        "include_drone": True,
        "output": {
            "synth_h5": str(out_synth_h5),
            "ground_truth_h5": str(out_gt_h5),
        },
    }


def _run_compose(cfg_dict: dict) -> None:
    """Schreibt cfg_dict in temp YAML und ruft martymicfly.synth.cli.compose."""
    import tempfile
    import subprocess
    import yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump(cfg_dict, f)
        cfg_path = f.name
    subprocess.run(
        ["uv", "run", "python", "-m", "martymicfly.synth.cli.compose",
         "--config", cfg_path],
        check=True,
    )


def ensure_scenario(
    *,
    scenario: str,
    base_drone_artifact: str,
    mic_geom_xml: str,
    out_synth_h5: Path,
    out_gt_h5: Path,
    seed: int = 0,
) -> None:
    """Generiert Szenario-h5-Dateien falls noch nicht vorhanden (caching).

    Für S0 ein No-Op (Baseline-Files existieren bereits). Für S1..S3 wird
    compose über die CLI gerufen wenn output-h5 fehlen."""
    if Path(out_synth_h5).exists() and Path(out_gt_h5).exists():
        return
    cfg = make_compose_config(
        scenario=scenario,
        base_drone_artifact=base_drone_artifact,
        mic_geom_xml=mic_geom_xml,
        out_synth_h5=str(out_synth_h5),
        out_gt_h5=str(out_gt_h5),
        seed=seed,
    )
    if cfg is None:
        return
    _run_compose(cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/separation_study/test_synth_scenarios.py -v
```
Expected: alle PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/synth_scenarios.py \
        tests/separation_study/test_synth_scenarios.py
git commit -m "feat(separation_study): S0-S3 scenario compose generator + caching"
```

---

## Phase C: Study Runner & Aggregator

### Task 6: YAML Override Patcher (Dotted-Key)

**Files:**
- Create: `analysis/separation_study/yaml_override.py`
- Test: `tests/separation_study/test_yaml_override.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_yaml_override.py`:
```python
def test_dotted_key_simple():
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"a": {"b": {"c": 1}}}
    out = apply_overrides(base, {"a.b.c": 42})
    assert out == {"a": {"b": {"c": 42}}}


def test_dotted_key_does_not_mutate_input():
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"a": {"b": 1}}
    out = apply_overrides(base, {"a.b": 2})
    assert base["a"]["b"] == 1


def test_stage_index_path():
    """stages[0].csm.nperseg patcht den ersten Stage."""
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"stages": [{"kind": "array_filter", "csm": {"nperseg": 512}}]}
    out = apply_overrides(base, {"stages[0].csm.nperseg": 1024})
    assert out["stages"][0]["csm"]["nperseg"] == 1024


def test_missing_path_raises():
    from analysis.separation_study.yaml_override import apply_overrides
    import pytest
    with pytest.raises(KeyError):
        apply_overrides({"a": 1}, {"b.c": 5})


def test_array_filter_stage_finds_first():
    """Convenience: 'array_filter.csm.nperseg' findet den ersten array_filter
    stage automatisch (Notch-Stage in mixed-configs überspringen)."""
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"stages": [
        {"kind": "notch", "pole_radius": 0.999},
        {"kind": "array_filter", "csm": {"nperseg": 512}},
    ]}
    out = apply_overrides(base, {"array_filter.csm.nperseg": 256})
    assert out["stages"][1]["csm"]["nperseg"] == 256
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_yaml_override.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement yaml_override**

`analysis/separation_study/yaml_override.py`:
```python
"""Dotted-key Patches auf Pipeline-YAML-dicts.

Unterstützte Pfad-Syntax:
  - "a.b.c"            normale dict-Verschachtelung
  - "a[0].b"           Liste mit numerischem Index
  - "array_filter.x.y" Convenience: erster Stage mit kind=array_filter

Mutationsfrei: gibt eine deep-copy zurück."""
from __future__ import annotations

import copy
import re
from typing import Any


_INDEX_RE = re.compile(r"^([^[]+)\[(\d+)\]$")


def _find_array_filter_index(stages: list) -> int:
    for i, s in enumerate(stages):
        if isinstance(s, dict) and s.get("kind") == "array_filter":
            return i
    raise KeyError("no stage with kind='array_filter' found")


def _navigate(node: Any, parts: list[str]) -> tuple[Any, str]:
    """Walk parts[:-1], return (parent, last_key)."""
    for p in parts[:-1]:
        m = _INDEX_RE.match(p)
        if m:
            key, idx = m.group(1), int(m.group(2))
            node = node[key][idx]
        else:
            node = node[p]
    return node, parts[-1]


def apply_overrides(base: dict, overrides: dict[str, Any]) -> dict:
    """Apply each (dotted_key, value) override to a deep-copy of base."""
    out = copy.deepcopy(base)
    for path, value in overrides.items():
        if path.startswith("array_filter."):
            af_idx = _find_array_filter_index(out["stages"])
            real_path = f"stages[{af_idx}]." + path[len("array_filter."):]
        else:
            real_path = path
        parts = real_path.split(".")
        try:
            parent, last = _navigate(out, parts)
            m = _INDEX_RE.match(last)
            if m:
                key, idx = m.group(1), int(m.group(2))
                parent[key][idx] = value
            else:
                if isinstance(parent, dict) and last not in parent:
                    raise KeyError(f"path not found: {path}")
                parent[last] = value
        except (KeyError, TypeError, IndexError) as exc:
            raise KeyError(f"failed to apply override {path}: {exc}") from exc
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/separation_study/test_yaml_override.py -v
```
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/yaml_override.py \
        tests/separation_study/test_yaml_override.py
git commit -m "feat(separation_study): dotted-key YAML override patcher"
```

---

### Task 7: Run-Cache via Config-Hash

**Files:**
- Create: `analysis/separation_study/run_cache.py`
- Test: `tests/separation_study/test_run_cache.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_run_cache.py`:
```python
def test_run_hash_deterministic():
    from analysis.separation_study.run_cache import config_hash
    cfg = {"a": 1, "b": [1, 2, 3]}
    assert config_hash(cfg) == config_hash(cfg)


def test_run_hash_changes_with_value():
    from analysis.separation_study.run_cache import config_hash
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_hash_dict_order_independent():
    from analysis.separation_study.run_cache import config_hash
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_run_already_complete_detects_metrics_json(tmp_path):
    from analysis.separation_study.run_cache import is_run_complete
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not is_run_complete(run_dir)
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "study_metrics.json").write_text("{}")
    assert is_run_complete(run_dir)
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_run_cache.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement run_cache**

`analysis/separation_study/run_cache.py`:
```python
"""Run-Caching: stabiler Hash über config-dict + Detection ob Run schon fertig."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def config_hash(cfg: Any) -> str:
    """Deterministischer 8-stelliger SHA1 über sortiertes JSON."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def is_run_complete(run_dir: Path) -> bool:
    """True wenn metrics.json und study_metrics.json beide existieren."""
    return ((run_dir / "metrics.json").is_file()
            and (run_dir / "study_metrics.json").is_file())
```

- [ ] **Step 4: Run tests to verify pass**

```
uv run pytest tests/separation_study/test_run_cache.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/run_cache.py \
        tests/separation_study/test_run_cache.py
git commit -m "feat(separation_study): run-cache via config-hash"
```

---

### Task 8: Pipeline-Wrapper (Subprocess + Metric-Augmentation)

**Files:**
- Create: `analysis/separation_study/pipeline_wrapper.py`
- Test: `tests/separation_study/test_pipeline_wrapper.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_pipeline_wrapper.py`:
```python
def test_augment_metrics_with_separation(tmp_path, monkeypatch):
    """Nach Pipeline-Run wird study_metrics.json mit Separation-Metriken
    angereichert (psd_post + ext/mix-GT-Pfade aus config)."""
    import json
    import numpy as np
    import h5py
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

    # Fake-residual_csm.h5 schreiben (genug für steer_to_psd)
    # Stattdessen: monkeypatch psd_post + freqs direkt einfließen
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
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_pipeline_wrapper.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement pipeline_wrapper**

`analysis/separation_study/pipeline_wrapper.py`:
```python
"""Wrapper um martymicfly.cli.run_pipeline + Studien-Metrik-Augmentation.

run_pipeline_with_overrides: schreibt Override-config, ruft die Pipeline,
ergänzt anschließend study_metrics.json + metrics_freq.h5 aus residual_csm.h5
und den GT-h5-Pfaden."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml

from analysis.separation_study.metric_extensions import compute_run_metrics
from analysis.separation_study.freq_resolved_io import write_freq_resolved
from analysis.separation_study.yaml_override import apply_overrides


def _load_psd_post_from_run(
    run_dir: Path, mic_positions: np.ndarray, target_point: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    from martymicfly.processing.steering import (
        steer_to_psd, range_compensation_factor,
    )
    csm_path = run_dir / "residual_csm.h5"
    with h5py.File(csm_path, "r") as f:
        csm = np.asarray(f["csm"][:], dtype=np.complex128)
        freqs = np.asarray(f["frequencies"][:], dtype=np.float64)
    psd_post = steer_to_psd(csm, freqs, mic_positions, target_point)
    cal = range_compensation_factor(mic_positions, target_point)
    return psd_post * cal, freqs


def augment_metrics(
    *,
    run_dir: Path,
    ext_gt_h5: Path,
    mixed_gt_h5: Path,
    bands: list[dict],
    mic_positions: np.ndarray,
    target_point: tuple[float, float, float],
    welch_nperseg: int,
    welch_noverlap: int,
    window: str,
    welch_floor_db: float,
) -> dict:
    psd_post, freqs = _load_psd_post_from_run(run_dir, mic_positions, target_point)
    out = compute_run_metrics(
        psd_post=psd_post, frequencies=freqs,
        ext_gt_h5=ext_gt_h5, mixed_gt_h5=mixed_gt_h5,
        welch_nperseg=welch_nperseg, welch_noverlap=welch_noverlap, window=window,
        bands=bands, welch_floor_db=welch_floor_db,
    )
    fr = out.pop("frequency_resolved")
    write_freq_resolved(run_dir / "metrics_freq.h5", fr)
    (run_dir / "study_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_pipeline_with_overrides(
    *,
    base_config_path: Path,
    overrides: dict,
    output_dir: Path,
    log_level: str = "INFO",
) -> Path:
    """Patcht base_config mit overrides, schreibt temp-yaml, ruft run_pipeline.
    Liefert das resultierende run_dir Pfad."""
    base_cfg = yaml.safe_load(base_config_path.read_text())
    cfg = apply_overrides(base_cfg, overrides)
    cfg["output"]["dir"] = str(output_dir)  # explicit, no template
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name
    subprocess.run(
        ["uv", "run", "python", "-m", "martymicfly.cli.run_pipeline",
         "--config", tmp_path, "--output-dir", str(output_dir),
         "--log-level", log_level],
        check=True,
    )
    return output_dir
```

- [ ] **Step 4: Run test to verify pass**

```
uv run pytest tests/separation_study/test_pipeline_wrapper.py -v
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/pipeline_wrapper.py \
        tests/separation_study/test_pipeline_wrapper.py
git commit -m "feat(separation_study): pipeline-wrapper + metric augmentation"
```

---

### Task 9: Study-YAML Schema + Run-Plan-Generator

**Files:**
- Create: `analysis/separation_study/study_runner.py`
- Create: `analysis/separation_study/studies/baseline.yaml`
- Test: `tests/separation_study/test_study_runner.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_study_runner.py`:
```python
import yaml
from pathlib import Path


def test_expand_run_plan_baseline(tmp_path):
    """baseline.yaml: 8 bestehende Configs × S0 = 8 Runs (keine Achsen-Sweeps)."""
    from analysis.separation_study.study_runner import expand_run_plan
    study_yaml = {
        "phase": "baseline",
        "configs": [
            "configs/pipeline_external_only_doa_target_cone.yaml",
            "configs/pipeline_mixed_doa_target_cone.yaml",
        ],
        "scenarios": ["S0"],
        "axes": {},
        "output_root": str(tmp_path / "out"),
    }
    plan = expand_run_plan(study_yaml)
    assert len(plan) == 2
    for r in plan:
        assert r["scenario"] == "S0"
        assert r["overrides"] == {}


def test_expand_run_plan_phase1_axes(tmp_path):
    """Phase 1: 1 Config × 2 Achsen × 2 Punkte + 1 Baseline × 1 Szenario = 5 Runs."""
    from analysis.separation_study.study_runner import expand_run_plan
    study_yaml = {
        "phase": "phase1",
        "configs": ["configs/pipeline_mixed_doa_target_cone.yaml"],
        "scenarios": ["S0"],
        "axes": {
            "array_filter.csm.nperseg": [256, 1024],
            "array_filter.clean_sc.damp": [0.3, 0.9],
        },
        "output_root": str(tmp_path / "out"),
    }
    plan = expand_run_plan(study_yaml)
    # 1 baseline + (2 × 2 axis-points) = 5
    assert len(plan) == 5
    baselines = [r for r in plan if not r["overrides"]]
    assert len(baselines) == 1


def test_expand_includes_unique_run_ids(tmp_path):
    from analysis.separation_study.study_runner import expand_run_plan
    plan = expand_run_plan({
        "phase": "baseline",
        "configs": ["configs/a.yaml", "configs/b.yaml"],
        "scenarios": ["S0", "S1"],
        "axes": {},
        "output_root": str(tmp_path / "out"),
    })
    ids = [r["run_id"] for r in plan]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/separation_study/test_study_runner.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement study_runner**

`analysis/separation_study/study_runner.py`:
```python
"""Study runner — expandiert Studie-YAML in Liste von Run-Specs und führt sie aus.

Schema einer Studie:
  phase: baseline | phase1 | phase2
  configs: list of base-config yaml paths
  scenarios: list of S0..S3
  axes: dict[dotted_key, list_of_values]   # leer für reine config × scenario sweeps
  output_root: results-dir

Pro Achse wird ein einzelner-Knopf-Sweep generiert (alle anderen Achsen bleiben
auf base-config-Defaults). Plus 1 Baseline-Run pro (config, scenario) ohne
Override."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from analysis.separation_study.run_cache import config_hash, is_run_complete

log = logging.getLogger("separation_study.runner")


def _slug(value: Any) -> str:
    return str(value).replace("/", "_").replace(".", "p").replace(" ", "_")


def expand_run_plan(study: dict) -> list[dict]:
    """Pro (config, scenario) ein Baseline-Run + pro (axis, value) ein Override-Run."""
    plan: list[dict] = []
    output_root = Path(study["output_root"])
    for cfg_path in study["configs"]:
        cfg_name = Path(cfg_path).stem
        for scenario in study["scenarios"]:
            plan.append({
                "config": cfg_path,
                "scenario": scenario,
                "overrides": {},
                "run_id": f"{cfg_name}__{scenario}__baseline",
                "output_dir": str(output_root /
                                  f"{cfg_name}__{scenario}__baseline"),
            })
            for axis, values in study.get("axes", {}).items():
                for v in values:
                    overrides = {axis: v}
                    h = config_hash(overrides)
                    rid = f"{cfg_name}__{scenario}__{_slug(axis)}_{_slug(v)}_{h}"
                    plan.append({
                        "config": cfg_path,
                        "scenario": scenario,
                        "overrides": overrides,
                        "run_id": rid,
                        "output_dir": str(output_root / rid),
                    })
    return plan


def execute_plan(plan: list[dict], scenario_paths: dict, *,
                 force: bool = False) -> list[dict]:
    """Führt jeden Run-Spec aus, überspringt bereits komplette Runs.

    `scenario_paths` mappt scenario → {audio_h5, ground_truth_h5, drone_artifact, mic_geom_xml}.
    """
    from analysis.separation_study.pipeline_wrapper import (
        augment_metrics, run_pipeline_with_overrides,
    )
    from analysis.separation_study.synth_scenarios import ensure_scenario
    import numpy as np
    from martymicfly.io.mic_geom import load_mic_geom_xml

    results: list[dict] = []
    for spec in plan:
        run_dir = Path(spec["output_dir"])
        if not force and is_run_complete(run_dir):
            log.info("skip %s — already complete", spec["run_id"])
            results.append({**spec, "status": "skipped"})
            continue

        scenario = spec["scenario"]
        sp = scenario_paths[scenario]
        ensure_scenario(
            scenario=scenario,
            base_drone_artifact=sp["drone_artifact"],
            mic_geom_xml=sp["mic_geom_xml"],
            out_synth_h5=Path(sp["audio_h5"]),
            out_gt_h5=Path(sp["ground_truth_h5"]),
        )

        cfg_path = Path(spec["config"])
        base_cfg = yaml.safe_load(cfg_path.read_text())
        overrides = dict(spec["overrides"])
        # Patch input paths to scenario-specific files
        overrides.setdefault("input.audio_h5", sp["audio_h5"])
        overrides.setdefault("input.ground_truth_h5", sp["ground_truth_h5"])

        log.info("run %s overrides=%s", spec["run_id"], overrides)
        run_pipeline_with_overrides(
            base_config_path=cfg_path,
            overrides=overrides,
            output_dir=run_dir,
        )

        # Mixed-GT-Pfad: aus scenario-paths-Mapping; ext-only-GT als zweiter Pfad
        mixed_gt = Path(sp["ground_truth_h5"])
        ext_gt = Path(scenario_paths[scenario]["ext_only_gt_h5"])
        bands_cfg = yaml.safe_load(cfg_path.read_text())["stages"]
        # Erste array_filter-Stage finden
        af_stage = next(s for s in bands_cfg if s.get("kind") == "array_filter")
        bands = af_stage["bands"]
        target_point = tuple(af_stage["target_point_m"])
        nperseg = int(af_stage["csm"]["nperseg"])
        noverlap = int(af_stage["csm"]["noverlap"])
        window = af_stage["csm"]["window"]
        mic_positions = load_mic_geom_xml(sp["mic_geom_xml"])
        augment_metrics(
            run_dir=run_dir, ext_gt_h5=ext_gt, mixed_gt_h5=mixed_gt,
            bands=bands, mic_positions=mic_positions, target_point=target_point,
            welch_nperseg=nperseg, welch_noverlap=noverlap, window=window,
            welch_floor_db=-50.0,
        )
        results.append({**spec, "status": "ok"})
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--scenario-paths", type=Path, required=True,
                   help="YAML mapping scenario → file paths (drone_artifact, mic_geom_xml, audio_h5, ground_truth_h5, ext_only_gt_h5)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    study = yaml.safe_load(args.study.read_text())
    scenario_paths = yaml.safe_load(args.scenario_paths.read_text())
    plan = expand_run_plan(study)
    log.info("expanded %d runs", len(plan))
    results = execute_plan(plan, scenario_paths, force=args.force)
    summary = {"total": len(results),
               "ok": sum(1 for r in results if r["status"] == "ok"),
               "skipped": sum(1 for r in results if r["status"] == "skipped")}
    log.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`analysis/separation_study/studies/baseline.yaml`:
```yaml
phase: baseline
output_root: results/separation_study/baseline
scenarios: [S0]
axes: {}
configs:
  - configs/pipeline_external_only_doa_rotor_cone.yaml
  - configs/pipeline_external_only_doa_drone_cone.yaml
  - configs/pipeline_external_only_doa_target_cone.yaml
  - configs/pipeline_external_only_nnls.yaml
  - configs/pipeline_mixed_doa_rotor_cone.yaml
  - configs/pipeline_mixed_doa_drone_cone.yaml
  - configs/pipeline_mixed_doa_target_cone.yaml
  - configs/pipeline_mixed_nnls.yaml
```

- [ ] **Step 4: Run tests to verify pass**

```
uv run pytest tests/separation_study/test_study_runner.py -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/study_runner.py \
        analysis/separation_study/studies/baseline.yaml \
        tests/separation_study/test_study_runner.py
git commit -m "feat(separation_study): study-runner with run-plan expansion + caching"
```

---

### Task 10: Phase-1-YAML + Aggregate-Results

**Files:**
- Create: `analysis/separation_study/studies/phase1_sensitivity.yaml`
- Create: `analysis/separation_study/aggregate_results.py`
- Test: `tests/separation_study/test_aggregate_results.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_aggregate_results.py`:
```python
import json
from pathlib import Path


def test_aggregate_long_format(tmp_path):
    from analysis.separation_study.aggregate_results import aggregate_dir
    # Fake study_metrics.json in 2 run_dirs
    for name, val in [("run_a", 1.0), ("run_b", 2.0)]:
        d = tmp_path / name
        d.mkdir()
        (d / "study_metrics.json").write_text(json.dumps({
            "bands": {"mid": {"spectrum_l1_db": val,
                              "drone_leakage_db_def2": val * 2,
                              "compensation_flag": False}},
        }))
        (d / "run_meta.json").write_text(json.dumps({
            "run_id": name, "config": "configs/test.yaml",
            "scenario": "S0", "overrides": {}, "method": "test",
        }))

    df = aggregate_dir(tmp_path)
    assert {"run_id", "method", "scenario", "band", "metric", "value"} \
        <= set(df.columns)
    assert (df["metric"] == "spectrum_l1_db").any()


def test_consistency_check_warns(tmp_path, caplog):
    from analysis.separation_study.aggregate_results import (
        aggregate_dir, check_consistency,
    )
    d = tmp_path / "run_x"
    d.mkdir()
    # Verletzung: recovery_signed - leakage_def1 sollte = -SNR ergeben.
    (d / "study_metrics.json").write_text(json.dumps({
        "bands": {"mid": {
            "spectrum_l1_db": -10.0, "over_subtraction_db": -20.0,
            "drone_leakage_db_def1": -5.0, "drone_leakage_db_def2": -25.0,
            "spectrum_rms_db": 1.0, "recovery_db_signed": -5.0,
            "e_excess": 0.001, "e_deficit": 0.0001, "e_gt": 1.0,
            "d_unflt": 100.0, "p_post": 0.5,
            "compensation_flag": False,
        }},
    }))
    (d / "run_meta.json").write_text(json.dumps({
        "run_id": "run_x", "config": "x.yaml", "scenario": "S0",
        "overrides": {}, "method": "test",
    }))
    df = aggregate_dir(tmp_path)
    with caplog.at_level("WARNING"):
        check_consistency(df)
    # The expected SNR_band = 10·log10(e_gt/d_unflt) = 10·log10(0.01) = -20 dB
    # recovery - def1 = -5 - (-5) = 0 ≠ -20 → warning erwartet.
    assert any("consistency" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_aggregate_results.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement aggregate_results + run_meta dump in study_runner**

Modify `analysis/separation_study/study_runner.py` — at end of `execute_plan` per-spec block, before `results.append`, add:
```python
        (run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": spec["run_id"],
            "config": spec["config"],
            "scenario": spec["scenario"],
            "overrides": spec["overrides"],
            "method": Path(spec["config"]).stem,
        }, indent=2))
```

`analysis/separation_study/aggregate_results.py`:
```python
"""Aggregiert study_metrics.json + run_meta.json über result-dirs zu long-format
parquet."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("separation_study.aggregate")


SCALAR_METRICS = (
    "spectrum_l1_db", "over_subtraction_db",
    "drone_leakage_db_def1", "drone_leakage_db_def2",
    "spectrum_rms_db", "recovery_db_signed",
    "e_excess", "e_deficit", "e_gt", "d_unflt", "p_post",
)


def aggregate_dir(root: Path) -> pd.DataFrame:
    rows = []
    for sm in sorted(Path(root).rglob("study_metrics.json")):
        meta_path = sm.parent / "run_meta.json"
        if not meta_path.exists():
            log.warning("no run_meta.json next to %s — skipped", sm)
            continue
        meta = json.loads(meta_path.read_text())
        sm_data = json.loads(sm.read_text())
        for band, vals in sm_data.get("bands", {}).items():
            for m in SCALAR_METRICS:
                if m in vals:
                    rows.append({
                        **{k: meta.get(k) for k in
                           ("run_id", "method", "scenario", "config")},
                        "overrides": json.dumps(meta.get("overrides", {})),
                        "band": band,
                        "metric": m,
                        "value": float(vals[m]),
                    })
            if "compensation_flag" in vals:
                rows.append({
                    **{k: meta.get(k) for k in
                       ("run_id", "method", "scenario", "config")},
                    "overrides": json.dumps(meta.get("overrides", {})),
                    "band": band,
                    "metric": "compensation_flag",
                    "value": float(bool(vals["compensation_flag"])),
                })
    return pd.DataFrame(rows)


def check_consistency(df: pd.DataFrame) -> None:
    """Verifiziert: recovery_signed - drone_leakage_def1 ≈ -SNR_band (±0.5 dB)."""
    import numpy as np
    if df.empty:
        return
    pivot = df.pivot_table(index=["run_id", "band"],
                          columns="metric", values="value", aggfunc="first")
    needed = {"recovery_db_signed", "drone_leakage_db_def1", "e_gt", "d_unflt"}
    if not needed.issubset(pivot.columns):
        return
    snr = 10.0 * np.log10(pivot["e_gt"] / pivot["d_unflt"])
    sanity = pivot["recovery_db_signed"] - pivot["drone_leakage_db_def1"] + snr
    bad = sanity[abs(sanity) > 0.5]
    if not bad.empty:
        log.warning(
            "consistency check failed for %d (run, band) entries: max |Δ|=%.3f dB",
            len(bad), float(bad.abs().max()),
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = aggregate_dir(args.root)
    check_consistency(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    log.info("aggregated %d rows → %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`analysis/separation_study/studies/phase1_sensitivity.yaml`:
```yaml
phase: phase1
output_root: results/separation_study/phase1_sensitivity
scenarios: [S0, S1, S2, S3]
axes:
  array_filter.csm.nperseg: [256, 1024]
  array_filter.csm.diag_loading_rel: [0.0, 1.0e-4]
  array_filter.doa_grid.rotor_cone_half_angle_deg: [20.0, 45.0]
  array_filter.doa_grid.drone_disk_half_width_deg: [8.0, 25.0]
  array_filter.doa_grid.target_cone_half_angle_deg: [30.0, 60.0]
  array_filter.doa_grid.focal_radius_m: [1.0, 2.0]
  array_filter.doa_grid.azimuth_step_deg: [3.0, 10.0]
  array_filter.doa_grid.elevation_step_deg: [3.0, 10.0]
  array_filter.clean_sc.damp: [0.3, 0.9]
  array_filter.clean_sc.n_iter: [50, 300]
  array_filter.clean_sc.r_diag: [false]
configs:
  - configs/pipeline_mixed_doa_rotor_cone.yaml
  - configs/pipeline_mixed_doa_drone_cone.yaml
  - configs/pipeline_mixed_doa_target_cone.yaml
```

- [ ] **Step 4: Run tests to verify pass**

```
uv run pytest tests/separation_study/test_aggregate_results.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/aggregate_results.py \
        analysis/separation_study/studies/phase1_sensitivity.yaml \
        analysis/separation_study/study_runner.py \
        tests/separation_study/test_aggregate_results.py
git commit -m "feat(separation_study): result aggregation + phase1 study yaml"
```

---

### Task 11: Sensitivity Computation

**Files:**
- Create: `analysis/separation_study/sensitivity.py`
- Test: `tests/separation_study/test_sensitivity.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_sensitivity.py`:
```python
import pandas as pd


def test_compute_sensitivity_max_abs():
    from analysis.separation_study.sensitivity import compute_sensitivity
    rows = [
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -20.0, "overrides": "{}", "run_id": "baseline"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -18.0, "overrides": '{"a.b": 1}', "run_id": "low"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -25.0, "overrides": '{"a.b": 2}', "run_id": "high"},
    ]
    df = pd.DataFrame(rows)
    out = compute_sensitivity(df, baseline_overrides_json="{}")
    row = out.iloc[0]
    assert row["axis"] == "a.b"
    assert abs(row["sensitivity_db"] - 5.0) < 1e-9


def test_dominant_flag_uses_threshold():
    from analysis.separation_study.sensitivity import compute_sensitivity
    rows = [
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -20.0, "overrides": "{}", "run_id": "b"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -19.95, "overrides": '{"x.y": 1}', "run_id": "low"},
        {"method": "M", "scenario": "S0", "band": "mid", "metric": "spectrum_l1_db",
         "value": -22.0, "overrides": '{"a.b": 2}', "run_id": "high"},
    ]
    df = pd.DataFrame(rows)
    out = compute_sensitivity(df, baseline_overrides_json="{}",
                             welch_floor_db=0.1)
    a_b = out[out["axis"] == "a.b"].iloc[0]
    x_y = out[out["axis"] == "x.y"].iloc[0]
    assert a_b["dominant_flag"] is True
    assert x_y["dominant_flag"] is False
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_sensitivity.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement sensitivity**

`analysis/separation_study/sensitivity.py`:
```python
"""Berechnet Sensitivities aus dem Phase-1-Aggregate.

sensitivity = max(|metric(low) - metric(baseline)|, |metric(high) - metric(baseline)|).
dominant_flag wird gesetzt wenn sensitivity > 3 · welch_floor_db."""
from __future__ import annotations

import json
import logging

import pandas as pd

log = logging.getLogger("separation_study.sensitivity")


def _axis_from_overrides(ov_json: str) -> str | None:
    ov = json.loads(ov_json)
    if not ov:
        return None
    if len(ov) != 1:
        raise ValueError(f"expected single-axis override, got {ov}")
    return next(iter(ov.keys()))


def compute_sensitivity(
    df: pd.DataFrame, *,
    baseline_overrides_json: str = "{}",
    welch_floor_db: float = 0.1,
) -> pd.DataFrame:
    """Pro (method, scenario, band, metric, axis): sensitivity_db + dominant_flag.

    df: long-format aggregate von aggregate_results.
    baseline_overrides_json: JSON-serialized empty dict {} markiert Baseline-Run.
    """
    out_rows: list[dict] = []
    df = df.copy()
    df["axis"] = df["overrides"].map(_axis_from_overrides)
    for (method, scenario, band, metric), grp in df.groupby(
        ["method", "scenario", "band", "metric"], dropna=False,
    ):
        baseline_rows = grp[grp["overrides"] == baseline_overrides_json]
        if baseline_rows.empty:
            continue
        baseline = float(baseline_rows["value"].iloc[0])
        for axis, ax_grp in grp[grp["axis"].notna()].groupby("axis"):
            deltas = (ax_grp["value"] - baseline).abs()
            sens = float(deltas.max())
            out_rows.append({
                "method": method, "scenario": scenario, "band": band,
                "metric": metric, "axis": axis,
                "baseline": baseline,
                "n_axis_points": len(ax_grp),
                "sensitivity_db": sens,
                "dominant_flag": bool(sens > 3.0 * welch_floor_db),
            })
    return pd.DataFrame(out_rows)


def top_axes_per_metric(sens_df: pd.DataFrame, *, k: int = 3) -> pd.DataFrame:
    """Pro (method, metric, band) die top-k Achsen nach sensitivity_db."""
    return (sens_df.sort_values("sensitivity_db", ascending=False)
                  .groupby(["method", "metric", "band"], as_index=False)
                  .head(k))
```

- [ ] **Step 4: Run tests to verify pass**

```
uv run pytest tests/separation_study/test_sensitivity.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/sensitivity.py \
        tests/separation_study/test_sensitivity.py
git commit -m "feat(separation_study): sensitivity computation + dominant-flag"
```

---

## Phase D: Plots

### Task 12: Sensitivity-Heatmap-Plot

**Files:**
- Create: `analysis/separation_study/plots/__init__.py` (leer)
- Create: `analysis/separation_study/plots/sensitivity_heatmap.py`
- Test: `tests/separation_study/test_plots_sensitivity_heatmap.py`

- [ ] **Step 1: Write failing test (smoke)**

`tests/separation_study/test_plots_sensitivity_heatmap.py`:
```python
import pandas as pd


def test_sensitivity_heatmap_writes_html(tmp_path):
    from analysis.separation_study.plots.sensitivity_heatmap import write_heatmap
    sens = pd.DataFrame([
        {"method": "rotor_cone", "axis": "csm.nperseg", "metric": "spectrum_l1_db",
         "band": "mid", "sensitivity_db": 1.5, "dominant_flag": True},
        {"method": "rotor_cone", "axis": "clean_sc.damp", "metric": "spectrum_l1_db",
         "band": "mid", "sensitivity_db": 0.2, "dominant_flag": False},
    ])
    out = tmp_path / "heatmap.html"
    write_heatmap(sens, out_path=out, band="mid")
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_plots_sensitivity_heatmap.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement sensitivity_heatmap**

`analysis/separation_study/plots/sensitivity_heatmap.py`:
```python
"""Heatmap: Achsen × Metriken pro Methode, Farbe = sensitivity_db."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp


def write_heatmap(sens_df: pd.DataFrame, *, out_path: Path, band: str) -> None:
    band_df = sens_df[sens_df["band"] == band]
    methods = sorted(band_df["method"].unique())
    fig = sp.make_subplots(
        rows=len(methods), cols=1,
        subplot_titles=[f"{m} — band={band}" for m in methods],
        vertical_spacing=0.05,
    )
    for i, method in enumerate(methods, start=1):
        m_df = band_df[band_df["method"] == method]
        pivot = m_df.pivot_table(
            index="axis", columns="metric", values="sensitivity_db",
            aggfunc="first",
        )
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="Viridis", colorbar={"title": "Δ [dB]"},
            zmin=0.0, showscale=(i == 1),
        ), row=i, col=1)
    fig.update_layout(
        title=f"Phase-1 Sensitivity Heatmap (band={band})",
        height=300 * len(methods) + 100,
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
```

- [ ] **Step 4: Run test to verify pass**

```
uv run pytest tests/separation_study/test_plots_sensitivity_heatmap.py -v
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/plots/__init__.py \
        analysis/separation_study/plots/sensitivity_heatmap.py \
        tests/separation_study/test_plots_sensitivity_heatmap.py
git commit -m "feat(separation_study): sensitivity heatmap plot"
```

---

### Task 13: Pareto-Plot

**Files:**
- Create: `analysis/separation_study/plots/pareto_plot.py`
- Test: `tests/separation_study/test_plots_pareto.py`

- [ ] **Step 1: Write failing test**

`tests/separation_study/test_plots_pareto.py`:
```python
import pandas as pd


def test_pareto_plot_writes_html(tmp_path):
    from analysis.separation_study.plots.pareto_plot import write_pareto
    df = pd.DataFrame([
        {"axis_value": 0.3, "over_subtraction_db": -10.0,
         "drone_leakage_db_def2": -25.0, "method": "rotor_cone"},
        {"axis_value": 0.6, "over_subtraction_db": -15.0,
         "drone_leakage_db_def2": -30.0, "method": "rotor_cone"},
        {"axis_value": 0.9, "over_subtraction_db": -8.0,
         "drone_leakage_db_def2": -22.0, "method": "rotor_cone"},
    ])
    out = tmp_path / "pareto.html"
    write_pareto(df, axis_label="clean_sc.damp", out_path=out,
                welch_floor_db=-40.0, band="mid")
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify failure**

```
uv run pytest tests/separation_study/test_plots_pareto.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement pareto_plot**

`analysis/separation_study/plots/pareto_plot.py`:
```python
"""Pareto-Scatter: over_subtraction_db × drone_leakage_db_def2,
Punktfarbe = Knob-Wert."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def write_pareto(
    df: pd.DataFrame, *,
    axis_label: str,
    out_path: Path,
    welch_floor_db: float,
    band: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["over_subtraction_db"], y=df["drone_leakage_db_def2"],
        mode="markers+text",
        marker={"size": 12, "color": df["axis_value"], "colorscale": "Viridis",
                "showscale": True, "colorbar": {"title": axis_label}},
        text=[f"{v:g}" for v in df["axis_value"]], textposition="top center",
    ))
    # Welch-Floor als hatched rectangle (rechteck wo beide < floor)
    fig.add_shape(type="rect",
                  x0=-100, x1=welch_floor_db, y0=-100, y1=welch_floor_db,
                  fillcolor="rgba(200,200,200,0.3)", line={"width": 0})
    fig.update_layout(
        title=f"Pareto — {axis_label} (band={band})",
        xaxis_title="over_subtraction_db (gepeelt)",
        yaxis_title="drone_leakage_db_def2 (Drohnen-Leck)",
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
```

- [ ] **Step 4: Run test to verify pass**

```
uv run pytest tests/separation_study/test_plots_pareto.py -v
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/plots/pareto_plot.py \
        tests/separation_study/test_plots_pareto.py
git commit -m "feat(separation_study): pareto plot"
```

---

### Task 14: Frequency-Resolved-Plot + Interaction-Plot

**Files:**
- Create: `analysis/separation_study/plots/frequency_resolved.py`
- Create: `analysis/separation_study/plots/interaction_plot.py`
- Test: `tests/separation_study/test_plots_frequency_resolved.py`
- Test: `tests/separation_study/test_plots_interaction.py`

- [ ] **Step 1: Write failing tests**

`tests/separation_study/test_plots_frequency_resolved.py`:
```python
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
```

`tests/separation_study/test_plots_interaction.py`:
```python
import pandas as pd


def test_interaction_plot_writes_html(tmp_path):
    from analysis.separation_study.plots.interaction_plot import write_interaction
    df = pd.DataFrame([
        {"axis_a_value": 0.3, "axis_b_value": 50, "spectrum_l1_db": -10.0},
        {"axis_a_value": 0.3, "axis_b_value": 200, "spectrum_l1_db": -12.0},
        {"axis_a_value": 0.9, "axis_b_value": 50, "spectrum_l1_db": -8.0},
        {"axis_a_value": 0.9, "axis_b_value": 200, "spectrum_l1_db": -15.0},
    ])
    out = tmp_path / "ia.html"
    write_interaction(df, axis_a="clean_sc.damp", axis_b="clean_sc.n_iter",
                     metric="spectrum_l1_db", out_path=out)
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify failure**

```
uv run pytest tests/separation_study/test_plots_frequency_resolved.py \
              tests/separation_study/test_plots_interaction.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement plots**

`analysis/separation_study/plots/frequency_resolved.py`:
```python
"""Frequency-resolved Plot: psd_post + ext_GT + D_ref + excess/deficit-Subplot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from analysis.separation_study.freq_resolved_io import read_freq_resolved


def write_frequency_resolved(
    h5_path: Path, *, out_path: Path, bands: list[dict],
) -> None:
    payload = read_freq_resolved(h5_path)
    f = payload["frequencies"]
    fig = sp.make_subplots(
        rows=2, cols=1, vertical_spacing=0.10,
        subplot_titles=("PSD am Target [dB re 1 Pa²/Hz]",
                       "Per-Bin excess (rot) / deficit (blau) [linear power]"),
    )
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
    for label, color, key in [
        ("psd_post", "#1f77b4", "psd_post"),
        ("ext_gt",   "#2ca02c", "ext_gt"),
        ("d_ref",    "#d62728", "d_ref"),
    ]:
        fig.add_trace(go.Scatter(
            x=f, y=db(payload[key]), mode="lines", name=label,
            line={"color": color, "width": 2},
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=f, y=payload["excess"], marker_color="#d62728",
        name="excess", opacity=0.6,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=f, y=payload["deficit"], marker_color="#1f77b4",
        name="deficit", opacity=0.6,
    ), row=2, col=1)
    for b in bands:
        for r in (1, 2):
            fig.add_vline(x=b["f_min_hz"],
                          line={"color": "#888", "width": 1, "dash": "dot"},
                          row=r, col=1)
            fig.add_vline(x=b["f_max_hz"],
                          line={"color": "#888", "width": 1, "dash": "dot"},
                          row=r, col=1)
    fig.update_layout(template="plotly_white", height=800,
                      title=str(h5_path.parent.name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
```

`analysis/separation_study/plots/interaction_plot.py`:
```python
"""Wechselwirkungs-Heatmap: 2D-Grid von Knob_a × Knob_b mit Metrik als Farbe."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def write_interaction(
    df: pd.DataFrame, *, axis_a: str, axis_b: str, metric: str, out_path: Path,
) -> None:
    pivot = df.pivot_table(
        index="axis_a_value", columns="axis_b_value", values=metric, aggfunc="first",
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdBu_r", colorbar={"title": metric},
    ))
    fig.update_layout(
        title=f"{metric}: {axis_a} × {axis_b}",
        xaxis_title=axis_b, yaxis_title=axis_a,
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
```

- [ ] **Step 4: Run tests to verify pass**

```
uv run pytest tests/separation_study/ -v
```
Expected: alle PASSED.

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/plots/frequency_resolved.py \
        analysis/separation_study/plots/interaction_plot.py \
        tests/separation_study/test_plots_frequency_resolved.py \
        tests/separation_study/test_plots_interaction.py
git commit -m "feat(separation_study): frequency-resolved + interaction plots"
```

---

## Phase E: Studienausführung (operational, kein TDD)

Diese Tasks sind **Daten-Sammel-Schritte**, kein TDD-Code. Jeder Schritt ist ein
Kommando + Verifikation des Outputs.

### Task 15: Scenario-Paths-Mapping erstellen

**Files:**
- Create: `analysis/separation_study/studies/scenario_paths.yaml`

- [ ] **Step 1: Schreibe scenario_paths.yaml**

```yaml
S0:
  drone_artifact: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml:   /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml
  audio_h5:       /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip_gt.h5
  ext_only_gt_h5:  /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_external_only_gaptip_gt.h5
S1:
  drone_artifact: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml:   /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml
  audio_h5:       /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S1_synth.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S1_synth_gt.h5
  ext_only_gt_h5:  /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S1_extonly_gt.h5
S2:
  drone_artifact: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml:   /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml
  audio_h5:       /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S2_synth.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S2_synth_gt.h5
  ext_only_gt_h5:  /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S2_extonly_gt.h5
S3:
  drone_artifact: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml:   /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml
  audio_h5:       /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S3_synth.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S3_synth_gt.h5
  ext_only_gt_h5:  /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/scenario_S3_extonly_gt.h5
```

Note: für S1-S3 müssen pro Szenario zwei compose-Läufe gemacht werden — einmal mixed (`include_drone: true`) und einmal ext_only (`include_drone: false`). Erweitere `ensure_scenario` falls nötig in einem Folge-PR um auch ein `_extonly_gt.h5` zu erzeugen — für die erste Studienausführung kann S0 verwendet werden bis S1-S3 explizit gebraucht werden (Phase 1 läuft auch nur mit S0 falls Robustheit nicht verifiziert wird).

- [ ] **Step 2: Commit**

```
git add analysis/separation_study/studies/scenario_paths.yaml
git commit -m "chore(separation_study): scenario paths mapping"
```

---

### Task 16: Baseline-Studie ausführen

- [ ] **Step 1: Run baseline**

```
uv run python -m analysis.separation_study.study_runner \
  --study analysis/separation_study/studies/baseline.yaml \
  --scenario-paths analysis/separation_study/studies/scenario_paths.yaml \
  --log-level INFO
```

Expected: 8 result_dirs unter `results/separation_study/baseline/`, jeder mit
`metrics.json`, `study_metrics.json`, `metrics_freq.h5`, `run_meta.json`.

- [ ] **Step 2: Aggregate**

```
uv run python -m analysis.separation_study.aggregate_results \
  --root results/separation_study/baseline \
  --out  results/separation_study/baseline/baseline_summary.parquet
```

- [ ] **Step 3: Welch-Floor aus ext_only-Runs ermitteln**

Mache eine kurze Skript-Sitzung (z.B. `uv run python -c "..."`):
```python
import pandas as pd
df = pd.read_parquet("results/separation_study/baseline/baseline_summary.parquet")
ext_only = df[df["method"].str.contains("external_only")]
floor = (ext_only[ext_only["metric"] == "spectrum_l1_db"]
         .groupby("band")["value"].max())
print(floor)
```
Notiere `welch_floor_db` pro Band als Tabellenwert in `results/separation_study/welch_floor.json`.

- [ ] **Step 4: Commit Welch-Floor**

```
git add results/separation_study/welch_floor.json
git commit -m "data(separation_study): empirical Welch floor from ext_only baseline"
```

---

### Task 17: Phase-1-Audit ausführen

- [ ] **Step 1: Run phase 1 (S0 only first, dann S1-S3 nach Bedarf)**

Modifiziere `phase1_sensitivity.yaml` falls nur S0 zuerst:
```yaml
scenarios: [S0]
```

```
uv run python -m analysis.separation_study.study_runner \
  --study analysis/separation_study/studies/phase1_sensitivity.yaml \
  --scenario-paths analysis/separation_study/studies/scenario_paths.yaml
```

Expected: ~24 audit-Runs × 3 DOA-Methoden = 72 result_dirs.

Erwartete Laufzeit: ~1-2h (1-2 min/Run für ~10s mixed-Audio durch Pipeline).

- [ ] **Step 2: Aggregate + Sensitivity**

```
uv run python -m analysis.separation_study.aggregate_results \
  --root results/separation_study/phase1_sensitivity \
  --out  results/separation_study/phase1_sensitivity/phase1_sensitivity.parquet
```

Dann `compute_sensitivity` interaktiv:
```python
import pandas as pd
import json
from analysis.separation_study.sensitivity import (
    compute_sensitivity, top_axes_per_metric,
)
df = pd.read_parquet("results/separation_study/phase1_sensitivity/phase1_sensitivity.parquet")
floor = json.loads(open("results/separation_study/welch_floor.json").read())
# Min floor across bands as conservative threshold
welch_floor_db = min(floor.values())
sens = compute_sensitivity(df, welch_floor_db=welch_floor_db)
sens.to_csv("results/separation_study/phase1_sensitivity/phase1_sensitivity_table.csv",
            index=False)
top = top_axes_per_metric(sens, k=3)
top.to_csv("results/separation_study/phase1_sensitivity/phase1_dominant_axes.csv",
           index=False)
```

- [ ] **Step 3: Heatmap-Plot pro Band**

```
uv run python -c "
from pathlib import Path
import pandas as pd
from analysis.separation_study.plots.sensitivity_heatmap import write_heatmap
sens = pd.read_csv('results/separation_study/phase1_sensitivity/phase1_sensitivity_table.csv')
for band in ('low','mid','high'):
    write_heatmap(sens, out_path=Path(f'results/separation_study/phase1_sensitivity/heatmap_{band}.html'), band=band)
"
```

- [ ] **Step 4: Commit Phase-1-Outputs**

```
git add results/separation_study/phase1_sensitivity/*.csv \
        results/separation_study/phase1_sensitivity/*.html
git commit -m "data(separation_study): phase1 sensitivity audit results"
```

---

### Task 18: Phase-2-YAML aus dominanten Achsen generieren

**Files:**
- Create: `analysis/separation_study/studies/phase2_sweep.yaml`

- [ ] **Step 1: Inspect phase1_dominant_axes.csv**

Lies die CSV in einer Python-Sitzung, identifiziere die Top-3 dominanten Achsen
über alle Methoden gemittelt:

```python
import pandas as pd
df = pd.read_csv("results/separation_study/phase1_sensitivity/phase1_dominant_axes.csv")
# Pro Achse: mittlere sensitivity_db über alle (method, metric, band)
axis_avg = df.groupby("axis")["sensitivity_db"].mean().sort_values(ascending=False)
print(axis_avg.head(5))
```

- [ ] **Step 2: Schreibe phase2_sweep.yaml**

Beispielinhalt (ANPASSEN je nach Phase-1-Ergebnis — der Engineer wählt die Top-3
und füllt die Sweep-Punkte ein):

```yaml
phase: phase2
output_root: results/separation_study/phase2_focused_sweep
scenarios: [S0]
configs:
  # Beste Methode aus Phase 1 wählen — z.B. doa_rotor_cone
  - configs/pipeline_mixed_doa_rotor_cone.yaml
axes:
  array_filter.clean_sc.damp: [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
  array_filter.clean_sc.n_iter: [30, 60, 100, 150, 200, 300, 500]
  array_filter.doa_grid.drone_disk_half_width_deg: [5, 10, 15, 20, 25, 35, 45]
```

(Tatsächliche dominante Achsen + Wertelisten ergeben sich aus dem Phase-1-Ergebnis.)

- [ ] **Step 3: Run phase 2**

```
uv run python -m analysis.separation_study.study_runner \
  --study analysis/separation_study/studies/phase2_sweep.yaml \
  --scenario-paths analysis/separation_study/studies/scenario_paths.yaml
```

- [ ] **Step 4: Aggregate + Pareto-Plots**

```
uv run python -m analysis.separation_study.aggregate_results \
  --root results/separation_study/phase2_focused_sweep \
  --out  results/separation_study/phase2_focused_sweep/phase2_sweep.parquet
```

Pro Achse + Band ein pareto-Plot:
```python
import json
import pandas as pd
from pathlib import Path
from analysis.separation_study.plots.pareto_plot import write_pareto

df = pd.read_parquet("results/separation_study/phase2_focused_sweep/phase2_sweep.parquet")
floor = json.loads(open("results/separation_study/welch_floor.json").read())

axes = df["overrides"].apply(lambda s: list(json.loads(s).keys())[0]
                            if json.loads(s) else None)
df["axis"] = axes
for axis_name, axis_grp in df[df["axis"].notna()].groupby("axis"):
    pivot = (axis_grp[axis_grp["metric"].isin(["over_subtraction_db",
                                              "drone_leakage_db_def2"])]
             .pivot_table(index=["run_id", "band"],
                          columns="metric", values="value", aggfunc="first")
             .reset_index())
    pivot["axis_value"] = pivot["run_id"].apply(
        lambda rid: float(rid.rsplit("_", 1)[0].rsplit("_", 1)[-1].replace("p", ".")
                          if "p" in rid else 0.0))
    for band in pivot["band"].unique():
        sub = pivot[pivot["band"] == band]
        if len(sub) < 2:
            continue
        out_path = Path(f"results/separation_study/phase2_focused_sweep/"
                        f"pareto_{axis_name.replace('.', '_')}_{band}.html")
        write_pareto(sub, axis_label=axis_name, out_path=out_path,
                     welch_floor_db=floor[band], band=band)
```

- [ ] **Step 5: Commit Phase-2-Outputs**

```
git add analysis/separation_study/studies/phase2_sweep.yaml \
        results/separation_study/phase2_focused_sweep/*.html
git commit -m "data(separation_study): phase2 focused sweep results"
```

---

### Task 19: Wechselwirkungs-Grid (Top-2-Achsen)

- [ ] **Step 1: Identifiziere Top-2 Achsen + schreibe interaction-yaml**

Aus Phase-1: Top-2 nach mittlerer sensitivity_db. Beispiel-yaml unter
`analysis/separation_study/studies/phase2_interaction.yaml`:

```yaml
phase: phase2
output_root: results/separation_study/phase2_interaction
scenarios: [S0]
configs:
  - configs/pipeline_mixed_doa_rotor_cone.yaml
axes_grid:
  - axis: array_filter.clean_sc.damp
    points: [0.3, 0.5, 0.7, 0.85, 0.95]
  - axis: array_filter.clean_sc.n_iter
    points: [50, 100, 200, 400, 800]
```

(`axes_grid` ist ein eigenes Feld — siehe nächster Schritt für die runner-
Erweiterung.)

- [ ] **Step 2: Erweitere study_runner um axes_grid (cartesian product)**

In `expand_run_plan`, falls `study.get("axes_grid")` gesetzt:
```python
if "axes_grid" in study:
    from itertools import product
    grid = study["axes_grid"]
    axis_a, axis_b = grid[0]["axis"], grid[1]["axis"]
    for a, b in product(grid[0]["points"], grid[1]["points"]):
        overrides = {axis_a: a, axis_b: b}
        h = config_hash(overrides)
        for cfg_path in study["configs"]:
            cfg_name = Path(cfg_path).stem
            for scenario in study["scenarios"]:
                rid = (f"{cfg_name}__{scenario}__"
                       f"{_slug(axis_a)}_{_slug(a)}_x_{_slug(axis_b)}_{_slug(b)}_{h}")
                plan.append({
                    "config": cfg_path, "scenario": scenario,
                    "overrides": overrides, "run_id": rid,
                    "output_dir": str(output_root / rid),
                })
```

(Inkl. Test in `tests/separation_study/test_study_runner.py`:
```python
def test_expand_run_plan_axes_grid_cartesian(tmp_path):
    from analysis.separation_study.study_runner import expand_run_plan
    plan = expand_run_plan({
        "phase": "phase2",
        "configs": ["c.yaml"], "scenarios": ["S0"],
        "axes_grid": [{"axis": "a.b", "points": [1, 2]},
                     {"axis": "c.d", "points": [10, 20, 30]}],
        "output_root": str(tmp_path),
    })
    assert len(plan) == 2 * 3  # 2 × 3 cartesian
```
)

- [ ] **Step 3: Run + Aggregate + Plot**

```
uv run pytest tests/separation_study/test_study_runner.py -v
uv run python -m analysis.separation_study.study_runner \
  --study analysis/separation_study/studies/phase2_interaction.yaml \
  --scenario-paths analysis/separation_study/studies/scenario_paths.yaml
uv run python -m analysis.separation_study.aggregate_results \
  --root results/separation_study/phase2_interaction \
  --out  results/separation_study/phase2_interaction/interaction.parquet
```

Plot interaktiv:
```python
import pandas as pd
import json
from pathlib import Path
from analysis.separation_study.plots.interaction_plot import write_interaction
df = pd.read_parquet("results/separation_study/phase2_interaction/interaction.parquet")
df["overrides_dict"] = df["overrides"].apply(json.loads)
df["axis_a_value"] = df["overrides_dict"].apply(lambda d: d.get("array_filter.clean_sc.damp"))
df["axis_b_value"] = df["overrides_dict"].apply(lambda d: d.get("array_filter.clean_sc.n_iter"))
sub = df[df["metric"] == "spectrum_l1_db"]
for band in sub["band"].unique():
    band_df = sub[sub["band"] == band].copy()
    write_interaction(
        band_df.rename(columns={"value": "spectrum_l1_db"}),
        axis_a="clean_sc.damp", axis_b="clean_sc.n_iter",
        metric="spectrum_l1_db",
        out_path=Path(f"results/separation_study/phase2_interaction/ia_{band}.html"),
    )
```

- [ ] **Step 4: Commit**

```
git add analysis/separation_study/study_runner.py \
        analysis/separation_study/studies/phase2_interaction.yaml \
        results/separation_study/phase2_interaction/*.html \
        tests/separation_study/test_study_runner.py
git commit -m "feat+data(separation_study): axes-grid interaction runs"
```

---

### Task 20: Robustheits-Verifikation auf S1-S3

- [ ] **Step 1: Erzeuge S1-S3 ext_only-Files**

Erweiterung in `synth_scenarios.ensure_scenario`: nach mixed-Compose ein zweiter
compose mit `include_drone: false` und `out_*_extonly_gt.h5`. Implementieren
falls noch nicht erfolgt (kann in einem Schritt mit Robustheits-Run kommen).

```python
def ensure_scenario_pair(scenario, ..., out_extonly_h5, out_extonly_gt_h5):
    ensure_scenario(scenario=scenario, ..., out_synth_h5=mixed_h5, out_gt_h5=mixed_gt_h5)
    cfg = make_compose_config(scenario=scenario, ..., out_synth_h5=str(out_extonly_h5),
                              out_gt_h5=str(out_extonly_gt_h5))
    if cfg is not None:
        cfg["include_drone"] = False
        if not Path(out_extonly_h5).exists():
            _run_compose(cfg)
```

- [ ] **Step 2: Schreibe robustness-yaml**

`analysis/separation_study/studies/robustness.yaml`:
```yaml
phase: phase2
output_root: results/separation_study/robustness
# Optimal-Config aus Phase 2 als Override-Set übergeben
scenarios: [S0, S1, S2, S3]
configs:
  - configs/pipeline_mixed_doa_rotor_cone.yaml   # Phase-2-Beste
axes:
  # Optimaler Punkt: ein einzelner Override-Set, sweep verwendet als axes
  # mit nur 1 Punkt erzwingt Override + Baseline (= 2 Runs/Methode/Szenario)
  array_filter.clean_sc.damp: [0.85]   # Phase-2-Optimum aus Pareto
```

- [ ] **Step 3: Run**

```
uv run python -m analysis.separation_study.study_runner \
  --study analysis/separation_study/studies/robustness.yaml \
  --scenario-paths analysis/separation_study/studies/scenario_paths.yaml
```

Expected: 4 Szenarien × 2 (Baseline + Optimum) = 8 Runs.

- [ ] **Step 4: Aggregate + Frequency-Resolved-Plots**

```
uv run python -m analysis.separation_study.aggregate_results \
  --root results/separation_study/robustness \
  --out  results/separation_study/robustness/robustness.parquet
```

Pro Szenario einen frequency-resolved-Plot:
```python
from pathlib import Path
import yaml
from analysis.separation_study.plots.frequency_resolved import (
    write_frequency_resolved,
)
bands = yaml.safe_load(
    open("configs/pipeline_mixed_doa_rotor_cone.yaml")
)["stages"][1]["bands"]
for run_dir in Path("results/separation_study/robustness").glob("*"):
    h5 = run_dir / "metrics_freq.h5"
    if h5.exists():
        write_frequency_resolved(h5, out_path=run_dir / "fr.html", bands=bands)
```

- [ ] **Step 5: Commit**

```
git add analysis/separation_study/studies/robustness.yaml \
        analysis/separation_study/synth_scenarios.py \
        results/separation_study/robustness/**/fr.html \
        results/separation_study/robustness/robustness.parquet
git commit -m "data(separation_study): robustness verification on S0-S3"
```

---

### Task 21: Final Recommendation Markdown

**Files:**
- Create: `results/separation_study/final_recommendation.md`

- [ ] **Step 1: Generate Markdown skeleton**

Generiere skeleton interaktiv mit Phase-2-Daten (anpassen je nach Studienergebnis):

```markdown
# Mixed-Separation: Empfohlene Konfiguration

**Datum:** 2026-MM-DD
**Studie:** docs/superpowers/specs/2026-05-08-mixed-separation-study-design.md
**Git:** <commit-sha>

## Vergleichstafel (alle 3 DOA-Methoden, optimaler Punkt auf S0)

| Methode | spectrum_l1_db (mid) | drone_leakage_db_def2 (mid) | over_subtraction_db (mid) |
|---|---|---|---|
| doa_rotor_cone  | -16.2 | -34.5 | -22.1 |
| doa_drone_cone  | -14.0 | -29.1 | -18.7 |
| doa_target_cone | -13.5 | -28.0 | -17.9 |

## Empfehlung pro Methode

### doa_rotor_cone (PRIMÄR)

| Knob | Baseline | Empfohlen |
|---|---|---|
| clean_sc.damp | 0.6 | 0.85 |
| clean_sc.n_iter | 100 | 200 |
| ... |

#### Metriken pro Band (S0_baseline)

| Metrik | low | mid | high |
|---|---|---|---|
| spectrum_l1_db | ... | -16.2 | ... |
| over_subtraction_db | ... |
| drone_leakage_db_def1 | ... |
| drone_leakage_db_def2 | ... |
| welch_floor_db | -19.0 | -22.0 | -18.0 |

#### Robustheit auf S1/S2/S3

| Szenario | spectrum_l1_db (mid) | drone_leakage_db_def2 (mid) |
|---|---|---|
| S0 | -16.2 | -34.5 |
| S1 | ... | ... |

#### Plots

- `results/separation_study/phase2_focused_sweep/pareto_clean_sc_damp_mid.html`
- `results/separation_study/robustness/<run>/fr.html`

### doa_drone_cone
... (analog)

### doa_target_cone
... (analog)

## Limitationen
- Multi-Source-Szenario nicht getestet
- Notch-Stage nicht in Studie
- NNLS nicht im Fokus
```

- [ ] **Step 2: Fülle alle Zahlen aus den parquet-Files**

Skript, das aus Aggregaten die Recommendations einsetzt; erfolgt iterativ
während die Phase-2-Ergebnisse vorliegen.

- [ ] **Step 3: Commit**

```
git add results/separation_study/final_recommendation.md
git commit -m "docs(separation_study): final recommendation"
```

---

### Task 22: README für analysis/separation_study/

**Files:**
- Create: `analysis/separation_study/README.md`

- [ ] **Step 1: Schreibe README**

```markdown
# Mixed-Separation-Study

Implementiert die mixed-Separation Parameter-Studie aus
`docs/superpowers/specs/2026-05-08-mixed-separation-study-design.md`.

## Ablauf

1. **Baseline:** `uv run python -m analysis.separation_study.study_runner --study studies/baseline.yaml --scenario-paths studies/scenario_paths.yaml`
2. **Welch-Floor kalibrieren:** aus ext_only-Runs der Baseline
3. **Phase 1 (Audit):** `--study studies/phase1_sensitivity.yaml`
4. **Phase 2 (Sweep):** `--study studies/phase2_sweep.yaml` (axes nach Phase-1-Ergebnis)
5. **Wechselwirkungs-Grid:** `--study studies/phase2_interaction.yaml`
6. **Robustheit:** `--study studies/robustness.yaml`
7. **Empfehlung:** `results/separation_study/final_recommendation.md`

## Tests

```
uv run pytest tests/separation_study/ -v
```

## Module

| Modul | Verantwortung |
|---|---|
| `metric_extensions.py` | Per-Bin excess/deficit, alle 6 Studien-Metriken |
| `drone_only_helper.py` | Time-Domain-Subtraktion mixed - ext |
| `freq_resolved_io.py`  | h5-Beilage pro Run |
| `synth_scenarios.py`   | S0-S3 compose-config-Generatoren + Caching |
| `yaml_override.py`     | Dotted-key YAML-Patcher |
| `run_cache.py`         | Config-Hash + Run-Complete-Detection |
| `pipeline_wrapper.py`  | Wrapper um martymicfly.cli.run_pipeline + Augmentation |
| `study_runner.py`      | Run-Plan-Expansion + Execution |
| `aggregate_results.py` | study_metrics.json → parquet, consistency-Check |
| `sensitivity.py`       | Sensitivity-Berechnung pro Achse + Top-K |
| `plots/`               | Heatmap, Pareto, Frequency-resolved, Interaction |
```

- [ ] **Step 2: Commit**

```
git add analysis/separation_study/README.md
git commit -m "docs(separation_study): README with workflow instructions"
```

---

## Self-Review-Checkliste (für Engineer nach Plan-Abschluss)

- [ ] Alle Tests grün: `uv run pytest tests/separation_study/ -v`
- [ ] `welch_floor_db` empirisch aus ext_only-Runs ermittelt und dokumentiert
- [ ] Konsistenz-Check (recovery − leakage_def1 ≈ −SNR) wirft keine Warnungen
- [ ] `final_recommendation.md` enthält für jede DOA-Methode: Empfohlene
      Knöpfe, S0-Metriken, S1-S3-Robustheit, Plot-Links
- [ ] `compensation_flag` in keinem optimalen Konfig-Punkt true
- [ ] Phase-2 Pareto-Plots zeigen klare Optimum-Punkte (kein flacher Verlauf)
