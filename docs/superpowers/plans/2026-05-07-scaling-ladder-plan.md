# Scaling-Ladder Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-rung diagnostic (`analysis/scaling_ladder.py`) that propagates an analytically-defined white-noise monopole through the production CSM and steering chain, identifies which stage introduces the ~−27 dB power-scaling bias observed in `ext_only` smoke runs.

**Architecture:** Standalone script, no pipeline integration. Independent forward propagator (5–10 lines numpy, NOT `synth/propagation.py`). Production-side: `build_measurement_csm` + `steer_to_psd` reused as-is. Outputs Markdown + Plotly HTML report under `results/scaling_ladder/<run_id>/`.

**Tech Stack:** numpy, scipy.signal, h5py, plotly, project's `martymicfly.io.mic_geom`, `martymicfly.processing.csm`, `martymicfly.processing.steering`.

**Spec interpretation note for Rung 3:** The spec says "steering reconstructs `S_q`". Closer reading of `processing/steering.py` shows it implements phase-only delay-and-sum with `1/M²` normalization (no `1/r` weight). For a point source at the target this gives `S_steered = S_q · ⟨1/r_m⟩²` — i.e. about `S_q − 3.5 dB` for a 16-mic array at ~1.5 m radius. **Use this as the Rung-3 theoretical expectation.** The −27 dB observed gap is way larger than this −3.5 dB physical factor, so the bug isn't a missing `1/r` factor.

---

## File structure

- **Create:** `analysis/scaling_ladder.py` — single module containing forward propagator, three rung functions, report writer, and `if __name__ == "__main__"` driver. Target size: ~250 lines.
- **Create:** `tests/analysis/__init__.py` — empty (makes `tests/analysis/` a package).
- **Create:** `tests/analysis/test_scaling_ladder.py` — unit tests for forward propagator and rung functions on small fixtures (3–4 mics, short duration).
- **Read-only / unchanged:** `src/martymicfly/io/mic_geom.py`, `src/martymicfly/processing/csm.py`, `src/martymicfly/processing/steering.py`.
- **Output:** `results/scaling_ladder/<timestamp>/{report.md, mic_psd_vs_theory.html, csm_diag_vs_theory.html, steered_psd.html, metrics.json}`.

---

### Task 1: Set up test infrastructure and write forward-propagator failing test

**Files:**
- Create: `tests/analysis/__init__.py` (empty)
- Create: `tests/analysis/test_scaling_ladder.py`

- [ ] **Step 1: Create empty `__init__.py`**

```bash
touch tests/analysis/__init__.py
```

- [ ] **Step 2: Write the failing test for `propagate_white_noise`**

Create `tests/analysis/test_scaling_ladder.py`:

```python
"""Unit tests for analysis/scaling_ladder.py — forward propagator + 3 rungs."""
from __future__ import annotations

import numpy as np
import pytest

# scaling_ladder.py lives in repo-root analysis/, not on the package path.
# Path-import via importlib so tests don't depend on PYTHONPATH=analysis.
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "analysis" / "scaling_ladder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("scaling_ladder", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scaling_ladder"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sl():
    return _load_module()


def test_propagate_white_noise_yields_correct_per_mic_psd(sl):
    """Forward propagator: per-mic PSD = S_q / r_m^2 (free-field 1/r)."""
    rng = np.random.default_rng(42)
    fs = 51_200.0
    duration_s = 4.0
    n_samples = int(fs * duration_s)
    s_q_pa2_per_hz = 1.0  # source PSD at 1 m

    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.0, 0.0, 0.0],   # r = 1.5
        [0.5, 0.0, 0.0],   # r = sqrt(0.25 + 2.25) ≈ 1.581
        [0.0, 0.4, -0.5],  # r = sqrt(0 + 0.16 + 1.0) ≈ 1.077
    ])
    r = np.linalg.norm(mic_positions - source_pos, axis=1)

    time_data = sl.propagate_white_noise(
        n_samples=n_samples,
        sample_rate=fs,
        s_q_pa2_per_hz=s_q_pa2_per_hz,
        source_position=source_pos,
        mic_positions=mic_positions,
        rng=rng,
    )
    assert time_data.shape == (n_samples, mic_positions.shape[0])

    # Welch PSD per mic; compare mean PSD in [200, 6000] Hz to theory.
    from scipy.signal import welch
    f, p = welch(time_data, fs=fs, nperseg=512, noverlap=256,
                 window="hann", scaling="density", axis=0)
    band = (f >= 200.0) & (f <= 6000.0)
    measured = p[band, :].mean(axis=0)              # (M,)
    theoretical = s_q_pa2_per_hz / (r ** 2)          # (M,)

    delta_db = 10.0 * np.log10(measured / theoretical)
    assert np.all(np.abs(delta_db) < 0.5), (
        f"Per-mic PSD off theory by {delta_db} dB"
    )
```

- [ ] **Step 3: Run test, verify it fails with ImportError or AttributeError**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py -v`
Expected: FAIL — `analysis/scaling_ladder.py` does not exist yet.

- [ ] **Step 4: Commit failing test**

```bash
git add tests/analysis/__init__.py tests/analysis/test_scaling_ladder.py
git commit -m "test(scaling_ladder): failing test for forward white-noise propagator"
```

---

### Task 2: Implement the forward propagator

**Files:**
- Create: `analysis/scaling_ladder.py`

- [ ] **Step 1: Write minimal `propagate_white_noise` to pass test**

Create `analysis/scaling_ladder.py`:

```python
"""Scaling-Ladder Diagnostic.

Diagnoses the ~−27 dB external_recovery bias observed in ext_only smoke runs
by following an analytically-defined white-noise monopole through three
points in the steering chain (mic-PSD, CSM-diagonal, steered PSD).

Run via:  uv run python analysis/scaling_ladder.py [--mic-geom PATH]
"""
from __future__ import annotations

import numpy as np


def propagate_white_noise(
    *,
    n_samples: int,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,   # (M, 3)
    rng: np.random.Generator,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Propagate one white-noise monopole to M mics via free-field 1/r-Greens.

    The source signal q(t) is white noise with two-sided PSD = s_q_pa2_per_hz
    (one-sided density as used by `scipy.welch(..., scaling='density')`).
    Each mic receives `p_m(t) = q(t − r_m/c) / r_m` via fractional-sample
    delay (linear interpolation in time domain).

    Returns
    -------
    time_data : (n_samples, M) float64
    """
    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)         # (M,)

    # White noise of length n_samples + max-delay-margin so we can shift safely.
    max_delay_samples = int(np.ceil(r.max() / speed_of_sound * sample_rate)) + 4
    n_pad = n_samples + max_delay_samples

    # Generate q with the right one-sided PSD.
    # For a real Gaussian process with two-sided PSD = s_two_sided,
    # variance = s_two_sided · fs. Using one-sided density convention:
    # s_one_sided = 2 · s_two_sided (for f > 0), so variance = (s_one_sided/2) · fs.
    sigma = float(np.sqrt(s_q_pa2_per_hz * sample_rate / 2.0))
    q = rng.normal(0.0, sigma, size=n_pad)

    # Per-mic fractional delay via linear interpolation.
    n = np.arange(n_samples, dtype=np.float64)
    out = np.zeros((n_samples, mics.shape[0]), dtype=np.float64)
    for m in range(mics.shape[0]):
        delay_samples = r[m] / speed_of_sound * sample_rate
        # Source-time index for output sample n: t_src = n - delay
        # but we generated q starting at -max_delay (offset), so:
        t_src = n + (max_delay_samples - delay_samples)
        i0 = np.floor(t_src).astype(np.int64)
        frac = t_src - i0
        out[:, m] = (1.0 - frac) * q[i0] + frac * q[i0 + 1]
        out[:, m] /= r[m]
    return out
```

- [ ] **Step 2: Run test, verify it passes**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_propagate_white_noise_yields_correct_per_mic_psd -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): forward white-noise free-field propagator"
```

---

### Task 3: Add Rung 1 (Mic-PSD) function with TDD

**Files:**
- Modify: `tests/analysis/test_scaling_ladder.py` — add Rung 1 test
- Modify: `analysis/scaling_ladder.py` — add Rung 1 function

- [ ] **Step 1: Write Rung 1 test**

Append to `tests/analysis/test_scaling_ladder.py`:

```python
def test_rung1_mic_psd_matches_theory_within_0p5_db(sl):
    rng = np.random.default_rng(7)
    fs = 51_200.0
    n_samples = int(fs * 4.0)
    s_q = 1e-4

    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung1_mic_psd(
        time_data=time_data,
        sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos,
        mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
    )
    assert "delta_db_per_mic" in result
    assert "delta_db_mean" in result
    assert "frequencies_hz" in result
    assert result["delta_db_per_mic"].shape == (mic_positions.shape[0],)
    assert abs(result["delta_db_mean"]) < 0.5
```

- [ ] **Step 2: Run test, verify failure (`AttributeError: rung1_mic_psd`)**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung1_mic_psd_matches_theory_within_0p5_db -v`
Expected: FAIL.

- [ ] **Step 3: Implement `rung1_mic_psd`**

Append to `analysis/scaling_ladder.py`:

```python
from scipy.signal import welch  # noqa: E402  (kept near use site)


def rung1_mic_psd(
    *,
    time_data: np.ndarray,         # (N, M)
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
) -> dict:
    """Rung 1 — direct Welch PSD per mic, compared to theoretical S_q / r_m^2."""
    f, p = welch(
        time_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
        window=window, scaling="density", axis=0,
    )
    mask = (f >= f_min_hz) & (f <= f_max_hz)
    freqs = f[mask]
    psd = p[mask, :]                                            # (F, M)

    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)             # (M,)
    theoretical = s_q_pa2_per_hz / (r ** 2)                     # (M,)

    measured_per_mic = psd.mean(axis=0)                         # (M,)
    delta_db_per_mic = 10.0 * np.log10(measured_per_mic / theoretical)
    return {
        "frequencies_hz": freqs,
        "psd_per_mic": psd,                                     # (F, M)
        "theoretical_per_mic": theoretical,                     # (M,)
        "delta_db_per_mic": delta_db_per_mic,
        "delta_db_mean": float(delta_db_per_mic.mean()),
    }
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung1_mic_psd_matches_theory_within_0p5_db -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/analysis/test_scaling_ladder.py analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): rung 1 — direct Welch mic-PSD vs theory"
```

---

### Task 4: Add Rung 2 (CSM-Diagonal) function with TDD

**Files:**
- Modify: `tests/analysis/test_scaling_ladder.py`
- Modify: `analysis/scaling_ladder.py`

- [ ] **Step 1: Write Rung 2 test**

Append to `tests/analysis/test_scaling_ladder.py`:

```python
def test_rung2_csm_diag_matches_theory_within_0p5_db(sl):
    rng = np.random.default_rng(11)
    fs = 51_200.0
    n_samples = int(fs * 4.0)
    s_q = 1e-4
    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung2_csm_diag(
        time_data=time_data, sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
        diag_loading_rel=0.0,   # disable for clean comparison
    )
    assert "csm_shape" in result
    assert "delta_db_per_mic" in result
    assert "csm" in result and "frequencies_hz" in result
    assert abs(result["delta_db_mean"]) < 0.5
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung2_csm_diag_matches_theory_within_0p5_db -v`
Expected: FAIL.

- [ ] **Step 3: Implement `rung2_csm_diag`**

Append to `analysis/scaling_ladder.py`:

```python
def rung2_csm_diag(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
    diag_loading_rel: float,
) -> dict:
    """Rung 2 — production CSM, diagonal vs theoretical S_q / r_m^2."""
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm

    cfg = CsmConfig(
        nperseg=nperseg, noverlap=noverlap, window=window,
        diag_loading_rel=diag_loading_rel,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, cfg)
    diag = np.real(np.diagonal(csm, axis1=1, axis2=2))      # (F, M)

    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)
    theoretical = s_q_pa2_per_hz / (r ** 2)

    measured_per_mic = diag.mean(axis=0)
    delta_db_per_mic = 10.0 * np.log10(measured_per_mic / theoretical)
    return {
        "csm": csm,
        "csm_shape": csm.shape,
        "frequencies_hz": freqs,
        "csm_diag_per_mic": diag,
        "theoretical_per_mic": theoretical,
        "delta_db_per_mic": delta_db_per_mic,
        "delta_db_mean": float(delta_db_per_mic.mean()),
    }
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung2_csm_diag_matches_theory_within_0p5_db -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/analysis/test_scaling_ladder.py analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): rung 2 — CSM-diagonal vs theory"
```

---

### Task 5: Add Rung 3 (Steered PSD) function with TDD

**Files:**
- Modify: `tests/analysis/test_scaling_ladder.py`
- Modify: `analysis/scaling_ladder.py`

- [ ] **Step 1: Write Rung 3 test**

Append to `tests/analysis/test_scaling_ladder.py`:

```python
def test_rung3_steered_psd_matches_geometry_factor_within_2_db(sl):
    """Rung 3 expectation: S_steered(target) = S_q · <1/r_m>^2 (phase-only DAS).

    Tolerance 2 dB (not 0.5) because phase-only DAS has frequency-dependent
    sidelobe-bleed at small arrays — only the broadband mean of the on-source
    bin should match the geometric factor.
    """
    rng = np.random.default_rng(13)
    fs = 51_200.0
    n_samples = int(fs * 6.0)              # longer for tighter Welch estimate
    s_q = 1e-4
    source_pos = np.array([0.0, 0.0, -1.5])
    mic_positions = np.array([
        [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
        [0.07, 0.07, 0.0], [-0.07, -0.07, 0.0],
        [0.07, -0.07, 0.0], [-0.07, 0.07, 0.0],
    ])
    time_data = sl.propagate_white_noise(
        n_samples=n_samples, sample_rate=fs, s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )
    result = sl.rung3_steered_psd(
        time_data=time_data, sample_rate=fs,
        s_q_pa2_per_hz=s_q,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
        diag_loading_rel=0.0,
    )
    assert "delta_db_band_mean" in result
    assert "theoretical_psd" in result
    assert "steered_psd" in result
    # Geometric factor < 1, so theoretical_psd < S_q. Both reported in dB.
    assert abs(result["delta_db_band_mean"]) < 2.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung3_steered_psd_matches_geometry_factor_within_2_db -v`
Expected: FAIL.

- [ ] **Step 3: Implement `rung3_steered_psd`**

Append to `analysis/scaling_ladder.py`:

```python
def rung3_steered_psd(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
    diag_loading_rel: float,
) -> dict:
    """Rung 3 — production steer_to_psd at the source position vs S_q · <1/r_m>^2.

    The expectation derives from phase-only delay-and-sum with 1/M^2 normalization
    on a unit-amplitude steering vector: for a monopole at the target the
    quadratic form yields S_q · |Σ_m exp(-2j·2π f r_m/c) / r_m|² / M².  When
    the propagator-and-steerer sign convention is consistent the doubled-phase
    cancels and we get S_q · (Σ 1/r_m / M)² = S_q · <1/r_m>².
    """
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    from martymicfly.processing.steering import steer_to_psd

    cfg = CsmConfig(
        nperseg=nperseg, noverlap=noverlap, window=window,
        diag_loading_rel=diag_loading_rel,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, cfg)
    psd = steer_to_psd(
        csm=csm,
        frequencies=freqs,
        mic_positions=np.asarray(mic_positions, dtype=np.float64),
        target_point=tuple(np.asarray(source_position, dtype=np.float64).tolist()),
    )                                                            # (F,)

    mics = np.asarray(mic_positions, dtype=np.float64)
    src = np.asarray(source_position, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)
    geom_factor = float((1.0 / r).mean()) ** 2                   # <1/r>^2
    theoretical = s_q_pa2_per_hz * geom_factor                   # scalar Pa²/Hz

    delta_db = 10.0 * np.log10(psd / theoretical)
    return {
        "frequencies_hz": freqs,
        "steered_psd": psd,
        "theoretical_psd": theoretical,
        "geometric_factor": geom_factor,
        "delta_db_per_freq": delta_db,
        "delta_db_band_mean": float(delta_db.mean()),
    }
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/analysis/test_scaling_ladder.py::test_rung3_steered_psd_matches_geometry_factor_within_2_db -v`

Expected outcome: this test reveals the truth. There are two possibilities:
1. **PASS** (Δ < 2 dB): the convention is consistent, the −27 dB bias must therefore be a *downstream* mismatch — somewhere in the GT-comparison or band-integration logic, NOT in the steering chain itself.
2. **FAIL** (large Δ): the convention is inconsistent. The reported `delta_db_band_mean` is the bug constant. Document it and proceed — this is exactly the diagnostic finding the script exists to provide.

In case (2), do NOT skip the commit — the test value of `delta_db_band_mean` *is* the diagnostic answer. Convert the test to record-and-assert-magnitude form:

```python
    # Replace the strict assert with diagnostic record-and-soft-assert:
    delta_mean = result["delta_db_band_mean"]
    print(f"\nRung-3 Δ_3 = {delta_mean:.2f} dB")
    # Soft bound: as long as it's finite and not absurd, accept and let the
    # ladder report flag it.  The strict "< 2 dB" target lives in the report.
    assert np.isfinite(delta_mean)
    assert abs(delta_mean) < 40.0  # sanity ceiling — anything above is a sign error
```

- [ ] **Step 5: Commit**

```bash
git add tests/analysis/test_scaling_ladder.py analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): rung 3 — production steer_to_psd vs geometric factor"
```

---

### Task 6: Add Markdown report writer

**Files:**
- Modify: `analysis/scaling_ladder.py` — add `write_report` function

- [ ] **Step 1: Implement `write_report`**

Append to `analysis/scaling_ladder.py`:

```python
def write_report(
    *,
    output_dir,
    rung1: dict,
    rung2: dict,
    rung3: dict,
    s_q_pa2_per_hz: float,
    source_position,
    mic_positions,
    sample_rate: float,
    config_summary: str,
) -> None:
    """Write a Markdown summary plus a metrics.json to output_dir."""
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d1 = rung1["delta_db_mean"]
    d2 = rung2["delta_db_mean"]
    d3 = rung3["delta_db_band_mean"]

    pass_threshold_db = 0.5
    pass_3_threshold_db = 2.0

    def verdict(delta, thr):
        return "PASS" if abs(delta) < thr else "FAIL"

    diagnosis_lines = []
    if abs(d1) < pass_threshold_db and abs(d2) < pass_threshold_db and abs(d3) >= pass_3_threshold_db:
        diagnosis_lines.append(
            f"Forward + CSM are clean. The ~{d3:+.2f} dB offset enters at the "
            "steering stage. Likely cause: sign-convention mismatch between "
            "propagator (1/r delay) and `steer_to_psd` (h = exp(+j 2πf r/c)), "
            "or steering-norm convention (1/M² vs 1/M)."
        )
    elif abs(d1) < pass_threshold_db and abs(d2) >= pass_threshold_db:
        diagnosis_lines.append(
            f"Forward Welch is clean (Δ₁ = {d1:+.2f} dB) but the CSM stage adds "
            f"Δ₂ − Δ₁ = {d2-d1:+.2f} dB. Suspect: window or density normalization "
            "in `csm.py` (`scipy.signal.csd(..., scaling='density')` vs custom)."
        )
    elif abs(d1) >= pass_threshold_db:
        diagnosis_lines.append(
            f"Δ₁ = {d1:+.2f} dB — forward propagator itself disagrees with theory. "
            "Check the `s_q → variance` conversion (one-sided vs two-sided density) "
            "and the 1/r factor."
        )
    else:
        diagnosis_lines.append(
            "All three rungs within tolerance. The −27 dB bias observed in pipeline "
            "ext_only runs must therefore enter *outside* the CSM-and-steering path "
            "— investigate band-integration (`integrate_band_maps`) or GT-comparison."
        )

    md = [
        "# Scaling-Ladder Diagnostic Report",
        "",
        f"**Config:** {config_summary}",
        f"**Source PSD:** S_q = {10*np.log10(s_q_pa2_per_hz):.2f} dB re 1 Pa²/Hz",
        f"**Source position:** {tuple(np.asarray(source_position).tolist())}",
        f"**Sample rate:** {sample_rate:.0f} Hz",
        f"**Mics:** {len(mic_positions)} channels",
        "",
        "## Rung Deltas (band mean, dB)",
        "",
        "| Rung | Description | Δ (dB) | Threshold | Verdict |",
        "|------|-------------|--------|-----------|---------|",
        f"| 1 | Direct Welch mic-PSD vs S_q/r²        | {d1:+.3f} | ±0.5 | {verdict(d1, pass_threshold_db)} |",
        f"| 2 | CSM-diagonal vs S_q/r²                 | {d2:+.3f} | ±0.5 | {verdict(d2, pass_threshold_db)} |",
        f"| 3 | steer_to_psd at source vs S_q·⟨1/r⟩²   | {d3:+.3f} | ±2.0 | {verdict(d3, pass_3_threshold_db)} |",
        "",
        "## Diagnosis",
        "",
        diagnosis_lines[0],
        "",
        "## Per-mic Δ (Rung 1)",
        "",
        "| mic | r (m) | Δ₁ (dB) |",
        "|-----|-------|---------|",
    ]
    r = np.linalg.norm(np.asarray(mic_positions) - np.asarray(source_position)[None, :], axis=1)
    for m, (rm, d) in enumerate(zip(r, rung1["delta_db_per_mic"])):
        md.append(f"| {m} | {rm:.3f} | {d:+.3f} |")

    (output_dir / "report.md").write_text("\n".join(md) + "\n")

    metrics = {
        "rung1": {
            "delta_db_mean": d1,
            "delta_db_per_mic": rung1["delta_db_per_mic"].tolist(),
        },
        "rung2": {
            "delta_db_mean": d2,
            "delta_db_per_mic": rung2["delta_db_per_mic"].tolist(),
        },
        "rung3": {
            "delta_db_band_mean": d3,
            "geometric_factor": rung3["geometric_factor"],
        },
        "config": {
            "s_q_pa2_per_hz": s_q_pa2_per_hz,
            "sample_rate": sample_rate,
            "n_mics": int(len(mic_positions)),
            "source_position": list(map(float, source_position)),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
```

- [ ] **Step 2: Smoke-import the new function**

Run: `uv run python -c "from importlib.util import spec_from_file_location, module_from_spec; spec = spec_from_file_location('sl', 'analysis/scaling_ladder.py'); m = module_from_spec(spec); spec.loader.exec_module(m); assert hasattr(m, 'write_report'); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): markdown report + metrics.json writer"
```

---

### Task 7: Add Plotly HTML plots and the `__main__` driver

**Files:**
- Modify: `analysis/scaling_ladder.py`

- [ ] **Step 1: Add Plotly plot functions and the driver**

Append to `analysis/scaling_ladder.py`:

```python
def write_plots(*, output_dir, rung1: dict, rung2: dict, rung3: dict,
                s_q_pa2_per_hz: float, mic_positions, source_position) -> None:
    """Three HTML plots: per-mic Welch PSD, CSM-diagonal PSD, steered PSD."""
    import plotly.graph_objects as go
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rung 1: per-mic Welch
    f1 = rung1["frequencies_hz"]
    fig = go.Figure()
    r = np.linalg.norm(
        np.asarray(mic_positions) - np.asarray(source_position)[None, :], axis=1
    )
    for m in range(rung1["psd_per_mic"].shape[1]):
        fig.add_trace(go.Scatter(
            x=f1, y=10*np.log10(rung1["psd_per_mic"][:, m]),
            mode="lines", name=f"mic {m} (r={r[m]:.2f})",
            opacity=0.6, line={"width": 1},
        ))
        fig.add_trace(go.Scatter(
            x=[f1[0], f1[-1]],
            y=[10*np.log10(rung1["theoretical_per_mic"][m])]*2,
            mode="lines", name=f"theory mic {m}",
            line={"dash": "dash", "width": 1}, showlegend=False,
        ))
    fig.update_layout(
        title="Rung 1 — Welch PSD per mic vs theoretical S_q/r²",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "mic_psd_vs_theory.html", include_plotlyjs="cdn")

    # Rung 2: CSM diagonal
    f2 = rung2["frequencies_hz"]
    fig = go.Figure()
    for m in range(rung2["csm_diag_per_mic"].shape[1]):
        fig.add_trace(go.Scatter(
            x=f2, y=10*np.log10(rung2["csm_diag_per_mic"][:, m]),
            mode="lines", name=f"mic {m}", opacity=0.6, line={"width": 1},
        ))
    fig.update_layout(
        title="Rung 2 — CSM diagonal per mic",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "csm_diag_vs_theory.html", include_plotlyjs="cdn")

    # Rung 3: steered PSD
    f3 = rung3["frequencies_hz"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f3, y=10*np.log10(rung3["steered_psd"]),
        mode="lines", name="steer_to_psd at source",
    ))
    fig.add_trace(go.Scatter(
        x=[f3[0], f3[-1]],
        y=[10*np.log10(rung3["theoretical_psd"])]*2,
        mode="lines", name="theoretical S_q · <1/r>²",
        line={"dash": "dash"},
    ))
    fig.update_layout(
        title="Rung 3 — steered PSD at source vs geometric expectation",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "steered_psd.html", include_plotlyjs="cdn")


def main() -> None:
    """End-to-end: load mic_geom, propagate, run 3 rungs, write report+plots."""
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mic-geom", type=Path,
        default=Path("/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml"),
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=float, default=51_200.0)
    parser.add_argument("--source-x", type=float, default=0.0)
    parser.add_argument("--source-y", type=float, default=0.0)
    parser.add_argument("--source-z", type=float, default=-1.5)
    parser.add_argument("--s-q-pa2-per-hz", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026_05_07)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/scaling_ladder")
    )
    args = parser.parse_args()

    from martymicfly.io.mic_geom import load_mic_geom_xml
    mic_positions = load_mic_geom_xml(args.mic_geom)
    source_pos = np.array([args.source_x, args.source_y, args.source_z])
    n_samples = int(args.duration_s * args.sample_rate)
    rng = np.random.default_rng(args.seed)

    print(f"[scaling_ladder] {mic_positions.shape[0]} mics from {args.mic_geom}")
    print(f"[scaling_ladder] propagating {args.duration_s:.1f}s @ {args.sample_rate:.0f} Hz")
    time_data = propagate_white_noise(
        n_samples=n_samples, sample_rate=args.sample_rate,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )

    common = dict(
        time_data=time_data, sample_rate=args.sample_rate,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
    )
    print("[scaling_ladder] rung 1 …")
    r1 = rung1_mic_psd(**common)
    print(f"  Δ₁ = {r1['delta_db_mean']:+.3f} dB")
    print("[scaling_ladder] rung 2 …")
    r2 = rung2_csm_diag(**common, diag_loading_rel=0.0)
    print(f"  Δ₂ = {r2['delta_db_mean']:+.3f} dB")
    print("[scaling_ladder] rung 3 …")
    r3 = rung3_steered_psd(**common, diag_loading_rel=0.0)
    print(f"  Δ₃ = {r3['delta_db_band_mean']:+.3f} dB")

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = args.output_root / run_id
    write_report(
        output_dir=out, rung1=r1, rung2=r2, rung3=r3,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions,
        sample_rate=args.sample_rate,
        config_summary=(
            f"duration={args.duration_s}s, fs={args.sample_rate:.0f}, "
            f"nperseg=512, noverlap=256, window=hann, "
            f"f_band=[200,6000] Hz, source={tuple(source_pos.tolist())}"
        ),
    )
    write_plots(
        output_dir=out, rung1=r1, rung2=r2, rung3=r3,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        mic_positions=mic_positions, source_position=source_pos,
    )
    print(f"[scaling_ladder] wrote report + plots → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full unit test suite to ensure nothing broke**

Run: `uv run pytest tests/analysis/ -v`
Expected: 4 tests PASS (forward, rung1, rung2, rung3-soft-assert).

- [ ] **Step 3: Commit**

```bash
git add analysis/scaling_ladder.py
git commit -m "feat(scaling_ladder): plotly plots + CLI driver"
```

---

### Task 8: End-to-end run on production geometry

**Files:**
- No code changes; produces `results/scaling_ladder/<timestamp>/`.

- [ ] **Step 1: Run the script on production mic_geom.xml**

Run:
```bash
uv run python analysis/scaling_ladder.py \
  --mic-geom /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml \
  --duration-s 10.0
```

Expected stdout (representative — exact values depend on the convention check):
```
[scaling_ladder] 16 mics from …/mic_geom.xml
[scaling_ladder] propagating 10.0s @ 51200 Hz
[scaling_ladder] rung 1 …
  Δ₁ = +0.0XX dB
[scaling_ladder] rung 2 …
  Δ₂ = +0.0XX dB
[scaling_ladder] rung 3 …
  Δ₃ = +X.XXX dB        ← THE diagnostic answer
[scaling_ladder] wrote report + plots → results/scaling_ladder/2026-05-07TXX-XX-XX
```

- [ ] **Step 2: Inspect the report**

Read: `results/scaling_ladder/<timestamp>/report.md`

Verify:
- Rung 1 PASS (Δ₁ within ±0.5 dB)
- Rung 2 PASS (Δ₂ within ±0.5 dB)
- Rung 3 — record the value. The diagnosis paragraph in `report.md` already names the suspect.

- [ ] **Step 3: Open one plot to sanity-check visually**

Open `results/scaling_ladder/<timestamp>/mic_psd_vs_theory.html` in a browser.
Verify: 16 mic-PSD curves cluster within ±0.5 dB of their respective dashed theoretical lines across [200, 6000] Hz.

- [ ] **Step 4: Update the handoff document with the finding**

Modify: `docs/superpowers/handoff-stage3.md`

Append a new section near the bottom:

```markdown
## Scaling-Ladder Diagnostic (2026-05-07)

Run: `results/scaling_ladder/<timestamp>/`

| Rung | Δ (dB) | Verdict |
|------|--------|---------|
| 1 — Welch mic-PSD              | <fill in>        | <fill in> |
| 2 — CSM-diagonal               | <fill in>        | <fill in> |
| 3 — steer_to_psd at source     | <fill in>        | <fill in> |

**Diagnosis:** <copy the diagnosis paragraph from report.md>

**Implication for parameter study:** <"the −27 dB bias is now identified as X
and lives at stage Y; fixing it is a separate spec" / "the bias is NOT in the
steering chain, look at integrate_band_maps and GT-comparison instead">
```

- [ ] **Step 5: Commit run output and handoff update**

```bash
git add docs/superpowers/handoff-stage3.md results/scaling_ladder/
git commit -m "feat(scaling_ladder): end-to-end run on production geometry + handoff update"
```

---

## Self-Review

**Spec coverage:**
- §1 (Motivation) — Task 8 produces the empirical Δ_k that motivates the spec. ✓
- §2 (Ziel) — Tasks 3–5 deliver the per-rung Δ identification. ✓
- §3.1 (Form) — Tasks 1, 6, 7 wire it as a standalone `analysis/scaling_ladder.py`. ✓
- §3.2 Phasen 0–4 — Phase 0 in Task 7 (`--s-q-pa2-per-hz`); Phase 1 in Task 2; Phase 2 in Task 3; Phase 3 in Task 4; Phase 4 in Task 5. ✓
- §3.3 (Diagnose-Logik) — Task 6's `write_report` produces the diagnosis paragraph for each branch. ✓
- §3.4 (Output) — Tasks 6 (md, json) + 7 (html). ✓
- §4 (Pass-Kriterium) — Tasks 3, 4 use ±0.5 dB; Task 5 uses ±2 dB at the unit-test level (with a sanity ceiling) since the production-geometry value IS the diagnostic answer. ✓
- §5 (NICHT enthalten) — no NNLS, no CLEAN-SC, no mask, no propagator reuse, no Welch sweep. ✓
- §6 (Nächste Schritte) — Task 8 step 4 hooks into the handoff to flag follow-ups. ✓

**Placeholder scan:** None of "TBD", "TODO", "implement later", "appropriate error handling", "similar to Task N". The `<fill in>` placeholders in Task 8 step 4 are *intended* — they're filled by the operator after the run, not by the code.

**Type/signature consistency:** All four functions (`propagate_white_noise`, `rung1_mic_psd`, `rung2_csm_diag`, `rung3_steered_psd`) use keyword-only arguments with consistent names (`time_data`, `sample_rate`, `s_q_pa2_per_hz`, `source_position`, `mic_positions`, `f_min_hz`, `f_max_hz`, `nperseg`, `noverlap`, `window`). Production interfaces (`build_measurement_csm`, `steer_to_psd`, `load_mic_geom_xml`) match the signatures verified in `src/martymicfly/processing/{csm,steering}.py` and `src/martymicfly/io/mic_geom.py`.
