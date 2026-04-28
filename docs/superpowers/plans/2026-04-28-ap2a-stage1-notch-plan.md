# AP2-A Stufe 1 — Notch-Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AP2-A Stufe 1 (RPM-getriebene Cascade-Notch-Filterung) End-to-End auf den synthetischen 16-Mic-Daten zum Laufen bringen — mit Pipeline-Skelett, Metriken und Plotly-Plots.

**Architecture:** Editable-Pip-Deps für `notchfilter` (Filter-Engine) und `drone-synthdata` (Synth-Erzeugung extern). Eigenes Subpaket `src/martymicfly/` mit IO-, Processing-, Eval- und CLI-Modulen. Stage-Protokoll macht spätere Stufen 2/3 plug-in-fähig.

**Tech Stack:** Python 3.13, `notchfilter.cascade.CascadeNotchFilter` (mode='external', zero_phase=True), Acoular-Sources, scipy.signal (Welch+Spectrogram), Plotly, h5py, pydantic-2, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-04-28-ap2a-stage1-notch-design.md`

---

## Task-Übersicht

1. Cleanup: Notch-Altcode aus `analysis/` entfernen
2. Editable Pip-Deps + neue Deps + Paket-Skelett
3. Pydantic-Konfig-Modelle (`config.py`)
4. Test-Fixtures: `make_tiny_synth.py` + `tiny_synth.h5` + `tiny_geom.xml`
5. `io/synth_h5.py` — Loader für Acoular-Synth-HDF5
6. `io/mic_geom.py` — XML-Loader für Mic-Geometrie
7. `processing/frequencies.py` — RPM→BPF, Harmonik-Matrix, r-Schedule
8. `processing/sources.py` — `ArraySamplesGenerator`, `ArrayFreqSource`
9. `processing/pipeline.py` — `Stage`-Protokoll, `PipelineContext`, `run_pipeline`
10. `processing/notch.py` — `NotchStage` (Cascade-Smoke-Test gegen `notchfilter`)
11. `eval/metrics.py` — `compute_metrics`
12. `eval/plots.py` — `plot_channel_html`
13. `io/write_filtered.py` — gefilterte HDF5 + Provenance
14. `cli/run_notch.py` + `configs/example_notch.yaml` + Pipeline-Integration-Test
15. Smoke-Run auf `ap2a_synthesis_16mic_gaptip.h5`

---

## Task 1: Cleanup Notch-Altcode aus `analysis/`

**Files:**
- Remove: `analysis/mimo_adaptive_iir_filter.py`
- Remove: `analysis/mimo_filter_analysis.py`
- Remove: `analysis/iir_filter_validation.py`
- Remove: `analysis/frequency_initialization.py`
- Remove: `analysis/MIMO_FILTER_README.md`
- Remove: `analysis/mimo_filter_config_schema.yaml`
- Remove: `analysis/PULL_PUSH_ANALYSIS_NOTES.md`

- [ ] **Step 1: Verifizieren, dass alle sieben Dateien existieren**

Run:
```bash
ls analysis/mimo_adaptive_iir_filter.py analysis/mimo_filter_analysis.py \
   analysis/iir_filter_validation.py analysis/frequency_initialization.py \
   analysis/MIMO_FILTER_README.md analysis/mimo_filter_config_schema.yaml \
   analysis/PULL_PUSH_ANALYSIS_NOTES.md
```
Expected: alle sieben Dateien werden ohne Fehler aufgelistet.

- [ ] **Step 2: Prüfen, ob andere `analysis/`-Dateien diese importieren**

Run:
```bash
grep -lE "mimo_adaptive_iir_filter|mimo_filter_analysis|iir_filter_validation|frequency_initialization" analysis/*.py
```
Expected: keine Treffer. Falls Treffer → Aufgabe stoppt, manuelle Klärung nötig (Spec sieht keine Cross-Imports vor; `sound_power_*`, `beamforming_comparison`, `spectral_comparison`, `data_loader` sind eigenständig).

- [ ] **Step 3: Dateien entfernen**

Run:
```bash
git rm analysis/mimo_adaptive_iir_filter.py \
       analysis/mimo_filter_analysis.py \
       analysis/iir_filter_validation.py \
       analysis/frequency_initialization.py \
       analysis/MIMO_FILTER_README.md \
       analysis/mimo_filter_config_schema.yaml \
       analysis/PULL_PUSH_ANALYSIS_NOTES.md
```

Hinweis: Die Dateien sind aktuell untracked (siehe `git status`). `git rm` schlägt dann fehl. In dem Fall:
```bash
rm analysis/mimo_adaptive_iir_filter.py analysis/mimo_filter_analysis.py \
   analysis/iir_filter_validation.py analysis/frequency_initialization.py \
   analysis/MIMO_FILTER_README.md analysis/mimo_filter_config_schema.yaml \
   analysis/PULL_PUSH_ANALYSIS_NOTES.md
```
und im Commit unten dann `git add analysis/` für die übrig gebliebenen tracked-Dateien (in dem Fall ist allerdings nichts zu committen, weil die Dateien nie tracked waren — dann Task 1 als „nichts zu commiten" abschließen, Cleanup ist trotzdem erfolgt).

- [ ] **Step 4: Commit (sofern tracked)**

Run:
```bash
git status --short analysis/
```
Wenn `D` (deleted)-Einträge sichtbar:
```bash
git commit -m "chore: remove legacy MIMO IIR notch implementation

Superseded by notchfilter package (editable pip dep introduced in
the next task). Independent analysis modules (sound_power_*,
beamforming_comparison, spectral_comparison, data_loader) bleiben
unangetastet."
```
Wenn keine `D`-Einträge: kein Commit nötig (Dateien waren untracked); weiter zu Task 2.

---

## Task 2: Editable Pip-Deps + Paket-Skelett

**Files:**
- Modify: `pyproject.toml`
- Create: `src/martymicfly/__init__.py`
- Create: `src/martymicfly/io/__init__.py`
- Create: `src/martymicfly/processing/__init__.py`
- Create: `src/martymicfly/eval/__init__.py`
- Create: `src/martymicfly/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: `pydantic` als Dependency hinzufügen**

Run:
```bash
uv add "pydantic>=2"
```
Expected: `pyproject.toml` enthält `"pydantic>=2"` in `dependencies`, `uv.lock` wird aktualisiert.

- [ ] **Step 2: `notchfilter` editable installieren**

Run:
```bash
uv add --editable /home/steffen/Code/NotchFilter
```
Expected: `pyproject.toml` enthält `"notchfilter"` in `dependencies` und einen Eintrag unter `[tool.uv.sources]`:
```toml
[tool.uv.sources]
notchfilter = { path = "/home/steffen/Code/NotchFilter", editable = true }
```

- [ ] **Step 3: `drone-synthdata` editable installieren**

Run:
```bash
uv add --editable /home/steffen/Code/drone_synthdata
```
Expected: ergänzt `"drone-synthdata"` in `dependencies` und einen weiteren Eintrag in `[tool.uv.sources]`.

- [ ] **Step 4: `pytest` als Dev-Dependency hinzufügen**

Run:
```bash
uv add --dev pytest
```
Expected: `[dependency-groups].dev = ["pytest>=…"]` (oder `[tool.uv].dev-dependencies`, je nach uv-Version).

- [ ] **Step 5: Imports der Editable-Pakete verifizieren**

Run:
```bash
uv run python -c "import notchfilter; from notchfilter.cascade import CascadeNotchFilter; import drone_synthdata; print('OK')"
```
Expected: `OK`. Wenn `drone_synthdata` failt: prüfen, ob das Verzeichnis im drone_synthdata-Repo so heißt — der Importname kann von `name` im pyproject abweichen (Distribution-Name `drone-synthdata` → Import-Name `drone_synthdata`).

- [ ] **Step 6: `src/`-Layout in `pyproject.toml` aktivieren**

Falls noch nicht gesetzt, in `pyproject.toml` ergänzen:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/martymicfly"]
```
(Falls Build-Backend nicht `hatchling` ist, das zum vorhandenen Backend passende Äquivalent — ohne diese Zeile findet `uv` das Paket im `src/`-Layout nicht.)

- [ ] **Step 7: Subpaket-`__init__.py`-Dateien anlegen**

Create `src/martymicfly/__init__.py`:
```python
"""MartyMicFly — DFG project 'Fliegendes Messmikrofon' processing code."""

__version__ = "0.1.0"
```

Create `src/martymicfly/io/__init__.py`, `src/martymicfly/processing/__init__.py`, `src/martymicfly/eval/__init__.py`, `src/martymicfly/cli/__init__.py` — jeweils leer (`""""""`-Header optional).

Create `tests/__init__.py` — leer.

Create `tests/conftest.py`:
```python
"""Pytest config for martymicfly tests."""

import sys
from pathlib import Path

# Ensure src/ layout is importable when running pytest directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

- [ ] **Step 8: Paket-Import verifizieren**

Run:
```bash
uv run python -c "import martymicfly; print(martymicfly.__version__)"
uv run pytest --collect-only -q
```
Expected: `0.1.0`, dann „no tests collected" (oder „0 tests collected") ohne Import-Fehler.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/martymicfly tests/__init__.py tests/conftest.py
git commit -m "build: add notchfilter+drone-synthdata editable deps, package skeleton

- pydantic>=2, pytest (dev), notchfilter, drone-synthdata
- src/martymicfly/{io,processing,eval,cli}/__init__.py
- tests/conftest.py erlaubt src-Layout-Imports im Test"
```

---

## Task 3: Pydantic-Konfig-Modelle

**Files:**
- Create: `src/martymicfly/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_config.py`:
```python
"""Tests for martymicfly.config."""

import pytest
from pydantic import ValidationError

from martymicfly.config import AppConfig, NotchConfig, PoleRadiusConfig


def _minimal_config_dict():
    return {
        "input": {"audio_h5": "/tmp/x.h5", "mic_geom_xml": "/tmp/m.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 20},
        "notch": {
            "pole_radius": {"mode": "scalar", "value": 0.9994},
            "multichannel": False,
            "block_size": 4096,
        },
        "metrics": {
            "welch_nperseg": 8192,
            "welch_noverlap": 4096,
            "bandwidth_factor": 1.0,
            "broadband_low_hz": None,
        },
        "plots": {
            "enabled": True,
            "fmax_hz": None,
            "spectrogram_window": 4096,
            "spectrogram_overlap": 2048,
            "channel_subset": None,
        },
        "output": {
            "dir": "results/notch/{run_id}",
            "filtered_h5": "filtered.h5",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "plots_subdir": "plots",
            "copy_config": True,
        },
    }


def test_minimal_config_validates():
    cfg = AppConfig(**_minimal_config_dict())
    assert cfg.rotor.n_blades == 2
    assert cfg.notch.pole_radius.mode == "scalar"
    assert cfg.notch.pole_radius.value == pytest.approx(0.9994)


def test_segment_explicit_requires_start_end():
    d = _minimal_config_dict()
    d["segment"] = {"mode": "explicit"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_requires_k_cover_and_margin():
    d = _minimal_config_dict()
    d["notch"]["pole_radius"] = {"mode": "linear"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_pole_radius_linear_full():
    d = _minimal_config_dict()
    d["notch"]["pole_radius"] = {
        "mode": "linear",
        "k_cover": 1.5,
        "margin_hz": 5.0,
        "delta_bpf_hz": None,
        "r_min": 0.90,
        "r_max": 0.9995,
    }
    cfg = AppConfig(**d)
    assert cfg.notch.pole_radius.k_cover == pytest.approx(1.5)


def test_channels_list_requires_list_field():
    d = _minimal_config_dict()
    d["channels"] = {"selection": "list"}
    with pytest.raises(ValidationError):
        AppConfig(**d)


def test_config_hash_is_deterministic():
    d = _minimal_config_dict()
    cfg1 = AppConfig(**d)
    cfg2 = AppConfig(**d)
    assert cfg1.config_hash() == cfg2.config_hash()
    assert len(cfg1.config_hash()) == 8
```

- [ ] **Step 2: Test fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: `ImportError: cannot import name 'AppConfig' from 'martymicfly.config'`.

- [ ] **Step 3: `config.py` implementieren**

Create `src/martymicfly/config.py`:
```python
"""Pydantic models for martymicfly notch-pipeline configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_h5: str
    mic_geom_xml: str


class SegmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["middle", "head", "tail", "explicit"]
    duration: float | None = None
    start: float | None = None
    end: float | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "SegmentConfig":
        if self.mode == "explicit":
            if self.start is None or self.end is None:
                raise ValueError("segment.mode=explicit requires start and end")
            if self.end <= self.start:
                raise ValueError("segment.end must be > segment.start")
        else:
            if self.duration is None or self.duration <= 0:
                raise ValueError(
                    f"segment.mode={self.mode} requires positive duration"
                )
        return self


class ChannelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selection: Literal["all", "list"]
    list: list[int] | None = None

    @model_validator(mode="after")
    def _check_list(self) -> "ChannelsConfig":
        if self.selection == "list" and not self.list:
            raise ValueError("channels.selection=list requires non-empty list")
        return self


class RotorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_blades: int = Field(ge=1)
    n_harmonics: int = Field(ge=1)


class PoleRadiusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["scalar", "linear"]
    value: float | None = None
    k_cover: float | None = None
    margin_hz: float | None = None
    delta_bpf_hz: float | None = None
    r_min: float = 0.90
    r_max: float = 0.9995

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "PoleRadiusConfig":
        if self.mode == "scalar":
            if self.value is None or not (0.0 < self.value < 1.0):
                raise ValueError("pole_radius.scalar requires value in (0, 1)")
        else:  # linear
            if self.k_cover is None or self.margin_hz is None:
                raise ValueError(
                    "pole_radius.linear requires k_cover and margin_hz"
                )
            if not (0.0 < self.r_min < self.r_max < 1.0):
                raise ValueError("require 0 < r_min < r_max < 1")
        return self


class NotchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pole_radius: PoleRadiusConfig
    multichannel: bool = False
    block_size: int = Field(default=4096, ge=64)


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    welch_nperseg: int = Field(ge=64)
    welch_noverlap: int = Field(ge=0)
    bandwidth_factor: float = Field(default=1.0, gt=0.0)
    broadband_low_hz: float | None = None


class PlotsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    fmax_hz: float | None = None
    spectrogram_window: int = Field(ge=64)
    spectrogram_overlap: int = Field(ge=0)
    channel_subset: list[int] | None = None


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str
    filtered_h5: str = "filtered.h5"
    metrics_json: str = "metrics.json"
    metrics_csv: str = "metrics.csv"
    plots_subdir: str = "plots"
    copy_config: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InputConfig
    segment: SegmentConfig
    channels: ChannelsConfig
    rotor: RotorConfig
    notch: NotchConfig
    metrics: MetricsConfig
    plots: PlotsConfig
    output: OutputConfig

    def canonical_json(self) -> str:
        return self.model_dump_json(exclude_none=False)

    def config_hash(self) -> str:
        payload = json.loads(self.canonical_json())
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: alle 6 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/config.py tests/test_config.py
git commit -m "feat(config): pydantic models for notch pipeline config"
```

---

## Task 4: Test-Fixtures (`make_tiny_synth.py`, `tiny_synth.h5`, `tiny_geom.xml`)

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/make_tiny_synth.py`
- Create: `tests/fixtures/tiny_synth.h5` (generiert)
- Create: `tests/fixtures/tiny_geom.xml`

- [ ] **Step 1: `tiny_geom.xml` schreiben**

Create `tests/fixtures/tiny_geom.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<MicArray name="tiny_4mic">
  <pos Name="Point 1" x="0.1" y="0.0" z="0.0"/>
  <pos Name="Point 2" x="0.0" y="0.1" z="0.0"/>
  <pos Name="Point 3" x="-0.1" y="0.0" z="0.0"/>
  <pos Name="Point 4" x="0.0" y="-0.1" z="0.0"/>
</MicArray>
```

- [ ] **Step 2: `__init__.py` anlegen**

Create `tests/fixtures/__init__.py` (leer).

- [ ] **Step 3: `make_tiny_synth.py` schreiben**

Create `tests/fixtures/make_tiny_synth.py`:
```python
"""Generate tests/fixtures/tiny_synth.h5 — deterministic Acoular-format synth."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

FIXTURE = Path(__file__).resolve().parent / "tiny_synth.h5"

SAMPLE_RATE = 16_000.0
DURATION_S = 1.0
N_CHANNELS = 4
N_BLADES = 2  # so BPF = rpm * 2 / 60

ESC_RPM = {"ESC1": 3000.0, "ESC2": 3600.0}  # → BPF 100 Hz / 120 Hz
N_HARMONICS = 5
EXTERNAL_TONE_HZ = 800.0
EXTERNAL_TONE_AMPLITUDE = 0.2
NOISE_RMS = 0.01

ESC_DT_S = 1e-3  # 1 kHz telemetry rate (matches real synth)


def build_signal(rng: np.random.Generator) -> np.ndarray:
    """Return time_data of shape (N, C) float64."""
    n = int(round(SAMPLE_RATE * DURATION_S))
    t = np.arange(n) / SAMPLE_RATE

    base = np.zeros(n, dtype=np.float64)
    for rpm in ESC_RPM.values():
        bpf = rpm * N_BLADES / 60.0
        for h in range(1, N_HARMONICS + 1):
            base += np.sin(2 * np.pi * h * bpf * t) / N_HARMONICS

    base += EXTERNAL_TONE_AMPLITUDE * np.sin(2 * np.pi * EXTERNAL_TONE_HZ * t)

    data = np.empty((n, N_CHANNELS), dtype=np.float64)
    for c in range(N_CHANNELS):
        noise = rng.standard_normal(n) * NOISE_RMS
        data[:, c] = base + noise
    return data


def build_telemetry() -> dict[str, dict[str, np.ndarray]]:
    n_t = int(round(DURATION_S / ESC_DT_S)) + 1
    timestamps = np.arange(n_t) * ESC_DT_S
    return {
        name: {
            "rpm": np.full(n_t, rpm, dtype=np.float64),
            "timestamp": timestamps.astype(np.float64),
        }
        for name, rpm in ESC_RPM.items()
    }


def write_fixture(path: Path = FIXTURE) -> Path:
    rng = np.random.default_rng(seed=42)
    data = build_signal(rng)
    telemetry = build_telemetry()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        td = f.create_dataset("time_data", data=data, dtype="float64")
        td.attrs["sample_freq"] = np.float64(SAMPLE_RATE)
        grp = f.create_group("esc_telemetry")
        for name, esc in telemetry.items():
            sub = grp.create_group(name)
            sub.create_dataset("rpm", data=esc["rpm"], dtype="float64")
            sub.create_dataset("timestamp", data=esc["timestamp"], dtype="float64")
    return path


if __name__ == "__main__":
    out = write_fixture()
    print(f"wrote {out}")
```

- [ ] **Step 4: Fixture generieren und in Repo aufnehmen**

Run:
```bash
uv run python tests/fixtures/make_tiny_synth.py
ls -la tests/fixtures/tiny_synth.h5
```
Expected: Datei existiert (~ wenige hundert KB).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/make_tiny_synth.py \
        tests/fixtures/tiny_synth.h5 tests/fixtures/tiny_geom.xml
git commit -m "test: deterministic tiny synth HDF5 fixture (4ch, 1s, 2 ESCs)"
```

---

## Task 5: `io/synth_h5.py` — Loader

**Files:**
- Create: `src/martymicfly/io/synth_h5.py`
- Test: `tests/test_io_synth_h5.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_io_synth_h5.py`:
```python
"""Tests for martymicfly.io.synth_h5."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from martymicfly.io.synth_h5 import load_synth_h5

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_synth.h5"


def test_load_returns_expected_keys():
    out = load_synth_h5(FIXTURE)
    assert set(out.keys()) == {"time_data", "sample_rate", "rpm_per_esc", "duration"}


def test_load_shape_and_dtype():
    out = load_synth_h5(FIXTURE)
    assert out["time_data"].shape == (16_000, 4)
    assert out["time_data"].dtype == np.float64
    assert out["sample_rate"] == pytest.approx(16_000.0)
    assert out["duration"] == pytest.approx(1.0)


def test_load_rpm_per_esc_structure():
    out = load_synth_h5(FIXTURE)
    rpe = out["rpm_per_esc"]
    assert set(rpe.keys()) == {"ESC1", "ESC2"}
    for esc in rpe.values():
        assert set(esc.keys()) == {"rpm", "timestamp"}
        assert esc["rpm"].dtype == np.float64
        assert esc["timestamp"].dtype == np.float64
        assert np.all(np.diff(esc["timestamp"]) > 0)
    assert np.allclose(rpe["ESC1"]["rpm"], 3000.0)
    assert np.allclose(rpe["ESC2"]["rpm"], 3600.0)


def test_load_missing_sample_freq_raises(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("time_data", data=np.zeros((10, 2), dtype=np.float64))
        f.create_group("esc_telemetry")
    with pytest.raises(ValueError, match="sample_freq"):
        load_synth_h5(bad)


def test_load_non_monotonic_timestamps_raises(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        td = f.create_dataset("time_data", data=np.zeros((10, 2), dtype=np.float64))
        td.attrs["sample_freq"] = 1000.0
        grp = f.create_group("esc_telemetry/ESC1")
        grp.create_dataset("rpm", data=np.array([100.0, 100.0]))
        grp.create_dataset("timestamp", data=np.array([0.0, 0.0]))  # not monotonic
    with pytest.raises(ValueError, match="monoton"):
        load_synth_h5(bad)
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_io_synth_h5.py -v
```
Expected: ImportError.

- [ ] **Step 3: `synth_h5.py` implementieren**

Create `src/martymicfly/io/synth_h5.py`:
```python
"""Load synthetic mic+telemetry data in Acoular HDF5 layout."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_synth_h5(path: str | Path) -> dict:
    """Load Acoular-format synth HDF5.

    Layout expected::

        /time_data    (N, C) float64  + attr sample_freq (Hz)
        /esc_telemetry/<ESC_NAME>/rpm       (T,) float64
                                  /timestamp (T,) float64 (monoton steigend, in Sekunden)

    Returns
    -------
    dict
        Schlüssel ``time_data``, ``sample_rate``, ``rpm_per_esc``, ``duration``.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "time_data" not in f:
            raise ValueError(f"{path}: missing 'time_data' dataset")
        td = f["time_data"]
        if td.ndim != 2:
            raise ValueError(f"{path}: time_data must be 2D, got {td.ndim}D")
        if "sample_freq" not in td.attrs:
            raise ValueError(f"{path}: time_data is missing 'sample_freq' attribute")

        sample_rate = float(td.attrs["sample_freq"])
        time_data = np.asarray(td[()], dtype=np.float64)

        rpm_per_esc: dict[str, dict[str, np.ndarray]] = {}
        if "esc_telemetry" in f:
            for esc_name, esc_grp in f["esc_telemetry"].items():
                rpm = np.asarray(esc_grp["rpm"][()], dtype=np.float64)
                ts = np.asarray(esc_grp["timestamp"][()], dtype=np.float64)
                if ts.size > 1 and not np.all(np.diff(ts) > 0):
                    raise ValueError(
                        f"{path}: timestamps for {esc_name} not strictly monoton increasing"
                    )
                if rpm.shape != ts.shape:
                    raise ValueError(
                        f"{path}: rpm/timestamp shape mismatch for {esc_name}: "
                        f"{rpm.shape} vs {ts.shape}"
                    )
                rpm_per_esc[esc_name] = {"rpm": rpm, "timestamp": ts}

    duration = time_data.shape[0] / sample_rate
    return {
        "time_data": time_data,
        "sample_rate": sample_rate,
        "rpm_per_esc": rpm_per_esc,
        "duration": duration,
    }
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_io_synth_h5.py -v
```
Expected: alle 5 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/synth_h5.py tests/test_io_synth_h5.py
git commit -m "feat(io): load_synth_h5 for Acoular-format synth recordings"
```

---

## Task 6: `io/mic_geom.py` — XML-Loader

**Files:**
- Create: `src/martymicfly/io/mic_geom.py`
- Test: `tests/test_io_mic_geom.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_io_mic_geom.py`:
```python
"""Tests for martymicfly.io.mic_geom."""

from pathlib import Path

import numpy as np
import pytest

from martymicfly.io.mic_geom import load_mic_geom_xml

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_geom.xml"


def test_load_returns_correct_shape_and_order():
    pos = load_mic_geom_xml(FIXTURE)
    assert pos.shape == (4, 3)
    assert pos.dtype == np.float64
    np.testing.assert_allclose(pos[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(pos[1], [0.0, 0.1, 0.0])
    np.testing.assert_allclose(pos[2], [-0.1, 0.0, 0.0])
    np.testing.assert_allclose(pos[3], [0.0, -0.1, 0.0])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mic_geom_xml(tmp_path / "nope.xml")


def test_load_malformed_xml_raises(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<MicArray><pos x='1' y='2'/></MicArray>")  # missing z
    with pytest.raises(ValueError, match="z"):
        load_mic_geom_xml(bad)
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_io_mic_geom.py -v
```
Expected: ImportError.

- [ ] **Step 3: `mic_geom.py` implementieren**

Create `src/martymicfly/io/mic_geom.py`:
```python
"""Load Acoular-format MicArray XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_mic_geom_xml(path: str | Path) -> np.ndarray:
    """Parse `<MicArray>`/`<pos>` XML, return positions as ``(M, 3)`` float64.

    Reihenfolge entspricht dem Auftreten der ``<pos>``-Elemente.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[tuple[float, float, float]] = []
    for el in root.findall("pos"):
        try:
            x = float(el.attrib["x"])
            y = float(el.attrib["y"])
            z = float(el.attrib["z"])
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"{path}: <pos> missing attribute {missing!r}"
            ) from exc
        rows.append((x, y, z))
    if not rows:
        raise ValueError(f"{path}: no <pos> elements found")
    return np.array(rows, dtype=np.float64)
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_io_mic_geom.py -v
```
Expected: alle 3 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/mic_geom.py tests/test_io_mic_geom.py
git commit -m "feat(io): load_mic_geom_xml for Acoular MicArray XML"
```

---

## Task 7: `processing/frequencies.py` — RPM→BPF, Harmonik-Matrix, r-Schedule

**Files:**
- Create: `src/martymicfly/processing/frequencies.py`
- Test: `tests/test_frequencies.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_frequencies.py`:
```python
"""Tests for martymicfly.processing.frequencies."""

import numpy as np
import pytest

from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
    linear_r_schedule,
)


def _const_rpm_telemetry(rpm: float, t_end: float = 1.5):
    return {"rpm": np.array([rpm, rpm]), "timestamp": np.array([0.0, t_end])}


def test_interpolate_constant_rpm_yields_constant_bpf():
    rpm_per_esc = {
        "ESC1": _const_rpm_telemetry(3000.0),
        "ESC2": _const_rpm_telemetry(3600.0),
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=1000, sample_rate=1000.0, n_blades=2,
    )
    assert bpf.shape == (1000, 2)
    np.testing.assert_allclose(bpf[:, 0], 100.0)  # ESC1 alphabetical
    np.testing.assert_allclose(bpf[:, 1], 120.0)


def test_interpolate_uses_alphabetical_esc_order():
    rpm_per_esc = {
        "ESC2": _const_rpm_telemetry(3600.0),
        "ESC1": _const_rpm_telemetry(3000.0),
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=10, sample_rate=1000.0, n_blades=2,
    )
    np.testing.assert_allclose(bpf[:, 0], 100.0)  # ESC1 first
    np.testing.assert_allclose(bpf[:, 1], 120.0)


def test_interpolate_linear_ramp():
    rpm_per_esc = {
        "ESC1": {"rpm": np.array([0.0, 6000.0]), "timestamp": np.array([0.0, 1.0])},
    }
    bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0, n_samples=11, sample_rate=10.0, n_blades=2,
    )
    expected = np.linspace(0.0, 6000.0 * 2 / 60.0, 11)
    np.testing.assert_allclose(bpf[:, 0], expected, rtol=1e-10)


def test_build_harmonic_matrix_shape_and_columns():
    bpf = np.array([[100.0, 120.0], [101.0, 121.0]])  # (N=2, S=2)
    M = build_harmonic_matrix(bpf, n_harmonics=3)
    assert M.shape == (2, 2 * 3)
    # column index s*M + (h-1)
    np.testing.assert_allclose(M[:, 0], [100.0, 101.0])  # s=0, h=1
    np.testing.assert_allclose(M[:, 1], [200.0, 202.0])  # s=0, h=2
    np.testing.assert_allclose(M[:, 2], [300.0, 303.0])  # s=0, h=3
    np.testing.assert_allclose(M[:, 3], [120.0, 121.0])  # s=1, h=1
    np.testing.assert_allclose(M[:, 4], [240.0, 242.0])
    np.testing.assert_allclose(M[:, 5], [360.0, 363.0])


def test_linear_r_schedule_shape_and_clipping():
    r = linear_r_schedule(
        n_harmonics=10, fs=10_000.0, delta_bpf_hz=10.0,
        k_cover=1.5, margin_hz=5.0, r_min=0.90, r_max=0.9995,
    )
    assert r.shape == (10,)
    assert (r >= 0.90).all()
    assert (r <= 0.9995).all()
    # bw monoton wächst → r monoton fällt
    assert (np.diff(r) <= 0).all()


def test_linear_r_schedule_clamps_to_rmax_at_h1():
    # very small bw should hit r_max
    r = linear_r_schedule(
        n_harmonics=5, fs=100_000.0, delta_bpf_hz=0.01,
        k_cover=0.01, margin_hz=0.01, r_min=0.5, r_max=0.999,
    )
    assert r[0] == pytest.approx(0.999)
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_frequencies.py -v
```
Expected: ImportError.

- [ ] **Step 3: `frequencies.py` implementieren**

Create `src/martymicfly/processing/frequencies.py`:
```python
"""RPM-to-BPF interpolation, harmonic matrix construction, r-schedule."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def interpolate_per_motor_bpf(
    rpm_per_esc: dict[str, dict[str, np.ndarray]],
    t_start: float,
    n_samples: int,
    sample_rate: float,
    n_blades: int,
) -> np.ndarray:
    """Interpolate ESC RPM onto an audio sample grid, convert to BPF.

    Parameters
    ----------
    rpm_per_esc : dict
        Returned by :func:`martymicfly.io.synth_h5.load_synth_h5`.
    t_start : float
        Start time of the audio segment in seconds (audio time = ESC time
        for synth data; no sync offset).
    n_samples : int
        Audio samples in the segment.
    sample_rate : float
        Audio sample rate in Hz.
    n_blades : int
        Blades per rotor (BPF = mechanical_rpm * n_blades / 60).

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, S)``, ESCs in alphabetical order.
    """
    esc_names = sorted(rpm_per_esc.keys())
    audio_times = np.arange(n_samples) / sample_rate + t_start
    out = np.empty((n_samples, len(esc_names)), dtype=np.float64)
    for s, name in enumerate(esc_names):
        esc = rpm_per_esc[name]
        f = interp1d(
            esc["timestamp"], esc["rpm"],
            kind="linear", fill_value="extrapolate", assume_sorted=True,
        )
        out[:, s] = f(audio_times) * n_blades / 60.0
    return out


def build_harmonic_matrix(per_motor_bpf: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Build ``(N, S*M)`` harmonic frequency matrix.

    Column index ``s*M + (h-1)`` for source ``s`` (0-based) and harmonic ``h``
    (1-based). Identical convention as
    ``notchfilter.cascade.CascadeNotchFilter`` and
    ``NotchFilter/validate/run_filter_comparison.py:build_harmonic_matrix``.
    """
    n_samples, n_sources = per_motor_bpf.shape
    out = np.empty((n_samples, n_sources * n_harmonics), dtype=np.float64)
    for s in range(n_sources):
        for h in range(1, n_harmonics + 1):
            out[:, s * n_harmonics + (h - 1)] = h * per_motor_bpf[:, s]
    return out


def linear_r_schedule(
    n_harmonics: int,
    fs: float,
    delta_bpf_hz: float,
    k_cover: float,
    margin_hz: float,
    r_min: float = 0.90,
    r_max: float = 0.9995,
) -> np.ndarray:
    """Return ``(M,)`` per-harmonic pole radii.

    ``BW(h) = k_cover * h * delta_bpf_hz + margin_hz``,
    ``r(h) = clip(1 - BW(h) * pi / fs, r_min, r_max)``.
    """
    h = np.arange(1, n_harmonics + 1, dtype=np.float64)
    bw = k_cover * h * delta_bpf_hz + margin_hz
    return np.clip(1.0 - bw * np.pi / fs, r_min, r_max)
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_frequencies.py -v
```
Expected: alle 6 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/frequencies.py tests/test_frequencies.py
git commit -m "feat(processing): RPM→BPF interpolation, harmonic matrix, r-schedule"
```

---

## Task 8: `processing/sources.py` — Acoular-In-Memory-Sources

**Files:**
- Create: `src/martymicfly/processing/sources.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_sources.py`:
```python
"""Tests for martymicfly.processing.sources."""

import numpy as np

from martymicfly.processing.sources import ArrayFreqSource, ArraySamplesGenerator


def test_samples_generator_traits():
    data = np.arange(2000, dtype=np.float64).reshape(1000, 2)
    src = ArraySamplesGenerator(data, sample_freq=4096.0)
    assert src.sample_freq == 4096.0
    assert src.num_samples == 1000
    assert src.num_channels == 2


def test_samples_generator_yields_full_blocks_then_remainder():
    data = np.arange(35, dtype=np.float64).reshape(35, 1)
    src = ArraySamplesGenerator(data, sample_freq=1.0)
    blocks = list(src.result(10))
    assert [b.shape for b in blocks] == [(10, 1), (10, 1), (10, 1), (5, 1)]
    np.testing.assert_array_equal(np.vstack(blocks), data)


def test_freq_source_yields_2d_blocks():
    matrix = np.arange(60, dtype=np.float64).reshape(20, 3)
    src = ArrayFreqSource(matrix)
    blocks = list(src.result(7))
    assert [b.shape for b in blocks] == [(7, 3), (7, 3), (6, 3)]
    np.testing.assert_array_equal(np.vstack(blocks), matrix)


def test_freq_source_accepts_1d_input():
    arr = np.arange(15, dtype=np.float64)
    src = ArrayFreqSource(arr)
    blocks = list(src.result(5))
    assert [b.shape for b in blocks] == [(5,), (5,), (5,)]
    np.testing.assert_array_equal(np.concatenate(blocks), arr)
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_sources.py -v
```
Expected: ImportError.

- [ ] **Step 3: `sources.py` implementieren**

Create `src/martymicfly/processing/sources.py`:
```python
"""Acoular-compatible in-memory sources for notch filtering.

Mirrors the pattern of MockSamplesGenerator/MockFreqSource from
NotchFilter/validate/run_filter_comparison.py — but lives here so we can
build the pipeline without importing from NotchFilter's validate/ tree.
"""

from __future__ import annotations

import numpy as np
from acoular.base import SamplesGenerator
from traits.api import Array, HasTraits


class ArraySamplesGenerator(SamplesGenerator):
    """Wrap an ``(N, K)`` ndarray as a streaming SamplesGenerator."""

    _signal_data = Array(dtype=np.float64, shape=(None, None))

    def __init__(self, data: np.ndarray, sample_freq: float, **kwargs) -> None:
        arr = np.ascontiguousarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected 2D (N, K) array, got {arr.ndim}D")
        super().__init__(
            sample_freq=float(sample_freq),
            num_samples=int(arr.shape[0]),
            num_channels=int(arr.shape[1]),
            **kwargs,
        )
        self._signal_data = arr

    def result(self, num: int):
        n = self._signal_data.shape[0]
        for i in range(0, n, num):
            yield self._signal_data[i:i + num]


class ArrayFreqSource(HasTraits):
    """Wrap a 1D ``(N,)`` or 2D ``(N, M)`` array as a streaming freq source.

    ``result(num)`` yields row-wise slices, matching the contract expected by
    ``CascadeNotchFilter`` (mode='external') and the underlying
    ``AdaptiveNotchFilter._result_external``.
    """

    _data = Array(dtype=np.float64)

    def __init__(self, data: np.ndarray, **kwargs) -> None:
        arr = np.asarray(data, dtype=np.float64)
        super().__init__(**kwargs)
        self._data = arr

    def result(self, num: int):
        n = self._data.shape[0]
        pos = 0
        while pos < n:
            end = min(pos + num, n)
            yield self._data[pos:end]
            pos = end
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_sources.py -v
```
Expected: alle 4 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/sources.py tests/test_sources.py
git commit -m "feat(processing): Array{Samples,Freq}Source for in-memory cascade input"
```

---

## Task 9: `processing/pipeline.py` — Stage-Protokoll

**Files:**
- Create: `src/martymicfly/processing/pipeline.py`
- Test: `tests/test_pipeline_skeleton.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_pipeline_skeleton.py`:
```python
"""Tests for martymicfly.processing.pipeline (skeleton only)."""

from dataclasses import replace

import numpy as np

from martymicfly.processing.pipeline import PipelineContext, run_pipeline


class _DoubleStage:
    name = "double"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        return replace(ctx, time_data=ctx.time_data * 2.0)


class _LogMetadataStage:
    name = "log_meta"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        meta = dict(ctx.metadata)
        meta["log_meta_seen_shape"] = ctx.time_data.shape
        return replace(ctx, metadata=meta)


def _empty_ctx():
    return PipelineContext(
        time_data=np.ones((10, 2), dtype=np.float64),
        sample_rate=1000.0,
        rpm_per_esc={},
        mic_positions=np.zeros((2, 3), dtype=np.float64),
        per_motor_bpf=np.zeros((10, 2), dtype=np.float64),
        harm_matrix=np.zeros((10, 4), dtype=np.float64),
        metadata={},
    )


def test_run_pipeline_chains_stages_in_order():
    ctx = run_pipeline([_DoubleStage(), _LogMetadataStage()], _empty_ctx())
    np.testing.assert_allclose(ctx.time_data, 2.0)
    assert ctx.metadata["log_meta_seen_shape"] == (10, 2)


def test_run_pipeline_empty_stage_list_is_identity():
    ctx_in = _empty_ctx()
    ctx_out = run_pipeline([], ctx_in)
    assert ctx_out is ctx_in
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_pipeline_skeleton.py -v
```
Expected: ImportError.

- [ ] **Step 3: `pipeline.py` implementieren**

Create `src/martymicfly/processing/pipeline.py`:
```python
"""Pipeline skeleton: PipelineContext + Stage protocol + run_pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PipelineContext:
    """State passed between stages."""

    time_data: np.ndarray            # (N, C) float64 — current signal
    sample_rate: float
    rpm_per_esc: dict
    mic_positions: np.ndarray        # (M, 3)
    per_motor_bpf: np.ndarray        # (N, S)
    harm_matrix: np.ndarray          # (N, S*M)
    metadata: dict = field(default_factory=dict)


class Stage(Protocol):
    """Pipeline stage contract.

    Implementations consume a :class:`PipelineContext` and return a new one
    (typically via :func:`dataclasses.replace`). They MUST NOT mutate the
    input ctx in place.
    """

    name: str

    def process(self, ctx: PipelineContext) -> PipelineContext: ...


def run_pipeline(stages: list[Stage], ctx: PipelineContext) -> PipelineContext:
    """Run stages in order, threading the context through."""
    for stage in stages:
        ctx = stage.process(ctx)
    return ctx
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_pipeline_skeleton.py -v
```
Expected: beide Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/pipeline.py tests/test_pipeline_skeleton.py
git commit -m "feat(processing): pipeline skeleton — Stage protocol + run_pipeline"
```

---

## Task 10: `processing/notch.py` — `NotchStage`

**Files:**
- Create: `src/martymicfly/processing/notch.py`
- Test: `tests/test_notch_stage.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_notch_stage.py`:
```python
"""Smoke + integration tests for martymicfly.processing.notch."""

import numpy as np
from scipy.signal import welch

from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
)
from martymicfly.processing.notch import NotchStage, NotchStageConfig
from martymicfly.processing.pipeline import PipelineContext


def _rpm_per_esc_constant(rpm: float, duration_s: float = 2.0):
    return {
        "ESC1": {
            "rpm": np.array([rpm, rpm], dtype=np.float64),
            "timestamp": np.array([0.0, duration_s], dtype=np.float64),
        }
    }


def _build_ctx(signal: np.ndarray, fs: float, rpm: float, n_blades: int = 2,
               n_harmonics: int = 5):
    rpm_per_esc = _rpm_per_esc_constant(rpm, duration_s=signal.shape[0] / fs)
    per_motor_bpf = interpolate_per_motor_bpf(
        rpm_per_esc, t_start=0.0,
        n_samples=signal.shape[0], sample_rate=fs, n_blades=n_blades,
    )
    harm_matrix = build_harmonic_matrix(per_motor_bpf, n_harmonics)
    return PipelineContext(
        time_data=signal,
        sample_rate=fs,
        rpm_per_esc=rpm_per_esc,
        mic_positions=np.zeros((signal.shape[1], 3), dtype=np.float64),
        per_motor_bpf=per_motor_bpf,
        harm_matrix=harm_matrix,
        metadata={},
    )


def _band_db(signal_1d: np.ndarray, fs: float, f_target: float, half_bw: float = 5.0):
    f, psd = welch(signal_1d, fs=fs, nperseg=4096, noverlap=2048,
                   window="hann", scaling="density")
    band = (f >= f_target - half_bw) & (f <= f_target + half_bw)
    energy = np.trapz(psd[band], f[band])
    return 10.0 * np.log10(max(energy, 1e-30))


def test_notchstage_suppresses_pure_tone():
    fs = 16_000.0
    rpm = 3000.0
    bpf = rpm * 2 / 60.0  # 100 Hz
    n = int(2.0 * fs)
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * bpf * t)[:, None]  # (N, 1)

    ctx = _build_ctx(signal, fs, rpm, n_harmonics=5)
    stage = NotchStage(NotchStageConfig(
        n_blades=2, n_harmonics=5, pole_radius=0.998,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)

    pre_db = _band_db(signal[:, 0], fs, bpf)
    post_db = _band_db(out.time_data[:, 0], fs, bpf)
    assert post_db < pre_db - 30.0, f"expected ≥30 dB suppression, got {pre_db - post_db:.1f} dB"


def test_notchstage_preserves_pre_signal_in_metadata():
    fs = 16_000.0
    n = int(0.5 * fs)
    signal = np.random.default_rng(0).standard_normal((n, 1))
    ctx = _build_ctx(signal, fs, rpm=3000.0, n_harmonics=2)
    stage = NotchStage(NotchStageConfig(
        n_blades=2, n_harmonics=2, pole_radius=0.99,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)
    np.testing.assert_array_equal(out.metadata["pre_notch"], signal)


def test_notchstage_per_channel_loop_handles_multichannel_input():
    fs = 16_000.0
    n = int(1.0 * fs)
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((n, 3)) * 0.01
    bpf = 100.0
    t = np.arange(n) / fs
    for c in range(3):
        signal[:, c] += np.sin(2 * np.pi * bpf * t)
    ctx = _build_ctx(signal, fs, rpm=3000.0, n_harmonics=3)
    stage = NotchStage(NotchStageConfig(
        n_blades=2, n_harmonics=3, pole_radius=0.998,
        multichannel=False, block_size=4096,
    ))
    out = stage.process(ctx)
    assert out.time_data.shape == signal.shape
    for c in range(3):
        pre_db = _band_db(signal[:, c], fs, bpf)
        post_db = _band_db(out.time_data[:, c], fs, bpf)
        assert post_db < pre_db - 20.0
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_notch_stage.py -v
```
Expected: ImportError.

- [ ] **Step 3: `notch.py` implementieren**

Create `src/martymicfly/processing/notch.py`:
```python
"""NotchStage — RPM-driven cascade notch filtering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from notchfilter.cascade import CascadeNotchFilter

from .pipeline import PipelineContext
from .sources import ArrayFreqSource, ArraySamplesGenerator


@dataclass
class NotchStageConfig:
    n_blades: int
    n_harmonics: int
    pole_radius: float | np.ndarray
    multichannel: bool = False
    block_size: int = 4096


class NotchStage:
    """Wrap CascadeNotchFilter (mode='external', zero_phase=True) as a Stage."""

    name = "notch"

    def __init__(self, cfg: NotchStageConfig) -> None:
        self.cfg = cfg

    def process(self, ctx: PipelineContext) -> PipelineContext:
        signal = ctx.time_data
        n, c = signal.shape
        S = ctx.per_motor_bpf.shape[1]
        M = self.cfg.n_harmonics

        f_inits = np.array(
            [[ctx.per_motor_bpf[0, s] * h for h in range(1, M + 1)] for s in range(S)],
            dtype=np.float64,
        )

        if self.cfg.multichannel:
            filtered = self._run_cascade(signal, ctx.sample_rate, ctx.harm_matrix,
                                         f_inits, S, M)
        else:
            filtered = np.empty_like(signal)
            for ch in range(c):
                filtered[:, ch:ch + 1] = self._run_cascade(
                    signal[:, ch:ch + 1], ctx.sample_rate, ctx.harm_matrix,
                    f_inits, S, M,
                )

        new_meta = dict(ctx.metadata)
        new_meta["pre_notch"] = signal
        return replace(ctx, time_data=filtered, metadata=new_meta)

    def _run_cascade(self, signal: np.ndarray, fs: float, harm_matrix: np.ndarray,
                     f_inits: np.ndarray, S: int, M: int) -> np.ndarray:
        cascade = CascadeNotchFilter(
            num_sources=S,
            harmonics_per_source=M,
            frequencies=f_inits,
            pole_radius=self.cfg.pole_radius,
            mode="external",
            zero_phase=True,
            source=ArraySamplesGenerator(signal, fs),
            freq_source=ArrayFreqSource(harm_matrix),
        )
        return np.vstack(list(cascade.result(self.cfg.block_size)))
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_notch_stage.py -v
```
Expected: alle 3 Tests grün. **Falls `test_notchstage_per_channel_loop_handles_multichannel_input` mit weniger als 20 dB Suppression durchfällt:** `pole_radius=0.999` und `n_harmonics=3` sind konservativ — auf engere Bandbreite oder mehr Harmonische gehen. Erst wenn `notchfilter` selbst den Test bricht, ist das ein Dep-Problem. Die ersten zwei Tests (Pure-Tone + Metadata) sind die Smoke-Asserts.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/notch.py tests/test_notch_stage.py
git commit -m "feat(processing): NotchStage über notchfilter.CascadeNotchFilter"
```

---

## Task 11: `eval/metrics.py` — `compute_metrics`

**Files:**
- Create: `src/martymicfly/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_metrics.py`:
```python
"""Tests for martymicfly.eval.metrics."""

import numpy as np
import pytest

from martymicfly.eval.metrics import compute_metrics


def _make_pre_post(fs=16_000.0, dur=1.0, bpf=100.0, n_harm=5,
                   suppression_db=30.0):
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    pre = np.zeros((n, 1), dtype=np.float64)
    for h in range(1, n_harm + 1):
        pre[:, 0] += np.sin(2 * np.pi * h * bpf * t)
    pre[:, 0] += np.random.default_rng(0).standard_normal(n) * 0.005
    post = pre.copy()
    # apply uniform attenuation in tonal bands by adding destructive replica
    factor = 10.0 ** (-suppression_db / 20.0)
    for h in range(1, n_harm + 1):
        post[:, 0] -= np.sin(2 * np.pi * h * bpf * t) * (1.0 - factor)
    return pre, post


def test_compute_metrics_basic_shape():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    assert "channels" in out
    assert len(out["channels"]) == 1
    ch = out["channels"][0]
    assert "tonal_per_harmonic" in ch
    assert len(ch["tonal_per_harmonic"]) == 5
    assert ch["channel"] == 0


def test_compute_metrics_reduction_matches_expected_db():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs, suppression_db=30.0)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    ch = out["channels"][0]
    # ±5 dB Toleranz wegen Welch-Fenster + Random-Noise
    assert ch["tonal_reduction_db"] == pytest.approx(30.0, abs=5.0)


def test_compute_metrics_per_motor_summary_present():
    fs = 16_000.0
    pre, post = _make_pre_post(fs=fs)
    bpf = np.full((pre.shape[0], 1), 100.0)
    out = compute_metrics(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf, n_harmonics=5,
        pole_radius=0.998, channels=[0],
        welch_nperseg=4096, welch_noverlap=2048,
        bandwidth_factor=1.0, broadband_low_hz=None,
        fmax_hz=fs / 2,
    )
    summary = out["per_motor_bpf_summary"]
    assert summary[0]["bpf_min_hz"] == pytest.approx(100.0)
    assert summary[0]["bpf_max_hz"] == pytest.approx(100.0)
```

- [ ] **Step 2: Tests fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_metrics.py -v
```
Expected: ImportError.

- [ ] **Step 3: `metrics.py` implementieren**

Create `src/martymicfly/eval/metrics.py`:
```python
"""Compute tonal and broadband energy metrics pre/post notch filtering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import welch


def _bw_from_pole_radius(pole_radius: float | np.ndarray, fs: float,
                        n_harmonics: int) -> np.ndarray:
    """Per-harmonic bandwidth in Hz: BW = (1 - r) * fs / pi."""
    if np.isscalar(pole_radius):
        r = np.full(n_harmonics, float(pole_radius))
    else:
        r = np.asarray(pole_radius, dtype=np.float64)
        if r.ndim == 1 and r.shape[0] == n_harmonics:
            pass
        elif r.ndim == 2:
            # (S, M) — collapse to per-harmonic by mean across S
            r = r.mean(axis=0)
        else:
            raise ValueError(
                f"pole_radius shape {r.shape} not understood for n_harmonics={n_harmonics}"
            )
    return (1.0 - r) * fs / np.pi


def _band_energy(f: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    if f_hi <= f_lo:
        return 0.0
    mask = (f >= f_lo) & (f <= f_hi)
    if not mask.any():
        # Schmale Bänder kleiner als Welch-Bin — ein Bin mitnehmen
        idx = np.argmin(np.abs(f - 0.5 * (f_lo + f_hi)))
        df = f[1] - f[0] if f.size > 1 else 1.0
        return float(psd[idx] * df)
    return float(np.trapz(psd[mask], f[mask]))


def _to_db(p: float, ref: float = 1.0) -> float:
    return 10.0 * np.log10(max(p, 1e-30) / ref ** 2)


def compute_metrics(
    *,
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    per_motor_bpf: np.ndarray,
    n_harmonics: int,
    pole_radius: float | np.ndarray,
    channels: Iterable[int],
    welch_nperseg: int,
    welch_noverlap: int,
    bandwidth_factor: float,
    broadband_low_hz: float | None,
    fmax_hz: float,
) -> dict:
    """Compute pre/post tonal and broadband metrics. dB are ref=1 (units²)."""
    channels = list(channels)
    n, c = pre.shape
    assert post.shape == pre.shape
    S = per_motor_bpf.shape[1]
    bw_per_h = _bw_from_pole_radius(pole_radius, fs, n_harmonics) * bandwidth_factor
    f_per_sh = np.array(
        [[per_motor_bpf[:, s].mean() * h for h in range(1, n_harmonics + 1)]
         for s in range(S)],
        dtype=np.float64,
    )

    if broadband_low_hz is None:
        broadband_low_hz = 0.5 * float(per_motor_bpf.min())

    esc_names = [f"ESC{s+1}" for s in range(S)]  # bookkeeping label only

    out_channels: list[dict] = []
    for ch in channels:
        f_pre, psd_pre = welch(pre[:, ch], fs=fs, window="hann",
                               nperseg=welch_nperseg, noverlap=welch_noverlap,
                               scaling="density")
        f_post, psd_post = welch(post[:, ch], fs=fs, window="hann",
                                 nperseg=welch_nperseg, noverlap=welch_noverlap,
                                 scaling="density")

        tonal_per_h: list[dict] = []
        tonal_pre_sum = 0.0
        tonal_post_sum = 0.0
        for s in range(S):
            for h in range(1, n_harmonics + 1):
                f_target = float(f_per_sh[s, h - 1])
                half = 0.5 * float(bw_per_h[h - 1])
                e_pre = _band_energy(f_pre, psd_pre, f_target - half, f_target + half)
                e_post = _band_energy(f_post, psd_post, f_target - half, f_target + half)
                tonal_pre_sum += e_pre
                tonal_post_sum += e_post
                tonal_per_h.append({
                    "motor": esc_names[s],
                    "h": h,
                    "f_hz": f_target,
                    "pre_db": _to_db(e_pre),
                    "post_db": _to_db(e_post),
                    "delta_db": _to_db(e_post) - _to_db(e_pre),
                })

        # Broadband = total energy in [low, fmax] minus all tonal bands
        full_pre = _band_energy(f_pre, psd_pre, broadband_low_hz, fmax_hz)
        full_post = _band_energy(f_post, psd_post, broadband_low_hz, fmax_hz)
        broad_pre = max(full_pre - tonal_pre_sum, 0.0)
        broad_post = max(full_post - tonal_post_sum, 0.0)

        out_channels.append({
            "channel": int(ch),
            "broadband_pre_db": _to_db(broad_pre),
            "broadband_post_db": _to_db(broad_post),
            "broadband_delta_db": _to_db(broad_post) - _to_db(broad_pre),
            "tonal_total_pre_db": _to_db(tonal_pre_sum),
            "tonal_total_post_db": _to_db(tonal_post_sum),
            "tonal_reduction_db": _to_db(tonal_pre_sum) - _to_db(tonal_post_sum),
            "tonal_per_harmonic": tonal_per_h,
        })

    return {
        "sample_rate": fs,
        "n_motors": S,
        "n_harmonics": n_harmonics,
        "per_motor_bpf_summary": [
            {
                "motor": esc_names[s],
                "bpf_min_hz": float(per_motor_bpf[:, s].min()),
                "bpf_max_hz": float(per_motor_bpf[:, s].max()),
                "bpf_mean_hz": float(per_motor_bpf[:, s].mean()),
            }
            for s in range(S)
        ],
        "broadband_low_hz": float(broadband_low_hz),
        "fmax_hz": float(fmax_hz),
        "db_reference": "1 (units squared) — relative, not calibrated to Pa",
        "channels": out_channels,
    }
```

- [ ] **Step 4: Tests laufen lassen**

Run:
```bash
uv run pytest tests/test_metrics.py -v
```
Expected: alle 3 Tests grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): compute_metrics — tonal + broadband pre/post in dB"
```

---

## Task 12: `eval/plots.py` — Plotly-HTML pro Kanal

**Files:**
- Create: `src/martymicfly/eval/plots.py`
- Test: `tests/test_plots.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_plots.py`:
```python
"""Tests for martymicfly.eval.plots."""

import json
from pathlib import Path

import numpy as np

from martymicfly.eval.plots import plot_channel_html


def test_plot_channel_html_writes_valid_file(tmp_path: Path):
    fs = 16_000.0
    n = int(0.5 * fs)
    t = np.arange(n) / fs
    pre = np.sin(2 * np.pi * 100 * t)[:, None]
    post = pre * 0.1
    bpf = np.full((n, 1), 100.0)
    out = tmp_path / "ch00.html"
    plot_channel_html(
        pre=pre, post=post, fs=fs, per_motor_bpf=bpf,
        channel=0, outpath=out, n_harmonics=5,
        fmax_hz=1000.0, spectrogram_window=2048, spectrogram_overlap=1024,
    )
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "Plotly" in content
    # Trace data presence (Plotly embeds JSON):
    assert "100" in content  # the BPF marker should appear
```

- [ ] **Step 2: Test fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_plots.py -v
```
Expected: ImportError.

- [ ] **Step 3: `plots.py` implementieren**

Create `src/martymicfly/eval/plots.py`:
```python
"""Plotly HTML output: per-channel spectrum + spectrogram before/after."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

_MOTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_channel_html(
    *,
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    per_motor_bpf: np.ndarray,
    channel: int,
    outpath: Path,
    n_harmonics: int,
    fmax_hz: float,
    spectrogram_window: int,
    spectrogram_overlap: int,
) -> None:
    """Write a standalone HTML with two stacked subplots for one channel."""
    pre_1d = np.asarray(pre[:, channel], dtype=np.float64)
    post_1d = np.asarray(post[:, channel], dtype=np.float64)
    S = per_motor_bpf.shape[1]

    f_pre, psd_pre = welch(pre_1d, fs=fs, window="hann",
                           nperseg=min(8192, pre_1d.size),
                           noverlap=min(4096, pre_1d.size // 2),
                           scaling="density")
    f_post, psd_post = welch(post_1d, fs=fs, window="hann",
                             nperseg=min(8192, post_1d.size),
                             noverlap=min(4096, post_1d.size // 2),
                             scaling="density")
    fmask = f_pre <= fmax_hz

    f_spec, t_spec, sxx = spectrogram(
        post_1d, fs=fs, window="hann",
        nperseg=spectrogram_window, noverlap=spectrogram_overlap,
        scaling="density",
    )
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-20))
    spec_mask = f_spec <= fmax_hz

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.10,
                        subplot_titles=("Welch PSD (pre vs post)",
                                        "Spectrogram (post)"))

    fig.add_trace(go.Scatter(
        x=f_pre[fmask], y=10.0 * np.log10(np.maximum(psd_pre[fmask], 1e-20)),
        name="pre", line=dict(color="#444", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f_post[fmask], y=10.0 * np.log10(np.maximum(psd_post[fmask], 1e-20)),
        name="post", line=dict(color="#d62728", width=1.5),
    ), row=1, col=1)

    for s in range(S):
        color = _MOTOR_COLORS[s % len(_MOTOR_COLORS)]
        bpf_mean = float(per_motor_bpf[:, s].mean())
        for h in range(1, n_harmonics + 1):
            f_h = bpf_mean * h
            if f_h > fmax_hz:
                break
            fig.add_vline(x=f_h, line=dict(color=color, dash="dot", width=1),
                          row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=t_spec, y=f_spec[spec_mask], z=sxx_db[spec_mask, :],
        colorscale="Viridis", colorbar=dict(title="dB", len=0.45, y=0.22),
        showscale=True,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="PSD [dB / Hz, ref=1]", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Frequency [Hz]", row=2, col=1)
    fig.update_layout(
        title=f"Channel {channel:02d} — pre/post notch",
        height=900, showlegend=True,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath, include_plotlyjs="cdn", full_html=True)
```

- [ ] **Step 4: Test laufen lassen**

Run:
```bash
uv run pytest tests/test_plots.py -v
```
Expected: Test grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/eval/plots.py tests/test_plots.py
git commit -m "feat(eval): plot_channel_html — Plotly spectrum + spectrogram"
```

---

## Task 13: `io/write_filtered.py` — Output-HDF5

**Files:**
- Create: `src/martymicfly/io/write_filtered.py`
- Test: `tests/test_write_filtered.py`

- [ ] **Step 1: Failing-Test schreiben**

Create `tests/test_write_filtered.py`:
```python
"""Tests for martymicfly.io.write_filtered."""

from pathlib import Path

import h5py
import numpy as np

from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.io.write_filtered import write_filtered


def test_roundtrip_preserves_telemetry_and_writes_attrs(tmp_path: Path):
    src = Path(__file__).parent / "fixtures" / "tiny_synth.h5"
    src_data = load_synth_h5(src)

    filtered = src_data["time_data"] * 0.5  # bogus filter
    out_path = tmp_path / "filtered.h5"
    write_filtered(
        out_path=out_path,
        filtered_time_data=filtered,
        sample_rate=src_data["sample_rate"],
        rpm_per_esc=src_data["rpm_per_esc"],
        attrs={
            "martymicfly_version": "0.1.0",
            "input_h5": str(src),
            "config_hash": "deadbeef",
            "segment_start_s": 0.0,
            "segment_duration_s": 1.0,
            "n_blades": 2,
            "n_harmonics": 5,
            "notch_mode": "rpm_external_zerophase",
            "pole_radius_repr": "scalar:0.998",
        },
    )

    out_data = load_synth_h5(out_path)
    np.testing.assert_allclose(out_data["time_data"], filtered)
    assert set(out_data["rpm_per_esc"].keys()) == set(src_data["rpm_per_esc"].keys())
    np.testing.assert_array_equal(
        out_data["rpm_per_esc"]["ESC1"]["rpm"],
        src_data["rpm_per_esc"]["ESC1"]["rpm"],
    )

    with h5py.File(out_path, "r") as f:
        assert f.attrs["config_hash"].decode() == "deadbeef" if isinstance(
            f.attrs["config_hash"], bytes) else f.attrs["config_hash"] == "deadbeef"
        assert f.attrs["notch_mode"].decode() == "rpm_external_zerophase" if isinstance(
            f.attrs["notch_mode"], bytes) else f.attrs["notch_mode"] == "rpm_external_zerophase"
```

- [ ] **Step 2: Test fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_write_filtered.py -v
```
Expected: ImportError.

- [ ] **Step 3: `write_filtered.py` implementieren**

Create `src/martymicfly/io/write_filtered.py`:
```python
"""Write filtered time_data + telemetry pass-through + provenance attrs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_filtered(
    *,
    out_path: str | Path,
    filtered_time_data: np.ndarray,
    sample_rate: float,
    rpm_per_esc: dict[str, dict[str, np.ndarray]],
    attrs: dict[str, str | int | float],
) -> None:
    """Write Acoular-format HDF5 with filtered signal + telemetry + attrs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        td = f.create_dataset(
            "time_data",
            data=np.ascontiguousarray(filtered_time_data, dtype=np.float64),
            dtype="float64",
        )
        td.attrs["sample_freq"] = np.float64(sample_rate)

        grp = f.create_group("esc_telemetry")
        for esc_name, esc in rpm_per_esc.items():
            sub = grp.create_group(esc_name)
            sub.create_dataset("rpm", data=esc["rpm"].astype(np.float64))
            sub.create_dataset("timestamp", data=esc["timestamp"].astype(np.float64))

        for k, v in attrs.items():
            f.attrs[k] = v
```

- [ ] **Step 4: Test laufen lassen**

Run:
```bash
uv run pytest tests/test_write_filtered.py -v
```
Expected: grün.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/write_filtered.py tests/test_write_filtered.py
git commit -m "feat(io): write_filtered HDF5 with provenance attrs"
```

---

## Task 14: `cli/run_notch.py` + `configs/example_notch.yaml` + Pipeline-Integration

**Files:**
- Create: `src/martymicfly/cli/run_notch.py`
- Create: `configs/example_notch.yaml`
- Test: `tests/test_pipeline_integration.py`
- Modify: `pyproject.toml` (Entry-Point)

- [ ] **Step 1: `configs/example_notch.yaml` schreiben**

Create `configs/example_notch.yaml`:
```yaml
input:
  audio_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synthesis_16mic_gaptip.h5
  mic_geom_xml: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml

segment:
  mode: middle
  duration: 10.0

channels:
  selection: all

rotor:
  n_blades: 2
  n_harmonics: 20

notch:
  pole_radius:
    mode: scalar
    value: 0.9994
  multichannel: false
  block_size: 4096

metrics:
  welch_nperseg: 8192
  welch_noverlap: 4096
  bandwidth_factor: 1.0
  broadband_low_hz: null

plots:
  enabled: true
  fmax_hz: null
  spectrogram_window: 4096
  spectrogram_overlap: 2048
  channel_subset: null

output:
  dir: results/notch/{run_id}
  filtered_h5: filtered.h5
  metrics_json: metrics.json
  metrics_csv: metrics.csv
  plots_subdir: plots
  copy_config: true
```

- [ ] **Step 2: Failing-Integration-Test schreiben**

Create `tests/test_pipeline_integration.py`:
```python
"""End-to-end: tiny fixture → CLI → outputs."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
FIXTURE_H5 = REPO / "tests" / "fixtures" / "tiny_synth.h5"
FIXTURE_XML = REPO / "tests" / "fixtures" / "tiny_geom.xml"


def _config_for_fixture(out_dir: Path) -> dict:
    return {
        "input": {
            "audio_h5": str(FIXTURE_H5),
            "mic_geom_xml": str(FIXTURE_XML),
        },
        "segment": {"mode": "middle", "duration": 0.5},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 5},
        "notch": {
            "pole_radius": {"mode": "scalar", "value": 0.998},
            "multichannel": False,
            "block_size": 4096,
        },
        "metrics": {
            "welch_nperseg": 2048,
            "welch_noverlap": 1024,
            "bandwidth_factor": 1.0,
            "broadband_low_hz": None,
        },
        "plots": {
            "enabled": True,
            "fmax_hz": 2000.0,
            "spectrogram_window": 1024,
            "spectrogram_overlap": 512,
            "channel_subset": [0, 1],
        },
        "output": {
            "dir": str(out_dir / "{run_id}"),
            "filtered_h5": "filtered.h5",
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "plots_subdir": "plots",
            "copy_config": True,
        },
    }


def test_end_to_end_on_tiny_fixture(tmp_path: Path):
    cfg = _config_for_fixture(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "martymicfly.cli.run_notch",
         "--config", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    runs = list(tmp_path.glob("*"))
    run_dirs = [p for p in runs if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "filtered.h5").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "config.yaml").exists()
    plots = list((run_dir / "plots").glob("ch*.html"))
    assert len(plots) == 2  # channel_subset

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_motors"] == 2
    assert metrics["n_harmonics"] == 5
    # Tonal reduction at the seeded harmonics should be substantial
    for ch in metrics["channels"]:
        assert ch["tonal_reduction_db"] > 15.0, ch
```

- [ ] **Step 3: Test fehlschlagen lassen**

Run:
```bash
uv run pytest tests/test_pipeline_integration.py -v
```
Expected: ModuleNotFoundError (`martymicfly.cli.run_notch`).

- [ ] **Step 4: `cli/run_notch.py` implementieren**

Create `src/martymicfly/cli/run_notch.py`:
```python
"""CLI entry point: YAML config → notch pipeline → outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from martymicfly.config import AppConfig
from martymicfly.eval.metrics import compute_metrics
from martymicfly.eval.plots import plot_channel_html
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.synth_h5 import load_synth_h5
from martymicfly.io.write_filtered import write_filtered
from martymicfly.processing.frequencies import (
    build_harmonic_matrix,
    interpolate_per_motor_bpf,
    linear_r_schedule,
)
from martymicfly.processing.notch import NotchStage, NotchStageConfig
from martymicfly.processing.pipeline import PipelineContext, run_pipeline

log = logging.getLogger("martymicfly.run_notch")


def _select_segment(cfg, n_total: int, fs: float) -> tuple[float, int]:
    mode = cfg.segment.mode
    if mode == "explicit":
        start = cfg.segment.start
        end = cfg.segment.end
        n_seg = int(round((end - start) * fs))
        return float(start), n_seg
    duration = cfg.segment.duration
    n_seg = int(round(duration * fs))
    if n_seg > n_total:
        raise ValueError(f"segment duration {duration}s > recording {n_total/fs:.2f}s")
    if mode == "head":
        return 0.0, n_seg
    if mode == "tail":
        return (n_total - n_seg) / fs, n_seg
    # middle (default)
    start_idx = (n_total - n_seg) // 2
    return start_idx / fs, n_seg


def _resolve_pole_radius(cfg, n_harmonics: int, fs: float,
                         per_motor_bpf: np.ndarray):
    pr = cfg.notch.pole_radius
    if pr.mode == "scalar":
        return float(pr.value), f"scalar:{pr.value:.4f}"
    delta_bpf = pr.delta_bpf_hz
    if delta_bpf is None:
        mean_bpf = per_motor_bpf.mean(axis=0)
        delta_bpf = float(mean_bpf.max() - mean_bpf.min())
    sched = linear_r_schedule(
        n_harmonics=n_harmonics, fs=fs,
        delta_bpf_hz=delta_bpf, k_cover=pr.k_cover, margin_hz=pr.margin_hz,
        r_min=pr.r_min, r_max=pr.r_max,
    )
    repr_ = (f"linear:k={pr.k_cover},margin={pr.margin_hz}"
             f",delta_bpf={delta_bpf:.2f}")
    return sched, repr_


def _resolve_channels(cfg, n_total_channels: int) -> list[int]:
    if cfg.channels.selection == "all":
        return list(range(n_total_channels))
    return list(cfg.channels.list)


def _resolve_plot_channels(cfg, channels: list[int]) -> list[int]:
    if cfg.plots.channel_subset is None:
        return channels
    subset = set(cfg.plots.channel_subset)
    return [c for c in channels if c in subset]


def _resolve_fmax(cfg, per_motor_bpf: np.ndarray, n_harmonics: int,
                  fs: float) -> float:
    if cfg.plots.fmax_hz is not None:
        return float(cfg.plots.fmax_hz)
    return min(1.1 * float(per_motor_bpf.max()) * n_harmonics, fs / 2.0)


def _write_metrics(metrics: dict, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fieldnames = [
        "channel", "broadband_pre_db", "broadband_post_db", "broadband_delta_db",
        "tonal_total_pre_db", "tonal_total_post_db", "tonal_reduction_db",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ch in metrics["channels"]:
            w.writerow({k: ch[k] for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run AP2-A stage 1 (notch) pipeline.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override output.dir from the YAML config.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = AppConfig(**cfg_dict)

    config_hash = cfg.config_hash()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + "_" + config_hash
    out_template = str(args.output_dir) if args.output_dir is not None else cfg.output.dir
    run_dir = Path(out_template.replace("{run_id}", run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run_id=%s, output=%s", run_id, run_dir)

    # 1. Load
    src = load_synth_h5(cfg.input.audio_h5)
    geom = load_mic_geom_xml(cfg.input.mic_geom_xml)
    if geom.shape[0] != src["time_data"].shape[1]:
        raise ValueError(
            f"mic_geom rows ({geom.shape[0]}) != channels ({src['time_data'].shape[1]})"
        )

    # 2. Segment
    seg_start_s, n_seg = _select_segment(cfg, src["time_data"].shape[0], src["sample_rate"])
    seg_start_idx = int(round(seg_start_s * src["sample_rate"]))
    signal = np.ascontiguousarray(
        src["time_data"][seg_start_idx:seg_start_idx + n_seg], dtype=np.float64,
    )

    # 3. Frequencies
    per_motor_bpf = interpolate_per_motor_bpf(
        src["rpm_per_esc"], t_start=seg_start_s,
        n_samples=n_seg, sample_rate=src["sample_rate"],
        n_blades=cfg.rotor.n_blades,
    )
    harm_matrix = build_harmonic_matrix(per_motor_bpf, cfg.rotor.n_harmonics)
    pole_radius, pole_repr = _resolve_pole_radius(
        cfg, cfg.rotor.n_harmonics, src["sample_rate"], per_motor_bpf,
    )

    # 4. Pipeline
    ctx = PipelineContext(
        time_data=signal,
        sample_rate=src["sample_rate"],
        rpm_per_esc=src["rpm_per_esc"],
        mic_positions=geom,
        per_motor_bpf=per_motor_bpf,
        harm_matrix=harm_matrix,
        metadata={},
    )
    notch_stage = NotchStage(NotchStageConfig(
        n_blades=cfg.rotor.n_blades,
        n_harmonics=cfg.rotor.n_harmonics,
        pole_radius=pole_radius,
        multichannel=cfg.notch.multichannel,
        block_size=cfg.notch.block_size,
    ))
    ctx = run_pipeline([notch_stage], ctx)

    # 5. Metrics
    channels = _resolve_channels(cfg, signal.shape[1])
    fmax = _resolve_fmax(cfg, per_motor_bpf, cfg.rotor.n_harmonics, src["sample_rate"])
    metrics = compute_metrics(
        pre=ctx.metadata["pre_notch"], post=ctx.time_data,
        fs=src["sample_rate"], per_motor_bpf=per_motor_bpf,
        n_harmonics=cfg.rotor.n_harmonics, pole_radius=pole_radius,
        channels=channels,
        welch_nperseg=cfg.metrics.welch_nperseg,
        welch_noverlap=cfg.metrics.welch_noverlap,
        bandwidth_factor=cfg.metrics.bandwidth_factor,
        broadband_low_hz=cfg.metrics.broadband_low_hz,
        fmax_hz=fmax,
    )
    metrics.update({
        "run_id": run_id,
        "config_hash": config_hash,
        "input_h5": str(cfg.input.audio_h5),
        "segment": {"start_s": seg_start_s, "duration_s": n_seg / src["sample_rate"]},
    })
    _write_metrics(metrics, run_dir / cfg.output.metrics_json,
                   run_dir / cfg.output.metrics_csv)

    # 6. Plots
    if cfg.plots.enabled:
        plot_channels = _resolve_plot_channels(cfg, channels)
        plots_dir = run_dir / cfg.output.plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        for ch in plot_channels:
            plot_channel_html(
                pre=ctx.metadata["pre_notch"], post=ctx.time_data,
                fs=src["sample_rate"], per_motor_bpf=per_motor_bpf,
                channel=ch, outpath=plots_dir / f"ch{ch:02d}.html",
                n_harmonics=cfg.rotor.n_harmonics, fmax_hz=fmax,
                spectrogram_window=cfg.plots.spectrogram_window,
                spectrogram_overlap=cfg.plots.spectrogram_overlap,
            )

    # 7. Filtered HDF5
    write_filtered(
        out_path=run_dir / cfg.output.filtered_h5,
        filtered_time_data=ctx.time_data,
        sample_rate=src["sample_rate"],
        rpm_per_esc=src["rpm_per_esc"],
        attrs={
            "martymicfly_version": "0.1.0",
            "input_h5": str(cfg.input.audio_h5),
            "config_hash": config_hash,
            "segment_start_s": float(seg_start_s),
            "segment_duration_s": float(n_seg / src["sample_rate"]),
            "n_blades": int(cfg.rotor.n_blades),
            "n_harmonics": int(cfg.rotor.n_harmonics),
            "notch_mode": "rpm_external_zerophase",
            "pole_radius_repr": pole_repr,
        },
    )

    # 8. Snapshot config
    if cfg.output.copy_config:
        shutil.copy(args.config, run_dir / "config.yaml")
        (run_dir / "config.hash").write_text(config_hash + "\n", encoding="utf-8")

    log.info("done. outputs in %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Test laufen lassen**

Run:
```bash
uv run pytest tests/test_pipeline_integration.py -v
```
Expected: grün. Wenn `tonal_reduction_db > 15.0` nicht erreicht wird auf der Fixture: `pole_radius=0.998` ist konservativ; auf `0.999` gehen oder die Fixture-Signalstärke erhöhen.

- [ ] **Step 6: Commit**

```bash
git add src/martymicfly/cli/run_notch.py configs/example_notch.yaml \
        tests/test_pipeline_integration.py
git commit -m "feat(cli): run_notch CLI + example config + integration test"
```

---

## Task 15: Smoke-Run auf realer Synth-Datei

**Files:**
- (kein Code) — manuelle Verifikation

- [ ] **Step 1: CLI gegen den echten Synth-File ausführen**

Run:
```bash
uv run python -m martymicfly.cli.run_notch \
    --config configs/example_notch.yaml \
    --output-dir results/notch_smoke/{run_id}
```
Expected: Exit-Code 0; ein Verzeichnis `results/notch_smoke/<timestamp>_<hash>/` mit `filtered.h5`, `metrics.json`, `metrics.csv`, 16 Plotly-HTMLs unter `plots/`, `config.yaml`, `config.hash`. Laufzeit auf der vollen 10s-Mitte: ein paar Minuten (per-Channel-Loop × 16 Kanäle).

- [ ] **Step 2: `metrics.csv` auf Plausibilität prüfen**

Run:
```bash
cat results/notch_smoke/*/metrics.csv | head -5
```
Expected: alle 16 Kanäle haben `tonal_reduction_db` deutlich positiv (≳ 20 dB), `broadband_delta_db` betragsmäßig klein (≲ 2 dB).

- [ ] **Step 3: Eine Plotly-HTML im Browser inspizieren**

Run:
```bash
ls results/notch_smoke/*/plots/ch00.html
```
Optional `xdg-open <pfad>`. Erwartung: deutliche Notches an den vorhergesagten Frequenzen (Linien je Motor), Spektrogramm zeigt die Notches als Linien-Senken.

- [ ] **Step 4: `filtered.h5` öffnen und Provenance verifizieren**

Run:
```bash
uv run python -c "
import h5py, json
import glob
p = sorted(glob.glob('results/notch_smoke/*/filtered.h5'))[-1]
with h5py.File(p, 'r') as f:
    for k in ['config_hash', 'notch_mode', 'pole_radius_repr',
              'segment_start_s', 'segment_duration_s']:
        print(k, '=', f.attrs[k])
    print('time_data', f['time_data'].shape, 'fs', f['time_data'].attrs['sample_freq'])
"
```
Expected: alle Attrs gesetzt, `notch_mode='rpm_external_zerophase'`, `pole_radius_repr='scalar:0.9994'`, `time_data.shape == (520_000, 16)` (10s @ 52 kHz).

- [ ] **Step 5: `results/notch_smoke/` zu `.gitignore` hinzufügen, NICHT committen**

Append zu `.gitignore` (falls nicht schon enthalten):
```
results/
```

Run:
```bash
grep -q "^results/$" .gitignore || echo "results/" >> .gitignore
git add .gitignore
git diff --cached .gitignore
```

- [ ] **Step 6: Commit der `.gitignore`-Änderung (nur falls geändert)**

```bash
git diff --cached --quiet .gitignore && echo "no change" || \
  git commit -m "chore: gitignore results/ run outputs"
```

---

## Self-Review (durchgeführt)

**Spec coverage:**
- §1.2 In-Scope: Loader (Tasks 5/6), Pipeline-Skelett (9), Stufe 1 Notch (10),
  Adapter-Helfer (7/8), Output-HDF5 (13), Plotly + Metriken (11/12), CLI + YAML (14),
  Tests (jeder Task), Cleanup-Commit (1) — alles abgedeckt.
- §1.4 Cleanup: Task 1.
- §2 Repo-Layout: Task 2 (Skelett), Tasks 3–14 (Module).
- §4 Komponenten-Verträge: jede Komponente hat ihren eigenen Task mit
  vollem Test+Code.
- §5 Konfiguration: Task 3 (pydantic) + Task 14 (YAML).
- §6 Output-Struktur: Task 14 (CLI) + Task 15 (Smoke verifiziert).
- §7 Tests: jede Komponente hat eigenen Test-Task; Integration in Task 14;
  Fixture in Task 4.
- §8 Abhängigkeiten: Task 2.

**Placeholder scan:** keine TBD/TODO/„similar to" — alle Code-Blöcke sind
vollständig.

**Type consistency:**
- `NotchStageConfig` (Task 10) hat Felder `n_blades`, `n_harmonics`,
  `pole_radius`, `multichannel`, `block_size` — die CLI-Konstruktion
  (Task 14, `_resolve_pole_radius` + `NotchStage(NotchStageConfig(...))`)
  passt 1:1.
- `compute_metrics` Keyword-Args (Task 11) ↔ CLI-Aufruf (Task 14):
  `pre`, `post`, `fs`, `per_motor_bpf`, `n_harmonics`, `pole_radius`,
  `channels`, `welch_nperseg`, `welch_noverlap`, `bandwidth_factor`,
  `broadband_low_hz`, `fmax_hz` — abgeglichen.
- `plot_channel_html` Keyword-Args (Task 12) ↔ CLI-Aufruf (Task 14):
  `pre`, `post`, `fs`, `per_motor_bpf`, `channel`, `outpath`, `n_harmonics`,
  `fmax_hz`, `spectrogram_window`, `spectrogram_overlap` — abgeglichen.
- `write_filtered` Keyword-Args (Task 13) ↔ CLI-Aufruf — abgeglichen.
- `interpolate_per_motor_bpf` Reihenfolge `t_start, n_samples, sample_rate, n_blades`
  in Task 7 ↔ Aufrufer in Tasks 10 (Test) und 14 (CLI) — abgeglichen.
