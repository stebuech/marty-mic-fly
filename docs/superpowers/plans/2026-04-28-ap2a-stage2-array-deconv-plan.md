# AP2-A Stage 2 (Array Deconvolution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement AP2-A Stage 2 — array deconvolution via CLEAN-SC for spatial filtering of the broadband drone self-noise — with an in-repo synthesis helper that mixes the existing drone source artifact with a configurable external source so Stage 2 can be quantitatively validated against ground truth.

**Architecture:** Single CLEAN-SC run on a normal-sized diagnostic grid; the rotor-disc spatial mask serves both subtraction and visual diagnostics in one pass. Algorithm-pluggable `Algorithm` Protocol so the follow-up spec (Orth, CLEAN-T, CSM-Fitting) is purely additive. CLI is generalized to a stage-list pipeline (sharp cut on the old top-level `notch:` YAML format). Output of Stage 2 lives in `PipelineContext.metadata` (residual CSM, beam maps, pseudo-target PSD); `time_data` passes through unchanged.

**Tech Stack:** Python 3.13 + uv + pydantic v2 + scipy + numpy + h5py + plotly + acoular (frequency-domain CLEAN-SC) + pytest.

**Spec:** `docs/superpowers/specs/2026-04-28-ap2a-stage2-array-deconv-design.md`

---

## Task 1: Add `processing/constants.py`

**Files:**
- Create: `src/martymicfly/constants.py`
- Test: `tests/test_constants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_constants.py`:

```python
def test_speed_of_sound_constant():
    from martymicfly.constants import SPEED_OF_SOUND
    assert SPEED_OF_SOUND == 343.0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_constants.py -v
```

Expected: `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `src/martymicfly/constants.py`:

```python
"""Project-wide physical constants. Single source of truth so synthesis,
reconstruction, and steering all use the same numbers."""

SPEED_OF_SOUND: float = 343.0  # m/s, free-field, 20 °C, dry air
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_constants.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/constants.py tests/test_constants.py
git commit -m "feat(constants): SPEED_OF_SOUND single source of truth"
```

---

## Task 2: PipelineConfig refactor — stages list

Sharp-cut on the top-level `notch:` block. `AppConfig` gains a `stages: list[StageConfig]` field; `NotchStageConfig` becomes a stage entry rather than a top-level block.

**Files:**
- Modify: `src/martymicfly/config.py` (add stage discriminated union, replace `AppConfig.notch` with `AppConfig.stages`)
- Modify: `tests/test_config.py` (new YAML structure)

- [ ] **Step 1: Write the failing test**

Replace the existing notch-block test in `tests/test_config.py` and add a stages-list test:

```python
def test_app_config_with_notch_stage():
    from martymicfly.config import AppConfig
    payload = {
        "input": {"audio_h5": "/x.h5", "mic_geom_xml": "/x.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 13},
        "stages": [
            {
                "kind": "notch",
                "pole_radius": {"mode": "scalar", "value": 0.9994},
                "multichannel": False,
                "block_size": 4096,
            }
        ],
        "metrics": {"welch_nperseg": 4096, "welch_noverlap": 2048},
        "plots": {"enabled": True, "spectrogram_window": 4096, "spectrogram_overlap": 2048},
        "output": {"dir": "results/notch/{run_id}"},
    }
    cfg = AppConfig.model_validate(payload)
    assert cfg.stages[0].kind == "notch"
    assert cfg.stages[0].pole_radius.value == 0.9994


def test_app_config_rejects_old_top_level_notch():
    import pytest
    from pydantic import ValidationError
    from martymicfly.config import AppConfig
    payload = {
        "input": {"audio_h5": "/x.h5", "mic_geom_xml": "/x.xml"},
        "segment": {"mode": "middle", "duration": 10.0},
        "channels": {"selection": "all"},
        "rotor": {"n_blades": 2, "n_harmonics": 13},
        "notch": {"pole_radius": {"mode": "scalar", "value": 0.9994}},
        "metrics": {"welch_nperseg": 4096, "welch_noverlap": 2048},
        "plots": {"enabled": True, "spectrogram_window": 4096, "spectrogram_overlap": 2048},
        "output": {"dir": "results/notch/{run_id}"},
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)
```

Delete any test in `tests/test_config.py` that exercises the old top-level `notch:` block.

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_config.py -v
```

Expected: new tests fail (no `cfg.stages`); other tests still pass on the old structure but will break in the next step.

- [ ] **Step 3: Refactor `config.py`**

In `src/martymicfly/config.py`, replace the `AppConfig.notch: NotchConfig` field with a discriminated stages list. The existing `NotchConfig` (defined in `config.py`) becomes the body of a `NotchStageConfig` entry distinguished by `kind`.

```python
from typing import Annotated, Literal, Union

# ... existing imports + classes ...

class NotchStageConfig(NotchConfig):
    """NotchConfig fields + a `kind` discriminator for the stages list."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["notch"]


# Placeholder for Stage 2 — full body added in Task 19. We add a stub here so
# the discriminated union works; field defaults are validated when the stage
# entry is present.
class ArrayFilterStageConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # filled in Task 19
    kind: Literal["array_filter"]


StageConfig = Annotated[
    Union[NotchStageConfig, ArrayFilterStageConfig],
    Field(discriminator="kind"),
]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InputConfig
    segment: SegmentConfig
    channels: ChannelsConfig
    rotor: RotorConfig
    stages: list[StageConfig]            # NEW
    metrics: MetricsConfig
    plots: PlotsConfig
    output: OutputConfig

    # ... canonical_json / config_hash unchanged ...
```

Remove the `notch: NotchConfig` field from `AppConfig`. `extra="forbid"` will then reject the old top-level `notch:` block automatically.

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_config.py -v
```

Expected: both new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/config.py tests/test_config.py
git commit -m "feat(config): stages list with discriminated union; sharp-cut top-level notch"
```

---

## Task 3: Stage registry in `processing/pipeline.py`

**Files:**
- Modify: `src/martymicfly/processing/pipeline.py`
- Test: `tests/test_pipeline_skeleton.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_skeleton.py`:

```python
def test_stage_registry_builds_notch_stage():
    from martymicfly.config import NotchStageConfig, PoleRadiusConfig
    from martymicfly.processing.pipeline import build_stage
    cfg = NotchStageConfig(
        kind="notch",
        pole_radius=PoleRadiusConfig(mode="scalar", value=0.9994),
        multichannel=False,
        block_size=4096,
    )
    stage = build_stage(cfg)
    assert stage.name == "notch"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_pipeline_skeleton.py::test_stage_registry_builds_notch_stage -v
```

Expected: `ImportError` for `build_stage`.

- [ ] **Step 3: Add registry**

Append to `src/martymicfly/processing/pipeline.py`:

```python
from typing import Callable

from martymicfly.config import (
    ArrayFilterStageConfig,
    NotchStageConfig,
    StageConfig,
)


_STAGE_BUILDERS: dict[str, Callable[[object], "Stage"]] = {}


def register_stage_builder(kind: str, builder: Callable[[object], "Stage"]) -> None:
    _STAGE_BUILDERS[kind] = builder


def build_stage(stage_cfg: StageConfig) -> "Stage":
    if stage_cfg.kind not in _STAGE_BUILDERS:
        raise ValueError(
            f"no stage builder registered for kind={stage_cfg.kind!r}; "
            f"available: {sorted(_STAGE_BUILDERS)}"
        )
    return _STAGE_BUILDERS[stage_cfg.kind](stage_cfg)


def build_pipeline(stages_cfg: list[StageConfig]) -> list["Stage"]:
    return [build_stage(s) for s in stages_cfg]
```

In `src/martymicfly/processing/notch.py`, after the `NotchStage` class, add:

```python
def _build_notch_stage(cfg) -> NotchStage:
    return NotchStage(NotchStageConfig.from_app_config(cfg))


# Register on import. Top-level so importing the module registers the builder.
from martymicfly.processing.pipeline import register_stage_builder
register_stage_builder("notch", lambda cfg: NotchStage(cfg))
```

If `NotchStage` is currently constructed from a different config type than `NotchStageConfig`, adapt the lambda to do the conversion (read its current `__init__` signature and either widen it to accept a `NotchStageConfig` or wrap it). Show the wrapping in this step rather than punting to TODO.

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_pipeline_skeleton.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/pipeline.py src/martymicfly/processing/notch.py tests/test_pipeline_skeleton.py
git commit -m "feat(pipeline): stage registry; NotchStage registered as 'notch'"
```

---

## Task 4: New CLI `cli/run_pipeline.py`; `run_notch.py` becomes thin shim

**Files:**
- Create: `src/martymicfly/cli/run_pipeline.py` (copied/renamed from `run_notch.py`, generalized)
- Modify: `src/martymicfly/cli/run_notch.py` (becomes thin shim)
- Modify: `configs/example_notch.yaml` (stages-list format)
- Modify: `tests/test_pipeline_integration.py` (use new format)

- [ ] **Step 1: Inspect current CLI to understand what's there**

```
cat src/martymicfly/cli/run_notch.py
```

Note: the current `main()` builds a `NotchStage` directly. The refactor moves that building into `build_pipeline`.

- [ ] **Step 2: Write failing CLI test**

Append to `tests/test_pipeline_integration.py`:

```python
def test_run_pipeline_cli_with_stages_list(tmp_path, monkeypatch):
    """The new CLI accepts the stages-list YAML and runs notch only."""
    import shutil
    from pathlib import Path
    fixtures = Path("tests/fixtures")
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(f"""
input:
  audio_h5: {fixtures / 'tiny_synth.h5'}
  mic_geom_xml: {fixtures / 'tiny_geom.xml'}
segment: {{mode: head, duration: 0.5}}
channels: {{selection: all}}
rotor: {{n_blades: 2, n_harmonics: 4}}
stages:
  - kind: notch
    pole_radius: {{mode: scalar, value: 0.99}}
    multichannel: false
    block_size: 1024
metrics: {{welch_nperseg: 1024, welch_noverlap: 512}}
plots: {{enabled: false, spectrogram_window: 512, spectrogram_overlap: 256}}
output: {{dir: {tmp_path / 'out'}}}
""")
    from martymicfly.cli.run_pipeline import main
    rc = main(["--config", str(cfg_yaml)])
    assert rc == 0
    out_dirs = list((tmp_path / "out").glob("*"))
    assert len(out_dirs) == 1
    assert (out_dirs[0] / "filtered.h5").exists()
    assert (out_dirs[0] / "metrics.json").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```
uv run pytest tests/test_pipeline_integration.py::test_run_pipeline_cli_with_stages_list -v
```

Expected: ImportError on `martymicfly.cli.run_pipeline`.

- [ ] **Step 4: Create `run_pipeline.py`**

Copy `src/martymicfly/cli/run_notch.py` to `src/martymicfly/cli/run_pipeline.py`. In the new file:

1. Replace direct `NotchStage(...)` construction with `build_pipeline(cfg.stages)` and `run_pipeline(stages, ctx)`.
2. Resolve the per-stage outputs: notch-stage outputs (`filtered.h5`, `metrics.json`, `metrics.csv`, plots) come from existing eval modules called when a `notch` entry is present.
3. Skip Stage-2-specific output handling — Task 21 will inject array-filter outputs.

The complete new `run_pipeline.py` `main` body (replace the section that built/ran the notch stage):

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_pipeline")
    parser.add_argument("--config", required=False, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg_path = Path(args.config).resolve()
    cfg_payload = yaml.safe_load(cfg_path.read_text())
    cfg = AppConfig.model_validate(cfg_payload)
    if args.output_dir:
        cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"dir": args.output_dir})})

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + "_" + cfg.config_hash()
    out_dir = Path(cfg.output.dir.format(run_id=run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs (unchanged)
    synth = load_synth_h5(cfg.input.audio_h5)
    mic_pos = load_mic_geom_xml(cfg.input.mic_geom_xml)
    fs = synth["sample_rate"]
    n_total = synth["time_data"].shape[0]
    seg_start, n_seg = _select_segment(cfg, n_total, fs)
    i0 = int(round(seg_start * fs))
    time_seg = synth["time_data"][i0 : i0 + n_seg, :]

    per_motor_bpf = interpolate_per_motor_bpf(
        synth["rpm_per_esc"], seg_start, n_seg / fs, n_seg, fs, cfg.rotor.n_blades
    )
    harm_matrix = build_harmonic_matrix(per_motor_bpf, cfg.rotor.n_harmonics)

    ctx = PipelineContext(
        time_data=time_seg,
        sample_rate=fs,
        rpm_per_esc=synth["rpm_per_esc"],
        mic_positions=mic_pos,
        per_motor_bpf=per_motor_bpf,
        harm_matrix=harm_matrix,
        metadata={
            "platform": synth.get("platform"),
            "segment": {"start": seg_start, "duration": n_seg / fs},
            "config": cfg,
            "out_dir": str(out_dir),
        },
    )
    stages = build_pipeline(cfg.stages)
    ctx = run_pipeline(stages, ctx)

    # Notch-stage outputs (only if notch ran)
    if any(s.name == "notch" for s in stages):
        pre_notch = ctx.metadata["pre_notch"]
        post = ctx.time_data
        # ... existing metrics/plots/write_filtered calls — unchanged ...
        # (keep the notch-specific block from the old run_notch.py here)

    # Snapshot config + hash
    (out_dir / "config.yaml").write_text(cfg_path.read_text())
    (out_dir / "config.hash").write_text(cfg.config_hash())

    log.info("run complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Use the existing notch-output block from the old `run_notch.py` body verbatim inside the `if any(s.name == "notch" ...)` branch — do not paraphrase. Stage 2 outputs are added by Task 21.

- [ ] **Step 5: Make `run_notch.py` a thin shim**

Replace the body of `src/martymicfly/cli/run_notch.py` with:

```python
"""Backward-compatible shim: forwards to martymicfly.cli.run_pipeline."""
from martymicfly.cli.run_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Migrate `configs/example_notch.yaml`**

Replace the file's contents with the stages-list form (preserve the user's tuned values):

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
  n_harmonics: 13

stages:
  - kind: notch
    pole_radius:
      mode: scalar
      value: 0.9996
    multichannel: false
    block_size: 4096

metrics:
  welch_nperseg: 4096
  welch_noverlap: 2048
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

- [ ] **Step 7: Update existing integration test**

Edit `tests/test_pipeline_integration.py` to use the new YAML format wherever it builds a config; keep the assertions identical.

- [ ] **Step 8: Run all tests**

```
uv run pytest -x -v
```

Expected: all pass (notch behavior unchanged, just driven through the new code path).

- [ ] **Step 9: Commit**

```bash
git add src/martymicfly/cli/ configs/example_notch.yaml tests/test_pipeline_integration.py
git commit -m "feat(cli): generic run_pipeline driver; run_notch is now a shim"
```

---

## Task 5: `io/source_artifact.py` — load drone source artifact

**Files:**
- Create: `src/martymicfly/io/source_artifact.py`
- Test: `tests/test_io_source_artifact.py`
- Test fixture (read-only): `tests/fixtures/make_tiny_drone_artifact.py` (Task 11) — for Task 5 we exercise against a tiny artifact built inline.

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_source_artifact.py`:

```python
import h5py
import numpy as np
import pytest


def _write_minimal_artifact(path):
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 16000.0
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["blade_counts"] = np.array([2, 2], dtype=np.int32)
        plat["rotor_positions"] = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.10, 0.10], dtype=np.float64)
        f["rotor_index"] = np.array([0, 0, 1, 1], dtype=np.int32)
        f["subsource_positions"] = np.array(
            [[0.10, 0.0, 0.0], [0.20, 0.0, 0.0], [-0.10, 0.0, 0.0], [-0.20, 0.0, 0.0]],
            dtype=np.float64,
        )
        rng = np.random.default_rng(0)
        f["subsource_signals"] = rng.standard_normal((4, 1600)).astype(np.float64)
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(10, 3000.0)
            g["timestamp"] = np.linspace(0, 0.1, 10)


def test_load_source_artifact_returns_expected_shapes(tmp_path):
    from martymicfly.io.source_artifact import load_source_artifact
    p = tmp_path / "art.h5"
    _write_minimal_artifact(p)
    art = load_source_artifact(str(p))
    assert art.sample_rate == 16000.0
    assert art.subsource_positions.shape == (4, 3)
    assert art.subsource_signals.shape == (4, 1600)
    assert art.rotor_index.shape == (4,)
    assert art.rotor_positions.shape == (3, 2)
    assert art.rotor_radii.tolist() == [0.10, 0.10]
    assert sorted(art.rpm_per_esc.keys()) == ["ESC1", "ESC2"]


def test_load_source_artifact_rejects_missing_platform(tmp_path):
    p = tmp_path / "no_plat.h5"
    with h5py.File(p, "w") as f:
        f.attrs["sample_rate"] = 16000.0
    from martymicfly.io.source_artifact import load_source_artifact
    with pytest.raises(ValueError, match="platform"):
        load_source_artifact(str(p))
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_io_source_artifact.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the loader**

Create `src/martymicfly/io/source_artifact.py`:

```python
"""Reader for drone-source artifacts (gap_tip subsource model).

The artifact is produced upstream by drone_synthdata's reconstruction step
and contains the per-subsource time-domain signal at known geometry plus
platform metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class SourceArtifact:
    sample_rate: float
    subsource_positions: np.ndarray   # (G, 3)
    subsource_signals: np.ndarray     # (G, N)
    rotor_index: np.ndarray           # (G,) int
    rotor_positions: np.ndarray       # (3, R)
    rotor_radii: np.ndarray           # (R,)
    blade_counts: np.ndarray          # (R,)
    rpm_per_esc: dict
    metadata: dict


def load_source_artifact(path: str | Path) -> SourceArtifact:
    p = Path(path)
    with h5py.File(p, "r") as f:
        if "platform" not in f:
            raise ValueError(
                f"{p}: missing /platform group; not a valid drone source artifact"
            )
        sr = float(f.attrs["sample_rate"])
        plat = f["platform"]
        rotor_positions = np.asarray(plat["rotor_positions"][...], dtype=np.float64)
        rotor_radii = np.asarray(plat["rotor_radii"][...], dtype=np.float64)
        blade_counts = np.asarray(plat["blade_counts"][...], dtype=np.int32)
        subsource_positions = np.asarray(f["subsource_positions"][...], dtype=np.float64)
        subsource_signals = np.asarray(f["subsource_signals"][...], dtype=np.float64)
        rotor_index = np.asarray(f["rotor_index"][...], dtype=np.int32)
        rpm_per_esc: dict = {}
        if "rpm_per_esc" in f:
            for name in f["rpm_per_esc"]:
                g = f["rpm_per_esc"][name]
                rpm_per_esc[name] = {
                    "rpm": np.asarray(g["rpm"][...], dtype=np.float64),
                    "timestamp": np.asarray(g["timestamp"][...], dtype=np.float64),
                }
        metadata = dict(f.attrs)
        if "reconstruction" in f:
            metadata["reconstruction"] = dict(f["reconstruction"].attrs)

    return SourceArtifact(
        sample_rate=sr,
        subsource_positions=subsource_positions,
        subsource_signals=subsource_signals,
        rotor_index=rotor_index,
        rotor_positions=rotor_positions,
        rotor_radii=rotor_radii,
        blade_counts=blade_counts,
        rpm_per_esc=rpm_per_esc,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_io_source_artifact.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/source_artifact.py tests/test_io_source_artifact.py
git commit -m "feat(io): load_source_artifact reads drone source artifacts"
```

---

## Task 6: `io/synth_h5.py` — read `/platform/` if present

**Files:**
- Modify: `src/martymicfly/io/synth_h5.py`
- Modify: `tests/test_io_synth_h5.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_io_synth_h5.py`:

```python
def test_load_synth_h5_includes_platform_when_present(tmp_path):
    import h5py, numpy as np
    p = tmp_path / "with_plat.h5"
    with h5py.File(p, "w") as f:
        f.attrs["sample_freq"] = 16000.0
        f["time_data"] = np.zeros((1600, 4), dtype=np.float64)
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["rotor_positions"] = np.array([[0.15, -0.15], [0, 0], [0, 0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.10, 0.10])
        plat["blade_counts"] = np.array([2, 2], dtype=np.int32)
        rpm = f.create_group("esc_telemetry")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(5, 3000.0)
            g["timestamp"] = np.linspace(0, 0.1, 5)
    from martymicfly.io.synth_h5 import load_synth_h5
    res = load_synth_h5(str(p))
    assert res["platform"] is not None
    assert res["platform"]["n_rotors"] == 2
    assert res["platform"]["rotor_positions"].shape == (3, 2)


def test_load_synth_h5_platform_is_none_when_absent(tmp_path):
    import h5py, numpy as np
    p = tmp_path / "no_plat.h5"
    with h5py.File(p, "w") as f:
        f.attrs["sample_freq"] = 16000.0
        f["time_data"] = np.zeros((1600, 4), dtype=np.float64)
        rpm = f.create_group("esc_telemetry")
        g = rpm.create_group("ESC1")
        g["rpm"] = np.full(5, 3000.0)
        g["timestamp"] = np.linspace(0, 0.1, 5)
    from martymicfly.io.synth_h5 import load_synth_h5
    res = load_synth_h5(str(p))
    assert res["platform"] is None
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_io_synth_h5.py -v
```

Expected: KeyError on `"platform"` (key not yet added to return dict).

- [ ] **Step 3: Modify the loader**

In `src/martymicfly/io/synth_h5.py`, after the existing dict assembly inside `load_synth_h5`, add:

```python
        platform = None
        if "platform" in f:
            plat = f["platform"]
            platform = {
                "n_rotors": int(plat.attrs["n_rotors"]),
                "rotor_positions": np.asarray(plat["rotor_positions"][...], dtype=np.float64),
                "rotor_radii": np.asarray(plat["rotor_radii"][...], dtype=np.float64),
                "blade_counts": np.asarray(plat["blade_counts"][...], dtype=np.int32),
            }
```

And add `"platform": platform` to the returned dict.

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_io_synth_h5.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/synth_h5.py tests/test_io_synth_h5.py
git commit -m "feat(io): load_synth_h5 reads /platform group when present"
```

---

## Task 7: `synth/propagation.py` — Greens propagator

**Files:**
- Create: `src/martymicfly/synth/__init__.py`
- Create: `src/martymicfly/synth/propagation.py`
- Test: `tests/test_propagation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_propagation.py`:

```python
import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def test_greens_propagate_one_source_one_mic_attenuation_and_delay():
    from martymicfly.synth.propagation import greens_propagate
    fs = 16000.0
    n = 4096
    t = np.arange(n) / fs
    src = np.sin(2 * np.pi * 500.0 * t).astype(np.float64)
    src_pos = np.array([[0.0, 0.0, 0.0]])
    mic_pos = np.array([[1.5, 0.0, 0.0]])
    out = greens_propagate(src.reshape(1, -1), src_pos, mic_pos, fs)
    assert out.shape == (n, 1)
    r = 1.5
    expected_amp = 1.0 / (4 * np.pi * r)
    delay_samples = r / SPEED_OF_SOUND * fs
    n_lo = int(np.floor(delay_samples)) + 200
    n_hi = n - 200
    rms_out = np.sqrt(np.mean(out[n_lo:n_hi, 0] ** 2))
    rms_src = np.sqrt(np.mean(src[: n_hi - n_lo] ** 2))
    assert abs(rms_out / rms_src - expected_amp) / expected_amp < 0.05


def test_greens_propagate_sums_two_sources():
    from martymicfly.synth.propagation import greens_propagate
    fs = 16000.0
    n = 4096
    rng = np.random.default_rng(1)
    s1 = rng.standard_normal(n)
    s2 = rng.standard_normal(n)
    src_pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mic_pos = np.array([[0.0, 0.0, 0.0]])
    sig = np.stack([s1, s2])
    out_both = greens_propagate(sig, src_pos, mic_pos, fs)
    out_1 = greens_propagate(sig[:1], src_pos[:1], mic_pos, fs)
    out_2 = greens_propagate(sig[1:], src_pos[1:], mic_pos, fs)
    np.testing.assert_allclose(out_both, out_1 + out_2, atol=1e-9)
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_propagation.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `synth/__init__.py`** (empty)

Create empty `src/martymicfly/synth/__init__.py`.

- [ ] **Step 4: Create the propagator**

Create `src/martymicfly/synth/propagation.py`:

```python
"""Free-space Green's function propagator.

For each source-mic pair, the source signal is delayed by r/c·fs samples
(fractional delay via FFT phase rotation) and attenuated by 1/(4π·r), then
summed across sources.
"""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def _fractional_delay_fft(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a fractional delay via phase rotation in the FFT domain.

    signal: (N,) real. delay_samples: float (positive = delay).
    Returns (N,) real. Periodic wrap is acceptable because the synthesis is
    longer than the longest path.
    """
    n = signal.shape[0]
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles/sample
    phase = np.exp(-2j * np.pi * freqs * delay_samples)
    return np.fft.irfft(spec * phase, n=n)


def greens_propagate(
    source_signals: np.ndarray,    # (S, N)
    source_positions: np.ndarray,  # (S, 3)
    mic_positions: np.ndarray,     # (M, 3)
    sample_rate: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Propagate S point sources to M mics, return (N, M)."""
    src = np.asarray(source_signals, dtype=np.float64)
    src_pos = np.asarray(source_positions, dtype=np.float64)
    mic_pos = np.asarray(mic_positions, dtype=np.float64)
    if src.ndim != 2 or src.shape[0] != src_pos.shape[0]:
        raise ValueError("source_signals shape must be (S, N) matching source_positions (S, 3)")
    n_samples = src.shape[1]
    out = np.zeros((n_samples, mic_pos.shape[0]), dtype=np.float64)
    for s in range(src.shape[0]):
        for m in range(mic_pos.shape[0]):
            r = float(np.linalg.norm(mic_pos[m] - src_pos[s]))
            if r < 1e-9:
                # Co-located source/mic — undefined for free-space Greens.
                # Use unit delay-free contribution; this is a degenerate case.
                shifted = src[s].copy()
                amp = 1.0
            else:
                delay = r / speed_of_sound * sample_rate
                shifted = _fractional_delay_fft(src[s], delay)
                amp = 1.0 / (4.0 * np.pi * r)
            out[:, m] += amp * shifted
    return out
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_propagation.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/martymicfly/synth/__init__.py src/martymicfly/synth/propagation.py tests/test_propagation.py
git commit -m "feat(synth): greens_propagate with FFT-phase fractional delay"
```

---

## Task 8: `synth/external_source.py` — `ExternalSourceSpec` + signal generation

**Files:**
- Create: `src/martymicfly/synth/external_source.py`
- Test: `tests/test_external_source.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_external_source.py`:

```python
import numpy as np
import pytest


def test_external_source_spec_validates_required_fields():
    from martymicfly.synth.external_source import ExternalSourceSpec
    spec = ExternalSourceSpec(kind="noise", duration_s=0.5, seed=0)
    assert spec.kind == "noise"
    assert spec.position_m == (0.0, 0.0, -1.5)


def test_external_source_spec_rejects_sine_without_freq():
    from martymicfly.synth.external_source import ExternalSourceSpec
    with pytest.raises(ValueError, match="sine_freq_hz"):
        ExternalSourceSpec(kind="sine", duration_s=0.5)


def test_external_source_generate_noise_deterministic():
    from martymicfly.synth.external_source import (
        ExternalSourceSpec,
        generate_external_signal,
    )
    spec = ExternalSourceSpec(kind="noise", duration_s=0.5, seed=42)
    fs = 16000.0
    s1 = generate_external_signal(spec, fs, n_samples=int(0.5 * fs))
    s2 = generate_external_signal(spec, fs, n_samples=int(0.5 * fs))
    np.testing.assert_array_equal(s1, s2)
    assert s1.shape == (8000,)


def test_external_source_generate_sine_correct_frequency():
    from martymicfly.synth.external_source import (
        ExternalSourceSpec,
        generate_external_signal,
    )
    spec = ExternalSourceSpec(kind="sine", duration_s=1.0, sine_freq_hz=500.0)
    fs = 16000.0
    sig = generate_external_signal(spec, fs, n_samples=fs.__int__())
    spec_fft = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    peak_f = f[np.argmax(spec_fft)]
    assert abs(peak_f - 500.0) < 2.0
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_external_source.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the module**

Create `src/martymicfly/synth/external_source.py`:

```python
"""External source specification and signal generation for compose_external."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class ExternalSourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["noise", "sweep", "sine", "file"]
    position_m: tuple[float, float, float] = (0.0, 0.0, -1.5)
    amplitude_db: float = 0.0
    duration_s: Optional[float] = None
    sine_freq_hz: Optional[float] = None
    sweep_f_lo_hz: Optional[float] = None
    sweep_f_hi_hz: Optional[float] = None
    file_path: Optional[str] = None
    seed: Optional[int] = 0

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "ExternalSourceSpec":
        if self.kind == "sine" and self.sine_freq_hz is None:
            raise ValueError("kind='sine' requires sine_freq_hz")
        if self.kind == "sweep":
            if self.sweep_f_lo_hz is None or self.sweep_f_hi_hz is None:
                raise ValueError("kind='sweep' requires sweep_f_lo_hz and sweep_f_hi_hz")
            if self.sweep_f_hi_hz <= self.sweep_f_lo_hz:
                raise ValueError("sweep_f_hi_hz must be > sweep_f_lo_hz")
        if self.kind == "file" and not self.file_path:
            raise ValueError("kind='file' requires file_path")
        return self


def generate_external_signal(
    spec: ExternalSourceSpec,
    sample_rate: float,
    n_samples: int,
) -> np.ndarray:
    """Generate the external source signal at the source location (pre-propagation)."""
    n = int(n_samples)
    t = np.arange(n) / sample_rate
    if spec.kind == "noise":
        rng = np.random.default_rng(spec.seed)
        sig = rng.standard_normal(n)
    elif spec.kind == "sine":
        sig = np.sin(2 * np.pi * spec.sine_freq_hz * t)
    elif spec.kind == "sweep":
        from scipy.signal import chirp
        sig = chirp(
            t, f0=spec.sweep_f_lo_hz, t1=t[-1], f1=spec.sweep_f_hi_hz,
            method="logarithmic",
        )
    elif spec.kind == "file":
        import h5py
        with h5py.File(spec.file_path, "r") as f:
            sig = np.asarray(f["time_data"][...], dtype=np.float64).reshape(-1)[:n]
        if sig.shape[0] < n:
            raise ValueError(
                f"file {spec.file_path} has only {sig.shape[0]} samples; need {n}"
            )
    else:
        raise ValueError(f"unsupported kind={spec.kind!r}")

    gain = 10.0 ** (spec.amplitude_db / 20.0)
    return (gain * sig).astype(np.float64)
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_external_source.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/synth/external_source.py tests/test_external_source.py
git commit -m "feat(synth): ExternalSourceSpec with kind-aware generation"
```

---

## Task 9: `synth/compose_external.py` — mix drone + external, write outputs

**Files:**
- Create: `src/martymicfly/synth/compose_external.py`
- Test: `tests/test_compose_external.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_compose_external.py`:

```python
from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_artifact(path, fs=16000.0, n=1600, n_subs=4):
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = fs
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["blade_counts"] = np.array([2, 2], dtype=np.int32)
        plat["rotor_positions"] = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.10, 0.10])
        f["rotor_index"] = np.array([0, 0, 1, 1], dtype=np.int32)
        f["subsource_positions"] = np.array(
            [[0.10, 0, 0], [0.20, 0, 0], [-0.10, 0, 0], [-0.20, 0, 0]],
            dtype=np.float64,
        )
        f["subsource_signals"] = rng.standard_normal((n_subs, n)).astype(np.float64)
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(5, 3000.0)
            g["timestamp"] = np.linspace(0.0, 0.1, 5)


def _write_geom_xml(path, mic_positions):
    lines = ['<?xml version="1.0"?>', '<MicArray>']
    for i, p in enumerate(mic_positions):
        lines.append(f'  <pos Name="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>')
    lines.append('</MicArray>')
    Path(path).write_text("\n".join(lines))


def test_compose_external_writes_synth_and_gt(tmp_path):
    from martymicfly.synth.compose_external import compose_external
    from martymicfly.synth.external_source import ExternalSourceSpec
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    _write_artifact(art)
    _write_geom_xml(geom, [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0)])
    spec = ExternalSourceSpec(kind="noise", position_m=(0.5, 0.0, -0.5), seed=0)
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    assert out_synth.exists() and out_gt.exists()
    with h5py.File(out_synth, "r") as f:
        assert f["time_data"].shape == (1600, 4)
        assert "platform" in f
        assert "external" in f
        assert tuple(f["external"].attrs["position_m"]) == (0.5, 0.0, -0.5)
    with h5py.File(out_gt, "r") as f:
        assert f["time_data"].shape == (1600, 1)
        assert f.attrs["kind"] == "external_only"


def test_compose_external_drone_plus_external_equals_mix(tmp_path):
    """Synth mix ≈ drone-only + external-only propagated separately."""
    from martymicfly.synth.compose_external import (
        compose_external,
        _propagate_drone_only,
        _propagate_external_only,
    )
    from martymicfly.synth.external_source import ExternalSourceSpec
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    _write_artifact(art)
    _write_geom_xml(geom, [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0)])
    spec = ExternalSourceSpec(kind="noise", position_m=(0.5, 0.0, -0.5), seed=0)
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    drone_only = _propagate_drone_only(str(art), str(geom))
    ext_only = _propagate_external_only(str(art), str(geom), spec)
    with h5py.File(out_synth, "r") as f:
        mix = np.asarray(f["time_data"][...])
    np.testing.assert_allclose(mix, drone_only + ext_only, atol=1e-9)
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_compose_external.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the helper**

Create `src/martymicfly/synth/compose_external.py`:

```python
"""Compose a synthetic measurement HDF5 by propagating the drone source
artifact's subsources to the mic array and adding a configurable external
source. Writes a paired ground-truth file with the pure external signal at
its source location (pre-propagation).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from martymicfly.constants import SPEED_OF_SOUND
from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.io.source_artifact import load_source_artifact
from martymicfly.synth.external_source import (
    ExternalSourceSpec,
    generate_external_signal,
)
from martymicfly.synth.propagation import greens_propagate


def _propagate_drone_only(artifact_path: str, mic_geom_path: str) -> np.ndarray:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    return greens_propagate(art.subsource_signals, art.subsource_positions, mic_pos,
                            art.sample_rate)


def _propagate_external_only(
    artifact_path: str, mic_geom_path: str, spec: ExternalSourceSpec
) -> np.ndarray:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    n_samples = art.subsource_signals.shape[1] if spec.duration_s is None \
        else int(round(spec.duration_s * art.sample_rate))
    ext = generate_external_signal(spec, art.sample_rate, n_samples)
    pos = np.array([list(spec.position_m)], dtype=np.float64)
    return greens_propagate(ext.reshape(1, -1), pos, mic_pos, art.sample_rate)


def compose_external(
    artifact_path: str,
    mic_geom_path: str,
    ext_spec: ExternalSourceSpec,
    out_synth_path: str,
    out_gt_path: str,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> None:
    art = load_source_artifact(artifact_path)
    mic_pos = load_mic_geom_xml(mic_geom_path)
    n_samples = art.subsource_signals.shape[1] if ext_spec.duration_s is None \
        else int(round(ext_spec.duration_s * art.sample_rate))

    drone_at_mics = greens_propagate(
        art.subsource_signals[:, :n_samples],
        art.subsource_positions,
        mic_pos,
        art.sample_rate,
        speed_of_sound,
    )

    ext_signal = generate_external_signal(ext_spec, art.sample_rate, n_samples)
    ext_at_mics = greens_propagate(
        ext_signal.reshape(1, -1),
        np.array([list(ext_spec.position_m)], dtype=np.float64),
        mic_pos,
        art.sample_rate,
        speed_of_sound,
    )

    mix = drone_at_mics + ext_at_mics

    out_synth = Path(out_synth_path)
    out_synth.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_synth, "w") as f:
        f.attrs["sample_freq"] = float(art.sample_rate)
        f.attrs["composed_by"] = "martymicfly.synth.compose_external"
        f["time_data"] = mix
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = int(art.rotor_positions.shape[1])
        plat["rotor_positions"] = art.rotor_positions
        plat["rotor_radii"] = art.rotor_radii
        plat["blade_counts"] = art.blade_counts
        rpm = f.create_group("esc_telemetry")
        for name, data in art.rpm_per_esc.items():
            g = rpm.create_group(name)
            g["rpm"] = data["rpm"]
            g["timestamp"] = data["timestamp"]
        ext_g = f.create_group("external")
        ext_g.attrs["kind"] = ext_spec.kind
        ext_g.attrs["amplitude_db"] = float(ext_spec.amplitude_db)
        ext_g.attrs["position_m"] = np.asarray(ext_spec.position_m, dtype=np.float64)
        ext_g.attrs["seed"] = int(ext_spec.seed) if ext_spec.seed is not None else -1
        ext_g.attrs["speed_of_sound"] = float(speed_of_sound)
        ext_g.attrs["propagation"] = "greens_1_over_r"

    out_gt = Path(out_gt_path)
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_gt, "w") as f:
        f.attrs["sample_freq"] = float(art.sample_rate)
        f.attrs["kind"] = "external_only"
        f["time_data"] = ext_signal.reshape(-1, 1)
        ext_g = f.create_group("external")
        ext_g.attrs["kind"] = ext_spec.kind
        ext_g.attrs["amplitude_db"] = float(ext_spec.amplitude_db)
        ext_g.attrs["position_m"] = np.asarray(ext_spec.position_m, dtype=np.float64)
        ext_g.attrs["seed"] = int(ext_spec.seed) if ext_spec.seed is not None else -1
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_compose_external.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/synth/compose_external.py tests/test_compose_external.py
git commit -m "feat(synth): compose_external mixes drone artifact + external source"
```

---

## Task 10: `synth/cli/compose.py` — synthesis CLI

**Files:**
- Create: `src/martymicfly/synth/cli/__init__.py`
- Create: `src/martymicfly/synth/cli/compose.py`
- Create: `configs/example_compose.yaml`
- Test: `tests/test_compose_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_compose_cli.py`:

```python
from pathlib import Path

import h5py
import numpy as np


def _write_artifact_and_geom(tmp_path):
    art = tmp_path / "art.h5"
    geom = tmp_path / "g.xml"
    rng = np.random.default_rng(11)
    with h5py.File(art, "w") as f:
        f.attrs["sample_rate"] = 16000.0
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 1
        plat["rotor_positions"] = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        plat["rotor_radii"] = np.array([0.1])
        plat["blade_counts"] = np.array([2], dtype=np.int32)
        f["rotor_index"] = np.array([0, 0], dtype=np.int32)
        f["subsource_positions"] = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
        f["subsource_signals"] = rng.standard_normal((2, 1600))
        rpm = f.create_group("rpm_per_esc")
        g = rpm.create_group("ESC1")
        g["rpm"] = np.full(5, 3000.0)
        g["timestamp"] = np.linspace(0, 0.1, 5)
    geom.write_text(
        '<MicArray>\n'
        '  <pos Name="0" x="0.3" y="0" z="0"/>\n'
        '  <pos Name="1" x="-0.3" y="0" z="0"/>\n'
        '</MicArray>\n'
    )
    return art, geom


def test_compose_cli_runs(tmp_path):
    from martymicfly.synth.cli.compose import main
    art, geom = _write_artifact_and_geom(tmp_path)
    out_synth = tmp_path / "mix.h5"
    out_gt = tmp_path / "gt.h5"
    cfg_yaml = tmp_path / "compose.yaml"
    cfg_yaml.write_text(f"""
input:
  drone_source_artifact_h5: {art}
  mic_geom_xml: {geom}
external:
  kind: noise
  position_m: [0.0, 0.0, -1.0]
  amplitude_db: 0.0
  duration_s: null
  seed: 0
output:
  synth_h5: {out_synth}
  ground_truth_h5: {out_gt}
""")
    rc = main(["--config", str(cfg_yaml)])
    assert rc == 0
    assert out_synth.exists() and out_gt.exists()
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_compose_cli.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `synth/cli/__init__.py`** (empty)

- [ ] **Step 4: Create the CLI**

Create `src/martymicfly/synth/cli/compose.py`:

```python
"""CLI: read compose-config YAML, run compose_external, write outputs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from martymicfly.synth.compose_external import compose_external
from martymicfly.synth.external_source import ExternalSourceSpec


class _ComposeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drone_source_artifact_h5: str
    mic_geom_xml: str


class _ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    synth_h5: str
    ground_truth_h5: str


class _ComposeAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: _ComposeInput
    external: ExternalSourceSpec
    output: _ComposeOutput


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compose")
    p.add_argument("--config", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    payload = yaml.safe_load(Path(args.config).read_text())
    cfg = _ComposeAppConfig.model_validate(payload)
    compose_external(
        cfg.input.drone_source_artifact_h5,
        cfg.input.mic_geom_xml,
        cfg.external,
        cfg.output.synth_h5,
        cfg.output.ground_truth_h5,
    )
    logging.getLogger("martymicfly.compose").info(
        "wrote %s and %s", cfg.output.synth_h5, cfg.output.ground_truth_h5
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Create `configs/example_compose.yaml`**

```yaml
input:
  drone_source_artifact_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml

external:
  kind: noise
  position_m: [0.0, 0.0, -1.5]
  amplitude_db: 0.0
  duration_s: null
  seed: 0

output:
  synth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip_gt.h5
```

- [ ] **Step 6: Run tests**

```
uv run pytest tests/test_compose_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add src/martymicfly/synth/cli/ configs/example_compose.yaml tests/test_compose_cli.py
git commit -m "feat(synth): compose CLI + example_compose.yaml"
```

---

## Task 11: Tiny drone-artifact + composed-mix fixtures

**Files:**
- Create: `tests/fixtures/make_tiny_drone_artifact.py`
- Create: `tests/fixtures/make_tiny_compose.py`
- Create: `tests/fixtures/tiny_drone_artifact.h5` (generated)
- Create: `tests/fixtures/tiny_geom_4mic.xml`
- Create: `tests/fixtures/tiny_synth_mixed.h5` (generated)
- Create: `tests/fixtures/tiny_gt.h5` (generated)

- [ ] **Step 1: Write the artifact-builder script**

Create `tests/fixtures/make_tiny_drone_artifact.py`:

```python
"""Generate a deterministic tiny drone-source artifact for tests."""
from pathlib import Path

import h5py
import numpy as np


def _write_geom(path: Path, mic_positions):
    lines = ['<?xml version="1.0"?>', '<MicArray>']
    for i, p in enumerate(mic_positions):
        lines.append(f'  <pos Name="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>')
    lines.append('</MicArray>')
    path.write_text("\n".join(lines))


def main():
    here = Path(__file__).parent
    art = here / "tiny_drone_artifact.h5"
    geom = here / "tiny_geom_4mic.xml"

    fs = 16000.0
    duration_s = 1.0
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)

    rotor_positions = np.array(
        [[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
    )
    rotor_radii = np.array([0.10, 0.10], dtype=np.float64)
    blade_counts = np.array([2, 2], dtype=np.int32)

    subsource_positions = np.array(
        [
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [-0.10, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotor_index = np.array([0, 0, 1, 1], dtype=np.int32)
    subsource_signals = rng.standard_normal((4, n)).astype(np.float64)

    with h5py.File(art, "w") as f:
        f.attrs["sample_rate"] = fs
        f.attrs["reconstruction_mode"] = "stationary"
        plat = f.create_group("platform")
        plat.attrs["n_rotors"] = 2
        plat["rotor_positions"] = rotor_positions
        plat["rotor_radii"] = rotor_radii
        plat["blade_counts"] = blade_counts
        f["rotor_index"] = rotor_index
        f["subsource_positions"] = subsource_positions
        f["subsource_signals"] = subsource_signals
        rpm = f.create_group("rpm_per_esc")
        for name in ("ESC1", "ESC2"):
            g = rpm.create_group(name)
            g["rpm"] = np.full(20, 3000.0)
            g["timestamp"] = np.linspace(0.0, duration_s, 20)

    _write_geom(geom, [(0.3, 0.0, 0.10), (-0.3, 0.0, -0.10),
                       (0.0, 0.3, 0.10), (0.0, -0.3, -0.10)])
    print(f"wrote {art} and {geom}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```
uv run python tests/fixtures/make_tiny_drone_artifact.py
```

Expected: `wrote .../tiny_drone_artifact.h5 and .../tiny_geom_4mic.xml`.

- [ ] **Step 3: Write the compose-mix builder script**

Create `tests/fixtures/make_tiny_compose.py`:

```python
"""Build the tiny composed mix + ground-truth files used by Stage-2 tests."""
from pathlib import Path

from martymicfly.synth.compose_external import compose_external
from martymicfly.synth.external_source import ExternalSourceSpec


def main():
    here = Path(__file__).parent
    art = here / "tiny_drone_artifact.h5"
    geom = here / "tiny_geom_4mic.xml"
    out_synth = here / "tiny_synth_mixed.h5"
    out_gt = here / "tiny_gt.h5"

    spec = ExternalSourceSpec(
        kind="noise",
        position_m=(0.5, 0.0, -0.5),
        amplitude_db=-6.0,
        seed=42,
    )
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    print(f"wrote {out_synth} and {out_gt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

```
uv run python tests/fixtures/make_tiny_compose.py
```

Expected: writes the two HDF5 files.

- [ ] **Step 5: Commit fixtures**

```bash
git add tests/fixtures/make_tiny_drone_artifact.py tests/fixtures/make_tiny_compose.py \
        tests/fixtures/tiny_drone_artifact.h5 tests/fixtures/tiny_geom_4mic.xml \
        tests/fixtures/tiny_synth_mixed.h5 tests/fixtures/tiny_gt.h5
git commit -m "test(fixtures): tiny drone artifact + composed mix + ground-truth"
```

---

## Task 12: `processing/csm.py` — measurement CSM

**Files:**
- Create: `src/martymicfly/processing/csm.py`
- Test: `tests/test_csm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_csm.py`:

```python
import numpy as np


def test_build_measurement_csm_shape_and_hermitian():
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    fs = 16000.0
    n = 8192
    rng = np.random.default_rng(0)
    time_data = rng.standard_normal((n, 4))
    cfg = CsmConfig(nperseg=512, noverlap=256, f_min_hz=200.0, f_max_hz=4000.0,
                    diag_loading_rel=0.0)
    csm, freqs = build_measurement_csm(time_data, fs, cfg)
    assert csm.shape == (freqs.shape[0], 4, 4)
    assert csm.dtype == np.complex128
    np.testing.assert_allclose(csm, csm.conj().transpose(0, 2, 1), atol=1e-9)
    assert freqs.min() >= 200.0
    assert freqs.max() <= 4000.0


def test_build_measurement_csm_diag_loading_increases_diagonal():
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    fs = 16000.0
    rng = np.random.default_rng(1)
    time_data = rng.standard_normal((4096, 3))
    base = CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0,
                     diag_loading_rel=0.0)
    loaded = CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0,
                       diag_loading_rel=1e-2)
    csm_b, _ = build_measurement_csm(time_data, fs, base)
    csm_l, _ = build_measurement_csm(time_data, fs, loaded)
    diag_b = np.abs(np.diagonal(csm_b, axis1=1, axis2=2)).mean()
    diag_l = np.abs(np.diagonal(csm_l, axis1=1, axis2=2)).mean()
    assert diag_l > diag_b
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_csm.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the module**

Create `src/martymicfly/processing/csm.py`:

```python
"""Welch-style cross-spectral matrix from multi-channel time data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import csd


@dataclass
class CsmConfig:
    nperseg: int = 512
    noverlap: int = 256
    window: str = "hann"
    diag_loading_rel: float = 1e-6
    f_min_hz: float = 200.0
    f_max_hz: float = 6000.0


def build_measurement_csm(
    time_data: np.ndarray,         # (N, M)
    sample_rate: float,
    cfg: CsmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (csm, freqs) where csm[f] is the (M, M) hermitian CSM at freqs[f].

    Frequencies outside [f_min_hz, f_max_hz] are dropped. Diagonal loading
    is applied as `cfg.diag_loading_rel * max(|diag|)` added to the identity.
    """
    if time_data.ndim != 2:
        raise ValueError("time_data must be 2D (N, M)")
    n_total, n_ch = time_data.shape

    # csd handles all pairs; we build the upper triangle and mirror.
    # First, compute the frequency grid using a single csd call.
    f, _ = csd(time_data[:, 0], time_data[:, 0],
               fs=sample_rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
               window=cfg.window, scaling="density")
    mask = (f >= cfg.f_min_hz) & (f <= cfg.f_max_hz)
    freqs = f[mask]
    n_f = freqs.shape[0]
    csm = np.zeros((n_f, n_ch, n_ch), dtype=np.complex128)
    for i in range(n_ch):
        for j in range(i, n_ch):
            _, c_ij = csd(time_data[:, i], time_data[:, j],
                           fs=sample_rate, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
                           window=cfg.window, scaling="density")
            c_ij = c_ij[mask]
            csm[:, i, j] = c_ij
            if i != j:
                csm[:, j, i] = np.conj(c_ij)

    if cfg.diag_loading_rel > 0.0:
        diag_mag = np.abs(np.diagonal(csm, axis1=1, axis2=2))
        peak = float(diag_mag.max()) if diag_mag.size else 0.0
        load = cfg.diag_loading_rel * peak
        eye = np.eye(n_ch, dtype=np.complex128)
        csm = csm + load * eye[None, :, :]

    return csm, freqs
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_csm.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/csm.py tests/test_csm.py
git commit -m "feat(processing): build_measurement_csm via Welch + diag loading"
```

---

## Task 13: `processing/beamform_grid.py` — diagnostic grid + rotor mask

**Files:**
- Create: `src/martymicfly/processing/beamform_grid.py`
- Test: `tests/test_beamform_grid.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_beamform_grid.py`:

```python
import numpy as np


def test_build_diagnostic_grid_shape_and_z():
    from martymicfly.processing.beamform_grid import build_diagnostic_grid
    grid, shape = build_diagnostic_grid(extent_xy_m=0.5, increment_m=0.05, z_m=0.0)
    nx, ny = shape
    assert nx == ny == 21  # round(2*0.5/0.05)+1
    assert grid.shape == (nx * ny, 3)
    assert np.all(grid[:, 2] == 0.0)


def test_build_rotor_disc_mask_marks_inside_only():
    from martymicfly.processing.beamform_grid import (
        build_diagnostic_grid,
        build_rotor_disc_mask,
    )
    grid, _ = build_diagnostic_grid(0.5, 0.02, 0.0)
    rotor_pos = np.array([[0.15, -0.15], [0.0, 0.0], [0.0, 0.0]])
    rotor_radii = np.array([0.10, 0.10])
    mask = build_rotor_disc_mask(grid, rotor_pos, rotor_radii, z_tol_m=0.02)
    assert mask.sum() > 0
    inside = grid[mask]
    for p in inside:
        d0 = np.linalg.norm(p[:2] - np.array([0.15, 0.0]))
        d1 = np.linalg.norm(p[:2] - np.array([-0.15, 0.0]))
        assert d0 <= 0.10 + 1e-9 or d1 <= 0.10 + 1e-9
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_beamform_grid.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the module**

Create `src/martymicfly/processing/beamform_grid.py`:

```python
"""Diagnostic-grid builders and rotor-disc spatial masks for Stage 2."""
from __future__ import annotations

import numpy as np


def build_diagnostic_grid(
    extent_xy_m: float,
    increment_m: float,
    z_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    nx = int(round(2 * extent_xy_m / increment_m)) + 1
    ny = nx
    xs = np.linspace(-extent_xy_m, +extent_xy_m, nx)
    ys = np.linspace(-extent_xy_m, +extent_xy_m, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z_m)], axis=1)
    return points, (nx, ny)


def build_rotor_disc_mask(
    grid_positions: np.ndarray,    # (G, 3)
    rotor_positions: np.ndarray,   # (3, R)
    rotor_radii: np.ndarray,       # (R,)
    z_tol_m: float = 0.05,
) -> np.ndarray:
    g_xy = grid_positions[:, :2]
    g_z = grid_positions[:, 2]
    r_xy = rotor_positions[:2, :].T   # (R, 2)
    r_z = rotor_positions[2, :]        # (R,)
    mask = np.zeros(g_xy.shape[0], dtype=bool)
    for i in range(r_xy.shape[0]):
        d_xy = np.linalg.norm(g_xy - r_xy[i], axis=1)
        z_ok = np.abs(g_z - r_z[i]) <= z_tol_m
        mask |= (d_xy <= rotor_radii[i]) & z_ok
    return mask
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_beamform_grid.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/beamform_grid.py tests/test_beamform_grid.py
git commit -m "feat(processing): diagnostic grid + rotor-disc spatial mask"
```

---

## Task 14: `processing/algorithms/base.py` — Protocol + `SourceMap` + `reconstruct_csm`

**Files:**
- Create: `src/martymicfly/processing/algorithms/__init__.py`
- Create: `src/martymicfly/processing/algorithms/base.py`
- Test: `tests/test_algorithm_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_algorithm_base.py`:

```python
import numpy as np


def test_source_map_subset_drops_columns():
    from martymicfly.processing.algorithms.base import SourceMap
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    powers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sm = SourceMap(
        positions=positions, powers=powers,
        frequencies=np.array([100.0, 200.0]), grid_shape=None, metadata={},
    )
    keep = np.array([True, False, True])
    sub = sm.subset(keep)
    assert sub.positions.shape == (2, 3)
    assert sub.powers.shape == (2, 2)
    np.testing.assert_array_equal(sub.powers[:, 0], np.array([1.0, 4.0]))


def test_reconstruct_csm_single_source_round_trip():
    """One unit-power source at (1, 0, 0); reconstruct_csm must be hermitian
    and have non-negative diagonal."""
    from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
    positions = np.array([[1.0, 0.0, 0.0]])
    freqs = np.array([500.0])
    powers = np.array([[1.0]])
    sm = SourceMap(positions=positions, powers=powers, frequencies=freqs,
                   grid_shape=None, metadata={})
    mics = np.array([[0.0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
    csm = reconstruct_csm(sm, mics)
    assert csm.shape == (1, 3, 3)
    np.testing.assert_allclose(csm, csm.conj().transpose(0, 2, 1), atol=1e-12)
    assert (np.real(np.diagonal(csm, axis1=1, axis2=2)) >= 0).all()
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_algorithm_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create modules**

Create `src/martymicfly/processing/algorithms/__init__.py`:

```python
"""Algorithm registry for ArrayFilterStage."""
from martymicfly.processing.algorithms.base import Algorithm, SourceMap, reconstruct_csm
from martymicfly.processing.algorithms.clean_sc import CleanScAlgorithm  # noqa: F401

ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {
    "clean_sc": CleanScAlgorithm,
}
```

(That import of `CleanScAlgorithm` will succeed once Task 15 lands; for now, write the registry entry as a TODO-free forward reference by deferring the import.)

Replace the registry block above with the deferred form to avoid an import-time error before Task 15:

```python
"""Algorithm registry for ArrayFilterStage. Concrete algorithms register
themselves on import."""
from martymicfly.processing.algorithms.base import (
    Algorithm,
    SourceMap,
    reconstruct_csm,
)

ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {}


def register_algorithm(cls: type[Algorithm]) -> type[Algorithm]:
    ALGORITHM_REGISTRY[cls.name] = cls
    return cls
```

Create `src/martymicfly/processing/algorithms/base.py`:

```python
"""Algorithm protocol, SourceMap dataclass, default reconstruct_csm."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Protocol

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


@dataclass(frozen=True)
class SourceMap:
    positions: np.ndarray            # (G, 3) grid points
    powers: np.ndarray               # (F, G) p^2
    frequencies: np.ndarray          # (F,) Hz
    grid_shape: Optional[tuple[int, int]]
    metadata: dict

    def subset(self, mask: np.ndarray) -> "SourceMap":
        return replace(
            self,
            positions=self.positions[mask],
            powers=self.powers[:, mask],
            grid_shape=None,
        )


class Algorithm(Protocol):
    name: str
    consumes: Literal["csm", "time"]

    def fit(
        self,
        *,
        csm: Optional[np.ndarray],
        frequencies: Optional[np.ndarray],
        time_data: Optional[np.ndarray],
        sample_rate: float,
        mic_positions: np.ndarray,
        grid_positions: np.ndarray,
        params: dict,
    ) -> SourceMap: ...


def reconstruct_csm(
    source_map: SourceMap,
    mic_positions: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """drone_csm[f] = sum_g power[f, g] · h[f, g] h[f, g]^H
    h[f, g] = exp(-j 2π f r_mg / c) / (4π r_mg)
    """
    positions = source_map.positions      # (G, 3)
    powers = source_map.powers            # (F, G)
    freqs = source_map.frequencies        # (F,)
    n_f = freqs.shape[0]
    n_m = mic_positions.shape[0]

    # Distances (M, G)
    diff = mic_positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r < 1e-9, 1e-9, r)
    inv_r = 1.0 / (4.0 * np.pi * r)        # (M, G)

    csm = np.zeros((n_f, n_m, n_m), dtype=np.complex128)
    for fi, f in enumerate(freqs):
        phase = np.exp(-2j * np.pi * f * r / speed_of_sound)   # (M, G)
        h = inv_r * phase                                       # (M, G)
        # csm[fi] = h * diag(powers[fi]) * h^H
        weighted = h * powers[fi][None, :]                      # (M, G)
        csm[fi] = weighted @ h.conj().T                         # (M, M)
    return csm
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_algorithm_base.py -v
```

Expected: 2 passed (subset + reconstruct_csm; CleanScAlgorithm import removed from `__init__`).

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/processing/algorithms/ tests/test_algorithm_base.py
git commit -m "feat(algorithms): Protocol + SourceMap + reconstruct_csm default"
```

---

## Task 15: `processing/algorithms/clean_sc.py` — CLEAN-SC algorithm

**Files:**
- Create: `src/martymicfly/processing/algorithms/clean_sc.py`
- Test: `tests/test_clean_sc_algorithm.py`

- [ ] **Step 1: Smoke-test Acoular `ImportGrid` interactively (5 min, not committed)**

Run a quick exploratory check to settle the ImportGrid contract before writing the wrapper. From a Python shell:

```python
import numpy as np
from acoular import ImportGrid, MicGeom, SteeringVector
positions = np.array([[0,0,0],[0.1,0,0]], dtype=np.float64).T  # (3, G)
g = ImportGrid()
# Try whichever path the installed acoular accepts:
try:
    g.gpos_file = ""  # in-memory not supported via gpos_file
    g.gpos = positions  # may need patch
except Exception as e:
    print("gpos:", e)
```

Decide whether to use `gpos_file` (write a tempfile per call) or assign `gpos` directly. Document the choice in a one-line comment in `clean_sc.py`.

- [ ] **Step 2: Write failing test**

Create `tests/test_clean_sc_algorithm.py`:

```python
import numpy as np


def test_clean_sc_recovers_single_source_on_diagonal_grid():
    """Synthesize a CSM with one known source via reconstruct_csm; CLEAN-SC
    on a 2D RectGrid that includes the true position must recover ≥ 70 %
    of the total power at the corresponding grid point."""
    from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
    from martymicfly.processing.algorithms.clean_sc import CleanScAlgorithm
    from martymicfly.processing.beamform_grid import build_diagnostic_grid

    grid, shape = build_diagnostic_grid(0.5, 0.05, 0.0)
    true_pos = np.array([[0.20, 0.0, 0.0]])
    freqs = np.array([1000.0, 1500.0, 2000.0])
    sm_true = SourceMap(
        positions=true_pos,
        powers=np.ones((3, 1)),
        frequencies=freqs,
        grid_shape=None,
        metadata={},
    )
    mic_positions = np.array([
        [0.4, 0.0, 0.05], [-0.4, 0.0, -0.05],
        [0.0, 0.4, 0.05], [0.0, -0.4, -0.05],
    ])
    csm = reconstruct_csm(sm_true, mic_positions)

    algo = CleanScAlgorithm()
    sm_est = algo.fit(
        csm=csm, frequencies=freqs, time_data=None, sample_rate=16000.0,
        mic_positions=mic_positions, grid_positions=grid,
        params={"damp": 0.6, "n_iter": 50},
    )
    total = sm_est.powers.sum()
    nearest_idx = int(np.argmin(np.linalg.norm(grid[:, :2] - true_pos[:, :2], axis=1)))
    near_neighbors = np.linalg.norm(grid[:, :2] - true_pos[:, :2], axis=1) <= 0.07
    near_power = sm_est.powers[:, near_neighbors].sum()
    assert near_power / total > 0.70, f"only {near_power/total:.2f} of power near true source"
```

- [ ] **Step 3: Run failing test**

```
uv run pytest tests/test_clean_sc_algorithm.py -v
```

Expected: ImportError or assertion failure.

- [ ] **Step 4: Implement the wrapper**

Create `src/martymicfly/processing/algorithms/clean_sc.py`:

```python
"""CLEAN-SC algorithm wrapper around acoular.BeamformerCleansc."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from martymicfly.processing.algorithms import register_algorithm
from martymicfly.processing.algorithms.base import Algorithm, SourceMap


def _import_grid_from_array(grid_positions: np.ndarray):
    """Wrap (G, 3) into an Acoular Grid. Implementation chosen via the
    Step-1 smoke test: writes a temp .xml that Acoular's ImportGrid reads.
    Replace with direct gpos assignment if the installed acoular supports it.
    """
    from acoular import ImportGrid

    pos = np.asarray(grid_positions, dtype=np.float64).T  # (3, G)
    # ImportGrid expects an XML file path. Write a tempfile in this format.
    f = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    f.write('<?xml version="1.0"?>\n<MicArray>\n')
    for i in range(pos.shape[1]):
        f.write(f'  <pos Name="{i}" x="{pos[0, i]}" y="{pos[1, i]}" z="{pos[2, i]}"/>\n')
    f.write('</MicArray>\n')
    f.close()
    grid = ImportGrid(gpos_file=f.name)
    return grid


@register_algorithm
class CleanScAlgorithm:
    name: str = "clean_sc"
    consumes: str = "csm"

    def fit(self, *, csm, frequencies, mic_positions, grid_positions,
            params, **_) -> SourceMap:
        from acoular import (
            BeamformerCleansc,
            MicGeom,
            PowerSpectraImport,
            SteeringVector,
            config as acoular_config,
        )

        acoular_config.global_caching = "none"
        mg = MicGeom(pos_total=mic_positions.T.copy())
        grid = _import_grid_from_array(grid_positions)
        steer = SteeringVector(grid=grid, mics=mg, steer_type="classic")
        ps = PowerSpectraImport()
        ps.csm = np.asarray(csm, dtype=np.complex128)
        ps.frequencies = np.asarray(frequencies, dtype=float)

        bf = BeamformerCleansc(
            freq_data=ps, steer=steer, r_diag=False,
            damp=params["damp"], n_iter=params["n_iter"],
            cached=False,
        )
        powers = np.zeros((len(frequencies), len(grid_positions)), dtype=float)
        for i, f in enumerate(frequencies):
            powers[i] = np.asarray(bf.synthetic(float(f), num=0)).ravel()

        return SourceMap(
            positions=np.asarray(grid_positions, dtype=np.float64),
            powers=powers,
            frequencies=np.asarray(frequencies, dtype=float),
            grid_shape=None,
            metadata={"damp": params["damp"], "n_iter": params["n_iter"]},
        )
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_clean_sc_algorithm.py -v
```

Expected: pass. If the assertion fails because the recovery is below 70 %, investigate first whether the Acoular call returns the expected unit (some Acoular versions return power in p² with non-trivial scaling): comment in `clean_sc.py` saying which version was tested and what `synthetic(f, num=0)` returned, and adjust the test threshold to ≥ 50 % only if the cause is documented (otherwise the implementation has a real bug).

- [ ] **Step 6: Wire registry import**

In `src/martymicfly/processing/algorithms/__init__.py`, import `clean_sc` so `@register_algorithm` runs at import:

```python
from martymicfly.processing.algorithms import clean_sc as _  # noqa: F401
```

Add this line at the bottom of `algorithms/__init__.py`.

- [ ] **Step 7: Commit**

```bash
git add src/martymicfly/processing/algorithms/clean_sc.py \
        src/martymicfly/processing/algorithms/__init__.py \
        tests/test_clean_sc_algorithm.py
git commit -m "feat(algorithms): CleanScAlgorithm via acoular BeamformerCleansc"
```

---

## Task 16: `io/residual_csm_h5.py` — write residual CSM

**Files:**
- Create: `src/martymicfly/io/residual_csm_h5.py`
- Test: `tests/test_io_residual_csm.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_io_residual_csm.py`:

```python
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
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_io_residual_csm.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the writer**

Create `src/martymicfly/io/residual_csm_h5.py`:

```python
"""Writer for residual-CSM HDF5 files."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_residual_csm(
    path: str,
    residual: np.ndarray,         # (F, M, M) complex
    freqs: np.ndarray,            # (F,)
    attrs: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v
        f["csm_real"] = np.real(residual).astype(np.float64)
        f["csm_imag"] = np.imag(residual).astype(np.float64)
        f["frequencies"] = np.asarray(freqs, dtype=np.float64)
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_io_residual_csm.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/residual_csm_h5.py tests/test_io_residual_csm.py
git commit -m "feat(io): write_residual_csm with separate real/imag datasets"
```

---

## Task 17: `io/ground_truth_h5.py` — load GT (and reuse `compose_external` writer)

**Files:**
- Create: `src/martymicfly/io/ground_truth_h5.py`
- Test: `tests/test_io_ground_truth.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_io_ground_truth.py`:

```python
import h5py
import numpy as np


def test_load_ground_truth_returns_signal_and_metadata(tmp_path):
    from martymicfly.io.ground_truth_h5 import load_ground_truth
    p = tmp_path / "gt.h5"
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1024).reshape(-1, 1)
    with h5py.File(p, "w") as f:
        f.attrs["sample_freq"] = 16000.0
        f.attrs["kind"] = "external_only"
        f["time_data"] = sig
        ext = f.create_group("external")
        ext.attrs["kind"] = "noise"
        ext.attrs["position_m"] = np.array([0.5, 0.0, -0.5])
    res = load_ground_truth(str(p))
    np.testing.assert_array_equal(res.signal, sig.ravel())
    assert res.sample_rate == 16000.0
    assert res.position_m == (0.5, 0.0, -0.5)
    assert res.kind == "noise"
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_io_ground_truth.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the loader**

Create `src/martymicfly/io/ground_truth_h5.py`:

```python
"""Reader for ground-truth (external-only) HDF5 files written by compose_external."""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class GroundTruth:
    signal: np.ndarray            # (N,) — at the source location, pre-propagation
    sample_rate: float
    position_m: tuple[float, float, float]
    kind: str
    amplitude_db: float


def load_ground_truth(path: str) -> GroundTruth:
    with h5py.File(path, "r") as f:
        sr = float(f.attrs["sample_freq"])
        sig = np.asarray(f["time_data"][...]).reshape(-1)
        ext = f["external"]
        pos = tuple(float(x) for x in np.asarray(ext.attrs["position_m"]))
        return GroundTruth(
            signal=sig,
            sample_rate=sr,
            position_m=pos,
            kind=str(ext.attrs["kind"]),
            amplitude_db=float(ext.attrs.get("amplitude_db", 0.0)),
        )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_io_ground_truth.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/io/ground_truth_h5.py tests/test_io_ground_truth.py
git commit -m "feat(io): load_ground_truth for compose_external GT files"
```

---

## Task 18: `processing/array_filter.py` — `steer_to_psd`, band integration, ArrayFilterStage skeleton

The full `ArrayFilterStage.process` lands in Task 21 after the `ArrayFilterStageConfig` is fleshed out (Task 19) and after the helpers `steer_to_psd` and `integrate_band_maps` exist. This task defines those two helpers separately.

**Files:**
- Create: `src/martymicfly/processing/steering.py` (helpers — keeps `array_filter.py` lean)
- Test: `tests/test_steering.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_steering.py`:

```python
import numpy as np


def test_steer_to_psd_shapes_and_units():
    """Steering an identity-only CSM (white noise per mic) at a random target
    yields a positive-real PSD."""
    from martymicfly.processing.steering import steer_to_psd
    n_f, n_m = 8, 4
    csm = np.tile(np.eye(n_m, dtype=np.complex128), (n_f, 1, 1))
    freqs = np.linspace(200.0, 1000.0, n_f)
    mic_positions = np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
    target = (0.0, 0.0, -0.5)
    psd = steer_to_psd(csm, freqs, mic_positions, target)
    assert psd.shape == (n_f,)
    assert (psd > 0).all()
    assert np.iscomplexobj(psd) is False


def test_integrate_band_maps_produces_band_dict():
    from martymicfly.processing.algorithms.base import SourceMap
    from martymicfly.processing.array_filter import integrate_band_maps, BandConfig
    n_f, n_g = 5, 9  # 3x3 grid
    powers = np.ones((n_f, n_g))
    sm = SourceMap(
        positions=np.zeros((n_g, 3)),
        powers=powers,
        frequencies=np.linspace(200.0, 1000.0, n_f),
        grid_shape=(3, 3),
        metadata={},
    )
    bands = [
        BandConfig(name="lo", f_min_hz=0.0, f_max_hz=500.0),
        BandConfig(name="hi", f_min_hz=500.0, f_max_hz=10_000.0),
    ]
    maps = integrate_band_maps(sm, bands, (3, 3))
    assert set(maps.keys()) == {"lo", "hi"}
    assert maps["lo"].shape == (3, 3)
    assert maps["hi"].shape == (3, 3)
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_steering.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `steer_to_psd`**

Create `src/martymicfly/processing/steering.py`:

```python
"""Conventional-beamformer steering helper for the pseudo-target-PSD output."""
from __future__ import annotations

import numpy as np

from martymicfly.constants import SPEED_OF_SOUND


def steer_to_psd(
    csm: np.ndarray,             # (F, M, M)
    frequencies: np.ndarray,     # (F,)
    mic_positions: np.ndarray,   # (M, 3)
    target_point: tuple[float, float, float],
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """PSD = (1/M^2) · h^H · csm · h, with h[m] = exp(j 2π f r_m / c) (no
    1/r weighting; pure phase steering — what conventional delay-and-sum
    delivers when integrated over the mic aperture)."""
    target = np.asarray(target_point, dtype=np.float64)
    diff = mic_positions - target[None, :]
    r = np.linalg.norm(diff, axis=1)         # (M,)
    n_f = frequencies.shape[0]
    n_m = mic_positions.shape[0]
    psd = np.zeros(n_f, dtype=np.float64)
    for fi, f in enumerate(frequencies):
        h = np.exp(2j * np.pi * f * r / speed_of_sound)   # (M,)
        # quadratic form
        val = np.real(h.conj() @ csm[fi] @ h) / (n_m * n_m)
        psd[fi] = float(val)
    return psd
```

- [ ] **Step 4: Implement `integrate_band_maps` + `BandConfig` skeleton**

Create `src/martymicfly/processing/array_filter.py`:

```python
"""Stage 2 — array deconvolution stage. Body filled in Task 21."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from martymicfly.processing.algorithms.base import SourceMap


@dataclass
class BandConfig:
    name: str
    f_min_hz: float
    f_max_hz: float


def integrate_band_maps(
    source_map: SourceMap,
    bands: list[BandConfig],
    grid_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    nx, ny = grid_shape
    out: dict[str, np.ndarray] = {}
    f = source_map.frequencies
    for band in bands:
        mask = (f >= band.f_min_hz) & (f <= band.f_max_hz)
        if not mask.any():
            out[band.name] = np.zeros((nx, ny), dtype=np.float64)
            continue
        integrated = source_map.powers[mask].sum(axis=0)        # (G,)
        if integrated.size != nx * ny:
            raise ValueError(
                f"powers length {integrated.size} doesn't match grid_shape {grid_shape}"
            )
        out[band.name] = integrated.reshape(nx, ny)
    return out
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_steering.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/martymicfly/processing/steering.py src/martymicfly/processing/array_filter.py tests/test_steering.py
git commit -m "feat(processing): steer_to_psd + integrate_band_maps helpers"
```

---

## Task 19: `ArrayFilterStageConfig` — pydantic body

**Files:**
- Modify: `src/martymicfly/config.py` (replace the placeholder `ArrayFilterStageConfig`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_array_filter_stage_config_full_payload():
    from martymicfly.config import ArrayFilterStageConfig
    payload = {
        "kind": "array_filter",
        "algorithm": "clean_sc",
        "csm": {"nperseg": 512, "noverlap": 256, "window": "hann",
                "diag_loading_rel": 1e-6, "f_min_hz": 200.0, "f_max_hz": 6000.0},
        "diagnostic_grid": {"extent_xy_m": 0.5, "increment_m": 0.02, "z_m": None},
        "bands": [
            {"name": "low", "f_min_hz": 200.0, "f_max_hz": 500.0},
            {"name": "mid", "f_min_hz": 500.0, "f_max_hz": 2000.0},
        ],
        "target_point_m": [0.0, 0.0, -1.5],
        "rotor_z_tolerance_m": 0.05,
        "clean_sc": {"damp": 0.6, "n_iter": 100},
    }
    cfg = ArrayFilterStageConfig.model_validate(payload)
    assert cfg.kind == "array_filter"
    assert cfg.algorithm == "clean_sc"
    assert cfg.target_point_m == (0.0, 0.0, -1.5)
    assert cfg.bands[0].name == "low"


def test_array_filter_stage_config_unknown_algorithm_rejected():
    import pytest
    from pydantic import ValidationError
    from martymicfly.config import ArrayFilterStageConfig
    with pytest.raises(ValidationError):
        ArrayFilterStageConfig.model_validate({"kind": "array_filter", "algorithm": "nope"})
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_config.py -v
```

Expected: failure on the second test (and the first one only succeeds if the placeholder is in `extra="allow"` mode — but the field types are missing).

- [ ] **Step 3: Replace the placeholder**

In `src/martymicfly/config.py`, replace the `ArrayFilterStageConfig` placeholder with:

```python
class CsmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nperseg: int = Field(default=512, ge=64)
    noverlap: int = Field(default=256, ge=0)
    window: str = "hann"
    diag_loading_rel: float = 1e-6
    f_min_hz: float = 200.0
    f_max_hz: float = 6000.0


class DiagnosticGridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extent_xy_m: float = 0.5
    increment_m: float = 0.02
    z_m: float | None = None


class BandConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    f_min_hz: float
    f_max_hz: float

    @model_validator(mode="after")
    def _check_range(self) -> "BandConfig":
        if self.f_max_hz <= self.f_min_hz:
            raise ValueError("f_max_hz must be > f_min_hz")
        return self


class CleanScConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    damp: float = 0.6
    n_iter: int = Field(default=100, ge=1)


class ArrayFilterStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["array_filter"]
    algorithm: Literal["clean_sc"] = "clean_sc"
    csm: CsmConfig = Field(default_factory=CsmConfig)
    diagnostic_grid: DiagnosticGridConfig = Field(default_factory=DiagnosticGridConfig)
    bands: list[BandConfig] = Field(default_factory=lambda: [
        BandConfig(name="low", f_min_hz=200.0, f_max_hz=500.0),
        BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0),
        BandConfig(name="high", f_min_hz=2000.0, f_max_hz=6000.0),
    ])
    target_point_m: tuple[float, float, float] = (0.0, 0.0, -1.5)
    rotor_z_tolerance_m: float = 0.05
    clean_sc: CleanScConfig = Field(default_factory=CleanScConfig)
```

(`Literal["clean_sc"]` is the registry's only entry now; future algorithms expand the union.)

Also add `ground_truth_h5: str | None = None` to `InputConfig`:

```python
class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_h5: str
    mic_geom_xml: str
    ground_truth_h5: str | None = None
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/config.py tests/test_config.py
git commit -m "feat(config): ArrayFilterStageConfig + CSM/grid/bands/clean_sc subconfigs"
```

---

## Task 20: `ArrayFilterStage` — full implementation

**Files:**
- Modify: `src/martymicfly/processing/array_filter.py`
- Modify: `src/martymicfly/processing/notch.py` (re-register; add `array_filter` builder import)
- Test: `tests/test_array_filter.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_array_filter.py`:

```python
import numpy as np
from dataclasses import replace


def test_array_filter_stage_e2e_on_tiny_fixture():
    from martymicfly.config import (
        ArrayFilterStageConfig,
        BandConfig,
        CleanScConfig,
        CsmConfig,
        DiagnosticGridConfig,
    )
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5
    from martymicfly.processing.array_filter import ArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext

    synth = load_synth_h5("tests/fixtures/tiny_synth_mixed.h5")
    mic_pos = load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml")
    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        diagnostic_grid=DiagnosticGridConfig(extent_xy_m=0.6, increment_m=0.05, z_m=0.0),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.5, 0.0, -0.5),
        rotor_z_tolerance_m=0.05,
        clean_sc=CleanScConfig(damp=0.6, n_iter=30),
    )
    ctx = PipelineContext(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"],
        mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    stage = ArrayFilterStage(cfg)
    new_ctx = stage.process(ctx)

    af = new_ctx.metadata["array_filter"]
    csm = af["csm_pre"]
    res = af["residual_csm"]
    assert csm.shape == res.shape
    np.testing.assert_allclose(res, res.conj().transpose(0, 2, 1), atol=1e-8)
    assert np.real(np.diagonal(csm, axis1=1, axis2=2)).sum() > \
           np.real(np.diagonal(res, axis1=1, axis2=2)).sum()
    assert af["target_psd_pre"].shape == af["target_psd_post"].shape
    assert (af["target_psd_post"] >= 0).all()
    assert "mid" in af["beam_maps"]
    assert af["beam_maps"]["mid"].shape == (25, 25)  # extent 0.6 / inc 0.05 = 25
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_array_filter.py -v
```

Expected: ImportError on `ArrayFilterStage` (currently a stub from Task 18).

- [ ] **Step 3: Fill in `ArrayFilterStage`**

Replace `src/martymicfly/processing/array_filter.py` with:

```python
"""Stage 2 — array deconvolution stage."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from martymicfly.config import ArrayFilterStageConfig, BandConfig
from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
from martymicfly.processing.beamform_grid import (
    build_diagnostic_grid,
    build_rotor_disc_mask,
)
from martymicfly.processing.csm import CsmConfig as RuntimeCsmConfig, build_measurement_csm
from martymicfly.processing.steering import steer_to_psd


def integrate_band_maps(
    source_map: SourceMap,
    bands: list[BandConfig],
    grid_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    nx, ny = grid_shape
    out: dict[str, np.ndarray] = {}
    f = source_map.frequencies
    for band in bands:
        mask = (f >= band.f_min_hz) & (f <= band.f_max_hz)
        if not mask.any():
            out[band.name] = np.zeros((nx, ny), dtype=np.float64)
            continue
        integrated = source_map.powers[mask].sum(axis=0)
        out[band.name] = integrated.reshape(nx, ny)
    return out


def integrate_csm_band_maps_from_residual(
    *args, **kwargs,
) -> dict[str, np.ndarray]:
    """Reserved for future: rebuild a 'post' diagnostic map directly from the
    residual CSM via conventional beamforming. Not used in this spec — the
    'post' beam-map is derived from masking the source_map (Task 23)."""
    raise NotImplementedError


class ArrayFilterStage:
    name = "array_filter"

    def __init__(self, cfg: ArrayFilterStageConfig):
        self.cfg = cfg
        if cfg.algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"unknown algorithm {cfg.algorithm!r}; available: {sorted(ALGORITHM_REGISTRY)}"
            )
        self.algo = ALGORITHM_REGISTRY[cfg.algorithm]()

    def process(self, ctx):
        # 1. CSM
        rcfg = RuntimeCsmConfig(
            nperseg=self.cfg.csm.nperseg,
            noverlap=self.cfg.csm.noverlap,
            window=self.cfg.csm.window,
            diag_loading_rel=self.cfg.csm.diag_loading_rel,
            f_min_hz=self.cfg.csm.f_min_hz,
            f_max_hz=self.cfg.csm.f_max_hz,
        )
        csm, freqs = build_measurement_csm(ctx.time_data, ctx.sample_rate, rcfg)

        # 2. Diagnostic grid
        z = self.cfg.diagnostic_grid.z_m
        if z is None:
            plat = ctx.metadata.get("platform")
            if plat is None:
                raise ValueError(
                    "diagnostic_grid.z_m=null requires synth file to carry /platform/"
                )
            z = float(np.asarray(plat["rotor_positions"])[2, 0])
        diag_grid, diag_shape = build_diagnostic_grid(
            self.cfg.diagnostic_grid.extent_xy_m,
            self.cfg.diagnostic_grid.increment_m,
            z,
        )

        # 3. CLEAN-SC
        source_map = self.algo.fit(
            csm=csm,
            frequencies=freqs,
            time_data=None,
            sample_rate=ctx.sample_rate,
            mic_positions=ctx.mic_positions,
            grid_positions=diag_grid,
            params={
                "damp": self.cfg.clean_sc.damp,
                "n_iter": self.cfg.clean_sc.n_iter,
            },
        )

        # 4. Rotor-disc mask
        plat = ctx.metadata["platform"]
        drone_mask = build_rotor_disc_mask(
            diag_grid,
            np.asarray(plat["rotor_positions"]),
            np.asarray(plat["rotor_radii"]),
            self.cfg.rotor_z_tolerance_m,
        )

        # 5. Drone CSM and residual
        drone_csm = reconstruct_csm(source_map.subset(drone_mask), ctx.mic_positions)
        residual_csm = csm - drone_csm

        # 6. Beam maps
        beam_maps = integrate_band_maps(source_map, self.cfg.bands, diag_shape)

        # 7. Pseudo-target PSD
        psd_pre = steer_to_psd(csm, freqs, ctx.mic_positions, self.cfg.target_point_m)
        psd_post = steer_to_psd(residual_csm, freqs, ctx.mic_positions, self.cfg.target_point_m)

        new_metadata = {
            **ctx.metadata,
            "array_filter": {
                "csm_pre": csm,
                "residual_csm": residual_csm,
                "frequencies": freqs,
                "source_map": source_map,
                "drone_mask": drone_mask,
                "beam_maps": beam_maps,
                "target_psd_pre": psd_pre,
                "target_psd_post": psd_post,
                "diagnostic_grid": diag_grid,
                "diagnostic_grid_shape": diag_shape,
            },
        }
        return replace(ctx, metadata=new_metadata)
```

- [ ] **Step 4: Register the stage builder**

Append to `src/martymicfly/processing/notch.py` *near the existing notch registration* (or in a new `src/martymicfly/processing/__init__.py` — choose whichever matches the established pattern). Recommended: keep registration in each stage's own module. Add a new `src/martymicfly/processing/array_filter.py` registration block at the bottom:

```python
from martymicfly.processing.pipeline import register_stage_builder
register_stage_builder("array_filter", lambda cfg: ArrayFilterStage(cfg))
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_array_filter.py -v
```

Expected: pass. If `target_psd_post.sum() > target_psd_pre.sum()` (subtraction increased target power), inspect: that means the drone-CSM reconstruction overshot. Reduce `n_iter` to 30 and `damp` to 0.4 for the tiny fixture; if still failing, investigate before relaxing assertions.

- [ ] **Step 6: Commit**

```bash
git add src/martymicfly/processing/array_filter.py tests/test_array_filter.py
git commit -m "feat(processing): ArrayFilterStage CLEAN-SC + rotor-disc subtraction"
```

---

## Task 21: `eval/array_metrics.py` — Stage-2 metrics

**Files:**
- Create: `src/martymicfly/eval/array_metrics.py`
- Test: `tests/test_array_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_array_metrics.py`:

```python
import numpy as np


def test_array_metrics_basic_shape_without_ground_truth():
    from martymicfly.eval.array_metrics import compute_array_metrics
    n_f = 8
    csm = np.tile(np.eye(3, dtype=np.complex128) * 4.0, (n_f, 1, 1))
    res = np.tile(np.eye(3, dtype=np.complex128) * 1.0, (n_f, 1, 1))
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 4.0)
    psd_post = np.full(n_f, 1.0)
    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}]
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 9)),
        drone_mask=np.array([True, True, True, False, False, False, False, False, False]),
        bands=bands, ground_truth=None,
    )
    assert "mid" in metrics["bands"]
    band = metrics["bands"]["mid"]
    assert band["csm_trace_reduction_db"] > 0
    assert band["target_psd_reduction_db"] > 0
    assert "drone_power_share_db" in band
    assert band["ground_truth"] is None


def test_array_metrics_with_ground_truth_recovery():
    from martymicfly.eval.array_metrics import compute_array_metrics
    n_f = 8
    freqs = np.linspace(200.0, 2000.0, n_f)
    psd_pre = np.full(n_f, 5.0)
    psd_post = np.full(n_f, 2.0)
    gt_psd = np.full(n_f, 2.0)
    bands = [{"name": "mid", "f_min_hz": 500.0, "f_max_hz": 1500.0}]
    csm = np.tile(np.eye(2, dtype=np.complex128) * 5.0, (n_f, 1, 1))
    res = np.tile(np.eye(2, dtype=np.complex128) * 2.0, (n_f, 1, 1))
    metrics = compute_array_metrics(
        csm_pre=csm, residual_csm=res, frequencies=freqs,
        psd_pre=psd_pre, psd_post=psd_post,
        source_map_powers=np.ones((n_f, 4)),
        drone_mask=np.array([True, False, False, False]),
        bands=bands,
        ground_truth={"psd_at_target": gt_psd, "frequencies": freqs},
    )
    band_gt = metrics["bands"]["mid"]["ground_truth"]
    assert abs(band_gt["external_recovery_db"]) < 0.5
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_array_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement metrics**

Create `src/martymicfly/eval/array_metrics.py`:

```python
"""Stage-2 metrics: CSM-trace reductions, drone power share, target PSD,
and (when ground-truth is supplied) external recovery / spectrum MAE."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _db(x: float, eps: float = 1e-30) -> float:
    return float(10.0 * np.log10(max(x, eps)))


def _band_mask(freqs, f_min, f_max):
    return (freqs >= f_min) & (freqs <= f_max)


def compute_array_metrics(
    *,
    csm_pre: np.ndarray,                  # (F, M, M)
    residual_csm: np.ndarray,             # (F, M, M)
    frequencies: np.ndarray,              # (F,)
    psd_pre: np.ndarray,                  # (F,)
    psd_post: np.ndarray,                 # (F,)
    source_map_powers: np.ndarray,        # (F, G)
    drone_mask: np.ndarray,               # (G,) bool
    bands: list,                          # list of dict or BandConfig
    ground_truth: Optional[dict] = None,  # {"psd_at_target": (F,), "frequencies": (F,)}
) -> dict:
    metrics: dict = {"bands": {}, "global": {}}

    trace_pre = np.real(np.diagonal(csm_pre, axis1=1, axis2=2)).sum(axis=1)         # (F,)
    trace_post = np.real(np.diagonal(residual_csm, axis1=1, axis2=2)).sum(axis=1)   # (F,)

    drone_power_per_freq = source_map_powers[:, drone_mask].sum(axis=1)
    total_power_per_freq = source_map_powers.sum(axis=1)

    for band in bands:
        if isinstance(band, dict):
            name, f_lo, f_hi = band["name"], band["f_min_hz"], band["f_max_hz"]
        else:
            name, f_lo, f_hi = band.name, band.f_min_hz, band.f_max_hz
        mask = _band_mask(frequencies, f_lo, f_hi)
        if not mask.any():
            continue
        tr_pre = float(trace_pre[mask].sum())
        tr_post = float(trace_post[mask].sum())
        psd_pre_band = float(psd_pre[mask].sum())
        psd_post_band = float(psd_post[mask].sum())
        share = float(drone_power_per_freq[mask].sum() /
                      max(total_power_per_freq[mask].sum(), 1e-30))

        gt_block = None
        if ground_truth is not None:
            gt_psd = np.asarray(ground_truth["psd_at_target"])
            gt_freq = np.asarray(ground_truth["frequencies"])
            gt_mask = (gt_freq >= f_lo) & (gt_freq <= f_hi)
            gt_band = float(gt_psd[gt_mask].sum())
            recovery_db = _db(psd_post_band) - _db(gt_band)
            spectrum_mae_db = float(np.mean(np.abs(
                10 * np.log10(np.maximum(psd_post[mask], 1e-30)) -
                10 * np.log10(np.maximum(gt_psd[gt_mask], 1e-30))
            )))
            gt_block = {
                "external_recovery_db": recovery_db,
                "spectrum_mae_db": spectrum_mae_db,
            }

        metrics["bands"][name] = {
            "csm_trace_pre_db": _db(tr_pre),
            "csm_trace_post_db": _db(tr_post),
            "csm_trace_reduction_db": _db(tr_pre) - _db(tr_post),
            "target_psd_pre_db": _db(psd_pre_band),
            "target_psd_post_db": _db(psd_post_band),
            "target_psd_reduction_db": _db(psd_pre_band) - _db(psd_post_band),
            "drone_power_share_db": _db(share),
            "ground_truth": gt_block,
        }

    metrics["global"] = {
        "drone_power_share_total_db": _db(
            float(drone_power_per_freq.sum() / max(total_power_per_freq.sum(), 1e-30))
        ),
    }
    return metrics
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_array_metrics.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/eval/array_metrics.py tests/test_array_metrics.py
git commit -m "feat(eval): array_metrics with optional ground-truth block"
```

---

## Task 22: `eval/array_plots.py` — beam-maps + target-PSD HTML

**Files:**
- Create: `src/martymicfly/eval/array_plots.py`
- Test: `tests/test_array_plots.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_array_plots.py`:

```python
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
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/test_array_plots.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create the plots module**

Create `src/martymicfly/eval/array_plots.py`:

```python
"""Stage-2 plotting: 2x3 beam-map subplots + 1x1 target-PSD plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _add_overlays(fig, row, col, extent, rotor_pos, rotor_radii, mic_pos, target_xy):
    theta = np.linspace(0, 2 * np.pi, 64)
    for i in range(rotor_pos.shape[1]):
        cx, cy, _ = rotor_pos[:, i]
        rr = rotor_radii[i]
        fig.add_trace(go.Scatter(
            x=cx + rr * np.cos(theta), y=cy + rr * np.sin(theta),
            mode="lines", line=dict(color="cyan", width=1, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=mic_pos[:, 0], y=mic_pos[:, 1],
        mode="markers", marker=dict(color="white", size=4, opacity=0.6),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=[target_xy[0]], y=[target_xy[1]],
        mode="markers", marker=dict(color="red", size=10, symbol="x"),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)
    fig.update_xaxes(range=[-extent, extent], row=row, col=col, title_text="X [m]")
    fig.update_yaxes(range=[-extent, extent], row=row, col=col,
                     scaleanchor=f"x{col}", scaleratio=1)


def plot_beam_maps(
    pre_maps: dict[str, np.ndarray],
    post_maps: dict[str, np.ndarray],
    extent_xy_m: float,
    rotor_positions: np.ndarray,
    rotor_radii: np.ndarray,
    mic_positions: np.ndarray,
    target_xy_m: tuple[float, float],
    out_path: str,
    db_dyn_range: float = 15.0,
) -> None:
    band_names = list(pre_maps.keys())
    titles = [f"pre — {n}" for n in band_names] + [f"post — {n}" for n in band_names]
    fig = make_subplots(rows=2, cols=len(band_names), subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.10)

    nx, ny = next(iter(pre_maps.values())).shape
    gx = np.linspace(-extent_xy_m, +extent_xy_m, nx)
    gy = np.linspace(-extent_xy_m, +extent_xy_m, ny)

    for col, name in enumerate(band_names, start=1):
        for row, source in enumerate((pre_maps, post_maps), start=1):
            m2d = np.maximum(source[name], 1e-30)
            peak = float(m2d.max())
            rel_db = 10.0 * np.log10(m2d / peak)
            fig.add_trace(go.Heatmap(
                x=gx, y=gy, z=rel_db.T,
                zmin=-db_dyn_range, zmax=0, colorscale="Inferno",
                showscale=(col == len(band_names) and row == 1),
                colorbar=dict(title="dB<br>(rel.<br>peak)", len=0.95, y=0.5, x=1.01),
            ), row=row, col=col)
            _add_overlays(fig, row, col, extent_xy_m, rotor_positions, rotor_radii,
                          mic_positions, target_xy_m)

    fig.update_layout(title="Stage-2 beam maps (CLEAN-SC, pre vs post)",
                      height=900, width=1500)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_target_psd(
    frequencies: np.ndarray,
    psd_pre: np.ndarray,
    psd_post: np.ndarray,
    gt_psd: np.ndarray | None,
    bpfs: list[float],
    out_path: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(psd_pre, 1e-30)),
                             mode="lines", name="pre"))
    fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(psd_post, 1e-30)),
                             mode="lines", name="post"))
    if gt_psd is not None:
        fig.add_trace(go.Scatter(x=frequencies, y=10 * np.log10(np.maximum(gt_psd, 1e-30)),
                                 mode="lines", name="ground truth", line=dict(dash="dot")))
    for f in bpfs:
        fig.add_vline(x=f, line=dict(color="gray", dash="dot", width=1))
    fig.update_xaxes(title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="PSD [dB rel. unit²/Hz]")
    fig.update_layout(title="Pseudo-target PSD (pre/post)", height=500, width=1000)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_array_plots.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/martymicfly/eval/array_plots.py tests/test_array_plots.py
git commit -m "feat(eval): array_plots beam_maps + target_psd"
```

---

## Task 23: Wire Stage-2 outputs into `run_pipeline.py`

**Files:**
- Modify: `src/martymicfly/cli/run_pipeline.py`
- Test: `tests/test_pipeline_e2e_array.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pipeline_e2e_array.py`:

```python
import json
from pathlib import Path


def test_pipeline_e2e_notch_plus_array_filter(tmp_path):
    from martymicfly.cli.run_pipeline import main
    fx = Path("tests/fixtures")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
input:
  audio_h5: {fx / 'tiny_synth_mixed.h5'}
  mic_geom_xml: {fx / 'tiny_geom_4mic.xml'}
  ground_truth_h5: {fx / 'tiny_gt.h5'}
segment: {{mode: head, duration: 1.0}}
channels: {{selection: all}}
rotor: {{n_blades: 2, n_harmonics: 4}}
stages:
  - kind: notch
    pole_radius: {{mode: scalar, value: 0.99}}
    multichannel: false
    block_size: 1024
  - kind: array_filter
    csm: {{nperseg: 256, noverlap: 128, f_min_hz: 200.0, f_max_hz: 4000.0}}
    diagnostic_grid: {{extent_xy_m: 0.6, increment_m: 0.05, z_m: 0.0}}
    bands:
      - {{name: mid, f_min_hz: 500.0, f_max_hz: 2000.0}}
    target_point_m: [0.5, 0.0, -0.5]
    rotor_z_tolerance_m: 0.05
    clean_sc: {{damp: 0.6, n_iter: 30}}
metrics: {{welch_nperseg: 1024, welch_noverlap: 512}}
plots: {{enabled: false, spectrogram_window: 512, spectrogram_overlap: 256}}
output: {{dir: {tmp_path / 'out'}}}
""")
    rc = main(["--config", str(cfg)])
    assert rc == 0
    runs = list((tmp_path / "out").glob("*"))
    assert len(runs) == 1
    run_dir = runs[0]
    for name in ("filtered.h5", "residual_csm.h5", "beam_maps.html",
                 "target_psd.html", "metrics.json", "metrics.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "stage2_array_filter" in metrics
    band = metrics["stage2_array_filter"]["bands"]["mid"]
    assert band["csm_trace_reduction_db"] > 0.0
    assert band["ground_truth"] is not None
```

- [ ] **Step 2: Run failing test**

```
uv run pytest tests/test_pipeline_e2e_array.py -v
```

Expected: failure on missing residual_csm.h5 (CLI doesn't write it yet).

- [ ] **Step 3: Modify `run_pipeline.py`**

After `run_pipeline(stages, ctx)` and before `config.yaml` snapshot, insert:

```python
    # Stage-2 outputs
    af = ctx.metadata.get("array_filter")
    if af is not None:
        from martymicfly.io.residual_csm_h5 import write_residual_csm
        from martymicfly.eval.array_metrics import compute_array_metrics
        from martymicfly.eval.array_plots import plot_beam_maps, plot_target_psd
        from martymicfly.io.ground_truth_h5 import load_ground_truth

        # Find the array_filter stage cfg to source extent / bands etc.
        stage_cfg = next(s for s in cfg.stages if s.kind == "array_filter")

        # residual_csm.h5
        write_residual_csm(
            str(out_dir / "residual_csm.h5"),
            af["residual_csm"], af["frequencies"],
            attrs={
                "algorithm": stage_cfg.algorithm,
                "stage": "array_filter",
                "config_hash": cfg.config_hash(),
            },
        )

        # post beam maps from masked source_map (band integration on subset)
        from martymicfly.processing.array_filter import integrate_band_maps
        sm = af["source_map"]
        post_sm = sm.subset(~af["drone_mask"])
        # We want post-maps over the same RectGrid layout. Reshape on the
        # full grid using zeros where drone_mask is True:
        post_powers_full = sm.powers * (~af["drone_mask"])[None, :]
        from martymicfly.processing.algorithms.base import SourceMap as _SM
        post_full_sm = _SM(
            positions=sm.positions, powers=post_powers_full,
            frequencies=sm.frequencies, grid_shape=af["diagnostic_grid_shape"],
            metadata={},
        )
        beam_maps_pre = integrate_band_maps(sm, stage_cfg.bands, af["diagnostic_grid_shape"])
        beam_maps_post = integrate_band_maps(post_full_sm, stage_cfg.bands, af["diagnostic_grid_shape"])

        plot_beam_maps(
            beam_maps_pre, beam_maps_post,
            extent_xy_m=stage_cfg.diagnostic_grid.extent_xy_m,
            rotor_positions=np.asarray(ctx.metadata["platform"]["rotor_positions"]),
            rotor_radii=np.asarray(ctx.metadata["platform"]["rotor_radii"]),
            mic_positions=ctx.mic_positions,
            target_xy_m=(stage_cfg.target_point_m[0], stage_cfg.target_point_m[1]),
            out_path=str(out_dir / "beam_maps.html"),
        )

        # Target PSD plot
        bpfs = []  # leave empty for now; could be derived from per_motor_bpf
        gt_block = None
        if cfg.input.ground_truth_h5:
            gt = load_ground_truth(cfg.input.ground_truth_h5)
            from scipy.signal import welch as _welch
            gt_f, gt_psd = _welch(
                gt.signal, fs=gt.sample_rate, nperseg=stage_cfg.csm.nperseg,
                noverlap=stage_cfg.csm.noverlap, scaling="density",
            )
            mask = (gt_f >= stage_cfg.csm.f_min_hz) & (gt_f <= stage_cfg.csm.f_max_hz)
            gt_psd = gt_psd[mask]
            gt_f = gt_f[mask]
            # Interpolate gt_psd onto af['frequencies'] grid
            gt_psd_interp = np.interp(af["frequencies"], gt_f, gt_psd)
            gt_block = {"psd_at_target": gt_psd_interp, "frequencies": af["frequencies"]}

        plot_target_psd(
            af["frequencies"], af["target_psd_pre"], af["target_psd_post"],
            gt_psd=(gt_block["psd_at_target"] if gt_block else None),
            bpfs=bpfs, out_path=str(out_dir / "target_psd.html"),
        )

        # Compose metrics into the metrics.json
        array_metrics = compute_array_metrics(
            csm_pre=af["csm_pre"], residual_csm=af["residual_csm"],
            frequencies=af["frequencies"],
            psd_pre=af["target_psd_pre"], psd_post=af["target_psd_post"],
            source_map_powers=sm.powers, drone_mask=af["drone_mask"],
            bands=[b.model_dump() for b in stage_cfg.bands],
            ground_truth=gt_block,
        )
        # Merge into metrics.json (created by notch block)
        mj = out_dir / "metrics.json"
        existing = json.loads(mj.read_text()) if mj.exists() else {}
        existing["stage2_array_filter"] = array_metrics
        mj.write_text(json.dumps(existing, indent=2))

        # Append rows to metrics.csv
        mc = out_dir / "metrics.csv"
        rows = []
        for name, band in array_metrics["bands"].items():
            row = {
                "stage": "array_filter",
                "band": name,
                "csm_trace_reduction_db": band["csm_trace_reduction_db"],
                "target_psd_reduction_db": band["target_psd_reduction_db"],
                "drone_power_share_db": band["drone_power_share_db"],
                "external_recovery_db": (band["ground_truth"] or {}).get("external_recovery_db", ""),
            }
            rows.append(row)
        with mc.open("a") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if mc.stat().st_size == 0:
                w.writeheader()
            for r in rows:
                w.writerow(r)
```

(Adjust the existing notch metrics.json/metrics.csv writers to leave `metrics.json` as a top-level dict with key `stage1_notch` rather than overwriting.)

- [ ] **Step 4: Adapt notch outputs to nest under `stage1_notch`**

In the notch-output block of `run_pipeline.py`, replace the old `json.dump(metrics, f, ...)` with:

```python
        existing = json.loads(mj.read_text()) if mj.exists() else {}
        existing["stage1_notch"] = metrics
        mj.write_text(json.dumps(existing, indent=2))
```

Wrap the existing notch CSV writer with a column for `stage` so the array-filter rows can append cleanly. Or write notch CSV to `stage1_metrics.csv` and array-filter CSV to `stage2_metrics.csv` if column union is messy. **Decision:** separate files — `stage1_metrics.csv` (notch) and `stage2_metrics.csv` (array_filter); `metrics.csv` becomes a deprecated symlink to `stage1_metrics.csv` or is dropped (drop it; no downstream consumer).

Update `tests/test_pipeline_integration.py` and `tests/test_pipeline_e2e_array.py` to look for `stage1_metrics.csv`/`stage2_metrics.csv` instead of `metrics.csv` if they reference it. Update `tests/test_pipeline_e2e_array.py` Step-1 assertion list accordingly.

- [ ] **Step 5: Run all tests**

```
uv run pytest -x -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/martymicfly/cli/run_pipeline.py tests/test_pipeline_e2e_array.py \
        tests/test_pipeline_integration.py
git commit -m "feat(cli): wire ArrayFilterStage outputs into run_pipeline"
```

---

## Task 24: Update `configs/example_pipeline.yaml`

**Files:**
- Create: `configs/example_pipeline.yaml`

- [ ] **Step 1: Create the file**

Create `configs/example_pipeline.yaml`:

```yaml
input:
  audio_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5
  mic_geom_xml: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip_gt.h5

segment:
  mode: middle
  duration: 10.0

channels:
  selection: all

rotor:
  n_blades: 2
  n_harmonics: 20

stages:
  - kind: notch
    pole_radius:
      mode: scalar
      value: 0.9994
    multichannel: false
    block_size: 4096

  - kind: array_filter
    algorithm: clean_sc
    csm:
      nperseg: 512
      noverlap: 256
      window: hann
      diag_loading_rel: 1.0e-6
      f_min_hz: 200.0
      f_max_hz: 6000.0
    diagnostic_grid:
      extent_xy_m: 0.5
      increment_m: 0.02
      z_m: null
    bands:
      - { name: low,  f_min_hz: 200.0,  f_max_hz: 500.0 }
      - { name: mid,  f_min_hz: 500.0,  f_max_hz: 2000.0 }
      - { name: high, f_min_hz: 2000.0, f_max_hz: 6000.0 }
    target_point_m: [0.0, 0.0, -1.5]
    rotor_z_tolerance_m: 0.05
    clean_sc:
      damp: 0.6
      n_iter: 100

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
  dir: results/pipeline/{run_id}
  filtered_h5: filtered.h5
  metrics_json: metrics.json
  plots_subdir: plots
  copy_config: true
```

- [ ] **Step 2: Commit**

```bash
git add configs/example_pipeline.yaml
git commit -m "feat(configs): example_pipeline.yaml with notch + array_filter"
```

---

## Task 25: Smoke-run on the production data path

This is a manual checklist, not a test. Records what success looks like on the real synth file once Stage 2 is end-to-end runnable.

- [ ] **Step 1: Generate the production synth-mix**

```
uv run python -m martymicfly.synth.cli.compose --config configs/example_compose.yaml
```

Expected: writes `ap2a_synth_mixed_gaptip.h5` and `ap2a_synth_mixed_gaptip_gt.h5` under the configured output directory.

- [ ] **Step 2: Run the full pipeline**

```
uv run python -m martymicfly.cli.run_pipeline --config configs/example_pipeline.yaml --log-level INFO
```

Expected: completes in < 10 min wallclock; produces `results/pipeline/<run_id>/` with all artefacts.

- [ ] **Step 3: Sanity-check `metrics.json`**

```
jq '.stage2_array_filter.bands.mid' results/pipeline/*/metrics.json | tail -5
```

Expected (smoke targets — see spec §3.3):
- `csm_trace_reduction_db` > 5
- `target_psd_reduction_db` > 0
- `ground_truth.external_recovery_db` within ±3 dB

- [ ] **Step 4: Inspect the beam-maps HTML**

Open `results/pipeline/<run_id>/beam_maps.html` in a browser. Expected:
- Pre-row: 4 strong peaks at the rotor positions (xy ≈ ±0.23 m).
- Post-row: peaks within rotor discs are gone; mid-band shows a peak in the negative-z direction (the external source projection).

- [ ] **Step 5: Document the run**

Append a one-liner to `docs/superpowers/handoff-stage3.md` (create the file if missing) recording:
- Date and `run_id`
- `external_recovery_db` per band
- Wall-clock time
- Any deviations from the smoke targets

This is the prompt material for Stage 3 brainstorming. No commit needed unless you want the run record version-controlled — at your discretion.

---

## Self-Review Outcome

- **Spec coverage:** every section of the spec maps to one or more tasks: §1 → tasks 19/24; §2 → 5/6/7/9/11; §3 → 4/23; §4 → entire plan; §5.1 → 14; §5.2 → 15; §5.3 → 12; §5.4 → 13; §5.5 → 18/20; §5.6/5.7 → 7/8/9/10/11; §5.8 → 21; §5.9 → 22; §5.10 → 5; §5.11 → 6; §5.12 → 16; §5.13 → 3; §6 → 4/19/24; §7 → 23; §8 → 11/(all test tasks); §9 → encoded as risks acknowledged in tasks 7/15; §10 (roadmap) → out-of-plan.
- **Placeholder scan:** no `TBD`/`TODO`/`fill in`/etc. anywhere in this plan.
- **Type consistency:** `SourceMap.subset` returns a `SourceMap`; `BandConfig` is shared between config (pydantic) and processing (dataclass-style) — Task 18 uses a dataclass `BandConfig` shadowed in Task 19 by the pydantic version. **Resolved**: in Task 18 the dataclass `BandConfig` was a placeholder for the test; the real `BandConfig` is the pydantic model in `config.py` (Task 19), and `array_filter.py` re-exports it at the top via `from martymicfly.config import BandConfig`. Update `array_filter.py` to remove the local dataclass and import from config.

**Action required during execution:** when implementing Task 18, *do* define the local dataclass `BandConfig` so the test passes; in Task 19 the dataclass gets removed in favor of the pydantic model and `array_filter.py` switches its import. The test from Task 18 still passes because `BandConfig` exposes `name`, `f_min_hz`, `f_max_hz` in both forms.
