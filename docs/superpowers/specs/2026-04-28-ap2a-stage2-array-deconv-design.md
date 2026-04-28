# AP2-A Stufe 2 — Array-Entfaltung (Design)

**Datum:** 2026-04-28
**Status:** Entwurf zur User-Review
**Vorhaben:** MartyMicFly / `marty-mic-fly`
**Vorgänger-Spec:** `docs/superpowers/specs/2026-04-28-ap2a-stage1-notch-design.md`

## 1. Ziel und Scope

### 1.1 Antragskontext

AP2-A im Antrag *FliegendesMessMikrofon* beschreibt eine dreistufige Pipeline
zur Trennung des Drohnen-Eigengeräuschs vom zu messenden Schall. Stufe 1
(tonale Notch-Filterung) ist abgeschlossen. **Stufe 2 — Räumliche Filterung
der breitbandigen Anteile** durch Arrayverfahren (Antragszeilen 506–524):
Entfaltung statt klassisches Beamforming, weil die Drohnen-Eigengeräusch-
Quellrichtungen relativ zum Mic-Array bekannt sind. Vorgesehene Verfahren:
Orthogonale Entfaltung, CLEAN-SC, CLEAN-T, optional CSM-Fitting.

**Diese Spec deckt CLEAN-SC ab**, mit einer Architektur, die die übrigen
Verfahren als reine Drop-Ins in einer Folge-Spec erwartet (Roadmap §10).

### 1.2 In-Scope

- **Neue Stage** `ArrayFilterStage` mit CLEAN-SC als einzigem Algorithmus,
  hinter einem `Algorithm`-Protocol als Erweiterungspunkt für Orth/CLEAN-T/CMF.
- **Synthese-Helper** im Repo, der das Drone-Source-Artifact
  (`drone_source_artifact_gaptip.h5`) mit einer konfigurierbaren externen
  Quelle propagiert und (a) ein Acoular-Format-Synth-File und (b) eine
  Ground-Truth-Datei für das reine externe Signal schreibt.
- **CLI-Refactor**: `cli/run_notch.py` → `cli/run_pipeline.py` mit
  `stages: [...]`-Liste in der YAML; bestehende Stage 1 wird unverändert ein
  Listen-Element. Alter CLI-Name bleibt als Shim erhalten.
- **Bewertung** mit zwei Stufen: Pflicht-Metriken ohne Ground-Truth (CSM-Spuren,
  Beam-Map-Diagnostik, Pseudo-Zielpunkt-Spektrum) plus optionale Ground-
  Truth-Metriken (`external_recovery_db`, `spectrum_mae_db`,
  `false_attribution_db`), aktiviert sobald die Synthese-Helper-Datei und ihre
  Ground-Truth-Datei vorliegen.
- **Outputs** pro Lauf: `residual_csm.h5` (per-Frequenzbin Rest-CSM) +
  `beam_maps.html` (2×3 Subplots: pre/post × low/mid/high) + `target_psd.html`
  (Welch-PSD am Zielpunkt) + `metrics.json` + `metrics.csv`. `filtered.h5` wird
  unverändert von Stufe 1 weitergereicht (Stage 2 fasst `time_data` nicht an).
- **Tests** auf einer Tiny-Fixture (4 Mics, 2 Rotoren × 2 Subsources,
  1 s @ 16 kHz), end-to-end durch Notch + ArrayFilter.

### 1.3 Out-of-Scope

- **Algorithmen Orth, CLEAN-T, CSM-Fitting** → Folge-Spec (Roadmap §10.1).
  Architektur ist explizit darauf ausgelegt, sie als Drop-Ins anzudocken.
- **Stufe 3** (Spektrum-Rekonstruktion auf der Rest-CSM) → eigene Spec.
- **Migration des Synthese-Helpers nach `drone_synthdata`** (q) → eigene
  Aufgabe nach Validierung. Der In-Repo-Helper ist als temporär markiert.
- **Real-Mess-Loader, Tan/Jiang-Frequenzschätzung, Multi-Quellen-Sweeps** →
  spätere Specs.
- **Spectrogramm am Pseudo-Zielpunkt** → nice-to-have, nur falls trivial
  während Implementierung dazustellbar; sonst Folge-Ticket.
- **Akustische Kalibrierung auf Pa**: Output-dB sind relativ zu 1 (Einheit²)
  wie in Stufe 1.

### 1.4 Output-Vertrag (Hybrid D)

Stufe 2 liefert **kein neues `time_data`**. Stattdessen wandert der eigentliche
Output in `PipelineContext.metadata["array_filter"]`:

- `csm_pre`, `residual_csm`: hermitesche Matrizen `(F, M, M)` complex128.
- `frequencies`: `(F,)` Hz.
- `source_map`: vollständige CLEAN-SC-Lokalisierung über das Diagnose-Grid.
- `drone_mask`: bool-Vektor `(G,)` — welche Grid-Punkte zählen als Drohne.
- `beam_maps`: dict `{band_name → (nx, ny) array}` für pre und post.
- `target_psd_pre`, `target_psd_post`: per-Frequenz-PSD am Pseudo-Zielpunkt.

`time_data` bleibt unverändert (post-Notch). Stufe 3 wird später aus
`residual_csm` schöpfen.

## 2. Quellgeometrie und Eingangsdaten

### 2.1 Drone-Source-Artifact

Pfad: `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5`

Das Artifact ist ein **`gap_tip`-Subsource-Modell** mit 8 Punktquellen
(2 pro Rotor, `tip_radius = 0.1905 m`):

```
/                              attrs: sample_rate=52000, reconstruction_mode='stationary'
  platform/                    attrs: n_rotors=4
    blade_counts        (4,) int32
    rotor_positions     (3, 4) float64   xyz columns; alle z=0
    rotor_radii         (4,)  float64   alle 0.1905 m
  reconstruction/              attrs: nperseg=4096, noverlap=2048, window='hann',
                                       reg_lambda=0.0, reg_type='none',
                                       subsource_model='{"kind":"gap_tip", ...}'
  rotor_index           (8,)  int32     mappt jede Subsource → Rotor-Index 0..3
  subsource_positions   (8, 3) float64  xyz rows; alle z=0
  subsource_signals     (8, N) float64  Zeitreihen, N≈4.6 M @ 52 kHz
  rpm_per_esc/{ESC1..ESC4}/{rpm,timestamp}
  source_measurement/          attrs: duration, file_path, n_channels
```

Dieses Artifact ist **kein Mess-File** im Sinne der Stage-1-Eingabe. Es ist
ein Source-Modell, aus dem ein Mic-Array-Signal *propagiert* werden muss
(Synthese-Helper, §6).

### 2.2 Synthese-Mix-Datei (Pflicht-Eingabe für Stufe 2)

Output des Synthese-Helpers, im Acoular-Format wie Stage 1:

```
/                              attrs: sample_rate=52000, ...
  time_data            (N, 16) float64    propagierte Drohne + ext. Quelle
  esc_telemetry/{ESC1..ESC4}/{rpm,timestamp}
  platform/                    attrs: n_rotors=4
    blade_counts, rotor_positions, rotor_radii   (Pass-through aus Artifact)
  external/                    attrs: kind, amplitude_db, position_m, seed,
                                       speed_of_sound, propagation='greens_1_over_r'
```

Die `/platform/`-Gruppe macht die Quellgeometrie selbst-beschreibend: keine
Code-Default-Annahme nötig.

### 2.3 Ground-Truth-Datei (optional)

Begleit-Datei zum Synthese-Mix, wird vom Synthese-Helper als zweites File
geschrieben:

```
/                              attrs: sample_rate=52000, kind='external_only'
  time_data            (N, 1) float64     reines externes Quellsignal
                                          AM Quellort (vor Propagation)
  external/                    attrs: kind, amplitude_db, position_m, seed
```

Wenn beim Stage-2-Lauf eine Ground-Truth-Datei in der Konfig steht, werden
die zusätzlichen Metriken (IV) berechnet.

## 3. Daten- und Pipeline-Fluss

```
       compose_external (offline, eigenes CLI)
              │
   drone_source_artifact ──┐         external_source_spec
                           ├──→ synth_mix.h5  +  ground_truth.h5
   mic_geom.xml ───────────┘
              │
              v
   ┌────────────── run_pipeline ──────────────┐
   │  load synth_mix.h5 + mic_geom.xml        │
   │  segment + frequencies                   │
   │  ↓                                       │
   │  NotchStage (Stage 1, unverändert)       │  time_data: post-notch
   │  ↓                                       │
   │  ArrayFilterStage                        │  time_data: passthrough
   │   1. build CSM                           │  metadata.array_filter:
   │   2. CLEAN-SC auf Diagnose-Grid          │    csm_pre,
   │   3. drone_mask = rotor-disc-mask        │    residual_csm,
   │   4. drone_csm = reconstruct(masked)     │    source_map,
   │   5. residual = csm − drone_csm          │    drone_mask,
   │   6. beam_maps pro Band                  │    beam_maps,
   │   7. target_psd via steering             │    target_psd_pre/post
   └──────────┬────────────────────────────────┘
              v
   eval.array_metrics + eval.array_plots
              │
              v
   write_outputs:
     results/<run_id>/
       filtered.h5          (post-notch time_data, von Stufe 1)
       residual_csm.h5
       beam_maps.html
       target_psd.html
       metrics.json
       metrics.csv
       config.yaml + config.hash + run.log
```

## 4. Repo-Layout

```
src/martymicfly/
  config.py                 [Δ]  PipelineConfig(stages: list),
                                 ArrayFilterConfig, CsmConfig, BandConfig,
                                 CleanScConfig, GroundTruthConfig
  constants.py              [+]  SPEED_OF_SOUND = 343.0
  io/
    synth_h5.py             [Δ]  liest /platform/-Gruppe falls vorhanden
    mic_geom.py             [=]
    write_filtered.py       [Δ]  generalisiert zu write_outputs(ctx, run_meta)
    source_artifact.py      [+]  load_source_artifact(path) → SourceArtifact
    residual_csm_h5.py      [+]  write_residual_csm(path, residual, freqs)
    ground_truth_h5.py      [+]  load_ground_truth(path) / write_ground_truth(path,...)
  processing/
    pipeline.py             [Δ]  Stage-Registry, run_pipeline ungeändert
    frequencies.py          [=]
    sources.py              [=]
    notch.py                [=]
    csm.py                  [+]  build_measurement_csm(time, fs, cfg)
    beamform_grid.py        [+]  build_diagnostic_grid, build_rotor_disc_mask
    array_filter.py         [+]  ArrayFilterStage(algorithm: Algorithm, …)
    algorithms/
      __init__.py           [+]  REGISTRY = {"clean_sc": CleanScAlgorithm}
      base.py               [+]  Algorithm Protocol, SourceMap dataclass,
                                 reconstruct_csm (default impl)
      clean_sc.py           [+]  CleanScAlgorithm
  synth/
    __init__.py             [+]
    compose_external.py     [+]  compose(artifact, mic_geom, ext_spec)
    propagation.py          [+]  greens_propagate (Sinc fractional shift + 1/r)
    cli/
      __init__.py           [+]
      compose.py            [+]  CLI für den Synthese-Helper
  eval/
    metrics.py              [=]  bleibt für Notch
    plots.py                [=]  bleibt für Notch
    array_metrics.py        [+]  I/II/III/IV Metriken
    array_plots.py          [+]  beam_maps.html, target_psd.html
  cli/
    run_notch.py            [Δ]  thin shim: ruft run_pipeline mit Notch-only
    run_pipeline.py         [+]  generischer Stage-Treiber
configs/
  example_notch.yaml        [Δ]  migriert auf stages-Format (Notch-only)
  example_pipeline.yaml     [+]  Notch + array_filter
  example_compose.yaml      [+]  Synthese-Helper-Default
tests/
  fixtures/
    make_tiny_drone_artifact.py [+]
    tiny_drone_artifact.h5      [+]  4 Mics, 2 Rotoren × 2 Subsources, 1 s @ 16 kHz
    make_tiny_compose.py        [+]
    tiny_synth_mixed.h5         [+]
    tiny_gt.h5                  [+]
    tiny_geom_4mic.xml          [+]
  test_csm.py                   [+]
  test_beamform_grid.py         [+]
  test_clean_sc_algorithm.py    [+]
  test_reconstruct_csm.py       [+]
  test_array_filter.py          [+]
  test_array_metrics.py         [+]
  test_compose_external.py      [+]
  test_pipeline_e2e_array.py    [+]
  test_algorithm_registry.py    [+]
```

`[+]` = neu, `[Δ]` = modifiziert, `[=]` = unverändert gegenüber Stage 1.

## 5. Komponenten-Verträge

### 5.1 `processing/algorithms/base.py`

```python
from typing import Literal, Protocol
from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class SourceMap:
    """Output einer Entfaltung: Power pro Grid-Punkt pro Frequenzbin."""
    positions: np.ndarray           # (G, 3) Grid-Punkte
    powers: np.ndarray              # (F, G) p^2 in (Einheit)²
    frequencies: np.ndarray         # (F,) Hz
    grid_shape: tuple[int, int] | None  # (nx, ny) für RectGrid; None für ImportGrid
    metadata: dict                  # algorithmenspezifisch (n_iter, damp, …)

    def subset(self, mask: np.ndarray) -> "SourceMap":
        """Behalt nur Grid-Punkte, an denen mask True ist. grid_shape → None."""
        return replace(
            self,
            positions=self.positions[mask],
            powers=self.powers[:, mask],
            grid_shape=None,
        )


class Algorithm(Protocol):
    name: str
    consumes: Literal["csm", "time"]   # für jetzt: alle "csm"; CLEAN-T später "time"

    def fit(
        self,
        *,
        csm: np.ndarray | None,
        frequencies: np.ndarray | None,
        time_data: np.ndarray | None,
        sample_rate: float,
        mic_positions: np.ndarray,    # (M, 3)
        grid_positions: np.ndarray,   # (G, 3)
        params: dict,
    ) -> SourceMap: ...


def reconstruct_csm(
    source_map: SourceMap,
    mic_positions: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:                       # (F, M, M)
    """Default-Rekonstruktion via Steering-Vektor-Summe.

    drone_csm[f] = Σ_g power[f,g] · h[f,g] h[f,g]^H
    h[f,g] = exp(-j·2π·f·r_mg/c) / (4π·r_mg)

    Algorithmen können diese Funktion via gleichnamige Methode überschreiben.
    """
```

### 5.2 `processing/algorithms/clean_sc.py`

```python
class CleanScAlgorithm:
    name = "clean_sc"
    consumes = "csm"

    def fit(self, *, csm, frequencies, mic_positions, grid_positions,
            params, **_) -> SourceMap:
        from acoular import (
            BeamformerCleansc, ImportGrid, MicGeom,
            PowerSpectraImport, SteeringVector, config as acoular_config,
        )
        acoular_config.global_caching = "none"

        mg = MicGeom(pos_total=mic_positions.T)              # (3, M)
        grid = _import_grid_from_array(grid_positions)       # (3, G) → Acoular Grid
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
            positions=grid_positions,
            powers=powers,
            frequencies=np.asarray(frequencies, dtype=float),
            grid_shape=None,
            metadata={"damp": params["damp"], "n_iter": params["n_iter"]},
        )
```

`_import_grid_from_array(grid_positions)` ist ein dünner Wrapper, der eine
Acoular-`Grid`-Instanz mit den gegebenen `(G, 3)`-Positionen liefert. Genaue
Implementierung (Acoular's `ImportGrid` mit Tempfile, oder eine kleine
`Grid`-Subklasse, die `gpos` direkt setzt) wird in der Plan-Phase entschieden,
nachdem ein Smoke-Test mit Acoular klärt, welche Variante stabil und
cache-frei läuft. Vertrag des Wrappers: Eingang `(G, 3)`, Ausgang Acoular-
`Grid` mit `gpos.shape == (3, G)`.

### 5.3 `processing/csm.py`

```python
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
    """Welch-style CSM via scipy.signal.csd über alle Mic-Paare.

    Δf = sample_rate / cfg.nperseg.
    Diagonal loading: cfg.diag_loading_rel * max(|diag|) auf Identität addiert.
    Frequenz-Maske: nur Bins in [f_min_hz, f_max_hz] werden zurückgegeben.

    Returns:
        csm:   (F, M, M) complex128, hermitesch
        freqs: (F,) float
    """
```

### 5.4 `processing/beamform_grid.py`

```python
def build_diagnostic_grid(
    extent_xy_m: float,
    increment_m: float,
    z_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rectangular grid auf der Rotor-Ebene.

    nx = ny = round(2*extent_xy_m / increment_m) + 1
    Returns (G, 3) and (nx, ny).
    """


def build_rotor_disc_mask(
    grid_positions: np.ndarray,    # (G, 3)
    rotor_positions: np.ndarray,   # (3, R)
    rotor_radii: np.ndarray,       # (R,)
    z_tol_m: float = 0.05,
) -> np.ndarray:                   # (G,) bool
    """True wherever a grid point lies within any rotor disc."""
```

### 5.5 `processing/array_filter.py`

```python
@dataclass
class DiagnosticGridConfig:
    extent_xy_m: float = 0.5
    increment_m: float = 0.02
    z_m: float | None = None       # None → aus rotor_positions ziehen


@dataclass
class BandConfig:
    name: str
    f_min_hz: float
    f_max_hz: float


@dataclass
class CleanScConfig:
    damp: float = 0.6
    n_iter: int = 100


@dataclass
class ArrayFilterConfig:
    algorithm: str = "clean_sc"
    csm: CsmConfig = field(default_factory=CsmConfig)
    diagnostic_grid: DiagnosticGridConfig = field(default_factory=DiagnosticGridConfig)
    bands: list[BandConfig] = field(default_factory=lambda: [
        BandConfig("low",  200.0,  500.0),
        BandConfig("mid",  500.0, 2000.0),
        BandConfig("high",2000.0, 6000.0),
    ])
    target_point_m: tuple[float, float, float] = (0.0, 0.0, -1.5)
    rotor_z_tolerance_m: float = 0.05
    clean_sc: CleanScConfig = field(default_factory=CleanScConfig)


class ArrayFilterStage:
    name = "array_filter"

    def __init__(self, config: ArrayFilterConfig,
                 registry: dict[str, type[Algorithm]]):
        self.cfg = config
        self.algo: Algorithm = registry[config.algorithm]()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        # 1. CSM
        csm, freqs = build_measurement_csm(ctx.time_data, ctx.sample_rate, self.cfg.csm)

        # 2. Diagnose-Grid auf Rotor-Ebene
        z = self.cfg.diagnostic_grid.z_m
        if z is None:
            z = float(ctx.metadata["platform"]["rotor_positions"][2, 0])
        diag_grid, diag_shape = build_diagnostic_grid(
            self.cfg.diagnostic_grid.extent_xy_m,
            self.cfg.diagnostic_grid.increment_m,
            z,
        )

        # 3. EIN CLEAN-SC-Lauf
        source_map = self.algo.fit(
            csm=csm, frequencies=freqs,
            time_data=None, sample_rate=ctx.sample_rate,
            mic_positions=ctx.mic_positions,
            grid_positions=diag_grid,
            params=asdict(self.cfg.clean_sc),
        )

        # 4. Räumliche Maske
        rotor_pos = ctx.metadata["platform"]["rotor_positions"]
        rotor_radii = ctx.metadata["platform"]["rotor_radii"]
        drone_mask = build_rotor_disc_mask(
            diag_grid, rotor_pos, rotor_radii, self.cfg.rotor_z_tolerance_m,
        )

        # 5. Drohnen-CSM und Subtraktion
        drone_csm = reconstruct_csm(source_map.subset(drone_mask), ctx.mic_positions)
        residual_csm = csm - drone_csm

        # 6. Beam-Maps pro Band
        beam_maps = integrate_band_maps(source_map, self.cfg.bands, diag_shape)

        # 7. Pseudo-Zielpunkt-PSD
        target_psd_pre  = steer_to_psd(csm,          freqs, ctx.mic_positions, self.cfg.target_point_m)
        target_psd_post = steer_to_psd(residual_csm, freqs, ctx.mic_positions, self.cfg.target_point_m)

        new_metadata = {
            **ctx.metadata,
            "array_filter": {
                "csm_pre": csm,
                "residual_csm": residual_csm,
                "frequencies": freqs,
                "source_map": source_map,
                "drone_mask": drone_mask,
                "beam_maps": beam_maps,
                "target_psd_pre": target_psd_pre,
                "target_psd_post": target_psd_post,
                "diagnostic_grid_shape": diag_shape,
            },
        }
        return replace(ctx, metadata=new_metadata)
```

`time_data` wird ausdrücklich **nicht** angefasst.

### 5.6 `synth/compose_external.py`

```python
@dataclass
class ExternalSourceSpec:
    kind: Literal["noise", "sweep", "sine", "file"]
    position_m: tuple[float, float, float] = (0.0, 0.0, -1.5)
    amplitude_db: float = 0.0      # rel. zu mean(|drone subsources|)
    duration_s: float | None = None  # None = match artifact
    sine_freq_hz: float | None = None
    sweep_f_lo_hz: float | None = None
    sweep_f_hi_hz: float | None = None
    file_path: str | None = None
    seed: int | None = 0


def compose_external(
    artifact_path: str,
    mic_geom_path: str,
    ext_spec: ExternalSourceSpec,
    out_synth_path: str,
    out_gt_path: str,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> None:
    """Read drone source artifact, propagate 8 subsources to mic_geom mics
    via Green's function (1/(4π·r) attenuation + Sinc-interpolated fractional
    delay = r/c samples at fs), generate external source signal at
    ext_spec.position_m, propagate, sum, write Acoular-format synth HDF5
    (incl. /platform/ pass-through) + 1-channel ground-truth HDF5."""
```

CLI: `python -m martymicfly.synth.cli.compose --config configs/example_compose.yaml`.

`ExternalSourceSpec` wird via pydantic mit `model_validator` validiert
(Kind-abhängige Pflichtfelder, analog `PoleRadiusConfig` aus Stage 1).

### 5.7 `synth/propagation.py`

```python
def greens_propagate(
    source_signals: np.ndarray,    # (S, N)
    source_positions: np.ndarray,  # (S, 3)
    mic_positions: np.ndarray,     # (M, 3)
    sample_rate: float,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:                   # (N, M)
    """Greenscher freier-Raum-Propagator: für jedes Source-Mic-Paar wird
    s_at_mic[m, n] = (1 / (4π·r_sm)) · sinc_shift(s, r_sm/c · fs)
    berechnet, dann über s aufsummiert.

    Sinc-shift via FFT-Phase-Rotation (analytisch korrekt, keine Truncation)
    oder kurze FIR-Sinc-Tabelle mit Hann-Fenster (~31 taps); Implementation
    wählt FFT-Phase wenn N nicht zu groß.
    """
```

### 5.8 `eval/array_metrics.py`

**Pflicht-Metriken** pro Band:

- `csm_trace_pre_db`: `10·log10(Σ_f Tr(csm_pre[f]))` über die Bandbreite.
- `csm_trace_post_db`: dito für `residual_csm`.
- `csm_trace_reduction_db`: pre − post.
- `drone_power_share_db`: `10·log10(Σ_f Σ_g_in_mask power[f,g] / Σ_f Σ_g power[f,g])`.
- `target_psd_pre_db`, `target_psd_post_db`, `target_psd_reduction_db`.
- `beam_map_peak_pre_xy`, `beam_map_peak_post_xy`: Peak-Position pro Band-Map.
- `beam_map_peak_in_rotor_disc_post`: bool — liegt Post-Peak innerhalb einer Rotor-Disc?

**Optional (Ground-Truth-Metriken)** pro Band, nur wenn `ground_truth_h5`
konfiguriert ist:

- `external_recovery_db`: `target_psd_post_db − gt_psd_at_target_db`
  (idealerweise nahe 0 dB).
- `spectrum_mae_db`: mean-absolute-error zwischen `target_psd_post(f)` und
  `gt_psd(f)` über die Bandbreite, in dB-Domäne.
- `false_attribution_db`: `10·log10(power_in_drone_mask_from_pure_external_csm
  / total_pure_external_csm_power)`. Mit reiner externer CSM ist das die
  Fehl-Zuordnung von externen Anteilen zu „Drohne".

`metrics.json` enthält alle Metriken vollständig pro Band; `metrics.csv` eine
Zeile pro Band mit den Hauptkennzahlen für schnelle Tabellen.

### 5.9 `eval/array_plots.py`

- `plot_beam_maps(...)`: Plotly 2×3-Subplot, Reihen = pre/post, Spalten =
  Bänder. Heatmap-Skala dB rel. Panel-Peak (analog `clean_sc_source_localization.py`).
  Overlay: Rotor-Discs (cyan dashed), Mic-Positionen (weiße Punkte),
  Pseudo-Zielpunkt-Projektion (rotes ×). `dB_dyn_range = 15`.
- `plot_target_psd(...)`: 1×1 Welch-PSD am Zielpunkt, pre/post, plus
  optional Ground-Truth-Linie. Gestrichelte vertikale Linien an
  `mean_BPF · h` für kleines `h` zur Orientierung.

### 5.10 `io/source_artifact.py`

```python
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


def load_source_artifact(path: str) -> SourceArtifact: ...
```

### 5.11 `io/synth_h5.py` Erweiterung

`load_synth_h5(path) -> dict` zusätzlich zu Stage-1-Feldern:

```python
{
    ...stage1 felder...,
    "platform": None | {
        "n_rotors": int,
        "blade_counts": np.ndarray,   # (R,)
        "rotor_positions": np.ndarray, # (3, R)
        "rotor_radii": np.ndarray,    # (R,)
    },
}
```

`platform: None` wenn das HDF5 die Gruppe nicht enthält (Backward-Compat
mit dem alten `ap2a_synthesis_16mic_gaptip.h5`). `ArrayFilterStage` hebt
in dem Fall einen Konfig-Fehler („run pipeline against a synth_mix file
produced by `compose_external`").

### 5.12 `io/residual_csm_h5.py`

```python
def write_residual_csm(path: str, residual: np.ndarray, freqs: np.ndarray,
                       attrs: dict) -> None: ...
```

Layout:

```
/                  attrs: martymicfly_version, config_hash, csm_nperseg,
                          algorithm='clean_sc', stage='array_filter'
  csm_real    (F, M, M) float64
  csm_imag    (F, M, M) float64
  frequencies (F,)      float64
```

(Real/Imag separat statt complex128, weil HDF5-Tools wie `h5dump` mit
Complex-Typen oft hadern.)

### 5.13 `processing/pipeline.py` Erweiterung

```python
STAGE_REGISTRY: dict[str, Callable[[Any], Stage]] = {
    "notch": lambda cfg: NotchStage(cfg),
    "array_filter": lambda cfg: ArrayFilterStage(cfg, ALGORITHM_REGISTRY),
}

def build_stages(stage_configs: list[StageConfig]) -> list[Stage]:
    return [STAGE_REGISTRY[s.kind](s.config) for s in stage_configs]
```

`run_pipeline` selbst bleibt unverändert.

## 6. Konfiguration

### 6.1 `configs/example_pipeline.yaml`

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
      z_m: null            # null → aus /platform/rotor_positions
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
```

### 6.2 `configs/example_compose.yaml`

```yaml
input:
  drone_source_artifact_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/drone_source_artifact_gaptip.h5
  mic_geom_xml: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml

external:
  kind: noise                # noise | sweep | sine | file
  position_m: [0.0, 0.0, -1.5]
  amplitude_db: 0.0
  duration_s: null           # null → match artifact
  seed: 0

output:
  synth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5
  ground_truth_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip_gt.h5
```

### 6.3 `configs/example_notch.yaml` Migration

Das alte Top-Level-`notch:`-Format wird scharf gecuttet. `example_notch.yaml`
wird auf das neue `stages:`-Format umgeschrieben:

```yaml
stages:
  - kind: notch
    ...
```

Die `pydantic`-Validierung lehnt den alten Top-Level-`notch:`-Block ab
(`extra="forbid"`). Repo ist jung — nur ein Beispiel-File und Stage-1-Tests
sind betroffen, beide werden im selben Implementierungs-Schritt umgestellt.

`cli/run_notch.py` bleibt **als Datei** erhalten (Muskel-Speicher-Backward-
Compat für den CLI-Aufruf), wird aber zu einem dünnen Shim, der intern
`run_pipeline.main(...)` mit dem unveränderten YAML-Pfad aufruft. Das YAML
selbst muss bereits das neue `stages:`-Format haben.

## 7. Output

```
results/pipeline/<run_id>/
  config.yaml
  config.hash
  filtered.h5            # post-notch time_data (von Stage 1)
  residual_csm.h5        # NEU
  beam_maps.html         # NEU: 2x3 Heatmaps pre/post × low/mid/high
  target_psd.html        # NEU: Welch am Zielpunkt
  metrics.json           # NEU: Notch-Metriken UND Array-Metriken
  metrics.csv            # NEU: erweitert um Array-Spalten pro Band
  plots/                 # Notch-Channel-Plots, unverändert
    ch00.html ...
  run.log
```

`metrics.json` Top-Level-Struktur:

```json
{
  "stage1_notch": { ...wie bisher... },
  "stage2_array_filter": {
    "bands": {
      "low":  { "csm_trace_pre_db": ..., ..., "ground_truth": { "external_recovery_db": ... } | null },
      "mid":  { ... },
      "high": { ... }
    },
    "global": { "drone_power_share_total_db": ..., ... }
  }
}
```

## 8. Tests

### 8.1 Tiny-Fixture

`tests/fixtures/make_tiny_drone_artifact.py` erzeugt deterministisch:

- 4-Mic-Geometrie (`tiny_geom_4mic.xml`): zwei Ringe (innerer r=0.15 m,
  äußerer r=0.30 m), z=±0.10 m abwechselnd.
- 2 Rotoren bei `(±0.15, 0, 0)`, jeweils mit 2 Subsources auf gleichem Schema.
- 1 s @ 16 kHz weißes Rauschen pro Subsource (Seed=0, Varianz 1.0).

`tests/fixtures/make_tiny_compose.py` ruft `compose_external` mit:

- Externe Quelle `kind=noise`, Position `(0.5, 0, -0.5)`, `amplitude_db=−6`.
- Output: `tiny_synth_mixed.h5` und `tiny_gt.h5`.

Fixtures sind zur CI-Reproduzierbarkeit deterministisch und werden bei
Bedarf via `make_*.py` neu erzeugt.

### 8.2 Tests pro Modul

- **`test_csm.py`**: Shape `(F, M, M)`, hermitesch (`csm[f] = csm[f].conj().T`),
  positive Diagonale, `diag_loading_rel`-Wirkung sichtbar (Diagonale steigt
  um den dokumentierten Betrag), Frequenz-Maske greift.
- **`test_beamform_grid.py`**: `build_diagnostic_grid` liefert erwartete
  `(nx*ny, 3)` und `(nx, ny)` Shape und liegt auf der angegebenen Z-Ebene.
  `build_rotor_disc_mask` markiert genau die Punkte innerhalb der Discs.
- **`test_clean_sc_algorithm.py`**: Synthetische CSM mit einem einzelnen
  bekannten Punktquellen-Beitrag (via `reconstruct_csm` aus einer Toy-
  `SourceMap` mit einer einzigen Quelle). `CleanScAlgorithm.fit` muss
  > 90 % der Total-Power an diesen Punkt zuordnen, in einer 5-mid-Frequenz-
  Auswahl.
- **`test_reconstruct_csm.py`**: 2-Quellen-Round-Trip
  (`SourceMap` → CSM → `CleanScAlgorithm.fit` → `SourceMap`) reproduziert
  Quell-Powers innerhalb 1.0 dB.
- **`test_array_filter.py`**: Tiny-Fixture, `ArrayFilterStage` end-to-end:
  - `csm.trace > residual_csm.trace`,
  - `residual_csm` ist hermitesch,
  - `target_psd_post < target_psd_pre` in mid-band,
  - kein NaN/Inf in irgendeinem Output.
- **`test_array_metrics.py`**: synthetisches Pre/Post-Paar mit bekannter
  Energie-Differenz; Bändermetriken innerhalb ±0.5 dB.
- **`test_compose_external.py`**:
  - Helper produziert Datei mit erwarteter Sample-Anzahl,
  - GT-File-Roundtrip (load, hash, compare),
  - Drohnen-Anteil + Externer-Anteil = Mix (innerhalb 1e-9 numerischer
    Toleranz, Greens-Propagation deterministisch).
- **`test_pipeline_e2e_array.py`**: Notch + ArrayFilter end-to-end auf
  `tiny_synth_mixed.h5`. Harte Erwartungen:
  1. `external_recovery_db` mid-band innerhalb ±5 dB von Ground-Truth.
  2. `csm_trace_reduction_db` mid-band > 3 dB.
  3. Post-Beam-Map-Peak-Position (mid) im Umkreis von 0.2 m um die externe
     Quelle bei `(0.5, 0, -0.5)`.
  4. Keine `NaN`/`Inf` in irgendeinem Metrik-Feld.
- **`test_algorithm_registry.py`**: `ALGORITHM_REGISTRY["clean_sc"]` ist
  `CleanScAlgorithm`; `ArrayFilterStage` instantiiert via Konfig.

### 8.3 Smoke-Run-Erwartungen (dokumentarisch, nicht im Test-Suite)

Auf `ap2a_synth_mixed_gaptip.h5` (Synthese-Helper-Output mit ext.
Quelle bei `(0,0,-1.5)`, weißes Rauschen):

- `external_recovery_db` mid-band (500–2000 Hz): innerhalb **±3 dB** von GT.
- `csm_trace_reduction_db` mid-band: > 5 dB.
- Post-Beam-Map: kein Peak innerhalb der vier Rotor-Discs; mind. ein Peak
  in `−z`-Projektion unterhalb der Drohne sichtbar.
- Wallclock pro Lauf: < 5 min auf einer aktuellen CPU.

## 9. Risiken und Annahmen

1. **CLEAN-SC auf normal-dimensioniertem Diagnose-Grid (~2600 Punkte).**
   Standard-Acoular-Pfad, kein Edge-Case. Risiko aus dem Brainstorming
   („CLEAN-SC auf 8-Punkt-ImportGrid — konvergiert das?") ist durch das
   Refinement *single-run-on-diagnostic-grid + spatial-mask* eliminiert.
2. **Sinc-Interpolation für fractional sample shifts.** Default. Bei
   `r=1.5 m, c=343 m/s, fs=52 kHz` sind das ~227.5 Samples Verzögerung —
   integer-rounding würde 0.5-Sample-Jitter (= ~9.6 µs = bei 1 kHz ~3.5°
   Phase) einführen. Sinc-Interpolation ist bei diesen Größenordnungen
   billig (FFT-Phase-Rotation auf 4.6 M Samples ist eine Sache von
   Sekunden pro Subsource).
3. **Speed-of-Sound-Konsistenz.** `processing/constants.py` mit
   `SPEED_OF_SOUND = 343.0` ist Single Source of Truth für Synthese-Helper,
   `reconstruct_csm` und Steering. Acoular's Default ist auch 343, aber
   wir setzen es explizit über `acoular.config.c0` (oder pro
   `SteeringVector(c=...)` falls verfügbar), um Drift zu vermeiden.
4. **Algorithmus-Heterogenität CLEAN-SC vs. CLEAN-T (Folge-Spec).**
   `Algorithm.consumes` ist im Protocol drin. `ArrayFilterStage` prüft
   `consumes` und füttert entsprechend (`csm` vs. `time_data`). In dieser
   Spec implementiert nur der `csm`-Pfad. Folge-Spec aktiviert den
   `time`-Zweig.
5. **Konstrained-Subtraktion könnte „zu viel" entfernen.** Wenn die
   Pseudo-Zielpunkt-Quelle leakage in die Rotor-Disc-Masken-Komponenten
   verursacht, wird sie mit-entfernt. Das ist die `false_attribution_db`-
   Metrik in (IV). Erwartung: klein bei klar getrennten Quellpositionen
   (vertikaler Versatz `(0,0,-1.5)`), größer bei in-plane-Quellen.
6. **Memory.** CSM `(F, M, M)` complex128 bei `F ≈ 110` (`nperseg=512`,
   Maske 200–6000 Hz, Δf=102 Hz) und `M=16` ≈ 0.5 MB. Trivial. Open
   Thread #3 aus Handoff entlastet. Selbst `nperseg=8192` (~5800 Bins)
   wäre 30 MB.
7. **Mic-Geometrie-Konsistenz Synthese ↔ Stage 2.** Der Synthese-Helper
   propagiert auf eine Mic-Geometrie aus `mic_geom.xml`; die Stage-2-
   Pipeline lädt `mic_geom.xml` separat. Beide müssen dasselbe File sein.
   Wir hashen die Mic-Geometrie und schreiben den Hash als Attr in beide
   Files; Stage 2 prüft die Hashes und bricht bei Mismatch ab.

## 10. Roadmap (Folge-Specs)

1. **Algorithmen-Erweiterung (iii)** — Orth, CLEAN-T, CSM-Fitting als
   `Algorithm`-Implementierungen in `processing/algorithms/`. CLI-Konfig
   `algorithm: orth | clean_t | cmf` ohne weitere Stage-Änderungen.
   Vergleichende Auswertung (mehrere Algos in einem Run, Comparison-Plot)
   wäre eine kleine Erweiterung von `array_metrics.py` und
   `array_plots.py`.
2. **Synthese-Migration nach `drone_synthdata`** (q) — `compose_external`
   und `propagation.py` wandern upstream. `martymicfly` importiert nur noch.
3. **Stufe 3 — Spektrum-Rekonstruktion** — operiert auf `residual_csm`,
   nutzt RPM-Schedule für „bei welchen Frequenzen wurde gefiltert →
   interpolieren" und/oder Mittelung über Drehzahlsegmente.
4. **Test-Sweeps** — der Synthese-Helper bekommt eine Loop-Konfig, die
   eine Suite verschiedener externer Positionen/Signal-Klassen erzeugt;
   Stage 2 wird gegen die Suite gebenchmarkt.
5. **Real-Mess-Loader** für AP3-Daten (transponiertes Layout). Pipeline
   bleibt unverändert, nur ein zweiter `io/`-Pfad.
