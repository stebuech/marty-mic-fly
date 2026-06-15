# AP2-A Stufe 1 — Notch-Filterung (Design)

**Datum:** 2026-04-28
**Status:** Entwurf zur User-Review
**Vorhaben:** MartyMicFly / `marty-mic-fly`

## 1. Ziel und Scope

### 1.1 Antragskontext

AP2-A im Antrag *FliegendesMessMikrofon* beschreibt eine dreistufige Pipeline zur Trennung des Drohnen-Eigengeräuschs vom zu messenden Schall (Antragsabschnitt „A — Getrennte Behandlung der tonalen und der breitbandigen Anteile", Zeilen 477–533):

1. **Stufe 1 — Tonal-Entfernung:** Cascade aus Notch-Filtern, adaptiv an momentane Rotor-Drehzahlen. Realisierung als IIR. Frequenz-Tracking: Variante 1 = RPM-Sensor (umgesetzt hier), Variante 2 = Tan/Jiang aus dem Mikrofonsignal (späterer Spec).
2. **Stufe 2 — Räumliche Filterung:** Orthogonale Entfaltung / CLEAN-SC / CLEAN-T (eigene Spec).
3. **Stufe 3 — Spektrum-Rekonstruktion:** Mittelung über Drehzahlsegmente bzw. Interpolation an gefilterten Frequenzen (eigene Spec).

Diese Spec deckt **nur Stufe 1** plus das Pipeline-Skelett ab, in das die späteren Stufen ohne Re-Design andocken.

### 1.2 In-Scope dieser Spec

- AP2-A Stufe 1 (Notch-Filterung, RPM-getrieben, zero-phase) End-to-End auf
  `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synthesis_16mic_gaptip.h5`.
- HDF5-Loader für Acoular-Format (Synth-Daten) und Mic-Geometrie-Loader (XML).
- Pipeline-Skelett mit klarer Stage-Schnittstelle für Stufen 2/3.
- `NotchFilter` (`/home/steffen/Code/NotchFilter`) und `drone_synthdata`
  (`/home/steffen/Code/drone_synthdata`) als editable Pip-Deps. `drone_synthdata`
  wird in dieser Spec nicht importiert; die Dep wird nur etabliert, damit
  Re-Synthese im selben Venv läuft.
- Adapter-Helfer (`interpolate_per_motor_bpf`, `build_harmonic_matrix`, Acoular-
  kompatible In-Memory-Sources, lineares r-Schedule) hier nachgebaut — keine
  Imports aus `NotchFilter/validate/`.
- Output: gefilterte HDF5 (gleiches Layout, Provenance-Attrs) +
  Plotly-HTML pro Kanal + `metrics.json` + `metrics.csv`.
- CLI + YAML-Konfiguration (Validierung mit `pydantic`).
- Tests: Unit (IO, Frequenzen, Sources, Notch-Smoke, Metriken, Plots) +
  Integration (End-to-End auf einer Mini-HDF5-Fixture).
- **Cleanup-Commit als erster Implementierungsschritt:** Alt-Notch-Code unter
  `analysis/` entfernen.

### 1.3 Out-of-Scope dieser Spec

- Re-Synthese läuft extern via `drone_synthdata`-Skripte. Bei Bedarf wird
  `drone_synthdata` upstream erweitert; Port in dieses Repo ist nicht geplant.
- Stufe 2 (Array-Entfaltung) und Stufe 3 (Spektrum-Rekonstruktion).
- Tan/Jiang-Frequenzschätzung (Variante 2 aus dem Antrag).
- Ground-Truth-Vergleich gegen die externe Quelle (hängt davon ab, dass die
  Synthese das Quellsignal mitspeichert — eigene Spec, ggf. mit
  Upstream-Patch in `drone_synthdata`).
- Performance-Benchmarks, GPU-Pfad, Vergleich mit Real-Messdaten,
  Gold-Master-Plots.
- Kalibrierung auf SPL in Pa: Output-dB sind relativ zu 1 (Einheit²) und
  werden so im Metrics-Header dokumentiert.

### 1.4 Bestehender Code

`analysis/` enthält eine ältere MIMO-IIR-Notch-Implementation (Harvey 2019),
die durch `NotchFilter` ersetzt wird, sowie davon unabhängige Module
(Sound-Power, Beamforming-Vergleich, Spectral-Comparison, Real-Mess-Data-Loader),
die für spätere APs/Stufen erhalten bleiben sollen.

**Cleanup-Commit am Anfang der Implementierung entfernt:**

- `analysis/mimo_adaptive_iir_filter.py`
- `analysis/mimo_filter_analysis.py`
- `analysis/iir_filter_validation.py`
- `analysis/frequency_initialization.py`
- `analysis/MIMO_FILTER_README.md`
- `analysis/mimo_filter_config_schema.yaml`
- `analysis/PULL_PUSH_ANALYSIS_NOTES.md` (sofern reine MIMO-Notch-Notizen)

**Bleibt unangetastet** (für Stufe 2 / AP5 / Real-Messdaten):

- `analysis/sound_power_calculation.py`, `analysis/sound_power_comparison.py`
- `analysis/beamforming_comparison.py`
- `analysis/spectral_comparison.py`
- `analysis/data_loader.py` (Real-Mess-Layout)
- `analysis/README.md`, `example_config.yaml`, `my_comparison.yaml`

## 2. Repo-Layout

```
src/martymicfly/
  __init__.py
  io/
    __init__.py
    synth_h5.py        # load_synth_h5(path) -> dict
    mic_geom.py        # load_mic_geom_xml(path) -> ndarray (M, 3)
    write_filtered.py  # write filtered HDF5 + Provenance-Attrs
  processing/
    __init__.py
    frequencies.py     # rpm->bpf, harmonic matrix, r-schedule
    sources.py         # ArraySamplesGenerator, ArrayFreqSource
    notch.py           # NotchStage über CascadeNotchFilter
    pipeline.py        # Stage-Protokoll, run_pipeline
  eval/
    __init__.py
    metrics.py         # tonal residual, broadband, SPL drop
    plots.py           # Plotly-Spektrum/Spektrogramm
  cli/
    __init__.py
    run_notch.py       # YAML -> Pipeline -> Outputs
configs/
  example_notch.yaml
tests/
  fixtures/
    make_tiny_synth.py
    tiny_synth.h5
    tiny_geom.xml
  test_io.py
  test_frequencies.py
  test_sources.py
  test_notch.py
  test_metrics.py
  test_plots.py
  test_pipeline.py
docs/superpowers/specs/
  2026-04-28-ap2a-stage1-notch-design.md   # diese Datei
```

`analysis/` bleibt parallel bestehen (siehe Cleanup-Liste oben für entfernte
Dateien).

## 3. Datenfluss

```
                 input HDF5 (Acoular-Format)              mic_geom.xml
                          |                                    |
                          v                                    v
                  +--------------+                     +--------------+
                  | io.synth_h5  |                     | io.mic_geom  |
                  +------+-------+                     +------+-------+
                         | time_data (N,C), fs,               | pos (M,3)
                         | rpm_per_esc                        |
                         v                                    |
              +----------------------+                        |
              | processing.frequencies|                       |
              |  - slice segment      |                       |
              |  - interp RPM->BPF    |                       |
              |  - build harm matrix  |                       |
              |  - r-schedule         |                       |
              +----------+-----------+                        |
                         | per_motor_bpf (N,S),               |
                         | harm_matrix (N,S*M),               |
                         | pole_radius                        |
                         v                                    |
              +----------------------+                        |
              | processing.notch     |                        |
              |  NotchStage          |                        |
              |  -> CascadeNotchFilter                        |
              |   (external,zero-ph) |                        |
              +----------+-----------+                        |
                         | filtered (N,C)                     |
                         v                                    |
                +--------+-----------+                        |
                v                    v                        |
        +----------------+   +----------------+               |
        | eval.metrics   |   | eval.plots     |<--------------+
        |  per channel   |   |  Plotly HTML   |  (geom für Anno-
        |  -> JSON + CSV |   |  pro Kanal     |   tation optional)
        +----------------+   +----------------+
                         |
                         v
                +-----------------+
                | io.write_filtered|
                | output HDF5 +   |
                | Provenance-Attrs |
                +-----------------+
```

## 4. Komponenten-Verträge

### 4.1 `io.synth_h5.load_synth_h5(path) -> dict`

Liest das Acoular-Format der Synth-Datei.

- **Input:** Pfad zur HDF5.
- **Output:**
  ```python
  {
      "time_data":   np.ndarray,   # (N, C) float64
      "sample_rate": float,
      "rpm_per_esc": dict[str, dict[str, np.ndarray]],
                                   # ESC-Name -> {"rpm": (T,), "timestamp": (T,)}
      "duration":    float,        # N / sample_rate
  }
  ```
- **Validierung:** `time_data` ist 2D float; `sample_freq`-Attr vorhanden;
  Timestamps streng monoton steigend; alle ESCs gleichberechtigt strukturiert.

### 4.2 `io.mic_geom.load_mic_geom_xml(path) -> np.ndarray`

Parst das Acoular-`<MicArray>`/`<pos>`-XML-Schema und liefert `(M, 3)` in
Reihenfolge des Auftretens (entspricht Mic-Index). Konsistenzprüfung gegen
`time_data.shape[1]` ist Aufgabe des Pipeline-Drivers, nicht des Loaders.

### 4.3 `processing.frequencies` (reine Funktionen)

#### `interpolate_per_motor_bpf(rpm_per_esc, t_start, duration, n_samples, sample_rate, n_blades) -> np.ndarray`

- ESCs werden alphabetisch sortiert (`sorted(rpm_per_esc.keys())`).
- `mic_times = np.arange(n_samples) / sample_rate + t_start`.
- Für jede ESC: `interp1d(timestamp, rpm, kind='linear', fill_value='extrapolate')`
  auf `mic_times`, dann `bpf = rpm * n_blades / 60.0`.
- **Synth-Spezifika:** Mic- und ESC-Zeit teilen sich `t=0`, kein Sync-Offset.
  Das ist der Unterschied zu `NotchFilter/validate/run_filter_comparison.py:291`
  und der Grund, warum diese Funktion hier neu lebt statt importiert wird.
- **Output:** `(n_samples, S)` float64.

#### `build_harmonic_matrix(per_motor_bpf, n_harmonics) -> np.ndarray`

- Spaltenindex für Motor `s`, Harmonik `h` (1-basiert): `s * n_harmonics + (h - 1)`.
  Diese Konvention ist identisch zu NotchFilter und wird im Docstring
  festgehalten, weil `CascadeNotchFilter` sich darauf verlässt.
- **Output:** `(n_samples, S * n_harmonics)` float64.

#### `linear_r_schedule(n_harmonics, fs, delta_bpf_hz, k_cover, margin_hz, r_min, r_max) -> np.ndarray`

- `bw(h) = k_cover * h * delta_bpf_hz + margin_hz`
- `r(h) = clip(1 - bw(h) * pi / fs, r_min, r_max)`
- **Output:** `(n_harmonics,)` float64.
- Werte werden separat getestet — der Code wird neu geschrieben, nicht aus
  NotchFilter importiert.

### 4.4 `processing.sources`

Acoular-kompatible In-Memory-Wrapper, API-kompatibel zu
`MockSamplesGenerator`/`MockFreqSource` aus NotchFilter:

#### `ArraySamplesGenerator(data, sample_freq)`

- `data`: `(N, K)` ndarray. `K=1` (Default-Modus per-Kanal-Loop) oder `K>1`
  (Multikanal-Modus).
- Erbt von `acoular.SamplesGenerator`.
- Pflicht-Traits: `sample_freq`, `numchannels` (Wert-Properties),
  `num_samples`, `num_channels` (Properties).
- `result(num)` yieldet `data[i:i+num]`-Blöcke.

#### `ArrayFreqSource(matrix)`

- `matrix`: `(N,)` oder `(N, M)` ndarray.
- `result(num)` yieldet zeilenweise Slices.
- Vertrag muss zu `AdaptiveNotchFilter._result_external` und
  `CascadeNotchFilter.result / _result_zero_phase` passen
  (block-aligned, sequenziell vorwärts).

### 4.5 `processing.notch.NotchStage`

```python
@dataclass
class NotchConfig:
    n_blades: int
    n_harmonics: int
    pole_radius: float | np.ndarray   # scalar oder (M,) oder (S, M)
    multichannel: bool = False
    block_size: int = 4096

class NotchStage:
    name = "notch"
    def __init__(self, config: NotchConfig): ...
    def process(self, ctx: PipelineContext) -> PipelineContext: ...
```

- `f_inits[s, h-1] = per_motor_bpf[0, s] * h` für `h in 1..M`.
- Baut intern für jeden Lauf:
  ```python
  CascadeNotchFilter(
      num_sources=S,
      harmonics_per_source=M,
      frequencies=f_inits,
      pole_radius=cfg.pole_radius,
      mode="external",
      zero_phase=True,
      source=ArraySamplesGenerator(signal, fs),
      freq_source=ArrayFreqSource(harm_matrix),
  )
  ```
  und sammelt das Ergebnis mit
  `np.vstack(list(cascade.result(cfg.block_size)))`.
- **Multikanal-Strategie:**
  - Default `multichannel=False` ⇒ pro Kanal sequenziell, deterministisch in
    Speicher und Laufzeit. `signal` ist `(N, 1)` pro Iteration, das gefilterte
    Resultat wird pro Kanal in `(N, C)` zusammengesetzt.
  - `multichannel=True` ⇒ One-Shot mit `signal=(N, C)`. Wird in der
    Implementierung via Smoke-Test gegen den Per-Kanal-Lauf verifiziert; bei
    Diskrepanz bleibt `multichannel` False im Default.
- **Pre/Post-Erhaltung:** `ctx.metadata["pre_notch"]` enthält das ungefilterte
  Segment (für Metriken/Plots benötigt). `ctx.time_data` wird durch das
  gefilterte Signal ersetzt.

### 4.6 `processing.pipeline`

```python
@dataclass
class PipelineContext:
    time_data: np.ndarray              # (N, C)
    sample_rate: float
    rpm_per_esc: dict
    mic_positions: np.ndarray          # (M, 3)
    per_motor_bpf: np.ndarray          # (N, S)
    harm_matrix: np.ndarray            # (N, S*M)
    metadata: dict                     # frei

class Stage(Protocol):
    name: str
    def process(self, ctx: PipelineContext) -> PipelineContext: ...

def run_pipeline(stages: list[Stage], ctx: PipelineContext) -> PipelineContext:
    for stage in stages:
        ctx = stage.process(ctx)
    return ctx
```

- Stufen 2/3 implementieren das gleiche `Stage`-Protokoll und werden später
  einfach an die `stages`-Liste angehängt — das Skelett ändert sich nicht.

### 4.7 `eval.metrics.compute_metrics(pre, post, fs, per_motor_bpf, n_harmonics, pole_radius, channels, cfg) -> dict`

Pro Kanal:

- **PSD vor/nach** via `scipy.signal.welch(window='hann', nperseg=cfg.welch_nperseg,
  noverlap=cfg.welch_noverlap, scaling='density')`.
- **Tonale Bandbreite pro Harmonik:** `BW(h) = (1 - r(h)) * fs / pi`,
  multipliziert mit `cfg.bandwidth_factor`.
- **Mittlere Frequenz pro Harmonik:** `f_h_s = mean(per_motor_bpf[:, s]) * h`.
- **Tonal-Energie an `(s, h)`:** PSD-Integral über
  `[f_h_s - BW(h)/2, f_h_s + BW(h)/2]` via Riemann-Summe in der
  Welch-Frequenzgitter-Auflösung. dB-Output relativ zu 1 (Einheit²),
  dokumentiert im JSON-Header.
- **Tonal-Total:** Summe aller `(s, h)`-Bänder, dann dB.
- **Broadband:** PSD-Integral über `[broadband_low_hz, fmax_plot]` minus alle
  Tonal-Bänder. `broadband_low_hz` Default `null` ⇒ `0.5 * min(BPF)`.

Ergebnis: zwei Artefakte:

- **`metrics.json`** — vollständige Metriken inkl. Per-Harmonik-Aufschlüsselung
  pro Kanal.
- **`metrics.csv`** — eine Zeile pro Kanal mit
  `channel, broadband_pre_db, broadband_post_db, broadband_delta_db,
  tonal_total_pre_db, tonal_total_post_db, tonal_reduction_db`.

### 4.8 `eval.plots.plot_channel_html(pre, post, fs, per_motor_bpf, outpath, cfg)`

Eine standalone HTML-Datei pro Kanal, zwei vertikal gestackte Subplots:

1. **Welch-PSD log-y, 0..fmax** mit zwei Linien (`pre`, `post`); gestrichelte
   vertikale Linien an den vorhergesagten Notch-Frequenzen (eigene Farbe je
   Motor).
2. **Spektrogramm „nach"** (oder als Tabs `pre`/`post`) via
   `scipy.signal.spectrogram(window='hann', nperseg=cfg.spectrogram_window,
   noverlap=cfg.spectrogram_overlap)`.

`fmax` Default = `min(1.1 * max(per_motor_bpf) * n_harmonics, fs / 2)`.

Default werden alle Kanäle geplottet; `plots.channel_subset` erlaubt eine
Whitelist (Pflicht bei größeren Arrays in späteren Phasen).

### 4.9 `io.write_filtered.write_filtered(out_path, ctx, run_meta)`

- `time_data` aus `ctx` ins gleiche Acoular-Layout schreiben
  (`(N, C)` float64, `EARRAY`/`EXTDIM=0`/`sample_freq` wie Eingabe).
- `esc_telemetry/`-Gruppen pass-through (unverändert kopiert).
- Root-Attrs: `martymicfly_version`, `input_h5`, `config_hash`,
  `segment_start_s`, `segment_duration_s`, `n_blades`, `n_harmonics`,
  `notch_mode='rpm_external_zerophase'`, `pole_radius_repr`.

### 4.10 `cli.run_notch`

Entrypoint: `python -m martymicfly.cli.run_notch --config <yaml> [--output-dir <path>] [--log-level INFO|DEBUG]`.

Ablauf:

1. YAML laden, mit `pydantic` validieren, `config_hash` (8 hex der gerenderten
   Konfig) berechnen.
2. `run_id = ISO-Timestamp + "_" + config_hash`.
3. `io.synth_h5.load_synth_h5` + `io.mic_geom.load_mic_geom_xml`,
   Konsistenz prüfen.
4. Segment auswählen (Mode aus `segment.mode`).
5. `processing.frequencies` ausführen → `per_motor_bpf`, `harm_matrix`,
   `pole_radius` (skalar oder per-Schedule).
6. `PipelineContext` aufbauen, `run_pipeline([NotchStage(...)])`.
7. `eval.metrics.compute_metrics` → JSON + CSV ins Output-Verzeichnis.
8. `eval.plots.plot_channel_html` für jeden Kanal in der Channel-Auswahl.
9. `io.write_filtered.write_filtered` → `filtered.h5`.
10. Konfig-Snapshot + `config.hash` ablegen, `run.log` flushen.

## 5. Konfiguration

`configs/example_notch.yaml`:

```yaml
input:
  audio_h5: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synthesis_16mic_gaptip.h5
  mic_geom_xml: /media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml

segment:
  mode: middle              # middle | head | tail | explicit
  duration: 10.0
  # start: 5.0              # nur bei mode=explicit
  # end: 15.0

channels:
  selection: all            # all | list
  # list: [0, 1, 4, 8]

rotor:
  n_blades: 2
  n_harmonics: 20

notch:
  pole_radius:
    mode: scalar            # scalar | linear
    value: 0.9994
    # k_cover: 1.5          # nur bei mode=linear
    # margin_hz: 5.0
    # delta_bpf_hz: null
    # r_min: 0.90
    # r_max: 0.9995
  multichannel: false
  block_size: 4096

metrics:
  welch_nperseg: 8192
  welch_noverlap: 4096
  bandwidth_factor: 1.0
  broadband_low_hz: null    # null = 0.5 * min(BPF)

plots:
  enabled: true
  fmax_hz: null             # null = 1.1 * max(BPF) * n_harmonics, gedeckelt bei fs/2
  spectrogram_window: 4096
  spectrogram_overlap: 2048
  channel_subset: null      # null = alle Kanäle, sonst Liste 0-basierter Indizes

output:
  dir: results/notch/{run_id}
  filtered_h5: filtered.h5
  metrics_json: metrics.json
  metrics_csv: metrics.csv
  plots_subdir: plots
  copy_config: true
```

**Konfig-Disziplin:**

- YAML ist Single Source of Truth. CLI-Overrides nur für `--output-dir`
  (überschreibt `output.dir`) und `--log-level`. Alles andere muss in der
  Konfig stehen.
- Jeder Lauf legt einen Konfig-Snapshot ab → reproduzierbar.
- `pydantic`-Modelle (`AppConfig`, `InputConfig`, `SegmentConfig`,
  `ChannelsConfig`, `RotorConfig`, `NotchConfig`, `MetricsConfig`,
  `PlotsConfig`, `OutputConfig`) mit Field-Validators für
  Mode-abhängige Pflichtfelder (z. B. `linear` → `k_cover` etc. erforderlich).

## 6. Output

```
results/notch/<run_id>/
  config.yaml
  config.hash
  filtered.h5
  metrics.json
  metrics.csv
  plots/
    ch00.html
    ch01.html
    ...
  run.log
```

- `<run_id>` = ISO-Timestamp + `_` + 8-hex-Konfig-Hash
  (`2026-04-28T15-22-04_a3f1c9d2`).
- `filtered.h5` enthält das gefilterte `time_data` plus pass-through
  `esc_telemetry/` (siehe 4.9).

## 7. Tests

```
tests/
  fixtures/
    make_tiny_synth.py     # generiert tiny_synth.h5 deterministisch (seed)
    tiny_synth.h5          # 4 Kanäle, 2 ESCs, 1 s @ 16 kHz
    tiny_geom.xml
  test_io.py
  test_frequencies.py
  test_sources.py
  test_notch.py
  test_metrics.py
  test_plots.py
  test_pipeline.py
```

**Fixture-Inhalt:** konstante BPF pro ESC (z. B. ESC1=100 Hz, ESC2=120 Hz;
`n_blades=2` ⇒ rpm 3000 / 3600), 5 saubere Sinusharmonische pro ESC,
weißes Rauschen mit bekannter Varianz, plus eine bekannte „externe Quelle"
(800 Hz Ton). Damit sind Erwartungen analytisch nachprüfbar.

**Pro Modul:**

- `test_io`: HDF5-Roundtrip identisch; XML mit 16 Punkten ⇒ `(16, 3)`-Array
  in dokumentierter Reihenfolge; fehlende `sample_freq` ⇒ klare Fehlermeldung.
- `test_frequencies`: konstante RPM ⇒ konstante BPF; lineare RPM-Rampe ⇒
  lineare BPF; `build_harmonic_matrix` liefert die dokumentierte Spaltenfolge
  `s*M + (h-1)`; `linear_r_schedule` Edge-Cases (Clipping bei `r_min`/`r_max`).
- `test_sources`: `ArraySamplesGenerator.result(num)` yieldet erwartete
  Blöcke und Restblock; `ArrayFreqSource` ditto; beide reproduzieren die
  Acoular-Trait-Properties (`num_samples`, `num_channels`, `sample_freq`).
- `test_notch`: reiner Sinus an bekannter Frequenz ⇒ Output-Energie an
  dieser Frequenz < −30 dB unter Eingang. Smoke-Test der Integration mit
  `notchfilter.cascade.CascadeNotchFilter`, kein Mock — wenn die Dep
  bricht, soll dieser Test es zeigen.
- `test_metrics`: PSD-Integration über bekanntes Sinussignal liefert die
  analytisch erwartete Energie ±0.5 dB; Tonal-Energie an einer Frequenz,
  an der nichts ist, ist nahe Null.
- `test_plots`: HTML wird geschrieben, ist gültiges UTF-8, enthält die
  erwarteten Plotly-Trace-Strukturen (kein Pixel-Vergleich).
- `test_pipeline`: End-to-End auf der Fixture; verifiziert die Felder in
  `metrics.json`, die Provenance-Attrs in `filtered.h5` und
  `tonal_reduction_db ≥ 25 dB` für die bekannten Sinusharmonischen.

## 8. Abhängigkeiten

Ergänzungen in `pyproject.toml` (End-Zustand nach `uv add` / `uv add --editable`):

```toml
[project]
dependencies = [
  # bestehende Deps bleiben
  "h5py",
  "numpy",
  "scipy",
  "plotly",
  "pyyaml",
  "pydantic>=2",
  "acoular",
  "notchfilter",        # via [tool.uv.sources] auf lokales Repo gemappt
  "drone-synthdata",    # via [tool.uv.sources] auf lokales Repo gemappt
]

[tool.uv.sources]
notchfilter = { path = "/home/steffen/Code/NotchFilter", editable = true }
drone-synthdata = { path = "/home/steffen/Code/drone_synthdata", editable = true }
```

Beide Editable-Deps werden in der Implementierung mit
`uv add --editable /home/steffen/Code/NotchFilter` und
`uv add --editable /home/steffen/Code/drone_synthdata` angelegt — `uv`
schreibt automatisch beide Sektionen. Die exakten Paketnamen ergeben sich
aus dem `pyproject.toml` der jeweiligen Repos (`notchfilter` und
`drone-synthdata`); falls dort anders benannt, werden die Namen oben
angepasst.

## 9. Risiken und offene Punkte

- **Multikanal-Verhalten von `CascadeNotchFilter` mit `K>1`:** In der
  Implementierung verifizieren; falls Diskrepanzen zum Per-Kanal-Lauf,
  bleibt `multichannel=False` Default.
- **Synth-Daten ohne Ground-Truth-Quellsignal:** Quantitative Bewertung der
  Quellsignal-Rückgewinnung erst nach Upstream-Patch in `drone_synthdata`
  möglich. Stufe-1-Bewertung beschränkt sich auf tonale Restenergie und
  Broadband-Erhaltung — beides ohne Ground-Truth aussagekräftig.
- **Akustische Kalibrierung:** Output-dB sind relativ zu 1 (Einheit²);
  echte SPL bräuchte separate Kalibrierungsspec.
- **Tonal-Bandbreite vs. PSD-Frequenzauflösung:** `BW(h)` kann bei großen
  `pole_radius` schmaler sein als die Welch-Bin-Breite. In dem Fall
  integriert die Riemann-Summe mindestens einen Bin; Test `test_metrics`
  deckt diesen Edge-Case ab.

## 10. Roadmap (Folge-Specs, nicht Teil dieses Plans)

1. **Stage 2 — Array-Entfaltung.** Orthogonale Entfaltung / CLEAN-SC /
   CLEAN-T, Acoular-basiert; Vorlage in `drone_synthdata/scripts/clean_sc_source_localization.py`.
2. **Stage 3 — Spektrum-Rekonstruktion.** Mittelung über Drehzahlsegmente /
   Interpolation an Notch-Frequenzen.
3. **Ground-Truth-Bewertung.** Sobald Synthese das Mess-Quellsignal
   mitspeichert: `signal_recovery_db`, Spektrum-MAE etc.
4. **Tan/Jiang-Frequenzschätzung** (Variante 2 aus dem Antrag).
5. **Real-Mess-Loader.** Zweiter Loader für das transponierte Real-Layout,
   sobald AP3-Messdaten vorliegen. Pipeline bleibt unverändert.
