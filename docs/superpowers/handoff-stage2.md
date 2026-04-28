# Handoff — AP2-A Stufe 2 (Array-Entfaltung)

**Stand:** 2026-04-28, nach Abschluss Stufe 1.

## Was ist erledigt

**Stufe 1 (Notch-Filter, RPM-extern, zero-phase) läuft End-to-End.** Spec und Plan sind committed:

- `docs/superpowers/specs/2026-04-28-ap2a-stage1-notch-design.md`
- `docs/superpowers/plans/2026-04-28-ap2a-stage1-notch-plan.md`

Implementierung unter `src/martymicfly/`:

```
src/martymicfly/
  config.py                   # pydantic AppConfig + Subkonfigs
  io/
    synth_h5.py               # load_synth_h5 (Acoular-HDF5 → dict)
    mic_geom.py               # load_mic_geom_xml → (M,3) ndarray
    write_filtered.py         # write_filtered (HDF5 + Provenance-Attrs)
  processing/
    frequencies.py            # interpolate_per_motor_bpf, build_harmonic_matrix, linear_r_schedule
    sources.py                # ArraySamplesGenerator, ArrayFreqSource (Acoular-kompatibel)
    pipeline.py               # PipelineContext, Stage Protocol, run_pipeline
    notch.py                  # NotchStage (wraps notchfilter.cascade.CascadeNotchFilter)
  eval/
    metrics.py                # compute_metrics (tonal + broadband pre/post, dB ref=1)
    plots.py                  # plot_channel_html (Welch PSD + pre/post Spektrogramme)
  cli/
    run_notch.py              # CLI mit DEFAULT_CONFIG = configs/example_notch.yaml
configs/example_notch.yaml    # 16-Mic-Synth-Default
tests/                        # 35 Tests, grün; tiny_synth.h5/tiny_geom.xml-Fixture
results/                      # gitignored
```

**Smoke-Run Ergebnisse** (10 s Mitte aus `ap2a_synthesis_16mic_gaptip.h5`,
16 Kanäle, scalar `pole_radius=0.9994`, `n_harmonics=20`):
- `tonal_reduction_db`: 14–18 dB pro Kanal
- `broadband_delta_db`: −6 bis −11 dB (Caveat — Tuning-Sache)
- ~52 s Wallclock pro Lauf (per-Kanal-Loop, 16 Kanäle × 80 Notches × zero-phase)

## Was Stufe 2 leisten soll

Aus Spec §1.1 / Antrag Zeilen 506–524:

> **Räumliche Filterung der breitbandigen Anteile** durch Arrayverfahren —
> Entfaltung statt klassisches Beamforming, weil die Drohnen-Eigengeräusch-
> Quellrichtungen bekannt sind (Rotor-Positionen relativ zum Mic-Array).
> Vorgesehene Verfahren:
> - **Orthogonale Entfaltung** [56]
> - **CLEAN-SC** [57]
> - **CLEAN-T** [58, 41]
> - Optional **CSM-Fitting** [59] als rechenintensives Vergleichsverfahren

Eingang in Stufe 2: das Notch-gefilterte Multikanal-Signal aus Stufe 1.
Ausgang: das vom Drohnen-Eigengeräusch räumlich befreite Signal +
Diagnose-Outputs (Beam-Maps, Quell-Power-Schätzungen).

Acoular bringt viele dieser Verfahren mit (siehe `acoular.fbeamform`,
`acoular.deconv` etc.). Im sibling-Repo
`/home/steffen/Code/drone_synthdata/scripts/clean_sc_source_localization.py`
gibt es bereits einen funktionierenden CLEAN-SC-Aufruf als Vorlage.

## Anker im Pipeline-Skelett

Stufe 2 ist eine eigene `Stage`-Implementierung nach dem Protokoll in
`src/martymicfly/processing/pipeline.py`. Sie konsumiert
`PipelineContext.time_data` (= post-Notch-Signal) plus
`PipelineContext.mic_positions` und liefert ein neues `time_data`.

Konkret braucht es:
- Ein neues Modul `src/martymicfly/processing/array_filter.py` (oder
  `deconv.py`) mit einer `ArrayFilterStage` für jeden Algorithmus
  (oder eine konfigurierbare `ArrayFilterStage` mit `algorithm: clean_sc | clean_t | orth | csm_fit`).
- Konfig-Erweiterung in `martymicfly.config` um eine `ArrayFilterConfig`,
  inkl. Rotor-Positionen relativ zum Array-Origin (kommen aus dem
  Synth-Artifact bei `drone_synthdata`).
- Metriken-Erweiterung in `eval/metrics.py` (Quell-Lokalisierung,
  Power-Verteilung pro Richtung) und/oder eine eigene `eval/maps.py`
  für Beam-Maps.
- Integration im CLI (`cli/run_notch.py` → `cli/run_pipeline.py`?
  oder Stage-Liste über Konfig steuerbar machen).

## Bekannte Open Threads für Stufe 2

1. **Rotor-Positionen.** Stufe 2 braucht die Schallquellen-Positionen.
   Synthesis-HDF5 (`ap2a_synthesis_16mic_gaptip.h5`) enthält aktuell **nur**
   `time_data` + `esc_telemetry`, **keine** Rotor-Geometrie und **kein**
   externes Quellsignal. Beides liegt in `drone_synthdata`-Artifacts und/oder
   muss bei der Synthese mit ausgeschrieben werden — möglicher Upstream-Patch
   in `drone_synthdata/io.py:write_acoular_h5` als Vorbedingung.

2. **Mic-Array-Mittelpunkt vs. Drohnen-Koordinatensystem.** `mic_geom.xml`
   hat 16 Punkte in Drohnen-relativen Koordinaten; Rotor-Positionen müssten
   im selben System vorliegen.

3. **Cross-Spectral-Matrix-Größe.** 16 Mics × FFT-Frames × Frequenzbins →
   bei 52 kHz und sinnvoller Auflösung wird das groß. Memory-Budget früh
   prüfen.

4. **Bewertung ohne Ground-Truth.** Wie bei Stufe 1: das Synth-HDF5 enthält
   das Mess-Quellsignal nicht separat. Quantitative Bewertung der
   Quell-Rückgewinnung erfordert Ground-Truth → Upstream-Patch in
   `drone_synthdata` (eigene Spec/Aufgabe).

5. **Frequenz-Bereich.** Beamforming mit 16 Mics ist erst sinnvoll oberhalb
   einer Mindestfrequenz (Aliasing-Grenze hängt von der Mic-Verteilung ab —
   das `gap_tip`-Layout hat einen Außenring r=0.575 m und einen Innenring,
   das gibt einen brauchbaren Bereich grob ab ~300 Hz aufwärts).

## Vorschlag für die nächste Session

Nach `/clear`:

> „Lass uns AP2-A Stufe 2 (Array-Entfaltung) brainstormen. Stufe 1 ist
> committed und dokumentiert in
> `docs/superpowers/specs/2026-04-28-ap2a-stage1-notch-design.md` und
> `docs/superpowers/plans/2026-04-28-ap2a-stage1-notch-plan.md`. Lies erst
> `docs/superpowers/handoff-stage2.md` für den Stand und die offenen Punkte,
> dann starte mit dem brainstorming-Skill."

Damit hat der frische Kontext alle Anker für eine zielgerichtete
Brainstorm-Plan-Implement-Runde nach demselben Workflow.
