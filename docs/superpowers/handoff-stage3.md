# Handoff — AP2-A Stufe 3 (nach Stufe 2 Smoke-Run)

**Stand:** 2026-04-28, Stufe 2 ist End-to-End-fähig, Smoke-Run auf Produktionsdaten zeigt grobe Abweichungen vom Spec-Target.

## Was ist erledigt

**Stufe 2 (CLEAN-SC Array-Entfaltung) läuft End-to-End.** Spec, Plan und 25 Tasks sind committed:

- `docs/superpowers/specs/2026-04-28-ap2a-stage2-array-deconv-design.md`
- `docs/superpowers/plans/2026-04-28-ap2a-stage2-array-deconv-plan.md`

Implementierung neu in `src/martymicfly/`:

```
config.py                            # AppConfig + ArrayFilterStageConfig + Sub-Configs
constants.py                         # SPEED_OF_SOUND
io/
  source_artifact.py                 # load_source_artifact (drone gap_tip)
  synth_h5.py                        # erweitert: liest /platform/ wenn vorhanden
  residual_csm_h5.py                 # write_residual_csm
  ground_truth_h5.py                 # load_ground_truth
processing/
  pipeline.py                        # Stage Registry mit kwargs forwarding
  csm.py                             # build_measurement_csm (Welch + diag loading)
  beamform_grid.py                   # build_diagnostic_grid + build_rotor_disc_mask
  steering.py                        # steer_to_psd
  array_filter.py                    # ArrayFilterStage + integrate_band_maps
  algorithms/
    base.py                          # Algorithm Protocol + SourceMap + reconstruct_csm
    clean_sc.py                      # CleanScAlgorithm (acoular BeamformerCleansc wrapper)
synth/
  propagation.py                     # Greens propagator
  external_source.py                 # ExternalSourceSpec + generate_external_signal
  compose_external.py                # mix drone + external, write synth + GT
  cli/compose.py                     # Compose CLI
eval/
  array_metrics.py                   # Stufe-2 Metriken
  array_plots.py                     # Beam-Maps + Target-PSD HTML
cli/
  run_pipeline.py                    # generischer Driver (notch + array_filter)
configs/
  example_pipeline.yaml              # 16-Mic Default mit beiden Stages
  example_compose.yaml               # Compose Default
tests/                               # 76 Tests, alle grün
```

## Smoke-Run auf Produktionsdaten (2026-04-29)

**Setup:**
- Synth-Mix: `compose_external` mit `drone_source_artifact_gaptip.h5` (gap_tip, 16 Mics) + externes Rauschen bei `(0, 0, -1.5) m`, 0 dB
- Output: `ap2a_synth_mixed_gaptip.h5` (884 MB) + `_gt.h5` (37 MB)
- Pipeline-Config: `configs/example_pipeline.yaml`
- Run: `2026-04-29T00-18-35_c162fe59`
- Wallclock: ~70 s (notch + CLEAN-SC auf 51×51 Grid, 3 Bänder, 100 Iterationen)

**Ergebnisse (`metrics.json`):**

| Band | csm_trace_reduction_db | target_psd_reduction_db | drone_power_share_db | external_recovery_db |
|------|---|---|---|---|
| low  | 0.045 | 0.045 | NaN | −26.3 |
| mid  | 0.051 | 0.033 | NaN | −26.9 |
| high | 0.080 | 0.068 | NaN | −23.2 |

**Vergleich mit Spec-Smoke-Targets (§3.3):**
- `csm_trace_reduction_db > 5` → **FAIL** (Faktor ~100 zu klein)
- `target_psd_reduction_db > 0` → marginal PASS (~0.05 dB statt mehrere dB)
- `external_recovery_db within ±3 dB` → **FAIL** (−23 bis −27 dB Abweichung)
- `drone_power_share_db` ist NaN — Edge-Case in der Berechnung (vermutlich `total_power_per_freq[mask].sum() == 0` für die mid/high-Bänder oder Division eines NaN-Werts).

## Diagnose-Hinweise aus dem Lauf

Während CLEAN-SC laufen drei Acoular-RuntimeWarnings:

1. `acoular/fbeamform.py:136 invalid value encountered in divide` — die `classic`-Steering-Formel `x / np.abs(x) / x.shape[-1]` divides by 0 wenn `|x|=0` (z.B. bei f=0 oder Grid-Punkt = Mic-Position).
2. `acoular/fbeamform.py:1455 invalid value encountered in divide` (`D1 = ... / hmax`) — `hmax=0` möglich.
3. `acoular/fbeamform.py:1459 invalid value encountered in divide` (`hh = ... / np.sqrt(1 + np.dot(ww, H))`) — Wurzel von negativ oder Division durch null in CLEAN-SC-Iteration.

Das passt zu einem CLEAN-SC, das numerisch durchläuft, aber praktisch keine Quellenenergie auf Grid-Zellen innerhalb der Rotor-Discs konzentriert: die Subtraktion `csm − drone_csm` reduziert die CSM-Spur dann fast nicht.

## Mögliche Ursachen (für Stufe 3 / Stufe-2-Tuning)

**Kalibrierung / Skalierung:**
- `bf.synthetic(f, num=0)` von Acoular gibt `p²` mit einer Konvention zurück, die u.U. nicht direkt mit `reconstruct_csm` (h h^H, p² als Powers) skaliert. Der Test-Threshold von 70 % auf der Tiny-Fixture ging zwar durch, aber dort war die Geometrie nahe ideal — Produktionsdaten sind anspruchsvoller.
- Die `target_psd_reduction_db` von ~0.05 dB auf ~10 dB Differenz im Tiny-Test-Fixture deutet auf einen anderen Skalierungsfaktor hin.

**Frequenz-Auflösung / nperseg:**
- 10 s Segment, fs=51200 Hz, nperseg=512 → ~1024 Welch-Frames pro Bin → die CSM-Schätzung sollte solide sein.
- Aber: bei `f_max=6000 Hz`, fs=51200 → Bin-Auflösung Δf=100 Hz → ~58 Bins von 200–6000 Hz. CLEAN-SC läuft pro Bin separat, das ist viel Iteration aber nicht ungewöhnlich.

**Diagnose-Grid Z:**
- `z_m: null` löst auf `platform.rotor_positions[2,0]`. Wenn das Drohnen-Origin nicht in der Rotor-Ebene liegt, ist das richtige Z für die Rotor-Disc-Suche, aber die externe Quelle bei `z=-1.5` liegt 1.5 m darunter — die Diagnose-Map sieht externe Quelle vermutlich gar nicht (sie trifft das Grid in `z=0` mit Phasen-Mismatch).
- **Der diag-Grid braucht z=0 ODER eine 3D-Suche.** Die externe Quelle liegt nicht in der Rotor-Ebene und wird daher von einem 2D-Grid bei z=Rotor-Höhe nicht gefunden.

**Rotor-Disc-Maske vs. Grid-Footprint:**
- 4 Rotor-Discs bei radius~0.1 m und Achsen-Abstand 0.23 m: das ist eine Fläche von ~0.13 m² in einem Grid von 1×1 m² (extent_xy_m=0.5, increment 0.02 → 51×51 = 2601 Cells, ~0.04 m² pro Cell). Pro Rotor ~8 Cells in der Maske, total ~32 Cells ≈ 1.2 % des Grids.
- Wenn CLEAN-SC die Rotor-Energie nicht auf diese 32 Zellen konzentriert sondern verschmiert, fängt die Maske fast nichts auf.

## Was funktioniert (Bestätigungen aus dem Run)

- Pipeline-Plumbing greift sauber: alle 7 erwarteten Output-Files entstehen (`filtered.h5`, `residual_csm.h5`, `beam_maps.html`, `target_psd.html`, `metrics.json`, `stage1_metrics.csv`, `stage2_metrics.csv`).
- Notch (Stage 1) läuft ohne Fehler — `stage1_metrics.csv` zeigt erwartete tonale Reduktion.
- CLEAN-SC läuft durch (keine Crashes), nur Numerik-Warnings.
- 76 Unit-Tests grün, inklusive E2E-Test über Tiny-Fixture, der `csm_trace_reduction_db > 0` und `target_psd_reduction_db > 0` korrekt validiert (die Tiny-Fixture-Geometrie ist günstiger als die Produktion).
- Beam-Maps HTML / Target-PSD HTML rendern (manuelle Inspektion empfohlen).

## Vorschlag für nächste Session

Stufe 2 ist **strukturell fertig** (Code, Tests, CLI, Configs, Plots), aber **numerisch nicht abgenommen** auf Produktionsdaten. Vor Stufe 3 sollte ein Tuning-Sprint erfolgen:

1. **Beam-Map manuell anschauen.** Öffnet `results/pipeline/2026-04-29T00-18-35_c162fe59/beam_maps.html`. Wo landet die Energie? Wenn die Pre-Map keine Peaks an den Rotor-Positionen hat, liegt ein fundamentaler Steering-Bug vor (Mic-Geo-Konvention `(M,3)` vs `(3,M)`, falsches Vorzeichen im Greens-Phasor, …). Wenn sie Peaks hat aber die Post-Map identisch aussieht, ist die Subtraktion das Problem (Acoular-Power-Skalierung).
2. **drone_power_share NaN fixen.** Edge-Case in `compute_array_metrics`: wenn `total_power_per_freq[mask].sum() == 0` oder `drone_power_per_freq` ein NaN enthält, schreibt JSON `null`. Defensives Min(eps) reicht.
3. **Diagnose-Grid 3D oder mindestens z=0 für externes Target.** Der aktuelle 2D-Grid bei Rotor-Z findet die externe Quelle bei z=-1.5 nie. Optionen: (a) `z_m: 0.0` setzen statt `null`, (b) zwei Grids (eines bei Rotor-Z für die Rotor-Maske, eines bei externem Z für die PSD-Schätzung), (c) echtes 3D-Grid (CLEAN-SC kann das, aber teurer).
4. **Acoular-Power-Konvention klären.** `bf.synthetic(f, num=0)` mit `r_diag=False` und `steer_type=classic`: ist die Rückgabe `p²` in `Pa²` oder bereits in `Pa²/Hz` (PSD)? Eine Skalierung-Off-By-Δf-Faktor erklärt einfach −20 bis −30 dB Mismatch.
5. **Wenn Punkte 1–4 das external_recovery_db nicht in den ±3 dB-Korridor bringen,** dann ist die Modellierung selbst zu prüfen: Greens-Free-Field ohne Drohnen-Streuung, ESC-Telemetrie-RPM-Schätzung, Kanal-Kalibrierung etc.

Erst danach Stufe 3 (z.B. Korrelations-basierte Trennung, Multi-Position-Mics, oder DOA-Tracking auf bewegten Quellen).

---

## Datei-Pointer

- Spec: `docs/superpowers/specs/2026-04-28-ap2a-stage2-array-deconv-design.md`
- Plan: `docs/superpowers/plans/2026-04-28-ap2a-stage2-array-deconv-plan.md`
- Smoke-Run-Outputs: `results/pipeline/2026-04-29T00-18-35_c162fe59/`
- Synth-Daten: `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip*.h5`
