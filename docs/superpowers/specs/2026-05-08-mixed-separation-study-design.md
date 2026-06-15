# Mixed-Separation-Studie: Design-Spec

**Datum:** 2026-05-08
**Status:** Design — Review pending
**Vorgänger:** `docs/superpowers/specs/2026-05-07-scaling-ladder-design.md` (Skalierungs-Bug isoliert + behoben via `range_compensation_factor`)

## Kontext

Nach dem Calibration-Fix in der ext_only-Pipeline liegt `external_recovery_db` bei
−0.42 dB (nnls) bis −0.75 dB (target_cone) — nahe des Welch-Floors, aber nicht
diagnostisch ausreichend für die eigentliche Projektaufgabe (Trennung der
externen Quelle von Drohnenlärm in `mixed`-Daten).

Die bestehenden Metriken sagen nichts darüber, **wie viel Drohnenleistung am
Target ankommt** — `external_recovery_db` ist außerdem bandintegriert vor
Verhältnisbildung und damit kompensationsanfällig (Over- und Undersubtraktion
in benachbarten Frequenzen heben sich auf). Bevor wir Knöpfe systematisch
sweepen, müssen wir das Bewertungsmaß repariert haben.

Die Studie geht in drei Etappen vor:
1. Metrik-Framework + Welch-Floor empirisch kalibrieren (Baseline)
2. Sensitivitäts-Audit: pro Knopf ein ±-Wackler, dominante Achsen identifizieren
3. Fokussierter Sweep: dominante Achsen verfeinern, Pareto-Optimum finden

## §1 Ziel und Erfolgskriterium

**Ziel:** Optimale Parameter für die räumliche Trennung von Drohne und externer
Quelle finden, gemessen am Pareto-Front zwischen Sender-Treue
(`spectrum_l1_db`) und Drohnenleckage (`drone_leakage_db_def2`) — auf den
vorhandenen synth-Daten plus drei Robustheits-Szenarien.

**Erfolgskriterium:**
- Sensitivitäts-Audit identifiziert ≤3 dominante Knöpfe pro Metrik
- Fokussierter Sweep liefert pro DOA-Methode eine Pareto-optimale Konfiguration
  mit
  - `spectrum_l1_db < welch_floor_db + 3 dB` pro Band
  - `drone_leakage_db_def2 < welch_floor_db` pro Band
- Reproduzierbares Empfehlungs-Markdown (`final_recommendation.md`) mit
  Zahlen + Plots als Endprodukt

## §2 Metriken (fixiert)

### Voraussetzungen

Drei Welch-PSDs in identischen Einheiten (Pa²/Hz, source-PSD am Target):

```
ext_GT(f)   = welch(ext_only_gt_h5["time_data"][:,0],   nperseg, noverlap, hann, density)
D_ref(f)    = welch(drone_only_gt_h5["time_data"][:,0], nperseg, noverlap, hann, density)
psd_post(f) = steer_to_psd(residual_csm, freqs, mics, target) · range_compensation_factor
```

Alle drei aus demselben Synth-Triple (ext + drone + mixed) am gleichen
`target_point_m`. `D_ref` wird per Time-Domain-Subtraktion `mixed − ext`
gewonnen (Linearität der Synth, exakt bis numerische Genauigkeit), keine
zusätzlichen Synth-Files nötig.

### Per-Bin Zerlegung (kompensationsfrei)

```
excess(f)  = max(psd_post(f) − ext_GT(f), 0)    # Unter-Subtraktion / Drohnen-Leck
deficit(f) = max(ext_GT(f) − psd_post(f), 0)    # Über-Subtraktion / extern gepeelt
|error|(f) = excess(f) + deficit(f)             # da nur einer pro f ≠ 0 ist
```

### Bandintegrale

```
E_excess  = Σ excess(f)  · Δf       (linear power)
E_deficit = Σ deficit(f) · Δf
E_GT      = Σ ext_GT(f)  · Δf
P_post    = Σ psd_post(f)· Δf
D_unflt   = Σ D_ref(f)   · Δf
```

### Reduzierter Metrik-Satz pro Band

| Metrik | Definition | Zweck |
|---|---|---|
| `spectrum_l1_db` | `10·log10((E_excess + E_deficit) / E_GT)` | Gesamt-Treuemaß, kompensationsfrei |
| `over_subtraction_db` | `10·log10(E_deficit / E_GT)` | Anteil gepeelten externen — Richtungs-Info |
| `drone_leakage_db_def1` | `10·log10(P_post / D_unflt)` | Konservativ, beide Argumente ≥ 0 |
| `drone_leakage_db_def2` | `10·log10(E_excess / D_unflt)` | Sensitiv auf reine Drohnenleckage |
| `spectrum_rms_db` | `√(mean((10·log10(psd_post/ext_GT))²))` | Log-Domain-Streuung (Komplement) |
| `recovery_db_signed` | `10·log10(P_post / E_GT)` | **Legacy/Sanity** — kompensationsanfällig, explizit markiert |

Plus zwei Diagnose-Felder pro Band:
- `welch_floor_db` (aus den ext_only-Sanity-Runs empirisch bestimmt)
- `compensation_flag` (bool): true gdw. **beide** Komponenten individuell über
  dem Welch-Floor liegen, d.h.
  ```
  10·log10(E_deficit / E_GT) > welch_floor_db + 3   (= over_subtraction_db)
  ∧ 10·log10(E_excess / E_GT) > welch_floor_db + 3   (= drone_leakage_def2 + SNR_band)
  ```
  Wenn beide Seiten signifikant sind, versteckt sich Kompensation in
  `recovery_db_signed`.

### Frequenzaufgelöste Beilage

`metrics_freq.h5` pro Run mit:
- `freqs` (F,)
- `psd_post`, `ext_GT`, `D_ref`, `excess`, `deficit` (alle (F,))

Für Plots, nicht im JSON-Aggregate.

### Sanity-Beziehungen (Test-relevant)

```
P_post = E_GT + E_excess − E_deficit                      (Definitionen)
recovery_db_signed − drone_leakage_db_def1 = −SNR_band    (unabhängig vom Filter)
spectrum_l1_db (linear) ≥ over_subtraction_db (linear)    (E_excess ≥ 0)
```

## §3 Baseline und Szenarien

### Bestehende Configs (8 Baseline-Runs)

|              | doa_rotor_cone | doa_drone_cone | doa_target_cone | nnls |
|---|---|---|---|---|
| **ext_only** | ✓ | ✓ | ✓ | ✓ |
| **mixed**    | ✓ | ✓ | ✓ | ✓ |

Alle existieren in `configs/pipeline_*.yaml`. Werden mit dem neuen
Metrik-Framework neu ausgewertet (Pipeline selbst unangetastet).

### Welch-Floor Kalibrierung

Die 4 ext_only-Runs liefern den Welch-Floor: kein Drohnenleck physikalisch
möglich, also `spectrum_l1_db` und `drone_leakage_db_def2` dort rein
Welch-Streuung + algorithmusinduzierte Schwankung. Diese Werte werden im
Report als Schraffur in Plots eingezeichnet und als JSON-Field
`welch_floor_db` mitgeliefert.

### Drohnen-only-Quelle

Per Time-Domain-Subtraktion:
```python
drone_only_at_array(t)  = mixed_h5["time_data"]    − ext_only_h5["time_data"]
drone_only_at_target(t) = mixed_gt_h5["time_data"] − ext_only_gt_h5["time_data"]
```

### Robustheits-Szenarien (Synth-on-demand)

| Szenario | Variation | Zweck |
|---|---|---|
| `S0_baseline` | target=(0,0,−1.5), amp=1.0 | bestehende synth files |
| `S1_offaxis` | target=(0.3, 0.0, −1.5) | Robustheit gegen laterale Position |
| `S2_low_snr` | source amp × 0.3 (≈10 dB tiefer) | rauschdominiertes Regime |
| `S3_far` | source bei z=−3.0 m (statt −1.5) | größere Quelldistanz |

S1–S3 werden via `src/martymicfly/synth/cli/compose.py` neu generiert.
Caching per (config_dict, scenario_h5_path) — bestehende h5 werden nicht
neu erzeugt.

## §4 Sensitivitäts-Audit (Phase 1)

Pro Achse ein ±-Wackler vom Baseline-Punkt, alle 5 essentiellen Metriken pro
Run erhoben.

### Knob-Achsen

| Knob-Achse | Baseline | Audit-Punkte |
|---|---|---|
| `csm.nperseg` | 512 | {256, 1024} |
| `csm.diag_loading_rel` | 1e-6 | {0, 1e-4} |
| `doa_grid.rotor_cone_half_angle_deg` | 30 | {20, 45} |
| `doa_grid.drone_disk_half_width_deg` | 15 | {8, 25} |
| `doa_grid.target_cone_half_angle_deg` | 45 | {30, 60} |
| `doa_grid.focal_radius_m` | 1.5 | {1.0, 2.0} |
| `doa_grid.az_step_deg × el_step_deg` | 5×5 | {3×3, 10×10} |
| `clean_sc.damp` | 0.6 | {0.3, 0.9} |
| `clean_sc.n_iter` | 100 | {50, 300} |
| `clean_sc.r_diag` | true | {false} |
| Quelle (synth) | S0 | {S1, S2, S3} |

**Out-of-Scope** (explizit nicht in der Studie):
- `notch.pole_radius` — Notch-Stage außerhalb dieser Studie
- NNLS-Algorithmus — nicht im Fokus, kein Audit-Lauf damit
- Multi-Source-Szenarien — separate Folge-Studie

### Run-Count

- 10 Achsen × 2 Punkte = 20 Audit-Runs
- 1 Achse (Quelle) × 3 Punkte = 3 Runs
- + 1 Baseline (S0)
- = **24 Runs pro DOA-Methode × 3 DOA-Methoden = 72 Runs**

### Empfindlichkeitsmaß

```
Δ_high = metric(high) − metric(baseline)
Δ_low  = metric(low)  − metric(baseline)
sensitivity = max(|Δ_high|, |Δ_low|)
dominant_flag = sensitivity > 3 · welch_floor_db
```

Pro Methode separat aufgeführt; für die Phase-2-Auswahl der dominanten
Achsen über die 3 DOA-Methoden gemittelt. Methodenspezifisch dominante
Achsen werden gesondert markiert.

### Output

- `phase1_sensitivity.csv` (long format: `method | axis | metric | baseline | low | high | sensitivity_db | dominant_flag`)
- `phase1_sensitivity_heatmap.html` (Subplots pro Methode, Achsen × Metriken,
  Farbe = sensitivity)
- `phase1_dominant_axes.json` ({method: [list]} für Phase 2)

### Welch-Floor

`welch_floor_db ≈ 0.1 dB` für mittlere Bänder bei nperseg=512, fs=51200, T=10s
(≈390 Mittlungen). Empfindlichkeiten <0.3 dB sind nicht detektierbar — wird
explizit dokumentiert.

## §5 Focused Sweep (Phase 2)

Pro dominanter Achse aus Phase 1 ein 5–7-Punkte-Sweep gegen die in Phase 1
beste DOA-Methode (Kriterium: niedrigstes `spectrum_l1_db` über alle Bänder
gemittelt auf S0_baseline; bei Gleichstand niedrigstes `drone_leakage_db_def2`).

### Sweep-Beispiele

| Achse | Phase-2-Sweep |
|---|---|
| `clean_sc.damp` | {0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95} |
| `clean_sc.n_iter` | {30, 60, 100, 150, 200, 300, 500} |
| `doa_grid.drone_disk_half_width_deg` | {5, 10, 15, 20, 25, 35, 45} |
| `csm.nperseg` | {128, 256, 512, 1024, 2048} |

### Wechselwirkungs-Grid

Top-2-Achsen (höchste mittlere sensitivity über die 5 essentiellen Metriken,
gemittelt über die 3 DOA-Methoden) als 5×5-Grid = 25 Runs. Identifiziert
paarweise Kopplung (z.B. `damp × n_iter`).

### Robustheits-Verifikation

Optimaler Konfig-Punkt aus Phase 2 wird gegen S1/S2/S3 getestet (3 Runs).
Wenn der Optimum-Punkt nicht generalisiert, wird das im Bericht als
Limitation dokumentiert.

### Run-Count Phase 2

15–28 Sweep + 25 Interaction + 3 Robustness ≈ **45–55 Runs**.

### Pareto-Auswertung

Pro dominanter Achse `pareto_<axis>.html`:
- X: `over_subtraction_db`
- Y: `drone_leakage_db_def2`
- Marker farb-codiert nach Knob-Wert (continuous scale)
- Welch-Floor als hatched rectangle
- Pareto-Frontpunkte hervorgehoben

### Endbericht

`final_recommendation.md` mit
- Empfohlener Konfig pro DOA-Methode (Tabelle aller 11 Knob-Werte)
- Erwartete Metriken auf S0 + Robustheits-Werte auf S1/S2/S3
- Diff zum Baseline-Punkt
- Plots eingebettet/verlinkt

## §6 Studien-Architektur

### Modulstruktur

```
analysis/separation_study/
├── __init__.py
├── README.md
├── metric_extensions.py       # spectrum_l1_db, over_subtraction_db,
│                              # drone_leakage_db_def1/def2, frequency-resolved arrays
├── drone_only_helper.py       # Time-Domain-Subtraktion + welch
├── synth_scenarios.py         # S1/S2/S3 generieren via compose_external/compose
├── study_runner.py            # YAML-Override → run_pipeline → metrics
├── aggregate_results.py       # metrics.json → study_db.parquet
├── plots/
│   ├── sensitivity_heatmap.py
│   ├── pareto_plot.py
│   ├── frequency_resolved.py
│   └── interaction_plot.py
└── studies/
    ├── phase1_sensitivity.yaml
    ├── phase2_sweep.yaml      # dynamisch nach Phase-1-Result
    └── baseline.yaml
```

### Integration in Pipeline-Code

Die separation-Metriken werden **nicht** direkt in
`src/martymicfly/eval/array_metrics.py` integriert, sondern als Decorator:

```python
def compute_array_metrics_with_separation(*, base_kwargs, drone_only_at_target, ...):
    base = compute_array_metrics(**base_kwargs)
    extra = compute_separation_metrics(...)
    return merge(base, extra)
```

Begründung: Studienmetriken brauchen Zugriff auf das drone-only-GT-Signal —
ein Studien-Konzept, kein Pipeline-Konzept. Falls sich die Metriken als
universell nützlich erweisen, in einem Folge-PR direkte Integration mit
optionalen `drone_only_*`-Argumenten.

### `study_runner.py` Konzept

```yaml
# studies/phase1_sensitivity.yaml
base_config: configs/pipeline_mixed_doa_target_cone.yaml
methods: [doa_rotor_cone, doa_drone_cone, doa_target_cone]
scenarios: [S0, S1, S2, S3]
axes:
  csm.nperseg:
    type: discrete
    points: [256, 1024]
  doa_grid.rotor_cone_half_angle_deg:
    type: discrete
    points: [20, 45]
  # ...
output_dir: results/separation_study/phase1
```

Runner-Schritte:
1. Pro Methode × Achse × Punkt × Szenario einen YAML-Override patchen
2. `martymicfly.cli.run_pipeline` aufrufen (Subprocess oder Python-API)
3. Ergebnis-`metrics.json` mit Studien-Metriken erweitern
4. In `study_db.parquet` (long-format) loggen

**Caching:** Run-Hash aus (config_dict, scenario_h5_path) bestimmt
Output-Verzeichnis. `--force` für Re-Runs.

**Parallelisierung:** Default sequentiell; optional `--n-workers N` für
Multiprocessing (Pipeline-Runs sind unabhängig, IO-leicht).

### Aggregation

Long-format Parquet:
```
columns: [study_phase, method, axis, axis_value, scenario, band, metric, value]
```

### Konsistenz-Tests am Output

Beim Aggregieren werden die Sanity-Beziehungen verifiziert:
```
recovery_db_signed − drone_leakage_db_def1 ≈ −SNR_band       (±0.2 dB)
spectrum_l1_db (linear) ≥ over_subtraction_db (linear)
```
Verstöße als WARN ins Log (Pipeline-Bug- oder Konventions-Indikator).

## §7 Output und Dokumentation

### Verzeichnisstruktur

```
results/separation_study/
├── baseline/
│   ├── ext_only_doa_*_S0/, mixed_doa_*_S0/, ...   (8 result_dirs)
│   └── baseline_summary.csv
├── phase1_sensitivity/
│   ├── runs/                                       (72 result_dirs)
│   ├── phase1_sensitivity.parquet
│   ├── phase1_sensitivity.csv
│   ├── phase1_sensitivity_heatmap.html
│   └── phase1_dominant_axes.json
├── phase2_focused_sweep/
│   ├── runs/                                       (~45–55 result_dirs)
│   ├── phase2_sweep.parquet
│   ├── pareto_<axis>.html                          (pro dominanter Achse)
│   ├── interaction_<a>_x_<b>.html                  (top-2)
│   └── frequency_resolved_<run>.html
├── final_recommendation.md
└── study_log.md                                    (chronologisch)
```

### `final_recommendation.md` (Schema)

Pro DOA-Methode:
- Tabelle: Knob | Baseline | Empfohlen | Δ
- Tabelle: Metrik × Band (low/mid/high) auf S0
- Tabelle: Robustheit auf S1/S2/S3
- Plot-Verlinkungen (Pareto + frequenzaufgelöst)
- Limitationen / offene Punkte

Plus eine Vergleichstafel vorne mit allen drei DOA-Methoden nebeneinander.

### Plot-Konventionen

Alle Plotly, `template="plotly_white"`.

| Plot-Typ | Achsen | Anmerkungen |
|---|---|---|
| Heatmap | Knob × Metrik | Subplots pro Methode; Diverging colormap |
| Pareto | over_subtraction_db × leakage_db_def2 | Welch-Floor als hatched; Knob-Wert farb-codiert |
| Wechselwirkungs-Grid | Knob_a × Knob_b | Optimum mit Pfeil annotated |
| Frequenzaufgelöst | f [Hz] × dB | psd_post + ext_GT + D_ref; excess/deficit als gestapeltes Subplot |

### `study_log.md`

Pro Studienschritt eine Zeile:
```
2026-05-10 14:23 | phase1 start | git=4757bd7 | base=...      | n_runs=72
2026-05-10 16:01 | phase1 done  | n_failed=0  | dominant_axes=[...]
```

### Reproduzierbarkeit

- `studies/phaseN_*.yaml` wird ins Repo committed bevor der Schritt startet
- result_dirs enthalten `config.yaml` (Pipeline-Override) + `git_sha.txt` +
  `study_yaml_path.txt`
- `aggregate_results.py` schreibt git-SHA in jede Parquet-Zeile

## §8 Tests

### 1. Metrik-Unit-Tests (`tests/separation_study/test_metric_extensions.py`)

| Test | Setup | Erwartung |
|---|---|---|
| `test_spectrum_l1_db_no_compensation` | Σexcess = Σdeficit, signed ≈ 0 | l1 ≫ 0; signed ≈ 0 — Compensation sichtbar |
| `test_spectrum_l1_db_perfect_match` | psd_post = ext_GT identisch | l1 → −∞ (oder ≤ welch_floor) |
| `test_over_subtraction_db_no_excess` | psd_post = α · ext_GT mit α<1 | over = 10·log10(1−α) |
| `test_drone_leakage_def1_floor` | psd_post = ext_GT, D_ref bekannt | def1 = −SNR |
| `test_drone_leakage_def2_zero_excess` | psd_post = ext_GT überall | def2 → −∞, repräsentiert als `< floor` |
| `test_recovery_minus_leakage_def1_eq_neg_snr` | beliebig | Sanity ±1e-10 |
| `test_compensation_flag_triggers` | excess + deficit beide groß, signed ≈ 0 | flag=true |
| `test_spectrum_rms_db_uniform_offset` | psd_post = β · ext_GT mit β konstant | rms_db = \|10·log10(β)\| (per-Bin gleich) |

### 2. Drone-only-Helper (`tests/separation_study/test_drone_only_helper.py`)

| Test | Setup | Erwartung |
|---|---|---|
| `test_subtraction_recovers_drone_audio` | mixed = ext + drone konstruiert | drone_only_at_target = mixed − ext bit-genau |
| `test_d_ref_matches_direct_welch_of_drone` | synthetisches drone-Signal bekannt | welch(D_ref) ≈ welch(drone_direct) ±1e-12 |

### 3. Study-Runner-Integration (`tests/separation_study/test_study_runner.py`)

| Test | Setup | Erwartung |
|---|---|---|
| `test_yaml_override_application` | base + axis_override-dict | dotted-key-Patch korrekt |
| `test_run_caching_skips_existing` | result_dir mit metrics.json existiert | runner skipped (kein Subprocess) |
| `test_aggregate_long_format_columns` | 3 fake metrics.json | parquet hat erwartete Spalten + Zeilenzahl |
| `test_consistency_check_warns` | metrics-Set mit verletzter Sanity | aggregate emittet WARN |

### 4. End-to-End-Smoke (`tests/separation_study/test_phase1_smoke.py`)

| Test | Setup | Erwartung |
|---|---|---|
| `test_phase1_minimal_runs` | mock-base + 2 Achsen × 2 Punkte | 1 Baseline + 4 Audit-Runs; metrics.json je 5 Metriken × n_bands |

`@pytest.mark.slow` — nicht in Default-CI.

### 5. Synth-Szenarien (`tests/separation_study/test_synth_scenarios.py`)

| Test | Setup | Erwartung |
|---|---|---|
| `test_S1_offaxis_target_position` | erzeuge S1 | Quelle bei (0.3, 0, −1.5) |
| `test_S2_low_snr_amplitude_correct` | erzeuge S2 | GT-PSD ~10 dB tiefer als S0 |
| `test_S3_far_source_position_correct` | erzeuge S3 | source_pos in synth-Metadata = (0,0,−3.0); steered psd_pre des externen Anteils am Array ≈ S0 −6 dB (1/r²-Greens) |
| `test_caching_skips_existing_h5` | h5 existiert | Re-Generation übersprungen |

### Coverage-Ziel

- Neue Metrik-Funktionen: 100% Branch
- Drone-only-Helper: 100%
- Study-Runner: ≥80%

### Folge-Issue (out-of-spec)

Nach Studienabschluss: empfohlener Konfig-Punkt als
`tests/test_separation_recommendation_e2e.py` einfrieren — slow,
skip-by-default, dient als Regression-Anker.

## Out-of-Scope

Explizit nicht Teil dieser Studie:
- `notch.pole_radius` Variation
- NNLS als Audit-Methode
- Multi-Source-Szenarien
- Direkte Integration der Studien-Metriken in `eval/array_metrics.py`
- Zeit-Domain SNR-Metriken (alle Metriken sind Welch-PSD-basiert)
- Real-data-Validierung (synth-only — separate Folge-Studie)
