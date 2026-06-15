# Scaling-Ladder Diagnostic — Design

**Status:** Draft, 2026-05-07
**Owner:** Steffen
**Project:** AP2-A Stage 2 Tuning Sprint, Vorlauf zur Parameterstudie

## 1. Motivation

Im Smoke-Run von Stage 2 (Array-Deconvolution) auf Produktionsdaten (`results/pipeline/2026-04-29T00-18-35_c162fe59`) liegt `external_recovery_db` bei −23 bis −27 dB neben dem Spec-Korridor von ±3 dB. Eine Auswertung der nachgereichten `ext_only`-Vergleichsläufe in `results/pipeline/ext_only_*` zeigt:

| Mask-Methode | Recovery low / mid / high (dB) |
|---|---|
| doa_drone_cone | −26.5 / −26.7 / −26.8 |
| doa_target_cone | −26.8 / −26.8 / −27.9 |
| drone_box | −28.5 / −27.2 / −28.3 |
| rotor_disc | −26.4 / −26.4 / −26.7 |

Der Mismatch ist über alle Mask-Methoden hinweg praktisch konstant ~−27 dB, sogar im `ext_only`-Szenario (kein Drohnen-Inhalt vorhanden, gar keine Trennung nötig). Das spricht stark dafür, dass der Großteil des Gaps kein algorithmisches Trennungsproblem ist, sondern eine **konstante Power-Skalierungs-Diskrepanz** zwischen `steered PSD` und `ground_truth PSD` darstellt.

Bevor die geplante Parameterstudie (Methoden + CSM- und Grid-Knöpfe) Sinn ergibt, muss diese Konstante identifiziert und entfernt werden, sonst sind alle relativen dB-Werte um einen unbekannten Offset verschoben und absolute Spec-Targets nicht zu prüfen.

## 2. Ziel

Ein eigenständiger Sanity-Check, der den Skalierungs-Pfad der Steering-Kette in drei Stufen prüft und identifiziert, an welcher Stufe der Faktor eintritt. Liefert eine Konstante (in dB) pro Stufe und damit die Antwort auf: „sitzt der Bias in CSM, im Steering, oder in der GT-Vergleichslogik?".

## 3. Architektur

### 3.1 Form

Skript `analysis/scaling_ladder.py`. Standalone, kein Pytest. Output: Markdown-Bericht plus Plotly-HTML in `results/scaling_ladder/<run_id>/`.

### 3.2 Phasen

**Phase 0 — Quelle.** Weißes Rauschen `q(t)` mit bekannter PSD `S_q` [Pa²/Hz @ 1 m] über 10 s bei `fs = 51200 Hz`. `S_q` so gewählt, dass mit `r ≈ 1.5 m` zur Quelle der Mic-PSD bei ~70 dB / Hz landet (typischer Pegelbereich der Produktionsdaten).

**Phase 1 — Forward (unabhängig).** Lese `mic_geom.xml` (16 Mics). Propagiere via händischer Free-Field-Greens-Funktion:

```
p_m(t) = q(t − r_m / c) / r_m
```

mit `c = 343 m/s`. Implementierung in 5–10 Zeilen Numpy, kein Aufruf von `synth/propagation.py`. Schreibe das Ergebnis als `ladder_synth.h5` im selben Format wie `ap2a_synth_external_only_gaptip.h5` (`/data/time_series` mit Shape `(n_samples, n_mics)`, `fs` als Attribut).

**Phase 2 — Rung 1: Mic-PSD.** Direkt-Welch pro Mic-Kanal mit Production-Parametern (`nperseg=512`, `noverlap=256`, `hann`, `density`). Theoretische Erwartung pro Mic *m*:

```
PSD_m(f) = S_q / r_m²    (flach über f)
```

Δ₁ = ⟨measured_dB − theoretical_dB⟩, gemittelt über Mics × Frequenzband [200, 6000] Hz. Diese Stufe verifiziert: ist der Forward-Pfad (Greens + Welch) konsistent.

**Phase 3 — Rung 2: CSM-Diagonale.** `build_measurement_csm` aus `processing/csm.py` mit identischen Welch-Parametern. Diagonale extrahieren, das ist per Definition die Mic-PSD. Δ₂ = ⟨CSM_diag_dB − theoretical_dB⟩. Sollte ≈ Δ₁ sein. Differenz Δ₂ − Δ₁ ist der CSM-Bauen-Faktor (Window-Norm, Density-Skalierung etc.).

**Phase 4 — Rung 3: Steered PSD.** Single-Cell-Grid an `(0, 0, −1.5)`. `steer_to_psd` aus `processing/steering.py` mit derselben Steering-Konvention wie in der Pipeline (`steer_type='classic'`, `r_diag=False`). Erwartung: das Beamforming auf der Quellen-Position rekonstruiert `S_q` (die Source-eigene PSD bei 1 m Referenz). Δ₃ = ⟨steered_PSD_dB − S_q_dB⟩, gemittelt über Frequenzband.

### 3.3 Diagnose-Logik

| Beobachtung | Verdächtiger |
|---|---|
| Δ₁ ≈ 0, Δ₂ ≈ 0, Δ₃ ≈ −27 dB | Steering / Acoular `bf.synthetic` p²-vs-PSD-Konvention. **Wahrscheinlichste Hypothese.** |
| Δ₁ ≈ 0, Δ₂ ≠ 0, Δ₃ trägt fort | `csm.py` Welch-Skalierung |
| Δ₁ ≠ 0 | Forward-Bug im Sanity-Skript selbst (sehr unwahrscheinlich, aber dann unmittelbar fixbar) |

### 3.4 Output

`results/scaling_ladder/<run_id>/`:
- `report.md` — Tabelle mit Δ₁, Δ₂, Δ₃ pro Frequenzband (low/mid/high), plus interpretierende Diagnose-Zeile.
- `mic_psd_vs_theory.html` — Plotly: Mic-PSD pro Kanal (16 Linien) plus Theorie-Linie.
- `csm_diag_vs_theory.html` — Analog für Rung 2.
- `steered_psd.html` — Steered PSD bei Quellen-Position vs `S_q`.
- `metrics.json` — Numerische Werte für maschinenlesbare Weiterverwendung.

## 4. Pass-Kriterium

Pro Rung: |Δ_k| < 0.5 dB innerhalb [200, 6000] Hz. Eine Konfiguration (Production-Welch). Wenn ein Rung den Korridor verlässt, ist die Differenz die gesuchte Bias-Konstante; der Diagnose-Bericht nennt sie explizit.

## 5. Bewusst NICHT enthalten

- Kein NNLS, kein CLEAN-SC, keine Maske.
- Keine Drohne, keine Mehrfach-Quellen.
- Kein Welch-Sweep (`nperseg`, Window) — falls Rung 2 fehlschlägt, ist das eine Folge-Untersuchung, nicht Teil dieses Tickets.
- Kein Reuse von `synth/propagation.py` im Forward — Unabhängigkeit ist die Pointe des Tests.

## 6. Nächste Schritte nach Abschluss

1. Konstante aus dem Bericht ablesen.
2. Entweder: Bug-Fix an der identifizierten Stelle und Re-Run der existierenden `ext_only_*`-Configs, um zu bestätigen dass `external_recovery_db` jetzt nahe 0 liegt.
3. Oder: Bias als bekannten Offset dokumentieren und in den Metrik-Vergleich einrechnen.
4. Erst danach: Parameterstudie über CSM-, Grid- und Methoden-Knöpfe (separater Spec).

## 7. Datei-Pointer

- Existierender Forward-Pfad (Referenz, nicht zu nutzen): `src/martymicfly/synth/propagation.py`, `src/martymicfly/synth/external_source.py`, `src/martymicfly/synth/compose_external.py`
- Production-CSM: `src/martymicfly/processing/csm.py` → `build_measurement_csm`
- Production-Steering: `src/martymicfly/processing/steering.py` → `steer_to_psd`
- Synth-H5-Format: `src/martymicfly/io/synth_h5.py`
- Mic-Geometrie: `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml`
- Vorhandene `ext_only`-Vergleichsläufe: `results/pipeline/ext_only_*`
- Stage-2-Spec (übergeordnet): `docs/superpowers/specs/2026-04-28-ap2a-stage2-array-deconv-design.md`
- Handoff (Hintergrund): `docs/superpowers/handoff-stage3.md`
