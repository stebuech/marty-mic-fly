# Messreihe: Einfluss des Rotor-Motorarm-Abstands auf Eigengeräusche

## Messdatum
**20. Januar 2026**

## Versuchsziel
Untersuchung des Einflusses des Abstands zwischen Rotor und Rotorarm einer Drohne auf die akustischen Eigengeräusche zur Optimierung der Mikrofonarray-Konfiguration für das "Fliegende Messmikrofon" Projekt.

---

## Versuchsaufbau

### Messumgebung
- **Ort**: Reflexionsarmer Raum, TU Berlin
- **Volumen**: 830 m³
- **Untere Grenzfrequenz**: 63 Hz
- **Bedingungen**: Praktisch reflexionsfrei, ohne Störschall

### Drohnenkonfiguration
- **Plattform**: Quadrocopter (auf Stativ fixiert)
- **Betriebsmodus**: Einzelmotor-Betrieb (nur ein Motor aktiv)
- **Ziel-Drehzahl**: ~3400 RPM (entspricht Schwebeflug-Drehzahl)
- **Motorposition bei Referenzabstand (0.5 cm)**: x=0, y=0, z=0

### Mikrofonarray-System

#### Array-Geometrie
- **Gesamtzahl Mikrofone**: 63 (verteilt auf 2 Arrays)
- **Linkes Array**: 31 Mikrofone bei z = +0.805 m
- **Rechtes Array**: 32 Mikrofone bei z = -0.805 m
- **Array-Abstand zur Drohne**: ±80.5 cm (symmetrisch)
- **Array-Typ**: Planare Arrays mit unregelmäßiger Mikrofonverteilung (optimiert für räumliche Auflösung)

#### Koordinatensystem
```
Referenzpunkt (Rotor bei Abstand 0.5 cm): (x=0, y=0, z=0)
Rechtes Array: z = -0.805 m
Linkes Array:  z = +0.805 m
```

---

## Messkampagne

### Versuchsparameter

#### Pull-Konfiguration (Rotor über dem Arm)
- **Abstandsbereich**: 0.5 cm bis 15.0 cm
- **Anzahl Messpunkte**: 16
- **Schrittweite**: 
  - 0.5 cm bis 1.0 cm: 0.5 cm
  - 1.0 cm bis 15.0 cm: 1.0 cm

| Abstand [cm] | Dateiname | Messzeit |
|--------------|-----------|----------|
| 0.5 | 2026-01-20_13-20-37_533829 | 13:20 |
| 1.0 | 2026-01-20_13-23-00_316033 | 13:23 |
| 2.0 | 2026-01-20_13-25-34_714255 | 13:25 |
| 3.0 | 2026-01-20_13-29-04_964424 | 13:29 |
| 4.0 | 2026-01-20_13-33-18_061777 | 13:33 |
| 5.0 | 2026-01-20_13-35-20_292953 | 13:35 |
| 6.0 | 2026-01-20_13-37-18_561751 | 13:37 |
| 7.0 | 2026-01-20_13-39-35_873257 | 13:39 |
| 8.0 | 2026-01-20_13-41-24_503980 | 13:41 |
| 9.0 | 2026-01-20_13-43-08_643826 | 13:43 |
| 10.0 | 2026-01-20_14-05-23_282697 | 14:05 |
| 11.0 | 2026-01-20_14-07-13_682714 | 14:07 |
| 12.0 | 2026-01-20_14-08-59_223846 | 14:08 |
| 13.0 | 2026-01-20_14-10-19_945097 | 14:10 |
| 14.0 | 2026-01-20_14-11-41_205454 | 14:11 |
| 15.0 | 2026-01-20_14-13-08_151342 | 14:13 |

#### Push-Konfiguration (Rotor unter dem Arm)
- **Abstandsbereich**: 0.0 cm bis -15.0 cm
- **Anzahl Messpunkte**: 16
- **Schrittweite**: 1.0 cm
- **Besonderheit**: Bei Abständen 0.0 cm, -1.0 cm und -2.0 cm keine Messung möglich (Motorhalterung kollidiert mit Beam-Rohr bei Abstand -3 cm)

| Abstand [cm] | Dateiname | Messzeit | Status |
|--------------|-----------|----------|--------|
| 0.0 | - | - | Keine Messung |
| -1.0 | - | - | Keine Messung |
| -2.0 | - | - | Keine Messung |
| -3.0 | 2026-01-20_14-50-00_813161 | 14:50 | ✓ |
| -4.0 | 2026-01-20_14-46-54_390032 | 14:46 | ✓ |
| -5.0 | 2026-01-20_14-45-29_051801 | 14:45 | ✓ |
| -6.0 | 2026-01-20_14-44-07_773468 | 14:44 | ✓ |
| -7.0 | 2026-01-20_14-40-40_398224 | 14:40 | ✓ |
| -8.0 | 2026-01-20_14-39-08_778867 | 14:39 | ✓ |
| -9.0 | 2026-01-20_14-37-54_221316 | 14:37 | ✓ |
| -10.0 | 2026-01-20_14-36-26_460959 | 14:36 | ✓ |
| -11.0 | 2026-01-20_14-35-15_226536 | 14:35 | ✓ |
| -12.0 | 2026-01-20_14-33-48_287676 | 14:33 | ✓ |
| -13.0 | 2026-01-20_14-32-17_254755 | 14:32 | ✓ |
| -14.0 | 2026-01-20_14-30-46_004267 | 14:30 | ✓ |
| -15.0 | 2026-01-20_14-28-44_985039 | 14:28 | ✓ |

### Messprotokoll
- **Messdauer pro Konfiguration**: ~10 Sekunden
- **Drehzahl**: Konstant bei ~3400 RPM
- **Sampling**: Vermutlich 48-96 kHz (Standard für akustische Arrays)
- **Gesamtmessungen**: 29 gültige Datensätze (16 Pull + 13 Push)

---

## Analyseziele

### Primäre Fragestellungen
1. **Abstandsabhängigkeit des Eigengeräusch-Spektrums**
   - Wie verändert sich das tonale Eigengeräusch (BPF und Harmonische)?
   - Wie verändert sich das breitbandige Eigengeräusch?
   - Gibt es einen optimalen Abstand zur Minimierung des Eigengeräuschs?

2. **Pull vs. Push Vergleich**
   - Unterschiede in der akustischen Signatur zwischen beiden Konfigurationen
   - Einfluss der Strömungsrichtung (Rotor schiebt vs. zieht)
   - Interaktion zwischen Rotor-Nachlauf und Rotorarm

3. **Räumliche Abstrahlcharakteristik**
   - Richtungsabhängigkeit der Schallemission
   - Unterschiede zwischen linkem und rechtem Array
   - Symmetrie-Eigenschaften

### Sekundäre Fragestellungen
4. **Frequenzbereich der Beeinflussung**
   - In welchem Frequenzbereich ist der Einfluss am stärksten?
   - Gibt es kritische Frequenzbereiche für die Schalldruckpegelmessung?

5. **Beamforming-Analyse**
   - Lokalisierung dominanter Schallquellen
   - Trennung zwischen Rotorgeräusch und Arm-induziertem Geräusch
   - Zeitliche Stabilität der Quellpositionen

6. **Implikationen für Filteralgorithmen**
   - Anforderungen an adaptive Notch-Filter (AP 2)
   - Effektivität räumlicher Filterung (Sparse Bayesian Learning)
   - Optimale Arrayposition für Eigengeräusch-Unterdrückung

---

## Erwartete Ergebnisse

### Physikalische Mechanismen
1. **Wirbelinteraktion**: Bei kleinen Abständen verstärkte Interaktion zwischen Rotor-Nachlauf und Arm
2. **Strömungsabriss**: Mögliche periodische Ablösung am Arm durch Rotor-induzierte Strömung
3. **Dipol-Quellen**: Arm als akustischer Dipol durch unsymmetrische Umströmung
4. **Tonale Verstärkung**: Mögliche Resonanzeffekte bei bestimmten Abständen

### Hypothesen
- **H1**: Eigengeräusch-Level sinkt monoton mit zunehmendem Abstand
- **H2**: Pull-Konfiguration erzeugt höhere breitbandige Komponente als Push
- **H3**: Optimaler Abstand liegt bei 10-15 cm (Kompromiss zwischen Eigengeräusch und Flugstabilität)
- **H4**: Tonale Komponenten (BPF) zeigen geringere Abstandsabhängigkeit als breitbandige

---

## Analyse-Workflow

### 1. Datenvorverarbeitung
```python
# Einlesen der h5-Dateien
# Zeitbereich-Analyse: RMS, Spektrogramme
# Frequenzbereich: FFT, Welch-Methode für PSD
```

### 2. Spektralanalyse
- **Schmalbandspektren**: FFT mit hoher Auflösung zur Identifikation tonaler Komponenten
- **Terz-Spektren**: Für Vergleich mit ISO 3744 Anforderungen
- **Spektrogramme**: Zeitliche Stabilität der Drehzahl und Frequenzkomponenten

### 3. Beamforming-Analyse (mit Acoular)
```python
import acoular as ac

# Standard Beamforming zur Quelllokalisierung
# CLEAN-SC für verbesserte Auflösung
# Orthogonal Beamforming zur Trennung kohärenter Quellen
```

### 4. Vergleichsanalysen
- Pull vs. Push für gleiche Abstände
- Abstandsabhängigkeit für jede Konfiguration
- Array-Vergleich (links vs. rechts)

### 5. Statistik und Unsicherheitsanalyse
- Messunsicherheit aus zeitlicher Variation
- Drehzahlstabilität (aus tonalen Komponenten)
- Wiederholbarkeit (falls mehrere Messungen pro Konfiguration)

---

## Relevanz für Projektarbeitspakete

### AP 1: Multicopter-Drohne mit Mikrofonarray
- **Erkenntnisse für Arrayposition**: Optimaler Abstand Mikrofon-Rotor
- **Mechanische Konstruktion**: Designvorgaben für Mikrofon-Befestigung
- **Trade-off**: Eigengeräusch vs. Flugstabilität vs. Array-Größe

### AP 2: Trennung des Eigengeräuschs
- **Notch-Filter Design**: Frequenzen und erforderliche Dämpfung
- **SBL-Anforderungen**: Räumliche Charakteristik des Eigengeräuschs
- **Validierungsdaten**: Ground-Truth für Algorithmen-Erprobung

### AP 3: Indoor-Experimente
- **Vergleichsbasis**: Referenzdaten für spätere Vollsystem-Tests
- **Messprozedur**: Lessons learned für experimentelles Protokoll
- **Unsicherheitsbudget**: Beitrag der Geometrie zur Gesamtunsicherheit

### AP 4: Messunsicherheit
- **Einflussgröße "Arrayposition"**: Quantifizierung für Monte-Carlo-Simulation
- **Modellvalidierung**: Reale Daten zur Absicherung des Unsicherheitsmodells

---

## Technische Details

### Dateien und Datenstruktur
- **Zeitstempel-Format**: YYYY-MM-DD_HH-MM-SS_NNNNNN
- **Dateiformat**: Vermutlich HDF5 (.h5) mit Acoular-Struktur
- **Kanalzahl**: 63 (synchrone Aufnahme aller Mikrofone)
- **Array-Definition**: `BVG_partial_vogel_left_right.xml`

### Koordinatensystem-Konventionen
```
x-Achse: [Orientierung tbd]
y-Achse: [Orientierung tbd]
z-Achse: Senkrecht zu Array-Ebenen
  - Rechtes Array: z = -0.805 m
  - Linkes Array:  z = +0.805 m
  - Rotor (Ref):    z = 0 m (bei Abstand 0.5 cm)
```

### Verarbeitungssoftware
- **Acoular**: Mikrofonarray-Auswertung
- **Python/NumPy/SciPy**: Signalverarbeitung
- **Matplotlib/Plotly**: Visualisierung

---

## Nächste Schritte

### Sofortige Aktionen
1. ✅ Dokumentation erstellt
2. ⬜ Datenintegrität prüfen (alle Dateien vorhanden und lesbar?)
3. ⬜ Quick-Look Analyse: Spektren visualisieren
4. ⬜ Drehzahlstabilität verifizieren

### Hauptanalyse
5. ⬜ Systematische Spektralanalyse aller Messpunkte
6. ⬜ Beamforming-Karten für ausgewählte Frequenzen
7. ⬜ Statistische Auswertung der Abstandsabhängigkeit
8. ⬜ Vergleichsplots Pull vs. Push

### Ergebnisse und Dokumentation
9. ⬜ Zusammenfassende Plots und Tabellen
10. ⬜ Empfehlung für optimale Array-Konfiguration
11. ⬜ Integration in Projektdokumentation
12. ⬜ Eventuelle Publikation der Ergebnisse

---

## Referenzen und Kontext

### Projektkontext
Siehe: `FliegendesMessMikrofon_Projektbeschreibung.pdf` und `Drone_Audition_Research_Landscape.md`

### Relevante Literatur
- Harvey PhD Thesis (adaptive IIR notch filtering)
- SAVE-MSBL-EM für Sparse Bayesian Learning
- ISO 3744 für Schallleistungsmessung

### Verwandte Arbeiten (TU Berlin)
- Herold et al.: Directivity measurements von UAVs
- Sarradj et al.: Acoular framework
- Vorarbeiten zu Multicopter-Akustik (siehe Projektbeschreibung)

---

## Kontakt und Verantwortlichkeiten
- **Projekt**: Fliegendes Messmikrofon (DFG-Antrag)
- **Antragsteller**: Prof. Ennes Sarradj, Dr.-Ing. Gert Herold
- **Institution**: TU Berlin, Fachgebiet Technische Akustik
- **Arbeitspaket**: AP 1 (Multicopter-Drohne mit Mikrofonarray)

---

**Dokumentversion**: 1.0  
**Erstellt**: 23. Januar 2026  
**Letztes Update**: 23. Januar 2026
