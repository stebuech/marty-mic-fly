# Machbarkeitsstudie: Low-RPM Drohne mit großen Festpropellern hoher Steigung

## Zusammenfassung

**Fazit: Das Design ist machbar, erfordert aber sorgfältige Kompromisse.**

Eine Drohne mit großen, langsam drehenden Festpropellern (Fixed-Pitch) kann stabil fliegen und 1 kg Nutzlast tragen. Der Ansatz ist akustisch sehr vielversprechend: Experimentell wurde gezeigt, dass ein 60,9 cm (24") Propeller bei 933 RPM den gleichen Schub erzeugt wie ein 15,2 cm Propeller bei 19.376 RPM -- bei **28,7 dBA weniger** Gesamtschalldruckpegel.

**Wichtigste Erkenntnis:** Die korrekte Strategie ist **großer Durchmesser mit moderatem Pitch** (P/D-Verhältnis 0.3-0.45), NICHT hohe Pitch-Winkel bei kleinem Durchmesser. Extrem hohe Pitch-Winkel (P/D > 0.5) führen im Hover zu Blattabriss (Stall), da der effektive Anstellwinkel ohne axiale Anströmung zu groß wird. Die D^4-Skalierung des Schubs ist der Hebel: Verdopplung des Durchmessers liefert 16x mehr Schub bei gleicher Drehzahl.

**Empfohlene Konfiguration:** Ein **Hexacopter** mit 22-24" oder 26-28" Propellern bietet gegenüber einem Quadcopter Redundanz und niedrigere Pro-Motor-Drehzahlen.

---

## 1. Akustische Vorteile großer, langsamer Rotoren

### Skalierungsgesetze

- Breitbandlärm skaliert mit der **4,5.--5,0. Potenz der Blattspitzen-Machzahl**
- Die Blattspitzengeschwindigkeit V_tip = pi * D * RPM / 60 ist der dominante Lärmparameter
- Durch Vergrößerung des Durchmessers und Reduktion der Drehzahl bei konstantem Schub werden dramatische Lärmminderungen erzielt

### Experimentelle Daten (gleicher Schub: 2,78 N)

| Propeller-Durchmesser | RPM    | A-gewichteter SPL (relativ) | Blade-Rate-Ton Reduktion |
|------------------------|--------|-----------------------------|--------------------------|
| 15,2 cm                | 19.376 | Referenz                    | Referenz                 |
| 20,3 cm                | ~15.000| -5,6 dBA                    | -13,7 dB                 |
| 30,5 cm                | 3.100  | -8,5 dBA                    | -33,3 dB                 |
| 60,9 cm                | 933    | -28,7 dBA                   | -76,6 dB                 |

Die Reduktion des Blade-Rate-Tons um 76,6 dB wird fast ausschließlich auf die geringere Blattspitzengeschwindigkeit zurückgeführt (nicht auf Frequenzverschiebung oder A-Bewertung).

### Psychoakustische Aspekte

- Menschen reagieren auf Drohnenlärm empfindlicher als auf Straßenverkehr gleicher Lautstärke (+5,6 dB Annoyance-Äquivalent)
- Das menschliche Gehör ist im Bereich 1-5 kHz am empfindlichsten -- genau dort, wo kleine Hochdrehzahl-Propeller Harmonische erzeugen
- Große, langsame Rotoren verschieben die Energie in tiefere Frequenzen (<500 Hz)
- **Achtung:** Tieffrequenter Lärm reist weiter, durchdringt Fenster und kann trotz niedrigerem dBA zu Beschwerden führen

---

## 2. Aerodynamik: Festpropeller mit hoher Steigung

### Grundprinzip

Propeller-Steigung (Pitch) = theoretischer Vorschub pro Umdrehung. Bei niedriger Drehzahl muss der Propeller pro Umdrehung einen größeren "Bissen" Luft nehmen, d.h. die Steigung muss hoch sein.

### Das Stall-Problem

- **Im Hover** (keine axiale Anströmung) arbeiten Hochsteigungspropeller bei sehr hohem Anstellwinkel
- Bei zu hoher Steigung **strömungsabriss (Stall)**: Auftriebsverlust, Effizienzeinbruch
- Faustregel: Pitch < 0,66 × Durchmesser für sicheren Betrieb (z.B. 26" x max ~17" Pitch)
- **Vortex Ring State (VRS):** Hohe Steigung erhöht das Risiko im Sinkflug -- gefährlicher Auftriebsverlust

### Optimales Pitch-zu-Durchmesser-Verhältnis

- Ein P/D-Verhältnis von ~1:1 (Pitch = Durchmesser) gilt theoretisch als maximal effizient für Cruise
- **Für Hover: P/D = 0.3-0.45 ist der praktische Sweet Spot** (z.B. 22x8 = P/D 0.36, 28x9.2 = P/D 0.33)
- P/D > 0.5 ist für Multirotor-Hover **problematisch** -- Teile des Blatts operieren im Stall-Bereich
- **Fazit: Durchmesser erhöhen statt Pitch erhöhen!**
- Verfügbare große Propeller mit moderatem Pitch für Multirotor:
  - **Xoar PJP-T-L 26x9.2** (26", Pitch 9.2, 63g, CF, <5000 RPM)
  - **Xoar PJP-T-L 28x8** (28", Pitch 8, 70g, CF, <4000 RPM)
  - **Xoar PJP-T-L 29x9.5** (29", Pitch 9.5, CF)
  - **RJXHOBBY 28x9.2** (28", Pitch 9.2, CF+Epoxy, max 27,5 kg Schub, 1500-3700 RPM)
  - **T-Motor G32x11** (32", Pitch 11, Foam Core CF, ~35 kg Schub, 1200-3000 RPM)
  - **Xoar PJP-T-L 40x10** (40", CF, <4000 RPM)

### Airfoil-Auswahl bei niedrigen Reynolds-Zahlen

Große Propeller bei niedriger Drehzahl operieren im Re-Bereich 10^4 bis 5×10^5:
- **NACA 4412**: Robuste Performance, hohe Wölbung, C_L,max ~1,35 bei Re=75.000
- **Eppler 1212**: Speziell für niedrige Re-Zahlen entwickelt
- Laminar-Separation-Bubbles sind ein Problem -- ggf. Grenzschicht-Tripping nötig

---

## 3. Schub bei niedrigen Drehzahlen

### Schubformel

Schub T proportional zu rho * n^2 * D^4 * C_T (mit n = Drehzahl [1/s], D = Durchmesser)

### Abschätzung für 1 kg Nutzlast

- **Gesamtgewicht (geschätzt):** ~2,5-3,0 kg (1 kg Nutzlast + 1,5-2,0 kg Drohne)
- **Schub pro Motor (Quadcopter):** 625-750 g bei Hover, ~1,25-1,5 kg Maximum (2:1 Schub/Gewicht-Verhältnis empfohlen)
- **Propellergröße vs. RPM für ~7,5 N Schub pro Motor:**

| Durchmesser | Geschätzte Hover-RPM | Blattspitzengeschwindigkeit |
|-------------|----------------------|-----------------------------|
| 18" (46 cm) | ~3000-4000           | ~72 m/s                     |
| 24" (61 cm) | ~1500-2500           | ~48 m/s                     |
| 28" (71 cm) | ~1200-2000           | ~44 m/s                     |
| 32" (81 cm) | ~900-1500            | ~40 m/s                     |

Zum Vergleich: Typische 5" FPV-Drohnen drehen bei ~20.000+ RPM, die Blattspitzengeschwindigkeit beträgt >130 m/s.

---

## 4. Stabilität und Steuerungsherausforderungen

### Kernproblem: Rotorträgheit

- Trägheitsmoment J steigt mit D^5 (oder stärker bei CF-Propellern)
- Größere Propeller reagieren langsamer auf Drehzahländerungen
- Reduzierte Steuerautorität = langsamere Korrektur von Störungen

### Flight-Controller-Kompatibilität

**ArduPilot/PX4 können große Propeller handhaben**, erfordern aber spezielle Tuning-Parameter:
- `MOT_THST_EXPO = 0.75` (für Propeller >20")
- `MOT_SPIN_ARM` / `MOT_SPIN_MIN` sorgfältig kalibrieren
- PID-Gains deutlich niedriger als bei kleinen Drohnen
- Meta-heuristische Optimierung (PSO, Genetic Algorithms) für PID-Tuning empfohlen

### ESC-Anforderungen

- **Active Braking (Damped Light)** ist essentiell für große Propeller
- Ohne aktives Bremsen: Motor verlangsamt sich zu langsam -> Instabilität bei Windböen
- DShot-Protokoll unterstützt Active Braking

### Fortgeschrittene Stabilisierungsstrategien

- **Model Predictive Control (MPC):** Prädiktive Regelung statt rein reaktiver PID
- **Tilt-Rotor:** Motoren kippen für seitliche Kräfte (entkoppelt Lage von Translation)
- **Control Moment Gyros (CMGs):** Kreiselbasierte Lagestabilisierung als Backup

---

## 5. Motor- und Antriebsauswahl

### Empfohlene Motorkategorien

Für 24-32" Propeller bei niedriger Drehzahl werden **Low-KV Outrunner** benötigt:

| Motor | KV | Max. Schub | Gewicht | Empf. Propeller |
|-------|----|------------|---------|-----------------|
| T-Motor U8 KV170 | 170 | ~5 kg | ~260g | 26-29" |
| T-Motor U10 Plus KV80 | 80 | 12 kg | 500g | 28-30" CF |
| T-Motor U15II KV80 | 80 | 36,5 kg | ~900g | 32-40" |

### Design-Merkmale für Low-KV Motoren

- **Outrunner-Konfiguration:** Außenläufer für mehr Drehmoment
- **Hohe Polzahl:** Mehr Pole = feinere Drehzahlkontrolle
- **Flat-Wire (Hairpin) Wicklung:** Höhere Kupferausnutzung, bessere Kühlung
- **Typischer KV-Bereich:** 80-400 KV für 24-40" Propeller

---

## 6. Existierende Referenzdesigns

### Industrielle/Kommerzielle Drohnen mit großen Propellern

- **DJI Agras T-Serie:** Agrar-Drohnen mit großen Propellern bei moderater Drehzahl
- **Freefly Alta X:** Kino-Drohne, 8 Rotoren, große Propeller für leisen Betrieb
- **Silent Sky (Korea):** Spezialisiert auf ultra-leise Propeller, 3D-gedruckte Formen, Carbon-Layup
- **Forschungsplattformen:** Diverse Universitäts-Quadcopter mit 24" Propellern für 1,36 kg Nutzlast nachgewiesen

### Noise-Reduction Innovationen

- **Toroidale Propeller (MIT Lincoln Lab):** Geschlossene Ringstruktur eliminiert Blattspitzenwirbel, +4,6% Effizienz, -4,9 dBA horizontal, -16,9 dBA longitudinal
- **Trailing-Edge Serrations:** 1,5-4 dB Reduktion (eulenflügelinspiriert)
- **Serration-Finlet Designs:** Bis zu 20 dB Reduktion im Labor
- **Gurney Flaps mit Serrations:** Erhöhen Auftriebsbeiwert bei gleichzeitiger Lärmreduktion

---

## 7. Empfohlene Spezifikationen für den Prototyp

### Option A: Hexacopter 22-24" (empfohlen für DFG-Projekt)

| Parameter | Empfehlung |
|-----------|------------|
| **Konfiguration** | Hexacopter (6 Motoren) -- Redundanz + niedrigere Pro-Motor-RPM |
| **Propeller** | 22-24" (56-61 cm), Pitch 8-9, 3-Blatt CF |
| **Konkret** | Xoar PJP-T-L 24x9 oder T-Motor G22x7.2 |
| **Hover-RPM** | ~3000-4000 RPM |
| **Motoren** | XOAR Titan T6015 (250KV) oder T-Motor U8 II KV150-190 |
| **ESC** | 40A+, DShot600, Active Braking, RPM-Telemetrie |
| **Flight Controller** | Pixhawk + ArduCopter |
| **Frame-Durchmesser** | ~1000-1100 mm (Motor-zu-Motor), ~1500 mm tip-to-tip |
| **Gesamtgewicht (mit Payload)** | ~4,5-5,5 kg |
| **Geschätzte Flugzeit** | 12-20 min (6S 5000-10000mAh) |
| **Blattspitzengeschw.** | ~38-47 m/s (Mach 0.11-0.14, sehr leise) |
| **BPF (3-Blatt)** | 150-200 Hz |

### Option B: Quadcopter 26-28" (kompakter, weniger Motoren)

| Parameter | Empfehlung |
|-----------|------------|
| **Konfiguration** | Quadcopter (X-Konfiguration) |
| **Propeller** | 26-28" (66-71 cm), Pitch 8-9.2, CF |
| **Konkret** | Xoar PJP-T-L 26x9.2 oder 28x8 |
| **Hover-RPM** | ~1500-2500 RPM |
| **Motoren** | T-Motor U8 KV170 oder äquivalent |
| **ESC** | 6S-12S, DShot600, Active Braking |
| **Flight Controller** | Pixhawk + ArduCopter |
| **Frame-Durchmesser** | ~800-900 mm (Motor-zu-Motor) |
| **Gesamtgewicht (mit Payload)** | ~3,0-3,5 kg |
| **Geschätzte Flugzeit** | 15-25 min (6S 5000mAh) |
| **Blattspitzengeschw.** | ~33-46 m/s (Mach 0.10-0.13, extrem leise) |
| **Vorteil** | Niedrigere RPM, kompakter, leichter |
| **Nachteil** | Keine Motor-Redundanz, trägere Steuerung |

### Option C: Hexacopter 28" (maximale Lärmreduktion)

| Parameter | Empfehlung |
|-----------|------------|
| **Propeller** | 28" (71 cm), Pitch 8-9.2, CF |
| **Hover-RPM** | ~2000-2500 RPM |
| **Motoren** | T-Motor U8 II KV100 |
| **Frame-Durchmesser** | ~1300 mm, ~1800 mm tip-to-tip |
| **Blattspitzengeschw.** | ~30-37 m/s |
| **Herausforderung** | Sehr groß, Transportproblematik, höheres Gewicht |

---

## 8. Risiken und Maßnahmen

| Risiko | Wahrscheinlichkeit | Gegenmaßnahme |
|--------|---------------------|----------------|
| Stall bei Hover (zu hohe Steigung) | Mittel | Pitch < 0,66×D; NACA 4412 Airfoil |
| Vortex Ring State im Sinkflug | Mittel | Sinkrate begrenzen; Piloten-Training |
| Unzureichende Steuerautorität | Mittel-Hoch | Active Braking ESC; niedrige PID-Gains; ggf. 6 oder 8 Rotoren |
| Zu hohes Gesamtgewicht | Niedrig | CF-Frame; 26" statt 32" Propeller |
| Windempfindlichkeit | Hoch | Nur bei schwachem Wind fliegen (<3 m/s für Messungen) |

---

## 9. Quellen

### Wissenschaftliche Paper

1. ["The reduction of quadcopter propeller noise"](https://www.researchgate.net/publication/335308628_The_reduction_of_quadcopter_propeller_noise) -- Kernpaper: Nachweis der 28,7 dBA Reduktion durch Durchmesservergrößerung
2. ["Small-Scale Rotor Aeroacoustics for Drone Propulsion: A Review"](https://www.mdpi.com/2311-5521/7/8/279) -- Umfassende Review zu Lärmquellen und Strategien
3. ["Characterization of the low-noise drone propeller with serrated Gurney flap"](https://www.frontiersin.org/journals/aerospace-engineering/articles/10.3389/fpace.2022.1004828/full) -- Gurney-Flap + Serrations
4. ["Propeller Design Requirements for Quadcopters"](https://commons.erau.edu/cgi/viewcontent.cgi?article=1840&context=publication) -- Embry-Riddle, Propeller-Auslegung
5. ["Bioinspired Drone Rotors for Reduced Aeroacoustic Noise"](https://arxiv.org/html/2501.01577v1) -- Bio-inspirierte Designs (Eulenflügel)
6. ["Drone Noise Reduction Using Serration-Finlet Blade Design"](https://www.mdpi.com/2071-1050/17/8/3451) -- Bis zu 20 dB Reduktion
7. ["Low-Noise Propeller Design with Enlarged Blade Area"](https://www.fujipress.jp/jrm/rb/robot003700040799/) -- 8,1 dBA Reduktion durch vergrößerte Blattfläche
8. ["Large-Sized Multirotor Design: Accurate Modeling"](https://www.mdpi.com/2504-446X/7/10/614) -- Modellierung großer Multirotorsysteme
9. ["Optimization of PID Controller for Quadcopter Using Meta-Heuristic Algorithms"](https://www.mdpi.com/2076-3417/11/14/6492) -- PID-Tuning für große Systeme
10. ["Investigation of Metrics for Assessing Human Response to Drone Noise"](https://pmc.ncbi.nlm.nih.gov/articles/PMC8954658/) -- NASA-Psychoakustik-Studie
11. ["Experimental Study of Quadcopter Acoustics"](https://www.bu.edu/ufmal/files/2016/07/aiaa-2016-2873.pdf) -- Boston University, Anechoic-Chamber Tests
12. ["Enhancing drone propellers: BEM-based pitch optimization"](https://www.sciencedirect.com/science/article/pii/S1270963825004031) -- Blade Element Momentum Pitch-Optimierung
13. ["Flow characteristics over NACA4412 airfoil at low Reynolds number"](https://www.epj-conferences.org/articles/epjconf/pdf/2016/09/epjconf_efm2016_02029.pdf) -- Airfoil-Daten

### Komponenten und Hersteller

14. [Xoar PJP-T-L Carbon Fiber Propellers](https://www.xoarintl.com/multicopter-propellers/precision-pair/PJP-T-L-Precision-Pair-Multicopter-Carbon-Fiber-Propeller-Low-Kv-Motor/) -- 26-40" CF-Propeller
15. [T-Motor U-Series Motors](https://shop.tmotor.com/collections/multirotor-motor-u-series) -- Low-KV Motoren für Heavy-Lift
16. [T-Motor Motor-Propeller Matching Guide](https://shop.tmotor.com/blog/drone-motor-propeller-matching-guide) -- Auslegungsleitfaden
17. [Large Drone Propellers Guide (LIG Power)](https://www.ligpower.com/blog/large-drone-propellers.html) -- Marktübersicht

### Flight Controller und Tuning

18. [ArduPilot Advanced Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)
19. [PX4 Multicopter PID Tuning Guide](https://docs.px4.io/main/en/config_mc/pid_tuning_guide_multicopter)
20. [ArduCopter Methodic Configurator](https://ardupilot.github.io/MethodicConfigurator/TUNING_GUIDE_ArduCopter.html)

### Innovative Ansätze

21. [MIT Lincoln Lab Toroidal Propeller](https://www.ll.mit.edu/partner-us/available-technologies/toroidal-propeller-0) -- Ring-Propeller ohne Blattspitzen
22. [Silent Sky -- Custom Low-Noise Propellers](https://www.silentsky.co.kr/)
23. [How to Reduce Drone Noise (Tyto Robotics)](https://www.tytorobotics.com/blogs/articles/how-to-reduce-drone-noise)

---

*Erstellt: 2026-04-07 | Kontext: DFG-Projekt "Flying Measurement Microphone" (MartyMicFly)*
*Quellen auch im NotebookLM hinterlegt: https://notebooklm.google.com/notebook/6affcef9-a9c3-49ac-ae6c-c9783e84fc88*
