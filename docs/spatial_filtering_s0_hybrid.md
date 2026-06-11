# Räumliche Filterung — konzeptuelles Vorgehen

Dieses Dokument erklärt, *wie* die Drohnen-Eigengeräusche in den
S0-Hybrid-Vergleichsergebnissen räumlich aus dem Mikrofonarray-Signal
herausgefiltert wurden, *ohne* in den Code zu schauen.

---

## 1. Ausgangslage und Ziel

Ein Mikrofonarray ist an einer Multikopter-Drohne montiert und soll
während des Flugs eine *externe* Schallquelle aufzeichnen, die unter der
Drohne liegt (Zielpunkt etwa 1.5 m unterhalb der Array-Ebene). Das
Problem: die Drohne ist gleichzeitig die lauteste Schallquelle in
unmittelbarer Nähe des Arrays — vier Rotoren plus deren breitbandiger
Verwirbelungslärm zwischen den Rotoren.

Wir wollen aus dem gemessenen Mehrkanalsignal das herausziehen, was *aus
der Richtung der externen Quelle* kommt, und alles unterdrücken, was
*aus der Richtung der Drohne selbst* stammt.

Drei Eingabe-Szenarien werden auf dieselbe Pipeline geworfen, damit die
Filterung selbst validiert werden kann:

| Szenario             | Was steckt im Signal                 | Erwartung an die Filterung                              |
|----------------------|--------------------------------------|---------------------------------------------------------|
| **ext_only**         | nur die externe Quelle               | Filter darf praktisch *nichts* abtragen                 |
| **mixed_with_notch** | Drohne + extern, BPF-Töne vorgekämmt | Filter muss noch das *breitbandige* Drohnenrauschen weg |
| **mixed_no_notch**   | Drohne + extern, ungekämmt           | Filter muss BPF-Töne **und** Breitband alleine schaffen |

Verglichen werden zwei klassische Entfaltungsverfahren der akustischen
Bildgebung — **CLEAN-SC** und **orthogonale Beamforming-Entfaltung
(Orth)** — wie sie im AP2-A des Projektantrags benannt sind.

---

## 2. Die zwei Bausteine

### 2.1 Optional: Notch-Vorfilterung (nur Szenario „mit Notch“)

Die Rotoren erzeugen ein scharfes, tonales Linienspektrum:
Blattfolgefrequenz (BPF) und deren Harmonische. Diese Töne sind
schmalbandig und sehr energiereich — sie liegen meist *außerhalb* des
Bands, das man später beobachten will, aber „lecken“ über
PSD-Schätzfehler in benachbarte Bänder.

Lösung: ein adaptiver **Kerbfilter** (Notch) entfernt vor der
räumlichen Verarbeitung die ersten 13 Harmonischen jeder
Motor-Blattfolgefrequenz. Die BPF-Spur kommt aus der mitgeloggten
Motor-Telemetrie (ESC-RPM), nicht aus dem Audio selbst, also ist die
Kerbe sehr eng und greift keine Breitband-Energie an.

Übrig bleibt nach dem Notch ein Signal, in dem die Drohne fast nur noch
*breitbandig* rauscht — und genau das ist die Domäne, in der räumliche
Filterung wirken kann.

### 2.2 Das Herzstück: räumliche Filterung über eine Schallquellen-Karte

Räumliche Filterung läuft hier nicht direkt im Zeitsignal, sondern über
den Umweg einer **Schallquellen-Karte (Source Map)**. Vier konzeptuelle
Schritte:

1. **CSM messen.** Aus dem Mehrkanalsignal wird pro Frequenzbin die
   Kreuzleistungsdichte-Matrix (Cross-Spectral Matrix, *CSM*)
   geschätzt. Sie beschreibt vollständig, wie das Schallfeld auf das
   Array trifft.
2. **Suchraum aufspannen.** Es wird eine endliche Menge von Kandidat-
   Quellpunkten definiert (siehe Abschnitt 3 — *das* ist die zentrale
   Designentscheidung).
3. **Energie verteilen (Deconvolution).** Ein Algorithmus zerlegt die
   gemessene CSM in einen Energie-Anteil pro Kandidatpunkt.
   Vorstellung: „wenn ich die Energie der CSM auf diese Kandidatpunkte
   so verteilen müsste, dass die akustische Abbildung am besten passt
   — wie viel landet wo?“ Heraus kommt eine **Power-Karte** über den
   Suchraum: pro Frequenz und pro Kandidat ein Pegel.
4. **Subtrahieren.** Aus der Teilmenge der Kandidaten, die *die Drohne
   selbst* repräsentieren, wird eine synthetische „Drohnen-CSM“
   zurückgebaut und von der gemessenen CSM abgezogen. Was übrig bleibt,
   ist die **Residual-CSM** — formal das Schallfeld minus
   Drohnenanteil.

Aus dieser Rest-CSM wird zum Schluss ein Beamformer auf den externen
Zielpunkt gerichtet — das ergibt die gefilterte
Leistungsdichteschätzung der externen Quelle.

Der Trick: jede Karten-Zelle hat eine **Richtung relativ zum Array**.
Wenn die Karte „weiß“, welche Zellen zur Drohne gehören und welche zur
Außenwelt, ist die Trennung im wesentlichen ein Buchhaltungs-Schritt.

---

## 3. Das Hybrid-Gitter — warum es so aussieht, wie es aussieht

Die räumliche Filterung steht und fällt damit, *welche* Kandidatpunkte
man der Karte überhaupt anbietet. Naive Wahl wäre ein dichtes 3D-Gitter
um den Drohnenkörper — sehr teuer und nicht nötig, wenn man weiß,
*woher* der Lärm typischerweise kommt.

Für diese Studie wird ein **hybrider Suchraum** verwendet, der zwei
Ideen kombiniert:

### 3.1 Eine DOA-Hemisphäre für die Außenwelt

Die externe Quelle liegt fern genug (≈ 1.5 m), dass nur ihre
**Einfallsrichtung** zählt, nicht ihr exakter Abstand. Daher wird die
halbe Sphäre *unterhalb* des Arrays mit Kandidatpunkten überzogen —
gleichmäßig in Azimut und Elevation alle 5°. Jeder dieser Punkte
codiert eine Richtung „aus der unten irgendetwas Schall sendet“.

Wichtige Eigenschaft: die Hemisphäre filtert nach **Richtung** (Direction
of Arrival, DOA), nicht nach Position. Die externe Quelle muss nicht
exakt auf der Hemisphäre liegen — solange ihre Einfallsrichtung mit
einem Sphärenpunkt übereinstimmt, kann ihr Beitrag dort gebunkert werden.

### 3.2 Diskrete Nahfeld-Atome für die Drohne

Die Drohne selbst ist *kein* Fernfeld-Problem. Sie liegt direkt am
Array — wenige Zentimeter weg. Eine reine Richtungs-Suche würde die
Drohnenenergie auf einen breiten Streifen der Hemisphäre verschmieren
(grob den Äquatorgürtel) und damit die Trennung von wirklich
horizontalen externen Richtungen unmöglich machen.

Deshalb werden dem Suchraum **acht zusätzliche Punkt-Kandidaten** in
der Array-Ebene beigemischt:

- die **vier Rotorpositionen** — dort sitzen die tonalen und tip-vortex
  Quellen,
- die **vier Mittelpunkte zwischen benachbarten Rotoren** — dort
  konzentriert sich der breitbandige Downwash, weil sich die
  Strömungen der Nachbarrotoren in der Lücke begegnen.

Diese acht Atome sind *im Nahfeld* des Arrays angesiedelt; sie haben
keine sinnvolle „Richtung“, sondern eine **Position**. Sie sind dazu
da, der Deconvolution einen Platz zu geben, an dem sie Drohnenenergie
unterbringen kann, ohne sie auf der Hemisphäre zu verteilen.

### 3.3 Warum „hybrid“

Der Suchraum besteht also aus zwei physikalisch unterschiedlich
motivierten Teilen:

```
Suchraum  =  DOA-Hemisphäre (Richtung, fern)  +  8 Nahfeld-Atome (Position, nah)
             └─ behalten für externe Quelle  ──┘    └─ subtrahieren: Drohne ──┘
```

Genau diese Trennung in „behalten“ vs. „subtrahieren“ ist die
räumliche Filterung. Welche Atome zu welcher Klasse gehören, ist
geometrisch fest verdrahtet — keine datengetriebene Klassifikation.

---

## 4. Die zwei verglichenen Entfaltungsverfahren

Beide Verfahren bekommen denselben Suchraum, dieselbe gemessene CSM und
dieselben Mikrofonpositionen. Sie unterscheiden sich nur darin, *wie*
sie die CSM-Energie auf die Atome verteilen:

- **CLEAN-SC** (CLEAN with Source Coherence) — *iterativ*. Findet
  wiederholt den stärksten Beitrag in der gerade gültigen „Dirty Map“
  des Beamformers, weist ihm einen Atom-Pegel zu, baut den
  zugehörigen kohärenten Schallfeld-Beitrag aus der CSM heraus und
  fängt von vorne an. Konvergiert nach 100 Iterationen mit
  Dämpfungsfaktor 0.6.
- **Orth** (Orthogonale Entfaltung) — *nicht-iterativ*. Zerlegt die
  CSM in ihre Eigenmoden, nimmt die acht stärksten und ordnet jede
  Eigenmode demjenigen Suchraum-Atom zu, an dem ihr individueller
  Beamformer-Output sein Maximum hat. Tendiert auf kleinen Arrays zu
  stabileren Ergebnissen als CLEAN-SC, weil keine Source-Coherence-
  Annahmen reinrutschen.

Beide Methoden gehören zu den drei in AP2-A explizit genannten
Verfahren (CLEAN-SC, CLEAN-T, orthogonale Entfaltung); CLEAN-T ist hier
nicht Teil des Vergleichs.

**Pegel-Konsistenz.** Beamforming-Tools liefern Power-Werte bis auf
einen geometrieabhängigen Array-Gain-Faktor. Damit die rekonstruierte
„Drohnen-CSM“ in der gleichen physikalischen Einheit ist wie die
gemessene CSM, wird die Map pro Frequenz so reskaliert, dass die
Summe ihrer Pegel der Spur der CSM entspricht. Erst danach ist die
Subtraktion energetisch sauber.

---

## 5. Was am Ende rauskommt

Pro Szenario × Algorithmus (insgesamt sechs Kombinationen) entsteht:

1. Eine **Schallquellen-Karte** über den Hybrid-Suchraum — sichtbar als
   DOA-Heatmap (Sphäre) und als Liste von Pegelwerten an den
   Nahfeld-Atomen.
2. Eine **Residual-CSM**: gemessene CSM minus synthetisierte
   Drohnen-CSM. Sie wird zusätzlich auf den Bereich positiv-semidefiniter
   Matrizen projiziert (Eigenwert-Clipping bei Null), damit
   unvermeidliche Überschwinger der Subtraktion an einzelnen Frequenzen
   nicht zu negativen Pegelschätzungen führen.
3. Eine **gefilterte PSD** am Zielpunkt — Beamformer-Output der
   Residual-CSM, gerichtet auf `(0, 0, −1.5) m`. Das ist *das* Ergebnis
   der Studie: was die externe Quelle nach der räumlichen Filterung
   noch beiträgt.

Die Auswertung vergleicht diese PSDs gegen die Ground Truth (das saubere
externe Signal, das in der Simulation bekannt ist), berechnet den
mittleren absoluten Fehler in vier Bändern (50–200, 200–500, 500–2000,
2000–6000 Hz) und stellt CLEAN-SC und Orth nebeneinander.

---

## 6. Ein-Satz-Zusammenfassung

> Die Mikrofonarray-CSM wird über ein hybrides Suchraster aus
> Fernfeld-Richtungen (untere Hemisphäre) **plus** acht
> Drohnen-Nahfeldpunkten (Rotoren + Rotor-Lücken) entfaltet; die den
> Nahfeldpunkten zugeordnete Energie wird als „Drohnenanteil“
> rekonstruiert und von der gemessenen CSM abgezogen, bevor zum
> Schluss auf den externen Zielpunkt gehört wird.
