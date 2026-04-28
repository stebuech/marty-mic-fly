Seite 1 von 21

Beschreibung des Vorhabens

Ennes Sarradj und Gert Herold, Berlin

Fliegendes Messmikrofon

1 Ausgangslage

Stand der Forschung und eigene Vorarbeiten

Einführung

Die Messung von Schalldruckpegeln ist eine sehr häuﬁge Messaufgabe im Zusammenhang mit der Cha-
rakterisierung von Schallemissionen und -immissionen. Dabei werden nach dem Stand der Technik Schall-
pegelmesser eingesetzt, die aus einem Messmikrofon (fast ausschließlich als Elektretmikrofon ausgeführt),
sowie der zugehörigen Signalerfassung und Signalverarbeitung mit Anzeige bestehen. Messgrößen sind
dabei der Schalldruckpegel (Effektivwert des Schalldrucks als Pegelgröße) oder davon abgeleitete Größen
wie zeit- oder frequenzbewertete Schalldruckpegel und das zugehörige Spektrum (Leistungsdichtespek-
trum). Die Messung des Schalldruckpegels ist Grundlage für weitere Messverfahren, bei denen andere
Messgrößen wie der Schallleistungspegel oder das Schalldämmmaß indirekt bestimmt werden. Dabei wird
gefordert, den Schalldruckpegel an einem oder mehreren festgelegten Messorten zu ermitteln.

Bei der Messung des Schalldruckpegels im Freien sind diese Messorte nicht immer einfach mit dem
Mikrofon zu erreichen. Das gilt insbesondere dann, wenn die Messorte weit über dem Boden oder oberhalb
einer größeren Schallquelle liegen. Dann ist ein erheblicher und oft nicht realisierbarer Aufwand durch
den Einsatz großer Stative, Gerüste oder Hebevorrichtungen nötig, um diese Orte zu erreichen und die
Messung durchzuführen (siehe Bild 1). Deshalb liegt der Gedanke nahe, für die Messung eine ﬂiegende
Plattform einzusetzen, die ein Mikrofon zur Messung des Schalldruckpegels mit geringerem Aufwand an
die einzelnen Messorte bringen kann.

Bild 1: Beispiele für Schalldruckpegelmessungen an schwer erreichbaren Messorten.

Stand der Forschung zu Mikrofonen auf ﬂiegenden Plattformen

Über die Anwendung von Mikrofonen auf ﬂiegenden Plattformen wird seit etwa 20 Jahren berichtet. Es exis-
tieren vielfältige Einsatzmöglichkeiten, wobei die Lokalisierung von Schallquellen eine der dominierenden
Anwendungen darstellt [11]. Diese umfasst unter anderem die Lokalisierung von Schallquellen in Such-
und Rettungssituationen [12, 13, 14], Sprecherlokalisierung [15], sowie militärische Zwecke [16].

Ein weiteres relevantes Anwendungsgebiet ist die Fehlerdetektion und Inspektion von Betriebszuständen
bei Windturbinen [17]. Es gibt erste Vorstellungen darüber, wie der Einsatz ﬂiegender Plattformen mit Mikro-
fonen dazu beitragen kann, den Zustand von Windkraftanlagen zu überwachen und potenzielle Probleme

Seite 2 von 21

frühzeitig zu erkennen. Die räumliche Verfolgung anderer Flugobjekte [18] ist eine weitere Anwendung, die
zur Kollisionsvermeidung [19] und Navigation genutzt wird. Eine umfassende Übersicht über die vielfältigen
Einsatzgebiete von Drohnen mit akustischer Messtechnik ﬁndet sich in [20]. Sofern Messungen durchge-
führt werden, ist die Messgröße bislang stets die Einfallsrichtung (Einfallswinkel) einer Schallquelle.

Mikrofone wurden bislang bei verschiedenen Typen und Konﬁgurationen ﬂiegender Plattformen einge-
setzt, die je nach den speziﬁschen Anforderungen und Aufgaben ausgewählt wurden. Unter den am häu-
ﬁgsten verwendeten ﬂiegenden Plattformen ﬁnden sich Quadrocopter [21, 11], Hexacopter [12] sowie Oc-
tocopter [22]. Diese Plattformen zeichnen sich durch ihre Flugstabilität und Wendigkeit aus, was sie ideal
für Aufgaben wie die Schallquellenlokalisierung macht. Unbemannte Flugzeuge sind eine weitere Katego-
rie von ﬂiegenden Plattformen für akustische Messungen [23]. Gleitﬂugzeuge [13] und Wing-Airframe sind
eine spezialisierte Variante und bieten besondere Vorteile mit kombinierten Flug- und Schwebefähigkeiten.
Die technische Realisierung von oft mehreren Mikrofonen (Mikrofonarray) als Drohnen-Nutzlast ist ein
Schlüsselaspekt, um akustische Messungen in verschiedenen Szenarien durchzuführen. Eine der ein-
fachsten Möglichkeiten zur Instrumentalisierung besteht darin, Einzelmikrofone auf den Drohnen zu plat-
zieren [17]. Im Vergleich zu Sensorarrays sind hier jedoch die Möglichkeiten der Signalnachbearbeitung
beschränkt. Eine häuﬁg realisierte Variante von Sensorarrays ist die Verwendung von zirkulären Mikrofon-
anordnungen [13, 11, 24]. Darüber hinaus gibt es Untersuchungen zu sphärischen [12] und T-förmigen [21]
Anordnungen. Die Platzierung der Mikrofone auf der Drohne kann variieren. Mikrofone können sowohl
unterhalb der Drohne [25] als auch oberhalb der Drohne [24, 15] positioniert werden. Eine vorgelagerte
Anordnung [21, 11, 22] bietet die Möglichkeit, durch erhöhten Abstand von den Rotoren die Einwirkung des
Eigengeräuschs der Drohne zu vermindern. Alternativ kann auch eine die Drohne umschließende Anord-
nung [14] realisiert werden. Eine umfassende Übersicht über die Hardware zur Umsetzung von Mikrofonar-
rays auf Drohnen ﬁndet sich in Tabelle 1 in der Arbeit von Clayton [11]. Die Anzahl der Mikrofone, die in
den Arrays auf Drohnen integriert werden, variiert von einzelnen Mikrofonen [17] bis hin zu umfangreichen
Arrays mit bis zu 32 Sensoren [26].

Um die Qualität der akustischen Messungen auf ﬂiegenden Plattformen zu gewährleisten, muss fast im-
mer eine Unterdrückung oder Trennung der Eigengeräusche der Drohne realisiert werden. Eine effektive
Methode zur Eigengeräuschunterdrückung ist die Abschaltung des Antriebs während des Gleitﬂugs [13].
Dieser Ansatz reduziert das vom Antrieb erzeugte Geräusch und ermöglicht präzisere Messungen in Flug-
phasen, in denen der Antrieb nicht aktiv ist. Filterung im Zeitbereich oder Zeit-Frequenz-Filterung sind
gängige Techniken zur Eigengeräuschunterdrückung und werden in mehreren Studien behandelt. Die Im-
plementierung eines Wiener-Filters [27, 11, 21] ist ein Beispiel für ein solches Filter, das dazu dient,
das Eigengeräusch gegenüber dem zu messenden Schall zu reduzieren. Weitere Methoden zur Eigen-
geräuschunterdrückung bieten der Einsatz von Notch-Filtern [19] von Deep-Learning-Techniken [15] und
Blind-Source Separation [28].

Zur Erprobung und Validierung von akustischen Messverfahren auf ﬂiegenden Plattformen sind bereits
einige Datensätze, so der DRone EGonoise and localizatiON (DREGON) [25] (Messung in einem Innen-
raum) und Audio-Visual dataset from a Quadcopter (AVQ) [15] (Messung im Freien) verfügbar.

Im Hinblick auf die Anwendung zur Messung des Schalldruckpegels sind insbesondere folgende offene

Fragen festzustellen:

• Ein Großteil der Arbeiten, die akustische Systeme auf ﬂiegenden Plattformen verwenden, zielt dar-
auf ab, qualitative Informationen zur Identiﬁkation und Klassiﬁkation von Schallquellen zu gewinnen.
(Quantitative) Messungen sind bislang nur für die Richtung, nicht jedoch für den Schalldruckpegel
bekannt und daher eine offene Frage.

• Bisher wurde die Möglichkeit zur Messung der Schallleistung mittels Mikrofonen auf Drohnen nicht
untersucht. Dies könnte in verschiedenen Anwendungsbereichen, wie der Umweltüberwachung oder
der Überwachung von Industrieanlagen, von großem Interesse sein.

• Es existiert keine allgemein anerkannte und einheitliche Lösung zur Implementierung von Mikrofonar-
rays auf ﬂiegenden Plattformen. Dies betrifft die Auswahl der Mikrofonanordnung, die eingesetzte
Messtechnik und die Positionierung der Mikrofone auf der Plattform.

• In [11] wird auf die Vorteile der vorgelagerten Positionierung des Mikrofonarrays hinsichtlich der Un-
terdrückung von Drohnengeräuschen hingewiesen. Allerdings kann dies gleichzeitig die Manövrier-

Seite 3 von 21

barkeit der Plattform beeinträchtigen. Daher besteht ein Bedarf an weiteren Untersuchungen zur
vorteilhaften Positionierung von Mikrofonarrays auf ﬂiegenden Plattformen.

Vorarbeiten der Antragsteller

Schallemission von Multicoptern Bei den Antragstellern liegen umfangreiche Vorerfahrungen mit der
Vermessung der Schallemissionen von Multicoptern vor. Durchgeführte Indoor- und Outdoor-Experimente
konzentrieren sich dabei auf frei ﬂiegende Drohnen in verschiedenen Flugzuständen wie Steig- und Sink-
ﬂug, Schwebﬂug sowie Geradeausﬂug. Bild 2 zeigt zwei Messaufbauten, mit denen entsprechende Multicopter-
Messungen durchgeführt wurden. Mithilfe von Mikrofonarrays wurden jeweils Schalldruck-Zeitdaten aufge-
nommen, die auf unterschiedliche Art weiterverarbeitet werden können.

Bild 2: Akustische Multicopter-Vorbeiﬂugmessungen im Freifeld (links) und im reﬂexionsarmen Raum (rechts).

Zum einen ist es möglich, mithilfe geeigneter Mikrofonarray-Algorithmen zu detektieren, an welcher Po-
sition im Raum sich dominante Schallquellen beﬁnden. Dies ist auch für die Verfolgung bewegter Schall-
quellen anwendbar. Bereits veröffentlichte Untersuchungen umfassen die Rekonstruktion des Flugpfades
von Multicoptern anhand gemessener Schalldruck-Zeitdaten [29, 10].

Darüber hinaus ist es ebenfalls möglich, die Signale gleichzeitig abstrahlender Schallquellen zu tren-
nen, wenn sie sich an unterschiedlichen Raumpositionen beﬁnden. Entsprechende Experimente wurden
mit unabhängig ﬂiegenden Quadcoptern durchgeführt [4], deren akustische Signaturen mithilfe geeigneter
Auswertealgorithmen getrennt wurden (siehe Bild 3).

Bild 3: Trennung der akustischen Signale zweier gleichzeitig über ein Mikrofonarray ﬂiegender Quadcopter. Links:
Ungeﬁltertes Spektrogramm, welches beide Signale enthält. Mitte/rechts: Entsprechend der individuellen Flugpfade
isolierte Spektrogramme [4].

Messaufbauten wie in Bild 2, die den Flugpfad der Drohnen ganz oder teilweise umschließen, erlauben
eine umfassende Beschreibung der akustischen Abstrahleigenschaften von Multicoptern im Flug [30, 7].

Messgrößen, die aus solchen Messungen gewonnen wurden, umfassen die Schallleistung und die Richt-
charakteristik (siehe Bild 4).

Seite 4 von 21

Bild 4: Richtcharakteristiken bei 4 kHz von zwei unterschiedlichen Quadcopter-Drohnen. Links: Indoor-Messung [7],
rechts: Outdoor-Messung [9].

Aktuelle Untersuchungen der Antragsteller auf diesem Gebiet befassen sich mit Fragen in Bezug auf
Messunsicherheiten, die in derartigen Experimenten auftreten. Gerade bei kurzer Messdauer stehen oft kei-
ne ausreichenden Daten zur Verfügung, um statistisch signiﬁkante Ergebnisse für Größen wie den Schall-
druckpegel abzuleiten [31], sodass hierfür wiederholte Messungen notwendig sind. Die Abweichungen zwi-
schen den aus Mikrofonarray-Messungen abgeleiteten Positionen eines vorbeiﬂiegenden Quadcopters und
seiner tatsächlichen Position wurden in Abhängigkeit von Mess- und Auswerteparametern untersucht [32,
10]. Diese Untersuchungen haben unter anderem verdeutlicht, dass die zeitlich adaptive Anpassung der
Auswerteparameter vorteilhaft sein kann.

Mikrofonarraymesstechnik Die Entwicklung von Mikrofonarrays und der zugehörigen Methoden für mess-
technische Zwecke ist einer der Hauptforschungsschwerpunkte beim Antragssteller [33]. Neben der Ent-
wicklung und Erweiterung neuer Mikrofonarraymethoden [1, 34, 35, 36, 8, 37] und der Optimierung der
Mikrofonanordnung [38] steht der Einsatz zur Charakterisierung von Schallquellen im Mittelpunkt zahlrei-
cher Arbeiten. Dabei werden Schalldruckpegelbeiträge, Spektren und Richtcharakteristiken gemessen.

Diese Forschung beinhaltet auch die Messung an Schallquellen in Bewegung, wie beispielsweise Vögel,
Kraftfahrzeuge und Eisenbahnzüge [39, 40, 41, 42]. Insbesondere wird auch die Messung der Schalldruckpegel-
Beiträge und Richtcharakteristiken einzelner Teile von rotierenden Schallquellen, die akustische Drehzahl-
messung von rotierenden schallerzeugenden Phänomenen [43, 5, 44, 45] und die Entwicklung von Metho-
den zum Trennen der Anteile rotierend bewegter und sich nicht bewegenden Schallquellen [46] betrachtet.
Bild 5 zeigt einige Anwendungsbeispiele.

Betrachtung der Messunsicherheit von mikrofonarraybasierten Messverfahren Um die Unsicherheit
der Messergebnisse zu untersuchen, die durch die bei jedem Arrayverfahren verwendeten Signalverarbei-
tungsalgorithmen und weiteren mathematischen Schätzverfahren entstehen, wurden in verschiedenen Pro-
jekten Monte-Carlo-Verfahren eingesetzt [3, 49]. Diese ermöglichen, die Unsicherheit Ergebnisse zu mo-
dellieren und für die jeweilige Messsituation beste Verfahren auszuwählen. Vergleichbare Untersuchungen
wurden auch zur Verringerung der Messunsicherheiten bei der Anwendung auf rotierende Schallquellen
durchgeführt [50, 51].

Acoular Bei den Antragstellern wird seit längerem die quelloffene Softwarebibliothek Acoular [2] gepﬂegt.
Der Hauptfokus von Acoular liegt auf Anwendungen von Arrays von Mikrofonen (oder anderen Senso-
ren) in der akustischen Messtechnik. Zahlreiche Arraymethoden sowie Funktionen zur Datenerfassung,
-verwaltung und -ausgabe sind implementiert. Durch verschiedene Maßnahmen zur Qualitätssicherung
gestattet die Software eine weitgehende Reproduzierbarkeit von Ergebnissen auch über unterschiedliche
Versionen und Plattformen hinweg. Acoular kann auf allen gängigen Betriebssystemen und Rechenplattfor-
men verwendet werden und bietet so eine einfache Möglichkeit, neu entwickelte Algorithmen mit geringem
Aufwand zu testen. Acoular ermöglicht auch die Erzeugung synthetischer Messdaten für Arrayanwendun-
gen, um die Eigenschaften von Algorithmen zur Messdatenverarbeitung kontrolliert zu untersuchen. Acou-
lar wird durch zwei weitere Softwarepakete ergänzt: SpectAcoular [52] zur Erstellung von Applikationen

xyz0π2π3π22π0π2πϕθLp/dBmsm,4000Hz50556090°±180°-90°0°90°0°90°180°Lp/dB4000 Hz, drone 3, 3 fly-bys (diff. dir., avg.)657075800°45°135°-135°-45°758085Lp/dB=90°Seite 5 von 21

(a) Überﬂug eines Turmfalken (falco tinninculus) [39]

(b) Dreidimensionale Kartierung eines Testaufbaus
für einen Hochgeschwindigkeits-Pantographen
in Originalgröße im aeroakustischen Windkanal
[47]

(c) Axialventilator bei 1050 U/min [6]

(d) Zugvorbeifahrt bei 200 km/h [41]

Bild 5: Mit Acoular [48] erstellte Karten der Beiträge einzelner Quellorte zum Gesamtschalldruckpegel am Messort.

und GUI Anwendungen und Acoupipe [53] zur efﬁzienten und reproduzierbaren Erzeugung großer Mengen
an synthetischen Datensätzen.

2 Ziele und Arbeitsprogramm

2.1 Voraussichtliche Gesamtdauer des Projekts

Für das hier beantragte Arbeitsprogramm ist eine Dauer von drei Jahren vorgesehen.

2.2 Ziele

Ziel des Vorhabens ist, eine Methode zur Messung des Schalldruckpegels auf der Basis einer Multicopter-
Drohne mit Mikrofonen zu etablieren. Die Methode soll einfach realisierbare Messungen auch an solchen
Orten ermöglichen, die vom Boden aus sonst nur mit großem Aufwand erreichbar sind. Die für das Mess-
verfahren erforderliche Technik und Signalverarbeitung soll entwickelt und beispielhaft in einem Demons-
trationsexperiment umgesetzt werden.

Dem Vorhaben liegen die folgenden Arbeitshypothesen zugrunde:

• Der von der Drohne selbst, insbesondere von den Rotoren ausgehende Schall (Eigengeräusch) über-
lagert den zu messenden Schall und beeinﬂusst ohne weitere Maßnahmen das Messergebnis in nicht
akzeptablem Umfang. Auf der Drohne beﬁndliche Mikrofone können darüber hinaus auch Einﬂüssen
durch die von den Rotoren verursachte Strömung („Pseudoschall“, Windgeräusche) ausgesetzt sein,
die das Messergebnis beeinﬂussen.

-1.0-0.50.00.51.0500 Hz-0246630 Hz-0246800 Hz-5-3-111 kHz-5-3-111.25 kHz-6-4-20-1.0-0.50.00.51.01.6 kHz-5-3-112 kHz-7-5-3-12.5 kHz-6-4-2-03.15 kHz-10-8-6-44 kHz-11-9-7-5-1-0.500.51-1.0-0.50.00.51.05 kHz-9-7-5-3-1-0.500.516.3 kHz-9-7-6-4-1-0.500.518 kHz-9-7-5-3-1-0.500.5110 kHz-6-4-20-1-0.500.51bird silhouette−0.4−0.20.00.20.4−0.4−0.20.00.20.44000Hz1618202224262830Lp/dBx/my/m  Third-octave, f_c = 1250 Hz, v = 200.44 km/h, direction = right-leftSeite 6 von 21

• Werden mehrere Mikrofone (ein Mikrofonarray) gleichzeitig für die Messung eingesetzt, erlauben die
damit erfassten Informationen im Vergleich zur Messung mit einem Mikrofon eine bessere Trennung
des zu messenden Schalls und des Eigengeräuschs.

• Das Eigengeräusch der Drohne besteht aus starken tonalen Anteilen bei den Blattfolgefrequenzen
der Rotoren und deren Vielfachen, sowie aus einem leistungsschwächeren breitbandigen Geräusch.
Die zur Trennung des Eigengeräuschs vom zu messenden Schall anzuwendenden Methoden kön-
nen diese unterschiedlichen Charakteristika gezielt ausnutzen, um bessere Ergebnisse zu erzielen.
Für den tonalen Anteil kann angenommen werden, dass der Charakter des Spektrums bekannt ist,
während für den breitbandigen Anteil eine mit dem Mikrofonarray realisierte Richtwirkung genutzt
werden kann.

• Die zur Erfassung der Mikrofonsignale und anschließende Signalverarbeitung notwendige Hardware
kann so ausgelegt werden, dass Masse und Energieverbrauch für die Nutzlast einer Multicopter-
Drohne ausreichend gering sind.

• Die Unsicherheit des Messergebnisses wird im wesentlichen von folgenden Faktoren bestimmt: Stär-
ke und Charakter des Eigengeräuschs der Drohne, abhängig von Art und Flugzustand, Stärke und
Charakter des zu messenden Schalls, angewendete Methoden zur Trennung von Eigengeräusch
und zu messendem Schall, Abweichung der Position der Drohne vom beabsichtigten Messort, Be-
wegungsgeschwindigkeit der Drohne während der Messung, Eigenschaften der eingesetzten Mikro-
fone.

• Die Messung der Schallleistung nach dem Hüllﬂächenverfahren ist ein gut geeigneter Anwendungs-
fall zur Demonstration des Messverfahrens, weil dazu der Schalldruckpegel an mehreren festgelegten
Orten gemessen werden muss.

Aus diesen Hypothesen lassen sich folgende Forschungsfragen ableiten, die im Vorhaben adressiert

werden sollen:

1. Wie kann ein Mikrofonarray mit zugehöriger Signalverarbeitung auf einer Drohne so realisiert werden,
dass eine Beeinﬂussung durch die Rotor-Strömung weitgehend vermieden wird, Energiebedarf und
Masse möglichst gering gehalten werden und gleichzeitig eine gute Trennung von Eigengeräusch
und zu messendem Schall möglich ist?

2. Welche konkreten Signalverarbeitungansätze sind am besten geeignet, um jeweils den tonalen und
den breitbandigen Anteil des Eigengeräuschs der Drohne von dem zu messenden Schall zu trennen?
3. Welche Unsicherheit hat das Messergebnis und wie stark wird diese von den einzelnen Einﬂussfak-

toren bestimmt?

4. Welche Messergebnisse lassen sich bei der praktischen Anwendung zur Messung der Schallleistung
mit dem Hüllﬂächenverfahren im Vergleich zu einer herkömmlichen Messung ohne Verwendung einer
ﬂiegenden Plattform erzielen?

Die erfolgreiche Beantwortung dieser Forschungsfragen würde nicht nur die Möglichkeiten von Emissions-
und Immissionsmessungen erweitern, sondern bildet auch die notwendige Grundlage für zukünftige An-
wendungen, wie beispielsweise die gleichzeitige Charakterisierung mehrerer Schallquellen ausgehend von
einer ﬂiegenden Plattform oder synchrone Messungen mit mehreren Drohnen.

2.3 Arbeitsprogramm inkl. vorgesehener Untersuchungsmethoden

Das Arbeitsprogramm zur Behandlung der zuvor deﬁnierten Forschungsfragen ist in fünf Arbeitspakete
unterteilt, deren zeitliche Abfolge in Abbildung 6 tabellarisch dargestellt ist.

Für das Vorhaben ist eine geeignete ﬂiegende Plattform nötig. Da Schalldruckmessungen nicht in Bewe-
gung, sondern an festen Orten durchgeführt werden sollen, scheint dafür eine Multicopter-Drohne geeignet.
Im Arbeitspaket 1 soll deshalb eine vorhandene Multicopter-Drohne als Basis genutzt und ein Mikrofonar-
ray mit der erforderlichen Signalerfassung, Signalverarbeitung und Kommunikation realisiert werden (For-
schungsfrage 1). Parallel dazu werden im Arbeitspaket 2 Signalverarbeitungs-Algorithmen zur Verfügung
gestellt, die das Eigengeräusch der Drohne vom zu messenden Schall trennen können (Forschungsfrage
2). Die im Arbeitspaket 1 realisierten Drohnenkonﬁgurationen, zusammen mit der im Arbeitspaket 2 imple-
mentierten Signalverarbeitung, werden im Arbeitspaket 3 genutzt, um im Rahmen von Indoor-Experimenten

Seite 7 von 21

1. Jahr

2. Jahr

3. Jahr

AP 1

AP 2

AP 3

AP 4

AP 5

Multicopter Mikrofonarray (ES)

Trennung Eigengeräusch (ES)

Indoor-Experimente (GH)

Modell Messunsicherheit (GH)

Demonstration Outdoor (GH)

Bild 6: Zeitplan für das Vorhaben. Der für ein Arbeitspaket hauptverantwortliche Antragsteller ist jeweils in Klammern
angegeben (ES = Ennes Sarradj, GH = Gert Herold)

im schallreﬂexionsarmen Raum unter kontrollierten akustischen Bedingungen Messungen durchzuführen
und dabei verschiedene Varianten der Signalverarbeitung mit Ergebnissen der herkömmlichen Messung
mit Mikrofonen auf Stativen zu vergleichen (Forschungsfragen 1 und 2). Darüber hinaus werden so Daten
für Kalibrationszwecke erhoben. Arbeitspaket 4 widmet sich der Modellierung und systematischen Betrach-
tung der Messunsicherheit (Forschungsfrage 3), während im Arbeitspaket 5 eine Schallleistungsmessung
im Freien als Demonstrationsexperiment durchgeführt wird (Forschungsfrage 4).

Über den Verlauf des gesamten Vorhabens hinweg sollen fertiggestellte Datensätze und Implementie-
rungen der Methoden zu bestehenden frei zugänglichen Bibliotheken hinzugefügt werden und die Validie-
rungsergebnisse in wissenschaftlichen Zeitschriften veröffentlicht werden.

AP 1: Multicopter-Drohne mit Mikrofonarray

Für die angestrebte Realisierung der Methode zur Messung des Schalldruckpegels wird eine ﬂiegende
Plattform mit installierter Mikrofonarray-Messtechnik, Signalerfassung und -verarbeitung benötigt. Diese
soll auf Basis einer Multicopter-Drohne realisiert werden. Bei den Antragstellern können dazu durch eine
Kooperation mit Fachgebiet Flugführung und Luftverkehr der TU Berlin (Prof. Marten Uijt de Haag) zwei ver-
schiedene Drohnen-Modelle für erste Erprobungen eingesetzt werden (Bild 7). Die Tarot X6 ist ein Carbon
Hexacopter mit einem Gewicht von 2 kg und bis zu 5 kg Nutzlast. Der Holybro S500 ist ein Quadcopter mit
einem Gewicht von 1,3 kg und einer maximalen Nutzlast von 1 kg. Beide Drohnen sind so konzipiert, dass
sie sowohl unterhalb ihrer Rotorplattform Platz für Messtechnik bieten als auch verschiedene Anbauten
ermöglichen.

Bild 7: Die beiden bereits verfügbaren Drohnen: Holybro S500 Quadcopter und Tarot X6 Hexacopter.

Bei einem kleineren Gesamtgewicht der Multicopter-Drohne wie beim Quadcopter kann tendenziell von
einem geringeren Eigengeräusch ausgegangen werden. Gleichzeitig führte eine vergrößerte Gesamtrotor-
ﬂäche wie beim Hexacopter zu einem tendenziell langsameren Schubstrahl der Rotoren, was sich ebenfalls
positiv auf ein niedriges Eigengeräusch auswirkt. Es gibt weitere Vor- und Nachteile bei Einsatz verschie-
dener Drohnen-Modelle. Deshalb ist vorgesehen, beide Drohnen für den Transport der Messtechnik zu
berücksichtigen und eine endgültige Bewertung der verschiedenen Konﬁgurationen erst nach den Experi-
menten im AP 3 vorzunehmen.

Seite 8 von 21

Anforderungen, Konzeption und Vergleich verschiedener Messtechnikkonzepte Die Anforderungen
an die auf der Drohne unterzubringenden Messtechnik umfassen ein möglichst geringes Gewicht, einen
geringen Energieverbrauch für einen etwa 30-minütigen autonomen Betrieb, Konnektivität zu mehreren
Mikrofonen, weiteren Sensoren, drahtlose Konnektivität zur Bodenstation zur Steuerung und Übertragung
von Messdaten sowie eine ausreichende Rechenleistung, um entweder alle oder einen wichtigen Teil der
in AP 2 zu entwickelnden Algorithmen mit den Mikrofonsignalen auszuführen. Dazu sind die in Tabelle 1
zusammengestellten Komponenten nötig. Für diese Komponenten gibt es jeweils neben den in der Ta-
belle aufgeführten Beispielen weitere Möglichkeiten. Deshalb sollen zunächst die Anforderungen genauer
präzisiert, verschiedene Konzepte im Detail ausgearbeitet und hinsichtlich der Erfüllung der Anforderun-
gen verglichen werden. Dabei wird insbesondere das Zusammenwirken der Komponenten hinsichtlich der
Hard- und Software berücksichtigt. Die Anordnung und mechanische Befestigung der Mikrofone an der
Drohne bleibt dabei zunächst unberücksichtigt. Ein oder maximal zwei verschiedene Konzepte werden für
eine Realisierung ausgewählt.

Tabelle 1: Komponenten, angestrebte Gesamtmasse < 400 g

Beispiel / Erläuterung

ca. Massenbudget

Komponente

Mikrofone

MEMS, z.B. Vesper VM3000 oder TDK-ICS-43434,
8–16 Stück

Mikrofonbefestigung

mechanische Teile und PCB

Erfassung Rotordrehzahl

Signalerfassung und Vor-
verarbeitung

Signalverarbeitung

elektrische oder optische Drehzahlmessung für Fil-
teralgorithmen in AP 2, z.B. Reﬂex-Optokoppler

z.B. XMOS-XCore Prozessor, PCB, ähnlich UMA8

Einplatinencomputer mit IEE802.11 (WiFi), Linux-
basiertes Betriebssystem, z.B. ODROID-C4

Akku

Li-Ion oder LiPO-Akku mit ca. 10 Wh

Gehäuse für Akku / PCBs

z.B. 3D-gedrucktes Gehäuse

10 g

100 g

40 g

40 g

80 g

80 g

50 g

Mikrofonanordnung Die Anordnung und Befestigung der Mikrofone des Arrays hat großen Einﬂuss auf
die Eigenschaften der Messplattform. Dabei sind Anforderungen in Bezug auf das Eigengeräusch der Droh-
ne zu berücksichtigen, wie ein ausreichender Abstand der Mikrofone von den Rotoren oder die Platzie-
rung der Mikrofone und ihrer Befestigungen außerhalb starker Strömungen, um zusätzliche Strömungs-
geräusche zu vermeiden. Die Befestigung der Mikrofone auf der Drohne kann unter Umständen auch die
Flugeigenschaften negativ beeinﬂussen. Ein weiterer Gesichtspunkt ist, dass die räumliche Filterung auf
der Basis der erfassten Mikrofonarraysignale (siehe AP 2) abhängig von der Anordnung der Mikrofone ist.
Somit ist es sinnvoll, diese optimal oder zumindest günstig zu wählen (siehe auch [38, 54]). Deshalb sollen
sowohl Simulationen als auch in begrenztem Umfang Experimente mit verschiedenen Mikrofonanordnun-
gen und -befestigungen durchgeführt werden. Dazu sind jeweils die benötigten mechanischen Teile und
PCBs auszulegen, zu fertigen und an den Drohnen zu testen. Auf der Basis eines Vergleichs hinsichtlich
der Eigenschaften wird ein Konzept der Mikrofonanordnung und -befestigung für die endgültige Realisie-
rung ausgewählt.

Drehzahlmessung Für die notwendige Drehzahlmessung aller Rotoren ist ein zuverlässig funktionieren-
des Konzept (elektrisch oder optisch) auszuwählen. Eine direkte Kommunikation mit der Drohnensteuerung
liefert voraussichtlich keine ausreichend präzisen, zeitaufgelösten Informationen.

Realisierung Zum Abschluss des Arbeitspaketes ist geplant, für jede Drohne mindestens ein Konzept
zu realisieren. Das umfasst den Entwurf und die Fertigung aller mechanischen und elektrischen Teile,

Seite 9 von 21

die notwendige Software für die Fernsteuerung der Messung von der Bodenstation (auf der Basis von
SpectAcoular [52]) sowie die abschließende Flugerprobung.

AP 2: Trennung des Eigengeräuschs vom zu messenden Schall

In diesem Arbeitspaket ist geplant, verschiedene Signalverarbeitungsalgorithmen zur Trennung des Eigen-
geräuschs der Multicopter-Drohne vom zu messenden Schall zu untersuchen und zu implementieren. Weil
zu Beginn nicht eingeschätzt werden kann, welcher mögliche Weg dazu der beste ist, werden verschiedene
Ansätze verfolgt. Diese lassen unterschiedlich gute Ergebnisse hinsichtlich der Trennung erwarten. Fast al-
le der in Erwägung gezogenen Ansätze benötigen vergleichsweise wenig Rechenleistung. Um einschätzen
zu können, wie praktikabel die Verwendung auf einer Drohne jeweils ist, soll neben der Funktionalität auch
der erforderliche Rechenaufwand bewertet werden.

Die im Folgenden unter A und B erläuterten Schritte werden in diesem Arbeitspaket entwickelt, imple-
mentiert und in die quelloffene Python-Software Acoular [2] integriert. Die rechenintensiven Komponenten
der Algorithmen sind kompiliert, sodass eine efﬁziente und schnelle Berechnung erreicht werden kann. Der
Einsatzbereich von Acoular umfasst sowohl leistungsstarke CPU- und GPU-Plattformen als auch Systeme
mit geringem Energieverbrauch und dadurch reduzierter Rechenleistung, wie sie als Nutzlast einer Droh-
ne infrage kommen. Viele der benötigten Bausteine, vor allem die Methoden zur räumlichen Filterung und
Entfaltung, sind bereits in Acoular vorhanden und stehen ohne weitere Arbeiten zur Verfügung.

A Getrennte Behandlung der tonalen und der breitbandigen Anteile Der erste Ansatz geht davon
aus, dass das durch die Rotoren verursachte Eigengeräusch aus zwei Komponenten besteht. Zum einen
gibt es einen tonalen Anteil mit einer höheren Schallleistung, der für jeden Rotor bei einer der Drehzahl ent-
sprechenden Frequenz und deren Vielfachen liegt, insbesondere bei den Blattfolgefrequenzen und ihren
Harmonischen. Diese Frequenzen ändern sich bei den permanent auftretenden Änderungen der Drehzahl.
Zum anderen entsteht durch die Umströmung der Rotorschaufelblätter an Hinterkante, Blattspitze und ge-
gebenenfalls auch an der Vorderkante ein Anteil mit im Allgemeinen geringerer Schallleistung, dafür aber
breitbandigem Charakter. Der tonale Anteil ist an allen Mikrofonen stark kohärent. Hingegen weist der breit-
bandige Anteil nur dann ein Kohärenzmaximum auf, wenn die unterschiedlichen Laufzeiten zwischen dem
Ort einer Quelle und den Mikrofonen berücksichtigt werden, was jedoch nur für einen Quellort gleichzei-
tig möglich ist. Davon ausgehend wird zunächst der tonale Anteil vom zu messenden Schall getrennt und
danach der breitbandige Anteil.

Notch-Filter Zur Trennung des tonalen Anteils ist vorgesehen, für jede aus dem Signal zu entfernende
Frequenz ein Notch-Filter (schmalbandige Bandsperre) zu implementieren, die adaptiv an die jeweils
aktuelle Frequenz entsprechend angepasst wird. Die Anzahl der hintereinander in den Signalweg
für jedes Mikrofonsignal einzubringenden Filter ist N h, wobei N die Anzahl der Rotoren und h die
Anzahl der relevanten Vielfachen der Blattfolgefrequenz ist. Zur praktischen Realisierung der Filter
gibt es verschiedene Möglichkeiten, wobei IIR-Filter einen geringen Rechenaufwand mit einer sehr
schmalbandigen Wirkung verbinden und deshalb hier realisiert werden sollen.

Anpassung an Momentandrehzahlen Zur Anpassung der zu ﬁlternden Frequenzen gibt es verschiede-
ne Möglichkeiten, von denen zwei umgesetzt werden sollen. Bei der ersten zu implementierenden
Methode soll die Drehzahl jedes Rotors mit den in AP 1 beschriebenen Drehzahlsensoren erfasst
und zur Anpassung der Filterkoefﬁzienten verwendet werden. Die zweite zu implementierende Me-
thode verwendet das mit den Mikrofonen erfasste Signal, um die Drehzahl zu bestimmen. Unter den
möglichen existierenden Ansätzen dazu soll hier zunächst der nach Tan und Jiang [55] umgesetzt
werden.

Arrayverfahren zur räumlichen Filterung Nach der Entfernung der tonalen Anteile verbleiben die breit-
bandigen Anteile des Eigengeräuschs im Signal. Durch Anwendung eines Arrayverfahrens werden
diese nun durch räumliche Filterung nach der Einfallsrichtung (direction of arrival, DOA) getrennt. Da
die Einfallsrichtung des Schalls für das Eigengeräusch der Rotoren bekannt ist, können diese An-
teile gezielt entfernt werden. Die klassischen auf Beamforming basierenden Methoden haben dabei

Seite 10 von 21

entweder eine frequenzabhängig ungenügende Filterwirkung (klassisches Delay-and-Sum) oder lie-
fern zwar die Richtung, aber keine zuverlässige Information über die Signalleistung (beispielsweise
MUSIC, Capon-Beamformer und Varianten). Deshalb sollen hier Entfaltungsmethoden zum Einsatz
kommen, die sowohl eine ausreichende Richtungsauﬂösung als auch eine zuverlässigere Schätzung
der Signalleistung ermöglichen. Folgende Methoden, die sich durch einen vergleichsweise geringen
Rechenaufwand auszeichnen, sollen eingesetzt werden: Orthogonale Entfaltung [56], Clean-SC [57]
und Clean-T [58, 41]. Zusätzlich zu den genannten Verfahren soll ebenfalls ein aufwändigeres inver-
ses Verfahren, beispielsweise Cross-spectral-Matrix-Fitting [59] für die Signalverarbeitungskette zur
Verfügung gestellt werden, um einschätzen zu können, welche Verbesserungen möglich sind, wenn
der Rechenaufwand keine Rolle spielt.

Rekonstruktion des Leistungsspektrums des zu messenden Schalls Bei der Entfernung des tonalen
Anteils werden gleichzeitig auch Teile des Leistungsspektrums des zu messenden Schalls aus den
Signalen heraus geﬁltert. Um dennoch das vollständige Leistungsspektrum rekonstruieren zu kön-
nen, sollen mehrere Zeitabschnitte, bei denen unterschiedliche Drehzahlen festgestellt und entspre-
chend geﬁltert wurden, so gemittelt werden, dass für jedes Frequenzintervall Informationen vorliegen.
Alternativ zum Notch-Filter können die Werte bei den entsprechenden Frequenzen auch durch Inter-
polation zwischen den Werten bei benachbarten Frequenzen ersetzt werden. Ein Algorithmus, der
beide Varianten ermöglicht und die Entfernung tonaler und breitbandiger Störanteile vereint, wird
implementiert.

B Zeit-Frequenz-Methode Der zweite Ansatz ignoriert die unterschiedlichen spektralen Eigenschaften
der tonalen und breitbandigen Anteile des Eigengeräuschs. Stattdessen wird ausgenutzt, dass sich die
Lage der Rotoren in Bezug auf die Mikrofone nicht verändert und im Vergleich zur Anzahl der möglichen
Schalleinfallsrichtungen nur aus wenigen Richtungen tatsächlich Schalleinfall zu erwarten ist. Damit ist die
Verteilung der insgesamt festgestellten Signalleistung auf die Einfallsrichtungen schwach besetzt. Mit den
Mikrofonsignalen als Eingangsgrößen soll das Verfahren des Sparse Bayesian Learning (SBL) [60, 61] zum
Einsatz kommen.

Dabei werden einzelne Zeitabschnitte der Mikrofonsignale Fourier-transformiert und alle Kreuzspektren
werden berechnet. Mit SBL werden beteiligte Schallquellen, deren Leistungsspektren sowie die Einfalls-
richtung geschätzt. Die Qualität der Schätzung steigt mit der Anzahl der betrachteten Zeitabschnitte. Daher
ist eine effektive Trennung, insbesondere bei langsamen Änderungen im Leistungsspektrum des Eigenge-
räuschs, wahrscheinlicher, wenn die Einfallsrichtungen des Eigengeräuschs bekannt sind. Das Verfahren
hat einen überschaubaren Rechenaufwand und benötigt keine explizite Information über die Drehzahl. In
gleicher Weise wie bei A kann nach der Anwendung von SBL das Leistungsspektrum des zu messenden
Schalls rekonstruiert werden.

C Test und Erprobung Die bei A und B implementierten Methoden und alle ihre Varianten werden auf
ihre Funktion hin getestet. Vorrangiges Ziel dabei ist, die korrekte Implementierung zu testen und eine
erste qualitative Einschätzung zur Leistung der verschiedenen Ansätze zu erhalten. Da im Rahmen des
Vorhabens erst in AP 3 Messdaten gewonnen werden, werden zunächst folgende Daten dazu verwendet:

Synthetische Daten Mit den in Acoular zur Verfügung stehenden Möglichkeiten zur Synthese von Mikro-
fonarray-Eingangssignalen werden entsprechende Daten für eine ausreichende Anzahl verschiede-
ner akustischer Szenarien realisiert. Die Szenarien beinhalten jeweils eine Schallquelle pro Rotor,
die ein Signal abgibt, dessen Frequenz sich im Laufe der Zeit leicht ändert und das über kurze Zeit-
abschnitte einem periodischen Signal ähnelt. Hinzu kommen weitere breitbandige Signale für jeden
Rotor sowie für die zu messende Schallquelle. Für die Synthese der Eingangssignale werden die
Signale aller Schallquellen an jedem Mikrofon überlagert.

DREGON Datensatz Messsignale eines von auf einem Multicopter montierten 8-Kanal-Mikrofonarrays
und weitere zugehörige Daten wie Rotordrehzahlen wurden vom INRIA-Forschungsinstitut in Form
eines frei zugänglichen Datensatzes DREGON [25] publiziert, der für die hier vorgesehenen Tests
als besonders geeignet erscheint.

Seite 11 von 21

AP 3: Indoor-Experiment unter kontrollierten akustischen Bedingungen

In diesem Arbeitspaket wird experimentell evaluiert, wie gut die in AP2 implementierten Methoden zur Un-
terdrückung des Eigengeräusches der Drohne geeignet sind. Die dafür vorgesehene Messkampagne soll
in einer praktisch von Reﬂexionen und Störschall freien Umgebung stattﬁnden und Szenarien mit einzel-
nen sowie mit mehreren gleichzeitig wirkenden Schallquellen für den zu messenden Schall berücksichti-
gen. Letzteres erlaubt eine Bewertung hinsichtlich der Möglichkeiten zur getrennten Schallpegelmessung
gleichzeitig emittierender Quellen. Zur Messung werden die in AP1 entwickelten drohnenbasierten Sensor-
systeme eingesetzt. Da bei der Anwendung in der Realität zu erwarten ist, dass der zu messende Schall
aus Richtungen entweder unterhalb oder seitlich der Drohne einfällt, sollen diese Fälle in den Experimenten
vorrangig berücksichtigt werden.

Als Messumgebung ist der große reﬂexionsarme Raum der TU Berlin vorgesehen. Mit einem freien Ge-
samtvolumen von V = 830 m³ und einer unteren Grenzfrequenz von fg = 63 Hz eignet sich der Raum auch
für akustische Untersuchungen mit Drohnen [30, 7]. Als Schallquellen sollen vorhandene Lautsprecher ver-
wendet werden, über die breitbandige Rauschsignale wiedergegeben werden. Wie in Bild 8 dargestellt,
sollen die Quellen auf einem Viertelkreis unterhalb und seitlich eines Messort angeordnet werden. Hierbei
ist ein Abstand von etwa 2,5 m zwischen Messort und Lautsprechern geplant. Zusätzlich sind weiter ent-
fernt liegende Messpunkte bis etwa 5 m Abstand möglich, sodass einerseits bei gleicher Quellamplitude
des zu messenden Schalls geringere Differenzen zum Eigengeräusch der Drohne und geringere Winkel
zwischen mehreren Schallquellen realisiert werden können.

weitere Messorte

Messort

2,5 m

≈

Lautsprecher

Bild 8: Schematische Darstellung des Messaufbaus. Für weiter entfernte Messpunkte verringert sich der Öffnungs-
winkel und der Störabstand.

Im Einzelnen sind folgende Messungen vorgesehen:

Referenzmessung: Das Terzspektrum des Schalldruckpegels wird für jeden Lautsprecher an mindestens
drei vorher deﬁnierten Messorten mit Hilfe eines Messmikrofons und Schallpegelmessers ermittelt.
Die Drohne wird dabei nicht verwendet. Als Eingangssignal für die Lautsprecher ist weißes Rauschen
vorgesehen. Die Schalldruckpegel an den Messorten sollen dabei in mehreren Stufen zwischen un-
gefähr 50 dB und 90 dB variiert werden.

Eigengeräuschmessung: Das Eigengeräusch der schwebenden Drohne wird mit dem an Bord vorhan-
denen Mikrofonarray aufgezeichnet. Daraus wird sowohl das schmalbandige Leistungsspektrum als
auch das Terzspektrum des Schalldruckpegels ermittelt. Es werden ebenfalls Messungen bei gerin-
ger Auf- und Abbewegung der Drohne durchgeführt, um so Veränderungen der Rotordrehzahlen zu
erreichen. Die Dauer einer Einzelmessung wird ausreichend lang gewählt (mehrere Minuten), um
später den Einﬂuss der Messdauer auf die Unsicherheit des Ergebnisses untersuchen zu können.
Die Messungen werden gegebenenfalls für unterschiedliche Konﬁgurationen der Drohne nach AP 1
wiederholt. Auf der Grundlage der Ergebnisse dieser Messungen kann die Differenz zwischen dem
zu messenden Schall und dem Eigengeräusch ermittelt und als eine der Grundlagen für die Model-
lierung der Messunsicherheit im AP 4 verwendet werden.

Seite 12 von 21

Betriebsmessung: Zunächst gibt jeder Lautsprecher einzeln ein Signal ab, danach werden mehrere Laut-
sprecher gleichzeitig mit unkorrelierten Eingangssignalen betrieben, um das Vorhandensein mehre-
rer Schallquellen zu simulieren. Es werden dieselben Schalldruckpegel wie bei der Referenzmes-
sung eingestellt. Die Drohne beﬁndet sich nacheinander an den zuvor deﬁnierten Messorten. Für
jeden Lautsprecher und jede Kombination von Lautsprechern, sowie jede Pegelstufe und jeden Mes-
sort, werden jeweils die gleichen Messungen wie bei der Eigengeräuschmessung durchgeführt, ein-
schließlich möglicher Wiederholungen für verschiedene Drohnenkonﬁgurationen. Für ausgewählte
Kombinationen werden die Messungen mehrfach wiederholt.
Der entstehende Datensatz enthält so neben den Referenz- und den Eigengeräuschmessungen eine
größere Anzahl von Messungen für verschiedene Messorte, bei denen jeweils auch eine Referenz-
messung für die Messgröße verfügbar ist. Außerdem stehen für einzelne Messorte und Quellsze-
narien mehrere Messungen zur Verfügung, um später die Unsicherheit bei der Wiederholung von
Messungen modellieren zu können.

Für die Bewertung der verschiedenen Signalverarbeitungsmethoden aus AP 2 und der verschiedenen
Drohnenkonﬁguration aus AP 1 hinsichtlich der Trennung des Eigengeräusches und der getrennten Mes-
sung mehrerer Schallquellen, wird jeweils die Differenz zwischen dem Schalldruckpegel der Referenzmes-
sungen und der bei der Betriebsmessung mit der Drohne gemessenen Schalldruckpegel der Lautsprecher
in Abhängigkeit der Frequenz betrachtet. Es wird erwartet, dass sich die Ergebnisse für die unterschiedli-
chen Ansätze unterscheiden, sodass auf der Basis dieser Untersuchungen der am besten geeignete An-
satz ausgewählt und in AP 4 und AP 5 für die Betrachtung der Messunsicherheit sowie zur Durchführung
der Demonstrationsmessung verwendet werden kann. Die sich aus den Experimenten ergebende günstige
Drohnenkonﬁguration soll dann auf der Basis einer eigens beschafften Drohne endgültig realisiert werden,
um damit im AP 5 Messungen auch außerhalb des Labors durchführen zu können.

Die in diesem Arbeitspaket erfassten Referenzmessungen sind ebenfalls für die Kalibrierung des droh-

nenbasierten Schalldruckpegelmesssystems mit dem Verfahren der Vergleichsmessung geeignet.

AP 4: Modell der Messung und Messunsicherheit

Um das ﬂiegende Messmikrofon als Messinstrument einsetzen zu können, ist es wichtig, den Grad der
Zuverlässigkeit der erhaltenen Messwerte systematisch zu quantiﬁzieren. Hierfür bietet eine Unsicherheits-
bewertung nach dem „Guide to the Expression of Uncertainty in Measurement“ (GUM) [62] ein geeignetes
Rahmenwerk, das eine systematische Abschätzung der erwartbaren Abweichungen der Messgröße vom
wahren Wert anhand der Eigenschaften der Einﬂussgrößen ermöglicht.

Zentraler Bestandteil des GUM und dieses Arbeitspaketes ist die Erstellung eines Modells, das den
Zusammenhang Y = f (X1, . . . , XN ) zwischen den gemessenen Eingangsgrößen Xi und der interessie-
renden Ausgangsgröße Y beschreibt. Da auch die Eingangsgrößen Ungenauigkeiten unterliegen, ist es
notwendig, deren Umfang abzuschätzen und mit geeigneten Wahrscheinlichkeitsverteilungen zu beschrei-
ben.

Für den hier geplanten Messaufbau ist die zu messende Ausgangsgröße der Schalldruckpegel Lp, der
frequenzabhängig betrachtet werden soll. Im Fall eines einzelnen idealen omnidirektionalen Mikrofons lässt
sich dieser aus dem gemessenen Schalldruck-Zeitsignal bzw. der Mittelung fouriertransformierter Zeitab-
schnitte (Welch-Methode zur Schätzung des Leistungsspektrums) berechnen. Bereits für Schallpegelmes-
ser mit Einzelmikrofonen [63] sind Einﬂussgrößen wie Frequenzgang des Mikrofons, Reﬂexionen am Ge-
häuse oder der veränderte Frequenzgang durch verwendetes Zubehör wie Windschirme zu beachten [64],
die auch in den Betrachtungen hier Eingang ﬁnden. Zusätzlich ergeben sich durch den Betrieb auf einer
Multicopter-Drohne sowie die Signalverarbeitung von Mikrofonarray-Messdaten (siehe AP 2) weitere Ein-
ﬂussgrößen, die betrachtet werden müssen.

Modellbildung In einem ersten Schritt erfolgt eine Erfassung und Einteilung aller denkbaren Einﬂussgrö-
ßen basierend darauf, ob deren Auswirkungen im Rahmen der Unsicherheitsbewertung statistisch analy-
siert werden (Typ A) oder ob die entsprechenden Wahrscheinlichkeitsverteilungen auf alternative Weise
ermittelt werden (Typ B). Einﬂussgrößen vom Typ A sind hier z. B.:

Seite 13 von 21

• Abweichung der tatsächlichen Position der Drohne von ihrer nominellen Position
• Schallbeugung/-reﬂexionen an festen Strukturen des der Drohne mit Array
• tonale und breitbandige Schallabstrahlung durch die Drohne
• Schallabstrahlung durch Störquellen abseits der Drohne
• Variation der Schalllaufzeiten zwischen den Einzelmikrofonen, z.B. durch selbst erzeugte Strömung
• Schalleinfallswinkel / relative Position zur Schallquelle

Einﬂussgrößen vom Typ B sind z. B.:

• Frequenzgang der Mikrofone
• Temperatur, Luftdruck und Luftfeuchtigkeit
• Positionierungsgenauigkeit der Mikrofone im Array
• Eigenschaften der verwendeten Array- und Filter-Algorithmen
• Messdauer

Es ist davon auszugehen, dass diese Aufzählung unvollständig ist und sich je nach dem Ergebnis der AP
1 und 2 weitere Einﬂussgrößen ergeben.

Neben der Beschreibung potenziell relevanter Einﬂussgrößen und deren Kategorisierung ist ein wei-
terer Gegenstand dieses Arbeitspaketes auch die Identiﬁkation gegenseitiger Abhängigkeiten. Hierbei ist
zu beachten, dass Typ-B-Einﬂussgrößen auch von Größen des Typs A abhängen können und umgekehrt
(z. B. hängen einige Eigenschaften der Array-Algorithmen von den Schalllaufzeiten ab, die wiederum von
Strömung und Temperatur abhängen). Diese Abhängigkeiten müssen ebenfalls mit modelliert werden.

Die Wahrscheinlichkeitsverteilungen von Einﬂussgrößen wie beispielsweise der tonalen und breitbandi-
gen Schallabstrahlung der Drohne oder die Variation von Schalllaufzeiten aufgrund des Rotor-Abwinds sind
nur mit erheblichem Aufwand vollständig numerisch modellierbar. Daher werden diese mit hybriden Ansät-
zen geschätzt, in denen die Eigenschaften der Strömung mit einfachen analytischen Modellen erfasst und
die Schallausbreitung mit schnellen numerischen Methoden [65] untersucht werden.

Monte-Carlo-Simulationen Das geplante Messsystem weist durch die vielen Einﬂussgrößen und die zu-
gehörigen Signalverarbeitungsalgorithmen eine hohe Komplexität auf und enthält außerdem Einﬂussgrö-
ßen, die sich gegenseitig beeinﬂussen können, sodass ein Modell nach dem Schema Y = f (X1, . . . , XN )
nicht leicht zu identiﬁzieren ist. Daher ist in diesem Arbeitspaket vorgesehen, Monte-Carlo-Simulationen
entsprechend GUM Supplement 1 [66] durchzuführen. Ein vergleichbares Vorgehen konnte bei der Unter-
suchung der Zuverlässigkeit von Array-Algorithmen aussagekräftige Ergebnisse liefern [3]. Das Vorgehen
hierfür ist wie folgt:

1. Es wird ein Datensatz mit konkreten Eingangsdaten erstellt, wobei diese entsprechend der Zufalls-

verteilungen der Eingangsgrößen gewürfelt werden.

2. Mit dem Eingangsdatensatz wird eine Simulation durchgeführt, die die Signalverarbeitungschritte von

Messdaten exakt abbildet.

3. Der sich ergebende Ausgangswert wird abgespeichert.
4. Die Schritte 1. bis 3. werden wiederholt, bis eine für eine statistische Auswertung ausreichende Men-
ge an Einzel-Messergebnissen für den Schalldruckpegel berechnet wurde (je nach Konvergenz der
Mittelwerte und Standardabweichungen bis zu mehrere Tausend).

Die Monte-Carlo-Simulationen werden unter Anwendung des bei den Antragstellern betreuten Softwarepa-
kets Acoular durchgeführt, das sowohl Methoden zu Erzeugung synthetischer Messdaten als auch Signal-
verarbeitungsalgorithmen für Mikrofonarray-Messdaten beinhaltet [2].

Eignung als Schallpegelmesser Der sich ergebende mittlere Schätzwert der Ausgangsgröße Lp wird
hinsichtlich seiner Plausibilität ausgewertet und die kombinierte Standardunsicherheit wird aus der Streu-
ung der Ergebniswerte berechnet. Hieraus wird die erweiterte Standardunsicherheit für eine Überdeckungs-
wahrscheinlichkeit von 95 % ermittelt und überprüft, inwiefern und unter welchen Bedingungen sich die
Akzeptanzgrenzen für Schallpegelmesser der Klassen 1 oder 2 (siehe DIN EN 61672 [63]) einhalten las-
sen. Ebenso wird der maximal zu erwartende Abweichung berechneter Messwerte vom optimalen Wert
quantiﬁziert.

Seite 14 von 21

Die Einﬂussfaktoren sind dahingehend zu analysieren, wie und in welchem Maße sie zur kombinierten
Standardunsicherheit beitragen. Dies kann durch die Untersuchung geeigneter Teilmengen der Monte-
Carlo-Zufallsauswahlen erfolgen. Außerdem ist abzuschätzen, ob und wie die jeweils zugehörigen Stan-
dardunsicherheiten reduziert werden können. Für den Fall einer deutlichen Überschreitung der Akzeptanz-
grenzen sind darauf basierend Empfehlungen zur Verbesserung des Messsystems abzuleiten.

Vereinfachung des Messmodells Auf Basis der Verteilungsfunktionen der Messunsicherheiten, die aus
den Monte-Carlo-Simulationen gewonnen werden, soll ein geeignetes Modell zur Beschreibung des Zu-
sammenhangs zwischen den Eingangsgrößen, einschließlich ihrer zugehörigen Standardunsicherheiten,
und der Ausgangsgröße, einschließlich der kombinierten Standardunsicherheit gewonnen werden. Da die-
ser Zusammenhang vor der Auswertung der Simulation nicht bekannt ist, kommen verschiedene Formen
der multiplen Regressionsanalyse zur Beschreibung infrage, die hier ausgewählt und angewendet werden
sollen.

Für einen einfachen Zusammenhang können lineare oder stochastische Regressionsverfahren zielfüh-
rende Modelle liefern. Für den Fall, dass die Zusammenhänge stark nichtlinear sind, kann auch ein Multi-
Layer-Perceptron oder das Verfahren der symbolischen Regression [67] verwendet werden. Das abgeleitete
Modell soll eine efﬁziente Berechnung der Ausgangsgröße ermöglichen. Des Weiteren soll es die schnelle
Abschätzung des Einﬂusses einzelner Parameter (z. B. Messdauer) auf die kombinierte Standardunsicher-
heit gestatten, ohne die Notwendigkeit erneuter Monte-Carlo-Simulationen.

Validierung Zusätzlich zu den Monte-Carlo-Simulationen mit rein synthetischen Daten werden die in AP
3 und AP 5 erfassten Daten hinsichtlich der Streuung der Messergebnisse bei wiederholten Messungen un-
ter nominell identischen Bedingungen sowie des Einﬂusses von Änderungen in den Auswerteparametern
(wie Algorithmen-Eigenschaften und Messdauer) untersucht. Damit wird überprüft, ob die vorhergesagten
Unsicherheiten in Abhängigkeit der Einﬂussgrößen (im beobachtbaren Rahmen) auch in der realen Mess-
praxis wiedergefunden werden.

AP 5: Demonstration Outdoor-Experiment: Schallleistungsmessung Hüllﬂächenverfahren

Im Rahmen dieses Arbeitspaket ist die Konzeption, Durchführung und Auswertung eines Demonstrator-
Experiments im Freien vorgesehen. Die dabei zu ermittelnde Messgröße ist die Schallleistung, welche im
Hüllﬂächenverfahren nach DIN EN ISO 3744 [68] mittels der Messung der Schalldruckpegel an verschie-
denen Positionen auf einer Hüllﬂäche um das Messobjekt bestimmt wird. So soll die Anwendung des zuvor
in AP 3 ausgewählten drohnenbasierten Schalldruckpegelmesssystems und der zugehörigen Signalverar-
beitungsansätze demonstriert werden.

Zusätzlich soll eine Vergleichsmessung an demselben Messobjekt mit demselben Verfahren zur Mes-
sung der Schallleistung, jedoch mit herkömmlicher Schallpegelmesstechnik und entsprechenden Stativen
durchgeführt werden, um eine Bewertung der Ergebnisse zu ermöglichen.

Bei dem gewähltem Messobjekt soll ein Kompromiss zwischen der Schwierigkeit in der Umsetzung der
drohnenbasierten und der Vergleichsmessung sowie praxisbezogener Anwendung der Messung gefunden
werden. So würde ein schwer zugängliches Messobjekt, wie etwa ein Schornstein, die Vorteile einer ﬂie-
genden Messplattform zeigen, jedoch wäre eine Vergleichsmessung mit festen Mikrofonen mit erheblichem
Aufwand verbunden. Als zu untersuchende Messobjekte kommen unter anderem in Frage:

• Lautsprecher auf einem hohen Stativ (gute Kontrolle über Stärke und Spektrum der Schallquelle,

jedoch praxisfern),

• Klimaanlage an der Außenwand oder größere (praxisrelevant, jedoch unter Umständen hoher Auf-

wand für die Vergleichsmessung).

Dementsprechend ist vorgesehen, die Entscheidung für ein konkretes Messobjekt erst während der Bear-
beitung des Vorhabens zu treffen und dabei auch die regelmäßig in anderen Projekten der Antragsteller
untersuchten Messobjekte in Erwägung zu ziehen.

Seite 15 von 21

Die Auslegung und Durchführung soll nach der in [68] dargestellten Fallstudie erfolgen. Dabei wird der
Schalldruckpegel an mehreren verschiedenen Positionen auf einer Halbkugel mit einem nach den Vorgaben
der Norm festzulegenden Radius (mehrere Meter) um das Messobjekt gemessen.

Die mit der Drohne und mit herkömmlichen Mikrofonen erfassten Schalldruckpegel werden anschließend
nach [68] ausgewertet und die Ergebnisse verglichen. So wird eine Aussage darüber möglich, inwieweit die
Drohnenmessung zur konventionellen Methode vergleichbare Ergebnisse aufweist.

2.4 Umgang mit Forschungsdaten

In den Arbeitspaketen 3 und 5 fallen im Rahmen von Messungen Messdaten für Mikrofone auf Drohnen
an. Ein Teil dieser Messdaten kann für andere Forschende von Interesse sein. Es ist deshalb geplant,
diese jeweils als Datensatz in einen frei zugänglichen Repositorium zu veröffentlichen. Im Arbeitspaket 2
ist geplant, zusätzliche Algorithmen im frei verfügbaren Softwarepaket Acoular zu implementieren. Diese
werden Bestandteil der von den Antragstellern gepﬂegten Software und können so von anderen frei genutzt
werden.

2.5 Relevanz von Geschlecht und/oder Vielfältigkeit

Für das beantragte Vorhaben und dessen Ergebnisse haben Geschlecht und Vielfältigkeit keinen unmittel-
baren Einﬂuss auf Ergebnisse und Bearbeitung.

3 Projekt- und themenbezogenes Literaturverzeichnis

[1] E. Sarradj. “A fast signal subspace approach for the determination of absolute levels from phased microphone

array measurements”. In: Journal of Sound and Vibration 329 (2010), S. 1553–1569.

[2] E. Sarradj und G. Herold. “A Python framework for microphone array data processing”. In: Applied Acoustics 116

(2017), S. 50–58.

[3] G. Herold und E. Sarradj. “Performance analysis of microphone array methods”. In: Journal of Sound and Vibration

401 (2017), S. 152–168.

[4] G. Herold u. a. “Flight Path Tracking and Acoustic Signature Separation of Swarm Quadcopter Drones Using Mi-
crophone Array Measurements”. en. In: Quiet Drones 2020 - International E-Symposium on UAV / UAS Noise. Paris,
France, 2020.

[5] G. Herold und E. Sarradj. “Detection of Rotational Speeds of Sound Sources Based on Array Measurements”. en.

In: Applied Acoustics 157 (2020), S. 107002.

[6] S. Jekosch und E. Sarradj. “An Extension of the Virtual Rotating Array Method Using Arbitrary Microphone Conﬁ-

gurations for the Localization of Rotating Sound Sources”. en. In: Acoustics 2.2 (2020), S. 330–342.

[7] G. Herold. “In-ﬂight directivity and sound power measurement of small-scale unmanned aerial systems”. In: Acta

Acustica 6.58 (2022), S. 1–13.

[8] A. Kujawski und E. Sarradj. “Fast grid-free strength mapping of multiple sound sources from microphone array
data using a Transformer architecture”. In: The Journal of the Acoustical Society of America 152.5 (2022), S. 2543–
2556.

[9] G. Herold und E. Sarradj. “Characterizing Multicopter UAV Noise Emission Through Outdoor Fly-by Measurements
With a Microphone Array”. In: Proceedings of the 30th AIAA/CEAS Aeroacoustics Conference. Rome, Italy, 2024.
[10] G. Herold, A. Kujawski, S. Jekosch und E. Sarradj. “Comparison of microphone array methods for small UAV ﬂight

path reconstruction”. In: Proceedings of the 10th Berlin Beamforming Conference. Berlin, Germany, 2024, S. 10.

[11] M. Clayton, L. Wang, A. McPherson und A. Cavallaro. “An Embedded Multichannel Sound Acquisition System for Drone

Audition”. In: IEEE Sensors Journal 23.12 (2023), S. 13377–13386.

[12] K. Hoshiba u. a. “Design of UAV-Embedded Microphone Array System for Sound Source Localization in Outdoor Environ-

ments”. In: Sensors 17.11 (2017), S. 2535.

[13] M. Kumon, H. G. Okuno und S. Tajima. “Alternating Drive-and-Glide Flight Navigation of a Kiteplane for Sound Source
Position Estimation”. In: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Prague,
Czech Republic: IEEE, 2021, S. 2114–2120.

[14] M. Wakabayashi, H. G. Okuno und M. Kumon. “Drone audition listening from the sky estimates multiple sound source
positions by integrating sound source localization and data association”. In: Advanced Robotics 34.11 (2020), S. 744–755.
L. Wang und A. Cavallaro. “Deep-Learning-Assisted Sound Source Localization From a Flying Drone”. In: IEEE Sensors
Journal 22.21 (2022), S. 20828–20838.

[15]

[16] R. B. Brookﬁeld. “Optimized Acoustic Sensing for Fixed-Wing Uncrewed Aerial Vehicles”. Magisterarb. The University of

Memphis, 2023.

Seite 16 von 21

[17] P. J. Bernalte Sánchez und F. P. Garcia Marquez. “New Approaches on Maintenance Management for Wind Turbines
Based on Acoustic Inspection”. In: Proceedings of the Fourteenth International Conference on Management Science and
Engineering Management. Hrsg. von J. Xu, G. Duca, S. E. Ahmed, F. P. García Márquez und A. Hajiyev. Bd. 1191. Cham:
Springer International Publishing, 2021, S. 791–800.

[18] A. Finn und S. Franklin. “Acoustic sense & avoid for UAV’s”. In: 2011 Seventh International Conference on Intelligent

Sensors, Sensor Networks and Information Processing. Adelaide, Australia: IEEE, 2011, S. 586–589.

[19] B. Harvey und S. O’Young. “A harmonic spectral beamformer for the enhanced localization of propeller-driven aircraft”. In:

[20]

Journal of Unmanned Vehicle Systems 7.2 (2019), S. 156–174.
J. Martinez-Carranza und C. Rascon. “A Review on Auditory Perception for Unmanned Aerial Vehicles”. In: Sensors 20.24
(2020), S. 7276.

[21] Y. Hioka, M. Kingan, G. Schmid, R. McKay und K. A. Stol. “Design of an unmanned aerial vehicle mounted system for quiet

audio recording”. In: Applied Acoustics 155 (2019), S. 423–427.

[22] R. R. Subramanyam, R. Castro Mota, K. Picker, V. Wittstock und S. Jacob. “Experimental Low-Frequency Noise Charac-
terization of an Octocopter Drone”. In: 30th AIAA/CEAS Aeroacoustics Conference (2024). Rome, Italy: American Institute
of Aeronautics and Astronautics, 2024.

[23] B. Harvey und S. O’Young. “Detection of continuous ground-based acoustic sources via unmanned aerial vehicles”. In:

[24]

Journal of Unmanned Vehicle Systems 4.2 (2016), S. 83–95.
L. Wang und A. Cavallaro. “Microphone-Array Ego-Noise Reduction Algorithms for Auditory Micro Aerial Vehicles”. In: IEEE
Sensors Journal 17.8 (2017), S. 2447–2455.

[25] M. Strauss, P. Mordel, V. Miguet und A. Deleforge. “DREGON: Dataset and Methods for UAV-Embedded Sound Source
Localization”. In: 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Madrid: IEEE, 2018,
S. 1–8.

[26] Y.-J. Go und J.-S. Choi. “An Acoustic Source Localization Method Using a Drone-Mounted Phased Microphone Array”. In:

Drones 5.3 (2021), S. 75.

[27] W. N. Manamperi, T. D. Abhayapala, P. N. Samarasinghe und J. ( Zhang. “Drone audition: Audio signal enhancement from
drone embedded microphones using multichannel Wiener ﬁltering and Gaussian-mixture based post-ﬁltering”. In: Applied
Acoustics 216 (2024), S. 109818.
L. Wang und A. Cavallaro. “A Blind Source Separation Framework for Ego-Noise Reduction on Multi-Rotor Drones”. In:
IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020), S. 2523–2537.

[28]

[29] G. Herold u. a. “Detection and Separate Tracking of Swarm Quadcopter Drones Using Microphone Array Measurements”.
en. In: Proceedings on CD of the 8th Berlin Beamforming Conference. Berlin: Gesellschaft zur Förderung angewandter
Informatik (GFaI), 2020, 19 pages.

[30] G. Herold, P. Testa, J. Foerster, M. Uijt de Haag und E. Sarradj. “Measurement of sound emission characteristics of
quadcopter drones under cruise condition”. In: Quiet Drones 2022 - 2nd International e-Symposium on UAV / UAS Noise.
Paris, France, 2022, S. 164–173.

[31] G. Herold und E. Sarradj. “Gerichtete Schallabstrahlung einer Quadkopter-Drohne bei unterschiedlichen Flugzuständen”.
In: Fortschritte der Akustik - DAGA 2023, 49. Jahrestagung für Akustik. Hamburg: Deutsche Gesellschaft für Akustik e.V.
(DEGA), 2023, S. 369–372.

[32] G. Herold und E. Sarradj. “Microphone array based trajectory reconstruction for small UAV ﬂy-by measurements”. In:

Proceedings of the 29th AIAA/CEAS Aeroacoustics Conference. San Diego, CA, 2023.

[33] R. Merino-Martínez u. a. “A Review of Acoustic Imaging Methods Using Phased Microphone Arrays”. In: CEAS Aeronautical

Journal 10.1 (2019), S. 197–230.

[34] G. Herold und E. Sarradj. “Microphone array method for the characterization of rotating sound sources in axial fans”. In:

Noise Control Engineering Journal 63.6 (2015), S. 546–551.

[35] A. Kujawski, G. Herold und E. Sarradj. “A Deep Learning Method for Grid-Free Localization and Quantiﬁcation of Sound

Sources”. en. In: The Journal of the Acoustical Society of America 146.3 (2019), EL225–EL231.

[36] S. Jekosch und E. Sarradj. “An Inverse Microphone Array Method for the Estimation of a Rotating Source Directivity”. en.

In: Acoustics 3.3 (2021), S. 462–472.

[37] G. Herold, F. Zenger und E. Sarradj. “Inﬂuence of blade skew on axial fan component noise”. In: International Journal of

Aeroacoustics 16.4-5 (2017), S. 418–430.

[38] E. Sarradj. “A generic approach to synthesize optimal array microphone arrangements”. In: Proceedings of the 6th Berlin

Beamforming Conference. Berlin, 2016.

[39] E. Sarradj, C. Fritzsche und T. Geyer. “Silent Owl Flight: Bird Flyover Noise Measurements”. In: AIAA Journal 49.4 (2011),

[40]

S. 769–779.
J. A. Ballesteros, E. Sarradj, M. D. Fernández, T. Geyer und M. J. Ballesteros. “Noise source identiﬁcation with Beamforming
in the pass-by of a car”. In: Applied Acoustics 93 (2015), S. 106–119.

[41] A. Kujawski und E. Sarradj. “Application of the CLEANT Method for High Speed Railway Train Measurements”. en. In:
Proceedings on CD of the 8th Berlin Beamforming Conference. Berlin: Gesellschaft zur Förderung angewandter Informatik
(GFaI), 2020, S. 1–13.

[42] M. Czuchaj, S. Jekosch, A. Kujawski und E. Sarradj. “Quantitative Charakterisierung von Schallquellen mit Mikrofonarrays
bei der Vorbeifahrt von Zügen”. de. In: Fortschritte der Akustik - DAGA 2023, 49. Jahrestagung für Akustik. Hamburg:
Deutsche Gesellschaft für Akustik e.V. (DEGA), 2023, S. 365–368.

Seite 17 von 21

[43] G. Herold, T. Geyer, P. Markus und E. Sarradj. “Simultaneous Sound Power Measurement of Engine Components”. In:

SAE International Journal of Passenger Cars-Mechanical Systems 9.2016-01-1774 (2016), S. 974–979.

[44] G. Herold. “One Ring to Find Them All – Detection and Separation of Rotating Acoustic Features with Circular Microphone

Arrays”. en. Diss. Technische Universität Berlin, 2021.

[45] S. Jekosch. “Methods for analyzing and characterizing sound sources in rotating systems”. Diss. Technische Universität

Berlin, 2023.

[46] G. Herold, S. Jekosch, T. Jüterbock und E. Sarradj. “Virtual Microphone Array Rotation in the Mode-Time Domain and
Separation of Stationary and Rotating Sound Sources in an Axial Fan”. en. In: Proceedings of the 9th Berlin Beamforming
Conference. Berlin: Gesellschaft zur Förderung angewandter Informatik (GFaI), 2022, S. 1–10.

[47] H. Brick, T. Kohrs, E. Sarradj und T. Geyer. “Noise from high-speed trains: Experimental determination of the noise radiation

of the pantograph”. In: Forum Acusticum 2011, Aalborg. 2011.

[48] E. Sarradj und G. Herold. “A Python framework for microphone array data processing”. In: Applied Acoustics 116 (2017),

S. 50–58.

[49] E. Sarradj, G. Herold und S. Jekosch. “Array Methods: which one is the best?” In: Proceedings of the 7th Berlin Beamfor-

ming Conference. BeBeC-2018-S01. 2018, S. 1–12.

[50] S. Jekosch, A. Kujawski und E. Sarradj. “Charakterisierung der Performance von inversen Mikrofonarrayverfahren für ro-
tierende Schallquellen”. de. In: Fortschritte der Akustik - DAGA 2021, 47. Jahrestagung für Akustik. Wien: Deutsche Ge-
sellschaft für Akustik e.V. (DEGA), 2021.

[51] G. Herold und E. Sarradj. “Vorhersage der Anwendungsgrenzen von virtuell rotierenden Mikrofonarrays”. de. In: Fortschritte
der Akustik - DAGA 2022, 48. Jahrestagung für Akustik. Stuttgart: Deutsche Gesellschaft für Akustik e.V. (DEGA), 2022,
S. 799–802.

[52] A. Kujawski, G. Herold, E. Sarradj und S. Jekosch. “SpectAcoular! an Extension for Acoular”. de. In: Fortschritte der Akustik

- DAGA 2021, 47. Jahrestagung für Akustik. Wien: Deutsche Gesellschaft für Akustik e.V. (DEGA), 2021, S. 824–827.

[53] A. Kujawski, A. J. R. Pelling, S. Jekosch und E. Sarradj. “A framework for generating large-scale microphone array data for

machine learning”. In: MultiMedia Tools and Applications (2023).

[54] E. Sarradj. “Optimal planar microphone array arrangements”. In: Fortschritte der Akustik-DAGA. 2015, S. 220–223.
[55]

L. Tan und J. Jiang. “Simpliﬁed Gradient Adaptive Harmonic IIR Notch Filter for Frequency Estimation and Tracking”. In:
American Journal of Signal Processing 5.1 (2015), S. 6–12.

[56] E. Sarradj, C. Fritzsche und T. Geyer. “Silent Owl Flight: Bird Flyover Noise Measurements”. In: 16th AIAA/CEAS Aeroa-

coustics Conference (31th AIAA Aeroacoustics Conference), AIAA paper 2010-3991. 2010.

[57] P. Sijtsma. “CLEAN based on spatial source coherence”. In: International Journal of Aeroacoustics 6.4 (2007), S. 357–374.
[58] R. Cousson, Q. Leclère, M. A. Pallas und M. Bérengier. “A time domain CLEAN approach for the identiﬁcation of acoustic

[59]

moving sources”. In: Journal of Sound and Vibration 443 (2019), S. 47–62.
T. Yardibi, J. Li, P. Stoica und L. N. Cattafesta. “Sparsity constrained deconvolution approaches for acoustic source map-
ping.” In: The Journal of the Acoustical Society of America 123.5 (2008), S. 2631–2642.

[60] P. Gerstoft, C. F. Mecklenbrauker, A. Xenaki und S. Nannuru. “Multisnapshot Sparse Bayesian Learning for DOA”. In: IEEE

Signal Processing Letters 23.10 (2016), S. 1469–1473.

[61] G. Ping, E. Fernandez-Grande, P. Gerstoft und Z. Chu. “Three-dimensional source localization using sparse Bayesian
learning on a spherical microphone array”. In: The Journal of the Acoustical Society of America 147.6 (2020), S. 3895–
3904.
Joint Committee for Guides in Metrology. JCGM 100:2008 - Evaluation of measurement data - Guide to the expression of
uncertainty in measurement. 2008.

[62]

[63] Deutsches Institut für Normung e.V. DIN EN 61672-1:2014-07 Schallpegelmesser. 2014.
[64] Deutsches Institut für Normung e.V. DIN EN 62585 - Elektroakustik – Verfahren zur Ermittlung von Korrekturwerten für die

Bestimmung des Freifeld-Frequenzgangs eines Schallpegelmessers. 2012.

[65] E. Sarradj, S. Jekosch und G. Herold. “An Efﬁcient Ray Tracing Approach for Beamforming on Rotating Sources in the Pre-
sence of Flow”. en. In: Proceedings on CD of the 8th Berlin Beamforming Conference. Berlin: Gesellschaft zur Förderung
angewandter Informatik (GFaI), 2020, 9 Pages.
Joint Committee for Guides in Metrology. JCGM 101:2008 - Evaluation of measurement data - Supplement 1 to the “Guide
to the expression of uncertainty in measurement” - Propagation of distributions using a Monte Carlo method. 2008.
[67] E. Sarradj und T. Geyer. “Symbolic regression modeling of noise generation at porous airfoils”. In: Journal of Sound and

[66]

Vibration 333.14 (2014), S. 3189–3202.

[68] Deutsches Institut für Normung e.V. DIN EN ISO 3744:2011-02 Akustik - Bestimmung der Schallleistungs- und Schall-
energiepegel von Geräuschquellen aus Schalldruckmessungen - Hüllﬂächenverfahren der Genauigkeitsklasse 2 für ein im
Wesentlichen freies Schallfeld über einer reﬂektierenden Ebene. 2011.

Seite 18 von 21

4 Begleitinformationen zum Forschungskontext

4.1 Angaben zu ethischen und/oder rechtlichen Aspekten des Vorhabens

4.1.1 Allgemeine ethische Aspekte

Es sind keine Risiken und/oder Belastungen für Personen bzw. Personengruppen und/oder mögliche wei-
tere negative Auswirkungen durch das Forschungsvorhaben zu erwarten.

4.1.2 Erläuterungen zu den vorgesehenen Untersuchungen am Menschen, an vom Menschen

entnommenem Material oder mit identiﬁzierbaren Daten

Das Forschungsvorhaben sieht keine Untersuchungen am Menschen, an vom Menschen entnommenem
Material oder mit identiﬁzierbaren Daten vor.

4.1.3 Erläuterungen zu den vorgesehenen Untersuchungen bei Versuchen an Tieren

Das Forschungsvorhaben sieht keine Versuche an Tieren vor.

4.1.4 Erläuterungen zu Forschungsvorhaben an genetischen Ressourcen (oder darauf bezogenem

traditionellem Wissen) aus dem Ausland

Genetische Ressourcen sind nicht Teil des Forschungsvorhabens.

4.1.5 Erläuterungen zu möglichen sicherheitsrelevanten Aspekten („Dual-Use Research of

Concern“; Außenwirtschaftsrecht)

Es gibt keine unmittelbaren Anhaltspunkte für möglichen schädlichen Missbrauch der Forschungsergeb-
nisse, Wissen oder Technologien. Zu beachten ist jedoch, dass eng verwandte Aufgabenstellungen, wie
sie sich unter dem Schlagwort „drone audition“ ﬁnden, deutliches Potenzial für eine sicherheitsrelevante
bzw. militärische Anwendung bieten. Dazu zählen beispielsweise die verdeckte Erfassung von Audiosigna-
len durch Drohnen oder auch die Ortung von Schallquellen wie anderen Drohnen, die bei militärischen
Angriffshandlungen unterstützen oder diese auch abwehren kann.

Im beantragten Vorhaben steht die Messung des Schalldruckpegels und nicht die Ortung oder die Erfas-
sung von (abhörbaren) Audiosignalen im Mittelpunkt. Damit ist nicht zu erwarten, dass die Projektergeb-
nisse für die erwähnten Anwendungen von Relevanz sind.

4.2 Angaben zur Dienststellung

• Ennes Sarradj: Universitätsprofessor (auf Lebenszeit)
• Gert Herold: wissenschaftlicher Mitarbeiter, haushaltsﬁnanziert, befristet bis 31.7.2027 mit Anschluss-

zusage für ein unbefristetes Arbeitsverhältnis

4.3 Angaben zur Erstantragstellung

-

4.4 Zusammensetzung der Projektarbeitsgruppe

Personen, die neben dem Antragsteller am Projekt arbeiten, aber nicht aus diesem ﬁnanziert werden:

• Dr.-Ing. Roman Tschakert, technischer Mitarbeiter, haushaltsﬁnanziert
• Torsten Daniel, Feinmechanikmeister, haushaltsﬁnanziert

Seite 19 von 21

4.5 Zusammenarbeit mit Wissenschaftlerinnen und Wissenschaftlern in Deutschland in

diesem Projekt

Arbeitsgruppe von Prof. Marten Uijt de Haag, TU Berlin hinsichtlich der für Vorversuche verwendeten Droh-
nen (siehe AP 1)

4.6 Zusammenarbeit mit Wissenschaftlerinnen und Wissenschaftlern im Ausland in

diesem Projekt

-

4.7 Wissenschaftlerinnen und Wissenschaftler, mit denen in den letzten drei Jahren

wissenschaftlich zusammengearbeitet wurde

Prof. Lars Enghardt, Dr. Thomas Geyer (DLR Cottbus), Prof. Klaus Höschler, Prof. Heiko Schmidt, Dr.
Sparsh Sharma (DLR Braunschweig), Prof. Niels Modler (TU Dresden), Dr. Danielle Moreau (UNSW, Syd-
ney), Prof. Lorna Ayrton (Cambridge University), Prof. Pieter Sijtsma, Prof. Mirijam Snellen (TU Delft), Dr.
Christopher Bahr (NASA Langley), Prof. Paolo Castellini (Uni Ancona), Prof. Mario Kupnik (TU Darmstadt),
Prof. Wei Ma (Shanghai Jiao Tong University), Prof. David Thompson (University of Southampton), Prof.
Mats Abom (KTH Stockholm), Prof. Ines Lopez (TU Eindhoven)

4.8 Projektrelevante Zusammenarbeit mit erwerbswirtschaftlichen Unternehmen

-

4.9 Projektrelevante Beteiligungen an erwerbswirtschaftlichen Unternehmen

-

4.10 Apparative Ausstattung

Die für das Projekt benötigte Ausstattung ist vorhanden:

• akustische Standardmesstechnik, Messmikrofone, Schallpegelmesser, Lautsprecher
• Reﬂexionsarmer Raum 1070 m³, untere Grenzfrequenz 63 Hz
• Werkstatt inkl. 3D-Drucker

Drohnen für die Vorversuche sind wie in AP 1 beschrieben durch eine Kooperation verfügbar.

4.11 Weitere Antragstellungen

-

4.12 Weitere Angaben

-

5 Beantragte Module/Mittel

5.1 Basismodul

Antragsteller: Ennes Sarradj = ES, Gert Herold = GH

Seite 20 von 21

5.1.1 Personalmittel

ein wissenschaftliche/r Mitarbeiter/in (Kategorie Doktorand/Doktorandin oder Vergleichbare)

• ES: 18 Monate, TV-L 13, 100% der regelmäßigen Arbeitszeit, AP 1 und 2
• GH: 18 Monate, TV-L 13, 100% der regelmäßigen Arbeitszeit, AP 3-5

Die Stellen dienen der Projektbearbeitung für sämtliche vorgesehenen Arbeitspakete. Es wird ange-
strebt, gemeinsam eine Person für die Dauer von drei Jahren einzustellen. Erforderlich sind fundierte
Kenntnisse der Technischen Akustik, der Signalverarbeitung sowie des Umgangs mit der hier erfor-
derlichen Mess- und Rechentechnik.

eine studentische Hilfskraft

• ES: 12 Monate, TV STUD, 60 h/Monat

11520 Euro

für Unterstützung bei der Auslegung, Konstruktion und Erprobung im AP 1

• GH: 12 Monate, TV STUD, 60 h/Monat

11520 Euro
für Unterstützung bei der Vorbereitung und Durchführung der experimentellen Arbeiten in AP 3
und AP 5

5.1.2 Sachmittel

Geräte bis 10.000 Euro, Software und Verbrauchsmaterial

• PCB-Fertigung und Bestückung als externer Auftrag:
• Material, Mechanikteile und Verbrauchsmaterial 3D-Druck:
• Elektronikbauteile und -baugruppen, Einplatinencomputer:

ES:

GH:

• Multicopter-Drohne einschließlich notwendiger Peripherie

(Steuerung, Transportbehälter etc.) für die endgültige Realisierung im
Anschluss an die Indoor-Versuche in AP3:

Reisemittel

ES:

• Konferenzbesuch international:
• Konferenzbesuch national:

GH:

• Konferenzbesuch international:
• Konferenzbesuch national:

Mittel für wissenschaftliche Gäste (ausgenommen Mercator-Fellow)

-

Mittel für Versuchstiere

-

3000 Euro
1000 Euro
1000 Euro

Summe: 5.000 Euro

2500 Euro

Summe: 2.500 Euro

2.000 Euro
800 Euro

Summe: 2.800 Euro

2.000 Euro
800 Euro

Summe: 2.800 Euro

Seite 21 von 21

Sonstige Mittel

-

Publikationsmittel

Veröffentlichungen in einer begutachteten Fachzeitschrift mit freiem Zugang (Open-Access, z.B. Acoustics,
Acta Acustica united with Acustica) sind vorgesehen,

• ES: APC für eine Veröffentlichung
• GH: APC für eine Veröffentlichung

1100 Euro
1100 Euro

5.1.3 Investitionsmittel

Geräte über 10.000 Euro

-

Großgeräte über 50.000 Euro

-

