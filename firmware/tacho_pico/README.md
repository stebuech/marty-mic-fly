# Tacho-Board-Firmware (Pico 2 / RP2350, Platine Rev. G3)

Zeitstempelt PDM-Takt, optische Tachopulse und ESC-Telemetrie in **einer** Zeitbasis und schickt alles über
UART an den Onboard-Pi. Host-Seite: `pico_link.py` (Empfänger), `daq.py` (HDF5). Konzept: `AP1_Rotorphasenerfassung_Umsetzungsplan.md`.

## Belegung (Rev. G3, as built)

| Pico | Signal | Erfassung |
|---|---|---|
| GP2 | PDM_CLK (J7 → U5) | PIO0 SM0 `clkdiv`: Ereignis alle `clk_div` steigenden Flanken (Default 16384 = 256 Samples) |
| GP3–GP8 | TACHO_1–6 (J1–J6 → U1–U4, U6, U7) | PIO0 SM1 `edge_watch`: jede Flanke, Pinzustand aller 6 Kanäle |
| GP9–GP14 | TEL1–6 (J10–J13, J22, J23) | PIO1 SM0–3, PIO2 SM0–1 `uart_rx`, 115200 8N1 |
| GP0 / GP1 | UART0 TX/RX ↔ Pi UART5 (`/dev/ttyAMA5`, Header 33/32) | 921600 8N1 |
| GP25 | Onboard-LED | blinkt mit 1 Hz (STATUS) |

## Zeitbasis

- TIMER1 zählt direkt `clk_sys` (RP2350-Register `TIMER.SOURCE`), also **150 MHz, 6,7 ns Auflösung**.
- PIO erkennt nur Ereignisse. Pro Quelle schiebt ein DMA-Kanalpaar das FIFO-Wort in einen Ringpuffer und kopiert
  danach `TIMERAWL` in einen Zeitstempel-Ring (Kette A → B → A). Die CPU ist nicht im Erfassungspfad, und die
  DMA-Latenz ist ein konstanter Offset.
- Erweiterung auf 64 Bit beim Auslesen mit `ts_extend()`: zustandslos gegen den aktuellen 64-Bit-Zählerstand,
  eindeutig, solange ein Eintrag jünger als 2³² Ticks (28,6 s) ist. Es gibt keinen Zustand pro Kanal, ein
  stehender Rotor kann also keine Epoche verlieren. Der Unit-Test geht über viele Überläufe
  (`tests/test_codec.c`, Plan-Test 4c).
- Gemessen am Board (Self-Test, PWM-Quellen aus demselben Quarz): Takt-Stützstellen σ = 0,7 Ticks (≈ 4,5 ns),
  Tachoperioden σ = 0,6 Ticks.

## Protokoll (Pico → Pi und Pi → Pico)

`0xA5 0x5A | type | len | payload | crc16 LE`, CRC-16/CCITT-FALSE über type, len und payload; Payload little endian.
Maßgeblich ist `src/protocol.h`, `pico_link.py` muss dazu passen (`PROTO_VERSION`).

| Typ | Richtung | Payload |
|---|---|---|
| 0x01 INFO | → Pi | proto u16, tick_hz u32, clk_div u32, n_tacho u8, n_tel u8, tel_baud u32, flags u8 (bit0 Self-Test), t_now u64, fw char[16]. Beim Start, alle 5 s und auf Anfrage |
| 0x02 STATUS | → Pi | 1 Hz: t_now, TIMER0-µs, Zähler (clk, tacho, link_drops, rx_crc, tacho_stalls, loop_max_us), pro TEL Bytes/Framing-Fehler/Überläufe |
| 0x10 CLK | → Pi | seq u32, t u64 |
| 0x11 TACHO | → Pi | seq u32, t u64, state u8, changed u8 (Bit n = TACHO n+1) |
| 0x12 TEL | → Pi | ch u8, flags u8, seq u32, t_first u64, n u8, data[n]: rohe ESC-Bytes, durch Leitungspausen (> 3 Bytezeiten) in Stücke geteilt. KISS-Framing und CRC8 prüft der Pi |
| 0x13 PONG | → Pi | id u32, t_rx u64 (Antwort auf PING, für die Abbildung Pi-Zeit ↔ Pico-Zeit) |
| 0x80 PING | → Pico | id u32 |
| 0x81 GET_INFO | → Pico | – |
| 0x82 SET_CLKDIV | → Pico | div u32 (Vielfaches von 64, ≥ 1024) |
| 0x8F BOOTSEL | → Pico | magic u32 0xB0075E1F: Neustart in den USB-Bootloader |

Last bei 6 Rotoren (beide Flanken), 6 ESCs à 100 Hz und Takt: ≈ 35 kB/s von 92 kB/s.

## Bauen

```bash
export PICO_SDK_PATH=~/.pico-sdk/sdk-2.2.0
export PICO_TOOLCHAIN_PATH=~/.pico-sdk/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi
cmake -S firmware/tacho_pico -B firmware/tacho_pico/build -G Ninja
ninja -C firmware/tacho_pico/build      # -> tacho_pico.uf2, tacho_pico_selftest.uf2
```

Host-Tests (Codec, CRC, 64-Bit-Erweiterung, Python-Decoder gegen C-Encoder): `uv run pytest tests/test_pico_link.py`.

## Flashen (Pico-USB am Pi)

```bash
python pico_link.py --bootsel                 # Pico meldet sich als Laufwerk RP2350
cp tacho_pico.uf2 /media/$USER/RP2350/
python pico_link.py --monitor 10              # Raten, Jitter, ESC-Frames, Fehlerzähler
```

Ohne laufende Firmware (leerer Pico) erscheint das Laufwerk von selbst, sonst BOOTSEL-Taste beim Einstecken.
`picotool reboot -f -u` geht ebenfalls, weil die Firmware das USB-Reset-Interface anbietet.

## Self-Test-Build (`tacho_pico_selftest.uf2`)

Braucht nichts angeschlossen. PWM auf GP22 (3,0 MHz) ersetzt PDM_CLK, PWM auf GP16–GP21 (≈ 57/56/54 Hz)
ersetzt die Sensoren, und eine PIO-UART auf GP26 sendet KISS-Frames mit 100 Hz, auf die alle sechs TEL-Empfänger
hören. INFO meldet `selftest: True`, daq.py zeigt beim Start `SELFTEST BUILD`.

## Bekannte Hardware-Punkte

- J7/U5-Eingang hat keinen Pull-down. Ohne gesteckten MCHStreamer-Takt kann GP2 Störflanken sehen
  (Rev.-H-Punkt, Notlösung 100 k von U5 Pin 2 nach Pin 3).
- TEL-Eingänge haben keinen Serienwiderstand und keine Klemmdiode, sie setzen also 3,3-V-Pegel vom ESC voraus.
