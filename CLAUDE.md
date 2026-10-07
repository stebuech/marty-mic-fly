# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MartyMicFly** is a DFG research project ("Fliegendes Messmikrofon"). The onboard Raspberry Pi 4 records a 16-channel PDM microphone array (miniDSP MCHStreamer), ESC telemetry and optical rotor tacho pulses; offline analysis (`src/martymicfly`, `analysis/`) separates the drone's own noise. Rotor phase capture concept: `AP1_Rotorphasenerfassung_Umsetzungsplan.md`.

## Development Commands

### Environment Setup
```bash
# The project uses uv for dependency management
uv sync                    # Install/sync dependencies from uv.lock
uv add <package>           # Add a new dependency
uv run pytest              # Tests (tests/test_pico_link.py covers the Pico protocol and HDF5 logger)
```

On the onboard Pi (`ssh steffen@130.149.163.48`, Debian trixie, PREEMPT_RT kernel) the working Python env is
`~/miniforge3/envs/rpm` (the repo's `.venv` there is not usable).

### Running the Main Applications
```bash
# Recorder: gate mode (RC switch on GPIO17 starts/stops one file per HIGH phase), mic array + Pico board
python daq.py
python daq.py --no-trigger-controlled --output-folder ~/MMFDataLogs/   # manual, Ctrl+C to stop
python daq.py --help                                                   # all options (BooleanOptional flags)

# Pico tacho board bench check (rates, jitter, ESC frames, error counters)
python pico_link.py --monitor 10

# Throttle control via DShot (separate Python process, pigpio)
python esc_throttle_set.py
python throttle_rpm_mapping.py         # calibration, telemetry via Pico by default
```

Pico firmware build/flash: `firmware/tacho_pico/README.md`.

## Architecture

### Hardware (Rev. G3 shield on the Pi 4)

- **Pico 2 (RP2350) tacho board** on `/dev/ttyAMA5` (Pi GPIO12/13 = UART5, 921600 8N1; Pico UART0 GP0/GP1).
  Pico GP2 = PDM_CLK from the MCHStreamer, GP3–GP8 = TACHO1–6, GP9–GP14 = ESC telemetry TEL1–6 (KISS, 115200).
  ESC telemetry is **no longer wired to Pi UARTs**; `/dev/ttyAMA0/2/3/4` only matter for older boards
  (`--telemetry-source uart`).
- **Gate/trigger** from the Pixhawk on Pi GPIO17 (RC PWM, decoded by pigpio).
- **DShot outputs** on Pi GPIO18, 19, 20, 21, 16, 26 (`esc_throttle_set.py`, pigpio waves). Motor n = DSHOT n = TEL n = TACHO n.
  Pin defaults live in `hw_config.py`; scripts take `--motors 4` (quadcopter, current default) or `--motors 6` (hexacopter).
- The heartbeat loopback (GPIO22 → 23) was removed in Rev. G3; the signal monitor/heartbeat options remain for old boards.
- `pigpiod` must run (`sudo pigpiod`); `dtoverlay=uart5` (plus uart2–4 for old boards) in `/boot/firmware/config.txt`.

### Pico firmware (`firmware/tacho_pico`)

PIO only detects events; DMA channel pairs copy the event word and TIMER1's raw counter (counting clk_sys, 150 MHz)
into ring buffers, so all channels share one timebase with constant latency. The CPU extends timestamps to 64 bit
(`ts_extend`, stateless) and frames packets. Protocol source of truth: `src/protocol.h`; host side
`pico_link.py` must match (`PROTO_VERSION`). A self-test build generates all signals on spare pins.

### Host side

- `pico_link.py`: `PicoLink` reader thread + frame parser, `KissStream` (KISS reassembly from the Pico's byte chunks),
  `ClockMap` (Pico ticks → Pi time via PING/PONG), `PicoESCChannel` (same interface as `daq.ESCTelemtry`).
- `daq.py`: `Timer` (time.perf_counter), `SignalMonitor` (pigpio gate/edges, logs pigpio ticks), `ESCTelemtry`
  (legacy UART), `MicArray` (sounddevice callback: copy + per-block record only), `HDF5Logger` (single writer
  thread), `DAQ` (manual and gate mode).

### Data Storage (HDF5, `format_version` 2)

```
/timing            attrs start_time_wall, start_time_perf ("timestamp" columns = s since start_time_perf)
/mic_array/audio_data (N, 16), /mic_array/blocks/{first_frame, frames, adc_time, current_time, callback_time, status}
/esc_telemetry/ESCn/{timestamp, pico_ticks, temperature, voltage, current, consumption, erpm, rpm}
/pico/clk/{seq, ticks}            one event per clk_div PDM_CLK edges (= clk_div/64 samples)
/pico/tacho/{seq, ticks, state, changed}
/pico/ping/{id, pi_send, pi_recv, pico_ticks}, /pico/status/*, /pico/tel_chunks/*, attrs tick_hz, clk_div, fw
/gate/{timestamp, state, tick}    decoded RC switch;  /trigger/* log-pin edges (old boards)
```

Per-sample mic timestamps are no longer stored; the sample clock plus `/pico/clk` is the time reference.
RPM: `rpm = erpm / pole_pairs` (default 12 pole pairs; `erpm` is stored raw).

### Threading Model

Pico reader + ping thread, PortAudio callback (no I/O), HDF5 writer thread (the only thread touching the file),
gate poll thread, pigpio callbacks. All Pi-side timestamps use one `Timer` (CLOCK_MONOTONIC).

## Dependencies

Core dependencies (from pyproject.toml):
- `pigpio`: Hardware GPIO control with precise timing
- `pyserial`: Serial communication
- `h5py`: HDF5 data logging
- `numpy`: Numerical operations
- `sounddevice`: MCHStreamer capture

Python requirement: >=3.13 (the Pi env runs 3.12, which works for the acquisition scripts)
