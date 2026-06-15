# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MartyMicFly** is a DFG research project focused on RPM monitoring and ESC (Electronic Speed Controller) telemetry for a "Flying Measurement Microphone" system. The project runs on Raspberry Pi hardware and interfaces with multiple ESCs to monitor motor RPM, control throttle, and synchronize data collection.

## Development Commands

### Environment Setup
```bash
# The project uses uv for dependency management
uv sync                    # Install/sync dependencies from uv.lock
uv add <package>           # Add a new dependency
```

### Running the Main Applications
```bash
# Monitor ESC telemetry (requires hardware connections)
python rpm_monitoring/kiss_tel_multi_esc_monitor.py

# With command-line options:
python rpm_monitoring/kiss_tel_multi_esc_monitor.py \
  --enable-trigger \
  --trigger-pin 17 \
  --data-logging \
  --output_folder ~/TelemetryLogs/

# Control ESC throttle via DShot protocol
python rpm_monitoring/esc_throttle_set.py
```

## Architecture

### Hardware Interface Layer

The system interfaces with Raspberry Pi GPIO using the **pigpio** library (migrated from RPi.GPIO). Key hardware interfaces:

- **Serial Communication**: Reads KISS telemetry protocol from ESCs via UART (default: /dev/ttyAMA0-4)
- **GPIO Interrupts**: Hardware-triggered event capture for synchronization (default: GPIO 17)
- **DShot Protocol**: Bidirectional digital motor control protocol (DShot150/300/600)

### Core Components

**1. Telemetry Monitoring (`kiss_tel_multi_esc_monitor.py`)**

Multi-threaded architecture for real-time ESC monitoring:

- `Timer`: High-precision timing using `time.perf_counter()` for synchronization
- `TriggerMonitor`: Hardware interrupt-based trigger event capture via pigpio callbacks
- `KISSTelemtryMonitor`: Serial protocol parser for KISS telemetry (temp, voltage, current, RPM)
  - Implements CRC8 checksum validation
  - Automatic packet synchronization
  - Per-ESC thread with circular buffer (`deque`)
- `TelemetryHDF5Logger`: Streaming HDF5 writer with periodic flushing
  - Extendable datasets with gzip compression
  - Separate groups for timing, triggers, and per-ESC telemetry
- `MultiESCMonitor`: Orchestrates multiple ESC monitors with optional trigger and logging

**Data Flow**: Serial Port → Buffer → CRC Validation → Thread-safe Deque → Periodic HDF5 Flush

**2. ESC Control (`esc_throttle_set.py`)**

DShot protocol implementation for ESC throttle control:

- `DShot`: Single ESC controller with frame generation and CRC
  - Generates precisely-timed waveforms using pigpio waves
  - Supports throttle range (48-2047) and special commands (0-47)
- `MultiESC`: Synchronized control of multiple ESCs
  - Parallel arming/disarming
  - Individual or collective throttle setting

**Critical Timing**: DShot600 requires 1.67μs bit timing; uses pigpio hardware-timed pulses

### Threading Model

- **ESC Monitor Threads**: One per ESC, continuously reads serial data using `select()` with 0.1s timeout
- **Logging Thread**: Periodic flush to HDF5 (default: 1.0s interval)
- **Main Thread**: User interface and coordination
- **Trigger Callbacks**: Hardware interrupt context (pigpio callback)

All threads share a single `Timer` instance for synchronized timestamps. Thread-safe access via `threading.Lock`.

### KISS Telemetry Protocol

10-byte packet structure per KISS specification:
- Byte 0: Temperature (°C)
- Bytes 1-2: Voltage (×0.01V, big-endian)
- Bytes 3-4: Current (×0.01A, big-endian)
- Bytes 5-6: Consumption (mAh, big-endian)
- Bytes 7-8: RPM (×100 × 2 / pole_pairs, big-endian)
- Byte 9: CRC8 checksum

Default configuration: 14 pole pairs, 115200 baud

### Data Storage

HDF5 hierarchical structure:
```
/timing
  - attrs: start_time_wall, start_time_perf
/trigger (optional)
  - timestamp[]: float64 array
  - state[]: int8 array
/esc_telemetry
  /ESC1, /ESC2, etc.
    - timestamp[], temperature[], voltage[], current[], consumption[], rpm[]
    - attrs: port, total_samples, avg_sample_rate, sample_rate_std
```

## Hardware Configuration

Default serial ports: `/dev/ttyAMA0`, `/dev/ttyAMA2`, `/dev/ttyAMA3`, `/dev/ttyAMA4`
Default trigger pin: GPIO 17 (BCM numbering)
Default DShot pins: GPIO 18, 19, 20, 21

Requires `pigpiod` daemon running:
```bash
sudo pigpiod  # Start pigpio daemon before running scripts
```

## Dependencies

Core dependencies (from pyproject.toml):
- `pigpio`: Hardware GPIO control with precise timing
- `pyserial`: Serial communication for KISS telemetry
- `h5py`: HDF5 data logging
- `numpy`: Numerical operations
- `ipython`: Interactive development

Python requirement: >=3.13