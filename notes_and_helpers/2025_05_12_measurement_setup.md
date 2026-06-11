# Measurement Setup Documentation
# RAR Drone Acoustic Measurements - December 2025

**Date:** 2025-12-05
**Location:** RAR (Reflektionsarmer Raum / Anechoic Chamber)
**Project:** MartyMicFly - Flying Measurement Microphone
**Measurement ID:** RAR_Drone_TRANSIT_BVG_Array

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Setup](#hardware-setup)
3. [Coordinate System](#coordinate-system)
4. [Test Configurations](#test-configurations)
5. [Test Signal Specifications](#test-signal-specifications)
6. [RPM Data Synchronization](#rpm-data-synchronization)
7. [Data Structure](#data-structure)
8. [Measurement Protocol](#measurement-protocol)
9. [Data Analysis Notes](#data-analysis-notes)

---

## Overview

This measurement campaign captures acoustic data from a quadcopter drone in an anechoic chamber using an external 96-channel microphone array. The primary objectives are:

- Characterize drone ego-noise patterns in controlled environment
- Validate noise suppression algorithms (notch filters, spatial filters)
- Measure acoustic transfer functions with known test signals
- Compare "pull" vs "push" rotor configurations

**Key Features:**
- Drone fixed in position (non-flying)
- Synchronized RPM playback ensures repeatability
- Multiple test signals at calibrated SPL
- Hardware synchronization via heartbeat signal

---

## Hardware Setup

### Drone Configuration

| Parameter | Value |
|-----------|-------|
| Type | Quadcopter |
| Mounting | Fixed on stand (non-flying) |
| Number of Motors | 4 |
| Number of Blades per Rotor | 2 |
| Pole Pairs | 14 |
| ESC Telemetry Protocol | KISS |
| Control Protocol | DShot 300 |

### Microphone Array

| Parameter | Value |
|-----------|-------|
| Number of Channels | 96 |
| Array Type | External (BVG TRANSIT array) |
| Array Geometry | See `mic_ref.xml` in measurement folder |
| Sampling Rate | 51.2 kHz |
| Data Format | HDF5 (.h5) |

**Special Channel Assignment:**
- **Channel 1:** Heartbeat synchronization signal (not acoustic data)
- **Channels 2-96:** Acoustic measurements

### Audio Playback

| Parameter | Value |
|-----------|-------|
| Speaker Type | External loudspeaker in anechoic chamber |
| Test Signal Generation | `generate_test_audio_signals.py` |
| Target SPL | 90 dB |
| Playback Start | ~15 seconds after RPM replay begins |

### Data Acquisition System

| Component | Details |
|-----------|---------|
| RPM Recording | Raspberry Pi via ESC telemetry |
| RPM Replay | `replay_and_record.py` script |
| Mic Array Recording | BVG TRANSIT system |
| Synchronization | GPIO heartbeat pin (monitored by both systems) |

---

## Coordinate System

### Drone Position (Fixed)

```
Origin: Center of microphone array
X-axis: [Specify direction]
Y-axis: [Specify direction]
Z-axis: Vertical (positive upward)

Drone Center Position:
  x = 0.000 m
  y = 0.375 m
  z = 0.850 m
```

### Rotor Orientation

**Pull Configuration:**
- Rotors pointing **upward** (standard quadcopter orientation)
- Propellers generate thrust in +Z direction
- Comments in protocol marked as "pull"

**Push Configuration:**
- Rotors pointing **downward** (inverted)
- Propellers remounted for correct push operation
- Propellers generate thrust in -Z direction
- Comments in protocol marked as "push"

---

## Test Configurations

Each measurement consists of one of the following configurations:

| Config ID | Rotors State | Test Signal | Notes |
|-----------|--------------|-------------|-------|
| Baseline | OFF | Various | Reference measurements without drone noise |
| Pull + Signal | ON (upward) | Various | Standard orientation with test signals |
| Push + Signal | ON (downward) | Various | Inverted orientation with test signals |
| Pull Only | ON (upward) | None | Drone noise only (no speaker) |
| Push Only | ON (downward) | None | Drone noise only (no speaker) |

See `2025_12_05_RAR_Drone_1st_test.ods` for complete measurement matrix.

---

## Test Signal Specifications

### RPM Profile

**Source:** Pre-recorded hover flight sequence
**Replay Method:** `replay_and_record.py` with throttle-RPM calibration
**RPM Range:** ~5500-6200 RPM (nominal hover)
**Duration:** Variable (typically 60-90 seconds)
**Repeatability:** Identical sequence replayed for all measurements

### Acoustic Test Signals

All signals generated using `generate_test_audio_signals.py`:

**Signal Parameters:**
- **Sampling Rate:** 48 kHz
- **Duration:** 60 seconds
- **Target SPL:** 90 dB (calibrated)
- **Format:** WAV (16-bit PCM)

**Signal Types Used:**

| Signal Name | Type | Purpose | Frequency Content |
|-------------|------|---------|-------------------|
| `white_noise_90dB.wav` | Broadband noise | Baseline reference | 20 Hz - 24 kHz (flat) |
| `pink_noise_90dB.wav` | 1/f noise | Realistic broadband | 20 Hz - 24 kHz (1/f) |
| `chirp_bpf_linear_90dB.wav` | Linear chirp | BPF sweep | 183-207 Hz (BPF range) |
| `chirp_realistic_rpm_ramp_90dB.wav` | Phase-continuous chirp | Realistic rotor simulation | RPM-synchronized |
| `multitone_bpf_90dB.wav` | Harmonic series | Notch filter testing | BPF harmonics (15 harmonics) |
| `multitone_bpf_8harm_90dB.wav` | Harmonic series | Reduced harmonics | BPF harmonics (8 harmonics) |

**BPF (Blade Passing Frequency) Reference:**
- Minimum: 183.3 Hz (at 5500 RPM)
- Maximum: 206.7 Hz (at 6200 RPM)
- Center: ~195 Hz (at 5850 RPM)

---

## RPM Data Synchronization

### Hardware Synchronization

```
┌─────────────┐      Heartbeat     ┌──────────────┐
│ Raspberry   │─────────GPIO───────▶│ Mic Array    │
│ Pi (RPM)    │      (10 Hz)       │ Recording    │
└─────────────┘                     └──────────────┘
       │                                   │
       │                                   │
   ESC Telemetry                    Channel 1 records
   (KISS protocol)                  heartbeat state
```

**Heartbeat Signal:**
- **Pin:** GPIO 22 (Raspberry Pi output)
- **Monitored by:** Microphone array Channel 1
- **Interval:** 10.0 seconds (configurable)
- **Purpose:** Synchronize RPM data with acoustic recordings

### Synchronization Workflow

1. **Start microphone array recording** (continuous)
2. **Start RPM replay script** (`replay_and_record.py`)
   - Initializes heartbeat generation
   - Records heartbeat state in telemetry HDF5
3. **Wait ~15 seconds** for rotor stabilization
4. **Start speaker playback** of test signal
5. **Continue until** RPM sequence completes
6. **Record 2 additional seconds** after sequence ends
7. **Stop all systems**

### Time Alignment Procedure

For data analysis, align the two data streams using:

1. **Find heartbeat edges** in microphone Channel 1
2. **Find corresponding edges** in RPM telemetry file
3. **Calculate time offset** between the two clocks
4. **Resample or interpolate** to common timebase

---

## Data Structure

### Data Location

```
/home/steffen/MartyMicFly/Messdaten/2025_12_05_RAR_Drone_TRANSIT_BVG_Array/
├── 2025_12_05_RAR_Drone_1st_test.ods    # Measurement protocol
├── mics_ref.xml                          # Microphone positions
├── rpm_data/                             # RPM telemetry files
│   ├── telemetry_data_20251205_150219.h5
│   ├── telemetry_data_20251205_150644.h5
│   └── ...
├── td/                                   # Microphone array data (time domain)
│   ├── 2025-12-05_16-00-07_396994.h5
│   ├── 2025-12-05_16-02-32_565258.h5
│   └── ...
└── test_signals/                         # Reference test signal files
    ├── white_noise_90dB.wav
    ├── pink_noise_90dB.wav
    └── ...
```

### RPM Data Format (HDF5)

**File:** `rpm_data/telemetry_data_YYYYMMDD_HHMMSS.h5`

```
/timing/
  - attrs:
      start_time_wall: (ISO 8601 timestamp)
      start_time_perf: (perf_counter reference)

/trigger/                               # Heartbeat signal
  - timestamp: [N] float64              # Timestamps of heartbeat events
  - state: [N] int8                     # State (0=low, 1=high)

/esc_telemetry/
  /ESC1/, /ESC2/, /ESC3/, /ESC4/
    - timestamp: [M] float64            # Sample timestamps
    - temperature: [M] float64          # °C
    - voltage: [M] float64              # Volts
    - current: [M] float64              # Amperes
    - consumption: [M] float64          # mAh
    - rpm: [M] float64                  # Revolutions per minute
    - attrs:
        port: (serial port)
        total_samples: (int)
        avg_sample_rate: (Hz)
```

**Typical Sample Rate:** ~100-200 Hz (asynchronous KISS telemetry)

### Microphone Array Data Format (HDF5)

**File:** `td/YYYY-MM-DD_HH-MM-SS_microseconds.h5`

**Structure:** (Specific to BVG TRANSIT array - verify with array documentation)

```
/td/                                    # Time domain data
  - dataset: [96, N_samples] float32    # 96 channels × time samples
  - attrs:
      fs: 51200.0                       # Sampling rate (Hz)
      ...
```

**Notes:**
- Channel 1 contains heartbeat signal (not microphone data)
- Channels 2-96 contain acoustic measurements
- Refer to `mics_ref.xml` for microphone positions

---

## Measurement Protocol

**Reference Document:** `2025_12_05_RAR_Drone_1st_test.ods`
(LibreOffice Calc spreadsheet in measurement data folder)

**Protocol Contains:**
- Measurement sequence numbers
- Configuration (pull/push, signal type)
- Start/end times
- File associations (RPM ↔ Mic data)
- Operator notes and observations
- Data quality flags

**Use this document to:**
- Map RPM files to corresponding microphone recordings
- Identify which test signal was used for each measurement
- Understand measurement conditions and any anomalies

---

## Data Analysis Notes

### Recommended Analysis Steps

1. **Load Measurement Protocol**
   - Parse `2025_12_05_RAR_Drone_1st_test.ods`
   - Create lookup table: measurement_id → (rpm_file, mic_file, config)

2. **Time Synchronization**
   - Extract heartbeat from microphone Channel 1
   - Extract heartbeat from `/trigger/` in RPM file
   - Calculate time offset using edge detection
   - Verify synchronization quality (should be <1ms jitter)

3. **RPM Analysis**
   - Load RPM data from all 4 ESCs
   - Calculate BPF for each time sample: `BPF(t) = RPM(t) / 60 * 2 blades`
   - Identify harmonics: `f_n(t) = n * BPF(t)` for n = 1, 2, 3, ...

4. **Acoustic Data Processing**
   - Load microphone array data (channels 2-96)
   - Apply microphone calibration if available
   - Beamforming / source localization
   - Spectral analysis aligned with RPM

5. **Ego-Noise Suppression Testing**
   - **Notch Filters:** Track BPF harmonics using RPM data
   - **Spatial Filters:** Use array geometry from `mics_ref.xml`
   - **Performance Metrics:** Compare with/without rotors for each test signal

### Key Analysis Questions

- **Signal Recovery:** How well can test signals be recovered in presence of drone noise?
- **Filter Performance:** How much attenuation at BPF harmonics vs. preservation of test signal?
- **Configuration Comparison:** Does pull vs. push significantly affect acoustic signature?
- **Source Localization:** Can individual rotors be spatially separated?

### Potential Challenges

⚠ **Channel 1 Not Usable for Acoustics:** Remember it's the heartbeat sync signal
⚠ **Time Drift:** Verify synchronization throughout entire recording
⚠ **RPM Variations:** Actual RPM may differ slightly from target sequence
⚠ **Speaker Directivity:** Account for loudspeaker radiation pattern
⚠ **Reflections:** Despite anechoic chamber, some reflections from drone/stand possible

---

## Related Scripts and Tools

### Data Generation
- **`generate_test_audio_signals.py`** - Test signal generator
  - Configurable RPM range, blade count, signal types
  - Outputs WAV files at various SPL levels

### Measurement Control
- **`replay_and_record.py`** - Main measurement orchestration
  - Replays RPM sequences with throttle control
  - Manages synchronization signals
  - Coordinates mic array recording

### RPM System (MartyMicFly Repository)
- **`rpm_monitoring/kiss_tel_multi_esc_monitor.py`** - ESC telemetry acquisition
- **`esc_throttle_set.py`** - DShot protocol motor control
- **`throttle_rpm_mapping.py`** - Throttle-RPM calibration
- **`daq.py`** - Unified data acquisition system

### Calibration
- **`process_calibration.py`** - Process throttle-RPM calibration data
- **`throttle_rpm_mapping.csv`** - Averaged calibration curve (in `/calibration_data/`)

---

## Contact and Project Information

**Project:** MartyMicFly (DFG Research Project)
**Institution:** TU Berlin, Fachgebiet Akustik
**Repository:** `/home/steffen/MartyMicFly/Code/marty-mic-fly/`

**For Questions:**
- Refer to project documentation in `CLAUDE.md`
- Review measurement protocol ODS file
- Check data file attributes and metadata

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-05-12 | 0.1 | Initial documentation (original German notes) |
| 2025-12-09 | 1.0 | Comprehensive English documentation with data analysis guidance |

---

**Document Status:** Ready for data analysis reference
**Last Updated:** 2025-12-09