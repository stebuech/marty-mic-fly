# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**Beamforming Framework (Optional):**
- `ego_noise_suppression` module from external drone-acoustics-analysis project
  - Integration: Optional import with graceful fallback (`analysis/beamforming_comparison.py` lines 21-28)
  - Purpose: Spatial acoustic mapping for rotor configuration analysis
  - Status: Optional dependency, works without it

**External APIs:**
- None detected - All processing is local

## Data Storage

**HDF5 Files:**
- Primary data format for recordings and analysis
  - Client: h5py 3.15.1+
  - Compression: gzip for datasets
  - Structure: Hierarchical groups with extendable datasets

**HDF5 Structure (from `daq.py`, `CLAUDE.md`):**
```
/timing
  - attrs: start_time_wall, start_time_perf
/trigger (optional)
  - timestamp[]: float64 array
  - state[]: int8 array
/mic_array
  - timestamp[], audio_data[]
  - timing_drift/: drift, drift_mean, drift_std, correction_factor, adc_timestamp, hw_timestamp
/esc_telemetry
  /ESC1, /ESC2, /ESC3, /ESC4
    - timestamp[], temperature[], voltage[], current[], consumption[], rpm[]
    - attrs: port, total_samples, avg_sample_rate, sample_rate_std
```

**File Storage:**
- Default output: `~/MMFDataLogs/` (configurable via `--output_folder`)
- Filenames: `telemetry_data_YYYYMMDD_HHMMSS.h5`

## Hardware Interfaces

**GPIO Control via pigpio:**
- Library: pigpio 1.78
- Files: `daq.py` (SignalMonitor, MicArray), `esc_throttle_set.py` (DShot, MultiESCControler)
- Daemon: `pigpiod` (must be running)
- Features: PWM pulse measurement, hardware interrupts, DShot waveforms

**Serial Communication (KISS Telemetry):**
- Library: pyserial 3.5+
- File: `daq.py` (ESCTelemtry class)
- Protocol: KISS 10-byte packet with CRC8
- Baud rate: 115200 (default)
- Ports: `/dev/ttyAMA0`, `/dev/ttyAMA2`, `/dev/ttyAMA3`, `/dev/ttyAMA4`
- Data: Temperature, voltage, current, consumption, RPM

**Audio Input (Microphone Array):**
- Library: sounddevice 0.5.3+
- File: `daq.py` (MicArray class)
- Device: MCHStreamer Kit from miniDSP (USB)
- Channels: 95 (configurable)
- Sample rate: 48000-51200 Hz
- Block size: 1024 samples

**DShot Motor Control:**
- Library: pigpio (hardware-timed waveforms)
- File: `esc_throttle_set.py`
- Protocol: DShot150/300/600
- Pins: GPIO 18, 19, 20, 21 (default)
- Throttle range: 48-2047 (0-47 are special commands)

## Monitoring & Observability

**Error Tracking:**
- Console output only (print statements)
- No external error tracking service

**Logging:**
- Print statements to stdout/stderr
- HDF5 files contain metadata (sample rates, timing info)

## Environment Configuration

**Development:**
- Required: Python 3.13+, uv package manager
- Hardware optional (can run analysis scripts without)
- Analysis scripts work on any platform

**Production (Raspberry Pi):**
- Required environment setup:
  ```bash
  sudo pigpiod  # Start GPIO daemon
  uv sync       # Install dependencies
  ```
- Serial port permissions (`dialout` group)
- USB audio device access

## Process Integration

**CPU Affinity Management:**
- `daq.py` (line 17): Pins to cores 1-3 (`os.sched_setaffinity(0, {1, 2, 3})`)
- `esc_throttle_set.py` (line 13): Pins to core 0 (`os.sched_setaffinity(0, {0})`)
- Purpose: Avoid pigpio/sounddevice timing interference
- Constraint: DAQ and ESC control must run in separate Python processes

**Threading Model:**
- ESC Monitor: One thread per ESC (4 threads)
- MicArray: sounddevice callback thread
- HDF5 Logger: Background flush thread
- SignalMonitor: Heartbeat thread (optional)
- All synchronized via single Timer instance

---

*Integration audit: 2026-01-23*
*Update when adding/removing external services*
