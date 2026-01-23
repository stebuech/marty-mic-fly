# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Python 3.13+ - All application code (`pyproject.toml` requires-python = ">=3.13")

**Secondary:**
- YAML - Configuration files (`analysis/example_config.yaml`, `analysis/mimo_filter_config_schema.yaml`)
- Markdown - Documentation (`CLAUDE.md`, `notes_and_helpers/*.md`)

## Runtime

**Environment:**
- Python 3.13+ (required)
- Raspberry Pi 4+ (target platform for hardware interface)
- Linux (required for pigpio daemon and CPU affinity)
- `pigpiod` daemon must be running for GPIO/DShot operations

**Package Manager:**
- uv (modern Python package manager)
- Lockfile: `uv.lock` (version 1, revision 3)

## Frameworks

**Core:**
- No web framework (standalone Python scripts for data acquisition)
- Multi-threaded architecture (Python `threading` module)
- Callback-driven hardware interface (pigpio callbacks)

**Testing:**
- Not detected - No formal test framework present
- Helper scripts exist: `notes_and_helpers/generate_test_audio_signals.py`

**Build/Dev:**
- uv for dependency management (`uv sync`, `uv add`)
- No build step required (interpreted Python)

## Key Dependencies

**Critical - Hardware Interface:**
- `pigpio` 1.78 - GPIO control, DShot protocol, PWM measurement (`daq.py`, `esc_throttle_set.py`)
- `pyserial` 3.5+ - Serial communication for KISS ESC telemetry (`daq.py`)
- `sounddevice` 0.5.3+ - USB audio input from MCHStreamer microphone array (`daq.py`)

**Critical - Data Processing:**
- `numpy` 2.3.4+ - Numerical computation (used throughout)
- `scipy` 1.16.3+ - Signal processing, IIR filters (`analysis/mimo_adaptive_iir_filter.py`)
- `h5py` 3.15.1+ - HDF5 data logging and storage (`daq.py`, `visualize_data.py`)

**Analysis:**
- `acoular` 25.4 - Beamforming and microphone array processing (`analysis/beamforming_comparison.py`)
- `scikit-learn` - Clustering for frequency initialization (`analysis/frequency_initialization.py`)
- `pandas` 2.2+ - Data manipulation (`analysis/sound_power_comparison.py`)
- `matplotlib` 3.10+ - Static visualization (`visualize_data.py`, analysis scripts)
- `plotly` 5.24+ - Interactive HTML visualization (`analysis/beamforming_comparison.py`)

**Configuration:**
- `pyyaml` 6.0.3 - YAML configuration parsing (`analysis/mimo_adaptive_iir_filter.py`)

**Development:**
- `ipython` 9.6.0 - Interactive development and debugging

## Configuration

**Environment:**
- No `.env` files required
- Hardware pins configured via CLI arguments
- Serial ports: `/dev/ttyAMA0`, `/dev/ttyAMA2`, `/dev/ttyAMA3`, `/dev/ttyAMA4`
- Default GPIO pins: 17 (control), 18-21 (DShot), 22 (output), 23 (log)

**Build:**
- `pyproject.toml` - Package metadata and dependencies
- No compilation required

**Analysis:**
- `analysis/example_config.yaml` - Complete measurement/analysis configuration
- `analysis/mimo_filter_config_schema.yaml` - MIMO filter parameter schema

## Platform Requirements

**Development:**
- Any platform with Python 3.13+ for analysis scripts
- Raspberry Pi 4+ for hardware interface testing
- `pigpiod` daemon for GPIO operations

**Production:**
- Raspberry Pi 4+ with:
  - 4× ESC UART connections
  - USB audio interface (MCHStreamer Kit from miniDSP)
  - GPIO pins for DShot, PWM, and trigger
- Linux with real-time capable kernel (for timing precision)
- `sudo pigpiod` before running scripts

---

*Stack analysis: 2026-01-23*
*Update after major dependency changes*
