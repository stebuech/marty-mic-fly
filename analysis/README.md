# Acoustic Analysis Module

Comprehensive analysis tools for comparing drone rotor configurations using measurements from the RAR anechoic chamber.

## Overview

This module provides a complete pipeline for analyzing and comparing acoustic signatures between different rotor configurations (Pull vs Push). The analysis includes:

- **Spectral Analysis**: Frequency-domain comparison with BPF harmonic extraction
- **Sound Power Measurement**: ISO 3744 compliant sound power calculation
- **Spatial Acoustic Mapping**: Beamforming with ego-noise suppression
- **Comprehensive Reporting**: Automated markdown/PDF report generation

## Installation

1. Install dependencies:
```bash
uv sync
```

This will install:
- `acoular` - Microphone array processing framework
- `plotly` - Interactive visualizations
- `pandas` - Data analysis
- `pyyaml` - Configuration file parsing
- And all other project dependencies

## Quick Start

### Method 1: Using YAML Configuration (Recommended)

This approach allows you to define all parameters once and reuse them across analyses.

#### 1. Prepare Configuration File

Copy and edit the example configuration:

```bash
cp analysis/example_config.yaml analysis/my_comparison.yaml
```

Edit `my_comparison.yaml` to specify your measurement files:

```yaml
comparison:
  pull_config:
    mic_h5: "/path/to/pull_microphone_data.h5"
    rpm_h5: "/path/to/pull_rpm_telemetry.h5"
    description: "Pull configuration (rotors upward)"

  push_config:
    mic_h5: "/path/to/push_microphone_data.h5"
    rpm_h5: "/path/to/push_rpm_telemetry.h5"
    description: "Push configuration (rotors downward)"

  shared:
    array_xml: "/path/to/mics_ref.xml"

  analysis_params:
    stable_duration: 30.0  # seconds
    analysis_frequencies: [500, 1000, 2000, 4000, 8000]

  output_dir: "results/my_comparison"
```

#### 2. Run Spectral Analysis

```bash
# Analyzes both Pull and Push configurations automatically
python analysis/spectral_comparison.py --config analysis/my_comparison.yaml
```

This will:
- Load both Pull and Push microphone and RPM data
- Synchronize datasets using heartbeat signals
- Extract stable hover segments
- Compute power spectra and extract BPF harmonics
- Generate comparison plots (PNG, PDF, HTML)
- Save raw data as CSV files

#### 3. Review Results

Results are saved in the `output_dir` specified in the configuration:

```
results/my_comparison/
├── spectral_analysis/
│   ├── spectral_comparison.png
│   ├── spectral_comparison.html (interactive)
│   ├── pull_spectrum.csv
│   ├── push_spectrum.csv
│   └── bpf_harmonics_comparison.csv
├── sound_power/              # Coming soon
│   ├── sound_power_comparison.png
│   └── *.json (ISO 3744 results)
├── beamforming/              # Coming soon
│   ├── 500Hz/, 1000Hz/, ... (maps for each frequency)
│   └── frequency_summary.png
└── report/                   # Coming soon
    └── comparison_report.md
```

### Method 2: Using Command-Line Arguments

For more control or one-off analyses, you can use command-line arguments directly:

```bash
# Spectral analysis
python analysis/spectral_comparison.py \
  --pull-mic /path/to/pull_td.h5 \
  --pull-rpm /path/to/pull_rpm.h5 \
  --push-mic /path/to/push_td.h5 \
  --push-rpm /path/to/push_rpm.h5 \
  --output-dir results/spectral \
  --duration 30.0

# Data loading and synchronization (Pull config)
python analysis/data_loader.py \
  --mic-h5 /path/to/pull_td.h5 \
  --rpm-h5 /path/to/pull_rpm.h5 \
  --output-dir results/data/pull
```

## Workflow Overview

The typical analysis workflow consists of:

1. **Data Loading & Synchronization** (`data_loader.py`)
   - Loads HDF5 files
   - Synchronizes microphone and RPM data using heartbeat signals
   - Extracts stable hover segments
   - Exports RPM to txt format for downstream analysis

2. **Spectral Analysis** (`spectral_comparison.py`) ✅ **YAML Config Supported**
   - Computes power spectral density (Welch's method)
   - Extracts BPF harmonics
   - Generates comparison plots
   - Exports data as CSV

3. **Sound Power Analysis** (`sound_power_comparison.py`) 🚧 **CLI Only**
   - ISO 3744 compliant measurement
   - Hemisphere envelope method
   - Third-octave band analysis

4. **Beamforming Analysis** (`beamforming_comparison.py`) 🚧 **CLI Only**
   - Spatial acoustic mapping
   - Ego-noise suppression
   - Source localization
   - Multi-frequency analysis

5. **Report Generation** (`report_generator.py`) 🚧 **Coming Soon**
   - Comprehensive markdown report
   - PDF export (optional)

## Module Components

### Core Modules

#### `data_loader.py`
Data loading and synchronization infrastructure.

**Key Functions:**
- `load_rpm_telemetry(h5_file)` - Extract RPM data from HDF5
- `load_microphone_data(h5_file)` - Load microphone array data
- `synchronize_datasets(mic_data, rpm_data)` - Align datasets using heartbeat
- `extract_stable_segment(rpm_data, duration)` - Find stable hover segment
- `export_rpm_as_txt(rpm_h5, output_txt)` - Convert to format for existing scripts

**Standalone Usage:**
```bash
# Using command-line arguments
python analysis/data_loader.py \
  --mic-h5 /path/to/mic.h5 \
  --rpm-h5 /path/to/rpm.h5 \
  --output-dir test_output

# Using YAML config (Pull configuration)
python analysis/data_loader.py --config my_comparison.yaml --use-pull

# Using YAML config (Push configuration)
python analysis/data_loader.py --config my_comparison.yaml --use-push
```

#### `spectral_comparison.py`
Frequency-domain analysis and BPF harmonic extraction.

**Key Functions:**
- `compute_power_spectrum(time_data, sample_rate)` - Welch's method PSD
- `extract_bpf_harmonics(psd, frequencies, rpm_avg)` - BPF peak extraction
- `plot_spectral_comparison(pull, push, output_dir)` - Comparison visualizations

**Standalone Usage:**
```bash
# Using command-line arguments
python analysis/spectral_comparison.py \
  --pull-mic /path/to/pull_mic.h5 \
  --pull-rpm /path/to/pull_rpm.h5 \
  --push-mic /path/to/push_mic.h5 \
  --push-rpm /path/to/push_rpm.h5 \
  --output-dir spectral_analysis

# Using YAML config (recommended - analyzes both configurations)
python analysis/spectral_comparison.py --config my_comparison.yaml
```

#### `sound_power_comparison.py`
ISO 3744 sound power measurement wrapper.

**Key Functions:**
- `run_sound_power_analysis(mic_h5, rpm_txt, array_xml, config)` - Run ISO 3744
- `compare_sound_power(pull_results, push_results)` - Compute differences
- `plot_sound_power_comparison(pull, push, comparison, output_dir)` - Visualizations

**Standalone Usage:**
```bash
python analysis/sound_power_comparison.py \
  --pull-mic /path/to/pull_mic.h5 \
  --pull-rpm-txt /path/to/pull_rpm.txt \
  --push-mic /path/to/push_mic.h5 \
  --push-rpm-txt /path/to/push_rpm.txt \
  --array-xml /path/to/mics_ref.xml \
  --output-dir sound_power
```

#### `beamforming_comparison.py`
Spatial acoustic mapping with ego-noise suppression.

**Key Functions:**
- `run_beamforming(mic_h5, rpm_txt, array_xml, freq, config)` - Run beamforming
- `plot_side_by_side_maps(pull_map, push_map, freq, grid, output_dir)` - Visualizations
- `compute_source_location(map_dB, grid)` - Find acoustic source peaks

**Standalone Usage:**
```bash
python analysis/beamforming_comparison.py \
  --pull-mic /path/to/pull_mic.h5 \
  --pull-rpm-txt /path/to/pull_rpm.txt \
  --push-mic /path/to/push_mic.h5 \
  --push-rpm-txt /path/to/push_rpm.txt \
  --array-xml /path/to/mics_ref.xml \
  --frequencies 500 1000 2000 4000 8000 \
  --output-dir beamforming
```

### Integration Modules

#### `compare_rotor_configs.py`
Main orchestrator script that runs the complete analysis pipeline.

#### `report_generator.py`
Generates comprehensive markdown reports with embedded figures.

## Configuration Reference

### Analysis Parameters

```yaml
analysis_params:
  # Data Processing
  stable_duration: 30.0      # Extract 30s stable segment
  max_rpm_cv: 0.05          # Max coefficient of variation for stability

  # Drone Configuration
  drone_config:
    n_rotors: 4             # Number of rotors
    n_blades: 2             # Blades per rotor
    pole_pairs: 14          # Motor pole pairs
    position: [0.0, 0.375, 0.85]  # [x, y, z] meters

  # Sound Power (ISO 3744)
  sound_power:
    radius: 2.0             # Hemisphere radius (m)
    n_points: 20            # Measurement points
    freq_range: [100, 10000]  # Frequency range (Hz)
    use_notch: true         # Enable ego-noise suppression

  # Beamforming
  beamforming:
    grid_size: 4.0          # Grid extent ±from center (m)
    grid_increment: 0.1     # Grid spacing (m)
    grid_z: 3.0             # Grid distance from array (m)
    target_band: 1          # 1=octave, 3=third-octave
```

## Data Format Requirements

### Microphone Array HDF5 (`mic_h5`)
Expected structure:
```
/time_data - [n_samples, 96] float32
  Channel 0 (index 0): Heartbeat synchronization signal
  Channels 1-95 (indices 1-95): Acoustic data

/metadata (optional)
  - temperature_celsius
  - humidity_percent
  - dewpoint_celsius
```

### RPM Telemetry HDF5 (`rpm_h5`)
Expected structure:
```
/esc_telemetry
  /ESC1, /ESC2, /ESC3, /ESC4
    - timestamp[]
    - rpm[]
    - temperature[]
    - voltage[]
    - current[]

/trigger
  - timestamp[]
  - state[]

/timing
  - attrs: start_time_wall, start_time_perf
```

### Microphone Array Geometry XML (`array_xml`)
XML file defining microphone positions in 3D space.

## Algorithms

### Data Synchronization

Uses heartbeat/trigger cross-correlation:

1. Extract heartbeat signal from microphone Channel 1
2. Detect rising edges in heartbeat (threshold crossing)
3. Extract trigger events from RPM `/trigger/` dataset
4. Cross-correlate edge sequences to find time offset
5. Validate synchronization quality (intervals should be consistent ~10s)

### BPF Calculation

```python
BPF = (RPM / 60) * n_blades
harmonic_n = n * BPF  # for n = 1, 2, 3, ..., 20
```

For 4-rotor configuration with 2 blades at ~5850 RPM:
- BPF ≈ 195 Hz
- Harmonics: 195 Hz, 390 Hz, 585 Hz, ...

### RPM Averaging

RPM data from 4 ESCs is interpolated to a common time base and averaged:

```python
timestamps_common = np.linspace(t_start, t_end, n_samples)
rpm_avg = np.mean([
    np.interp(timestamps_common, esc1_time, esc1_rpm),
    np.interp(timestamps_common, esc2_time, esc2_rpm),
    np.interp(timestamps_common, esc3_time, esc3_rpm),
    np.interp(timestamps_common, esc4_time, esc4_rpm)
], axis=0)
```

## Troubleshooting

### Import Errors

If you see "Could not import sound_power_calculation" or "ego_noise_suppression":
- Ensure you're running from the project root directory
- Check that `.claude/skills/drone-acoustics-analysis/scripts/` contains these files

### Synchronization Quality Low

If sync_quality < 0.8:
- Check that microphone Channel 1 contains heartbeat signal
- Verify RPM file has `/trigger/` dataset with events
- Ensure heartbeat interval is consistent (~10 seconds)
- Try manually specifying time offset if auto-sync fails

### Memory Issues

For large datasets (>1 GB):
- Reduce `stable_duration` (use 15s instead of 30s)
- Reduce `beamforming.grid_increment` (use 0.2m instead of 0.1m)
- Process fewer frequencies simultaneously

### Acoular Errors

Common issues:
- Channel count mismatch: Ensure `array_xml` matches actual channel count
- Sample rate mismatch: Verify microphone data is at expected 51.2 kHz
- CSM computation fails: Reduce `spectral.nperseg` to 4096 or 2048

## Current Status & Features

### ✅ Ready to Use
- **Spectral Analysis** (`spectral_comparison.py`)
  - YAML configuration support
  - Automated comparison workflow
  - Interactive and static visualizations
  - CSV data export

- **Data Loading** (`data_loader.py`)
  - YAML configuration support for individual configs
  - Heartbeat synchronization
  - Stable segment extraction
  - RPM export to txt format

### 🚧 CLI Only (No YAML support yet)
- **Sound Power Analysis** (`sound_power_comparison.py`)
- **Beamforming Analysis** (`beamforming_comparison.py`)

### 🔮 Coming Soon
- Main orchestrator script (`compare_rotor_configs.py`)
- Report generator (`report_generator.py`)
- Visualization utilities (`visualization_utils.py`)

## Advanced Usage

### Custom Analysis Frequencies

Edit `analysis_frequencies` in config.yaml to analyze specific frequencies:

```yaml
analysis_params:
  # Analyze BPF and first few harmonics only
  analysis_frequencies: [183, 195, 207, 390, 585, 780]

  # Or standard analysis frequencies
  analysis_frequencies: [500, 1000, 2000, 4000, 8000]
```

### Adjust Stable Segment Duration

If your measurements have shorter stable regions:

```yaml
analysis_params:
  stable_duration: 15.0  # Use 15 seconds instead of default 30
```

### Custom Output Directory Structure

```yaml
comparison:
  output_dir: "results/2025-12-05_RAR_comparison"  # Timestamped directory
```

### Processing Multiple Comparisons

Create multiple config files for different measurement sets:

```bash
# Process different rotor speeds
python analysis/spectral_comparison.py --config configs/5500rpm_comparison.yaml
python analysis/spectral_comparison.py --config configs/6000rpm_comparison.yaml
python analysis/spectral_comparison.py --config configs/6500rpm_comparison.yaml
```

## Practical Examples

### Example 1: Quick Spectral Analysis

```bash
# 1. Copy and edit config
cp analysis/example_config.yaml analysis/pull_vs_push.yaml
# Edit paths in pull_vs_push.yaml

# 2. Run analysis
python analysis/spectral_comparison.py --config analysis/pull_vs_push.yaml

# 3. View results
open results/my_comparison/spectral_analysis/spectral_comparison.html
```

### Example 2: Test Data Loading First

Before running full analysis, verify data loading and synchronization:

```bash
# Test Pull configuration
python analysis/data_loader.py \
  --config analysis/pull_vs_push.yaml \
  --use-pull \
  --output-dir test_pull

# Check sync quality in output
# Look for: "Sync quality: 0.XXX" (should be > 0.8)

# Test Push configuration
python analysis/data_loader.py \
  --config analysis/pull_vs_push.yaml \
  --use-push \
  --output-dir test_push
```

### Example 3: Different Segment Duration

If default 30s is too long or too short:

```bash
# Use 15 second segments
python analysis/spectral_comparison.py \
  --config analysis/pull_vs_push.yaml \
  --duration 15.0
```

## Contributing

When adding new analysis modules:

1. Follow existing module structure
2. Include standalone `if __name__ == '__main__'` testing
3. Add YAML config support using the pattern from `spectral_comparison.py`
4. Document key functions with docstrings
5. Save results in consistent format (JSON/CSV + PNG/HTML)
6. Update this README with usage instructions

## References

- ISO 3744:2010 - Acoustics - Determination of sound power levels of noise sources
- Acoular documentation: https://acoular.org/
- Measurement setup: `notes_and_helpers/2025_05_12_measurement_setup.md`
- Project overview: `CLAUDE.md`

## Support

For questions about this analysis module:
- Check the measurement setup documentation: `notes_and_helpers/2025_05_12_measurement_setup.md`
- Review the project documentation: `CLAUDE.md`
- Examine the original measurement protocol (`.ods` file in measurement data folder)

## Quick Reference

### YAML Config Structure
```yaml
comparison:
  pull_config:
    mic_h5: "path/to/pull_microphone.h5"
    rpm_h5: "path/to/pull_rpm.h5"
  push_config:
    mic_h5: "path/to/push_microphone.h5"
    rpm_h5: "path/to/push_rpm.h5"
  shared:
    array_xml: "path/to/mics_ref.xml"
  analysis_params:
    stable_duration: 30.0
  output_dir: "results/comparison"
```

### Common Commands
```bash
# Spectral analysis with YAML
python analysis/spectral_comparison.py --config config.yaml

# Data loading for Pull config
python analysis/data_loader.py --config config.yaml --use-pull

# Data loading for Push config
python analysis/data_loader.py --config config.yaml --use-push

# Spectral analysis with CLI
python analysis/spectral_comparison.py \
  --pull-mic pull.h5 --pull-rpm pull_rpm.h5 \
  --push-mic push.h5 --push-rpm push_rpm.h5
```
