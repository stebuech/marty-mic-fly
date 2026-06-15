# Pull vs Push Rotor Configuration Analysis

## Overview

This document provides context for the acoustic analysis comparing Pull (rotors upward) and Push (rotors downward) configurations from RAR anechoic chamber measurements (December 2025).

## Project Goal

Compare spectral characteristics, sound power, and beamforming maps between two drone configurations:
- **Pull**: Rotors pointing upward (standard orientation)
- **Push**: Rotors pointing downward (inverted orientation)

## Key Technical Details

### RPM Correction Factor: 7/12

Due to the pole pair configuration in the motor/ESC system, raw RPM values from KISS telemetry require correction:
- **Correction**: `rpm_corrected = rpm_raw * (7/12)`
- **Applied in**: `data_loader.py::load_rpm_telemetry()` at data load time
- **Reason**: Centralizes correction so all downstream analysis uses clean data

### BPF (Blade Passing Frequency) Calculation

Theoretical BPF calculation (no peak searching):
```python
bpf = (rpm_corrected / 60.0) * n_blades
```

- **n_blades**: 2 per motor
- **n_motors**: 4 total
- **Harmonics**: Calculate up to 20 harmonics per motor (limited by Nyquist frequency)

### Per-Motor Analysis

The system tracks individual motor performance:
- Each of 4 motors has its own BPF based on individual RPM
- Stable segment extraction finds optimal 30s window using average RPM
- Per-motor statistics are extracted from the **same** stable window for consistency

### Data Synchronization

- **Microphone array**: 96 channels @ 51.2 kHz sampling rate
- **RPM telemetry**: Variable rate (~100-200 Hz), interpolated to common time base
- **Synchronization method**: Cross-correlation of heartbeat signal with RPM trigger events
- **Time offset**: Calculated automatically to align mic and RPM data

## File Structure

### Analysis Modules

#### `analysis/data_loader.py` (567 lines)

Core data loading infrastructure:

**Key Functions**:
- `load_rpm_telemetry(h5_file, rpm_correction_factor=7/12)`:
  - Loads RPM data from HDF5 ESC telemetry file
  - Applies RPM correction to all motors
  - Returns: timestamp, rpm_avg, rpm_per_esc, trigger_events, metadata

- `load_mic_array_data(h5_file, array_xml)`:
  - Loads microphone array time-domain data
  - Returns: time_data (samples × channels), sample_rate, mic_positions, metadata

- `synchronize_data(mic_data, rpm_data, heartbeat_freq=1000.0)`:
  - Cross-correlates heartbeat signal with RPM triggers
  - Returns: time_offset, correlation_coefficient, synchronized mic/rpm data

- `extract_stable_segment(rpm_data, duration=30.0, max_cv=0.05)`:
  - Finds stable hover segment with minimal RPM variance
  - Uses sliding window to minimize coefficient of variation
  - **Returns**: (start_time, end_time, mean_rpm, std_rpm, per_motor_stats)
  - **per_motor_stats**: dict with 'mean_rpm' and 'std_rpm' arrays for each motor

**Example Usage**:
```python
from analysis.data_loader import load_rpm_telemetry, extract_stable_segment

rpm_data = load_rpm_telemetry(Path("telemetry.h5"), rpm_correction_factor=7/12)
start, end, mean_rpm, std_rpm, per_motor = extract_stable_segment(rpm_data, duration=30.0)
```

#### `analysis/spectral_comparison.py` (~560 lines)

Main spectral analysis script:

**Key Functions**:
- `calculate_bpf_harmonics(rpm_avg, rpm_per_motor, n_blades=2, n_harmonics=20)`:
  - Calculates theoretical BPF and harmonics from corrected RPM
  - No peak searching - pure theoretical calculation
  - Returns: bpf_avg, bpf_per_motor, rpm values, harmonic frequencies

- `compute_psd(time_data, fs, nperseg=8192, noverlap=4096)`:
  - Welch's method power spectral density estimation
  - Returns: frequencies, psd (linear scale)

- `plot_spectral_comparison(frequencies, pull_psd, push_psd, pull_harmonics, push_harmonics)`:
  - Creates 3-panel matplotlib figure:
    - Panel 1: Pull spectrum with solid per-motor harmonic lines
    - Panel 2: Push spectrum with dashed per-motor harmonic lines
    - Panel 3: Difference (Pull - Push) with all motor lines
  - Color scheme: 4 unique colors for 4 motors
  - Line styles: solid=pull, dashed=push

- `create_interactive_plot(...)`:
  - Generates Plotly HTML interactive plots
  - Same layout as matplotlib version

**Execution**:
```bash
python analysis/spectral_comparison.py --config analysis/my_comparison.yaml
```

#### `analysis/sound_power_comparison.py`

ISO 3744 sound power level calculation:
- Hemisphere measurement surface
- Adaptive notch filtering for ego-noise suppression
- Frequency-dependent analysis

#### `analysis/beamforming_comparison.py`

Acoustic source mapping using Acoular framework:
- Delay-and-sum beamforming
- Octave/third-octave band analysis
- 2D visualization on planar grid

### Configuration File

#### `analysis/my_comparison.yaml`

YAML configuration specifying:
- **Pull config**: mic h5, rpm h5, description
- **Push config**: mic h5, rpm h5, description
- **Shared resources**: array_xml, measurement_protocol
- **Analysis params**: stable_duration, max_rpm_cv, frequencies, drone_config, etc.
- **Output**: directory path, plot options

**Critical Parameters**:
```yaml
analysis_params:
  stable_duration: 30.0
  max_rpm_cv: 0.05
  drone_config:
    n_rotors: 4
    n_blades: 2
    pole_pairs: 24  # Not used in current analysis, but for reference
  spectral:
    nperseg: 8192    # PSD segment length
    noverlap: 4096   # PSD overlap
```

## Measurement Data Paths

**Location**: `/home/steffen/MartyMicFly/Messdaten/2025_12_05_RAR_Drone_TRANSIT_BVG_Array/`

### Pull Configuration
- Mic: `td/2025-12-05_16-26-27_468418.h5`
- RPM: `rpm_data/telemetry_data_20251205_152626.h5`

### Push Configuration
- Mic: `td/2025-12-05_16-51-06_934842.h5`
- RPM: `rpm_data/telemetry_data_20251205_155105.h5`

### Shared Resources
- Array geometry: `mics_ref.xml`
- Protocol: `2025_12_05_RAR_Drone_1st_test.ods`

## Visualization Design

### Color Scheme (Per-Motor)
```python
motor_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
# Motor 1: Red
# Motor 2: Blue
# Motor 3: Green
# Motor 4: Purple
```

### Line Styles
- **Pull configuration**: Solid lines (`'-'`)
- **Push configuration**: Dashed lines (`'--'`)

### Plot Layout

**Matplotlib Output** (`spectral_comparison.png`):
```
┌─────────────────────────────────────┐
│ Panel 1: Pull Configuration         │
│ - Black spectrum line               │
│ - 4 motors × 20 harmonics (solid)   │
│ - Legend: Motor 1-4                  │
├─────────────────────────────────────┤
│ Panel 2: Push Configuration         │
│ - Black spectrum line               │
│ - 4 motors × 20 harmonics (dashed)  │
│ - Legend: Motor 1-4                  │
├─────────────────────────────────────┤
│ Panel 3: Difference (Pull - Push)   │
│ - Pull harmonics (solid)            │
│ - Push harmonics (dashed)           │
│ - Legend: Pull M1-4, Push M1-4      │
│ - Shows all 8 sets of lines         │
└─────────────────────────────────────┘
```

**Plotly Output** (`spectral_comparison_interactive.html`):
- Same layout, interactive hover
- Zoom, pan, toggle traces

## Implementation Timeline

### Initial Implementation (Previous Session)
1. Created `data_loader.py` with HDF5 loading and synchronization
2. Created `spectral_comparison.py` with peak-based harmonic extraction
3. Created `sound_power_comparison.py` for ISO 3744 analysis
4. Created `beamforming_comparison.py` for source mapping
5. Added YAML configuration support

### Refactoring (Current Session)

**Change 1**: Theoretical BPF Calculation
- **Before**: Peak searching near theoretical harmonics
- **After**: Pure theoretical calculation with vertical line markers
- **Benefit**: Simpler, more reliable, guaranteed to show all harmonics

**Change 2**: Per-Motor Consistency
- **Before**: extract_stable_segment() returned only average stats
- **After**: Returns per_motor_stats dict with individual motor values
- **Benefit**: All BPF calculations use same stable time window

**Change 3**: Per-Motor Visualization
- **Before**: Only average harmonics plotted
- **After**: Each motor plotted with unique color, both configs visible
- **Benefit**: Shows motor-to-motor variation clearly

**Change 4**: Centralized RPM Correction
- **Before**: RPM correction in analysis functions
- **After**: Correction in load_rpm_telemetry() at data load time
- **Benefit**: Clean separation, single source of truth, stored in metadata

## Running the Analysis

### Prerequisites
```bash
# Ensure uv environment is synced
uv sync

# Verify data paths in analysis/my_comparison.yaml
```

### Execution
```bash
# Run spectral comparison
python analysis/spectral_comparison.py --config analysis/my_comparison.yaml

# Outputs:
# - results/pull_vs_push_comparison/spectral_comparison.png
# - results/pull_vs_push_comparison/spectral_comparison_interactive.html
# - Console output with RPM stats and BPF values
```

### Expected Console Output
```
Loading Pull configuration...
  Microphone data: 96 channels, 51.2 kHz, 163.84 s
  RPM data: 4 ESCs, 163.83 s duration

Loading Push configuration...
  Microphone data: 96 channels, 51.2 kHz, 163.84 s
  RPM data: 4 ESCs, 163.83 s duration

Extracting stable segments...

Pull Configuration:
  Start time: XX.XX s
  End time: YY.YY s
  Average RPM: 3398.1
  Average BPF: 113.27 Hz
  BPF per motor:
    Motor 1: 106.12 Hz (RPM: 3183.6)
    Motor 2: 122.08 Hz (RPM: 3662.3)
    Motor 3: 118.93 Hz (RPM: 3567.8)
    Motor 4: 105.96 Hz (RPM: 3178.8)

Push Configuration:
  [Similar output for push]

Computing PSDs...
Creating plots...
```

## Key Files Modified

### `analysis/data_loader.py`
- **Line 21-128**: `load_rpm_telemetry()` - added rpm_correction_factor parameter and application logic
- **Line 335-415**: `extract_stable_segment()` - changed return signature to include per_motor_stats
- **Line 572-582**: Main block - updated to handle 5-tuple return

### `analysis/spectral_comparison.py`
- **Line 89-145**: Replaced `extract_bpf_harmonics()` with `calculate_bpf_harmonics()`
- **Line 147-269**: `plot_spectral_comparison()` - per-motor harmonics with color coding
- **Line 271-432**: `create_interactive_plot()` - Plotly version of above
- **Line 509-514, 546-551**: Main execution - unpack 5-tuple, pass per-motor stats

## Technical Decisions Made

1. **RPM Correction Factor**: 7/12
   - Based on pole pair configuration
   - Applied at data load time for consistency

2. **BPF Calculation Method**: Theoretical (not peak-based)
   - More robust than searching for peaks
   - Guaranteed to show all harmonics
   - Simpler implementation

3. **Stable Segment Selection**: 30s duration, max CV 0.05
   - Balance between statistical stability and data availability
   - CV threshold ensures hover conditions

4. **Per-Motor vs Average**: Both tracked
   - Average used for stable segment selection
   - Per-motor used for individual BPF visualization
   - Same time window ensures consistency

5. **Visualization Strategy**: Color per motor, style per config
   - 4 colors (red, blue, green, purple)
   - Solid = pull, dashed = push
   - All combinations visible in difference plot

## Common Issues and Solutions

### Issue 1: Different RPM per Motor
**Symptom**: Large spread in BPF values across motors
**Cause**: Individual motor characteristics, ESC calibration, mechanical load
**Solution**: Plot all motors separately to visualize variation

### Issue 2: Harmonics Overlap
**Symptom**: Vertical lines cluster together in plots
**Cause**: Similar RPMs across motors
**Solution**: Alpha blending (0.4-0.5) to show overlapping lines

### Issue 3: Low Frequency Resolution
**Symptom**: Can't resolve closely-spaced harmonics
**Solution**: Increase nperseg parameter (e.g., 8192 → 16384)
- Note: Can also apply bandpass filter + decimation for 50-8000 Hz range

## Next Steps (Potential)

1. **Sound Power Analysis**: Run `sound_power_comparison.py` for ISO 3744 levels
2. **Beamforming**: Run `beamforming_comparison.py` for source maps
3. **Frequency Resolution**: Increase nperseg if needed for harmonic resolution
4. **Multiple Configurations**: Add more YAML configs for different test conditions
5. **Automated Reporting**: Generate PDF reports with all analysis results

## References

- **ISO 3744**: Acoustics - Determination of sound power levels and sound energy levels of noise sources using sound pressure
- **Acoular**: Python library for acoustic beamforming
- **KISS Telemetry Protocol**: 10-byte packet structure for ESC data
- **Welch's Method**: Power spectral density estimation with overlapping segments
- **BPF**: Blade Passing Frequency = (RPM / 60) × number_of_blades

## Metadata

- **Project**: MartyMicFly - Flying Measurement Microphone
- **Measurement Date**: 2025-12-05
- **Location**: RAR Anechoic Chamber
- **Array**: TRANSIT BVG 96-channel microphone array
- **Analysis Date**: 2025-12-15
- **Primary Developer**: Steffen (with Claude Code assistance)

---

*This document captures the complete context for continuing Pull vs Push rotor configuration analysis on another machine. All file paths are absolute to ensure reproducibility.*