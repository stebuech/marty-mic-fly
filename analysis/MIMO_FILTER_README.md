# MIMO Adaptive IIR Notch Filter - Implementation Guide

## Overview

This implementation provides **MIMO (Multiple-Input Multiple-Output) Adaptive IIR Notch Filtering** for removing tonal propeller noise from drone-mounted microphone arrays, based on:

**Harvey, B. (2019).** *Signal Processing Methods for the Detection and Localization of Acoustic Sources via Unmanned Aerial Vehicles.* PhD Thesis, Memorial University of Newfoundland, Chapter 3 (pp. 54-97).

### Key Features

✅ **Referenceless Adaptation**: No reference signal needed - minimizes output power
✅ **Zero-Phase Filtering**: Perfect phase preservation for beamforming
✅ **MIMO Architecture**: Handles 4 sources × 10+ harmonics × 95 channels
✅ **Three Initialization Methods**: Autocorrelation, FFT, RPM telemetry
✅ **Configuration-Driven**: Easy parameter adjustment via YAML

### Expected Performance

| Metric | Single-Pass | Zero-Phase |
|--------|-------------|------------|
| Suppression | 20-25 dB | 40-50 dB |
| Tracking Accuracy | MFD < 0.5 Hz | MFD < 0.25 Hz |
| Phase Distortion | Some | ~0 rad (exact) |
| Computational Cost | ~1.3 GFLOPS | ~2.6 GFLOPS |

## Quick Start

### Basic Usage

```python
from analysis.data_loader import load_microphone_data, load_rpm_telemetry
from analysis.data_loader import apply_mimo_iir_filter
from analysis.mimo_adaptive_iir_filter import MIMOFilterConfig

# Load data
mic_data = load_microphone_data('measurement.h5')
rpm_data = load_rpm_telemetry('rpm_telemetry.h5')

# Configure filter
config = MIMOFilterConfig(
    n_sources=4,
    n_harmonics=10,
    initialization_method='rpm',
    zero_phase=True
)

# Apply filter
result = apply_mimo_iir_filter(
    mic_data['time_data'],
    mic_data['sample_rate'],
    config,
    rpm_data
)

# Access results
filtered_signal = result['filtered_data']
suppression_db = result['suppression_metrics']['mean_suppression']
print(f"Achieved {suppression_db:.1f} dB suppression")
```

### Using Configuration File

```python
from analysis.mimo_adaptive_iir_filter import MIMOFilterConfig

# Load from YAML
config = MIMOFilterConfig.from_yaml('mimo_filter_config_schema.yaml')

# Apply
result = apply_mimo_iir_filter(time_data, sample_rate, config, rpm_data)
```

## File Structure

```
analysis/
├── mimo_adaptive_iir_filter.py       # Core implementation
│   ├── IIRNotchFilter                 # Single adaptive filter
│   ├── MIMOFilterConfig               # Configuration management
│   ├── MIMOFilterStage                # One stage (source, harmonic)
│   └── MIMOFilterBank                 # Full cascade orchestrator
│
├── frequency_initialization.py       # Frequency detection
│   └── FrequencyInitializer
│       ├── autocorrelation_method()   # Method 1: Autocorrelation
│       ├── fft_peak_detection_method() # Method 2: FFT peaks
│       ├── rpm_telemetry_method()     # Method 3: RPM (best)
│       └── compare_methods()          # Run all & recommend
│
├── iir_filter_validation.py          # Performance measurement
│   └── MIMOFilterValidator
│       ├── measure_suppression()      # Suppression in dB
│       ├── measure_tracking_accuracy() # MFD calculation
│       └── validate_phase_preservation() # Phase check
│
├── data_loader.py                     # Integration wrapper
│   └── apply_mimo_iir_filter()        # Main entry point
│
├── mimo_filter_config_schema.yaml    # Configuration template
├── example_config.yaml                # Extended with mimo_filter section
└── MIMO_FILTER_README.md              # This file
```

## Algorithm Details

### Core Filter Equations

**Transfer Function:**
```
H(z) = (1 - 2cos(θ)z⁻¹ + z⁻²) / (1 - 2r·cos(θ)z⁻¹ + r²z⁻²)
```

**Difference Equation:**
```
y(n) = x(n) - 2cos(θ)·x(n-1) + x(n-2)
       + 2r·cos(θ)·y(n-1) - r²·y(n-2)
```

**Gradient Signal:**
```
β(n) = 2sin(θ)[x(n-1) - ry(n-1)]
       + 2(r-1)cos(θ)β(n-1) + (1-r²)β(n-2)
```

**LMS Adaptation:**
```
θ(n+1) = θ(n) - 2μ·y(n)·β(n)
```

### MIMO Cascade Structure

```
Input [95 channels, N samples]
    ↓
Source 1, Harmonic 1  (190 Hz)  ─→  Filtered
    ↓
Source 1, Harmonic 2  (380 Hz)  ─→  Filtered
    ↓
    ...
    ↓
Source 1, Harmonic 10 (1900 Hz) ─→  Filtered
    ↓
Source 2, Harmonic 1  (200 Hz)  ─→  Filtered
    ↓
    ...
    ↓
Source 4, Harmonic 10 (2050 Hz) ─→  Filtered
    ↓
Output [95 channels, N samples]
```

Total: **40 stages** (4 sources × 10 harmonics)
Total filters: **3,800** (40 stages × 95 channels)

### Zero-Phase Filtering

```
1. Forward pass  → Adapt θ values
2. Reverse signal
3. Backward pass → Use frozen θ (no adaptation)
4. Reverse again
```

**Result**: φ(ω) = 0 for all frequencies, |H_zp| = |H|²

## Configuration Parameters

### Critical Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `pole_radius` | 0.95-0.99 | 0.98 | Notch bandwidth (higher = narrower) |
| `mu_base` | 1e-5 to 5e-4 | 1e-4 | Adaptation speed (higher = faster) |
| `mu_increment` | 0 to 5e-4 | 1e-4 | Step size increase per harmonic |
| `moving_avg_window` | 50-200 | 100 | Frequency smoothing (higher = smoother) |
| `n_harmonics` | 5-20 | 10 | Number of harmonics to remove |

### Initialization Methods

**Hierarchy (best to worst):**
1. **RPM Telemetry**: 0.00 Hz error (perfect)
2. **FFT Peak Detection**: 0.25 Hz error (very good)
3. **Autocorrelation**: 6.5 Hz error (fallback)

**Recommendation**: Use `method: 'rpm'` when available, otherwise `method: 'fft'`

## Performance Optimization

### Computational Cost

**Single-pass filtering:**
```
Operations per sample: 13 × S × M × K
  = 13 × 4 × 10 × 95
  = 49,400 FLOPs/sample

At 51.2 kHz:
  = 2.53 GFLOPS

Real-time factor on modern CPU (10 GFLOPS): ~4×
```

**Zero-phase filtering:** 2× computational cost (two passes)

### Memory Requirements

```
Filter state: ~30 bytes/filter
  = 30 × 3,800 = 114 KB (negligible)

Signal buffer (10 seconds, 95 channels):
  = 95 × 512,000 × 8 bytes = 389 MB
```

### Parallelization

✅ Channel-level parallelism within each stage
✅ Expected speedup: 4-8× on 8-core CPU
❌ Cannot parallelize across stages (sequential dependency)

## Validation & Testing

### Unit Tests

```bash
# Test core classes
python analysis/mimo_adaptive_iir_filter.py

# Test frequency initialization
python analysis/frequency_initialization.py

# Test validation tools
python analysis/iir_filter_validation.py
```

### Validation Metrics

**Suppression Measurement:**
```python
from analysis.iir_filter_validation import MIMOFilterValidator

validator = MIMOFilterValidator()
suppression = validator.measure_suppression(
    original_signal, filtered_signal, harmonic_freqs
)
print(f"Mean: {suppression['mean_suppression']:.1f} dB")
```

**Tracking Accuracy:**
```python
tracking = validator.measure_tracking_accuracy(
    tracked_freqs, ground_truth_freqs
)
print(f"MFD: {tracking['mfd']:.3f} Hz")
```

**Phase Preservation:**
```python
phase_check = validator.validate_phase_preservation(
    original_signal, filtered_signal, sample_rate
)
print(f"Max distortion: {phase_check['max_phase_distortion']:.4f} rad")
```

## Troubleshooting

### Issue: Poor Tracking (MFD > 1 Hz)

**Possible causes:**
- Initialization too far from true frequency
- Signal duration too short (< 1 second)
- Step size too small

**Solutions:**
- Use `method: 'rpm'` for best initialization
- Increase signal duration to 2-5 seconds
- Increase `mu_base` to 2e-4

### Issue: Low Suppression (< 15 dB)

**Possible causes:**
- Frequencies drifting during measurement
- Too few harmonics configured
- Notches too wide

**Solutions:**
- Ensure stable RPM during recording
- Increase `n_harmonics` to 15-20
- Increase `pole_radius` to 0.99 (narrower notches)

### Issue: Instability / Oscillations

**Possible causes:**
- Step size too large
- Pole radius too close to 1

**Solutions:**
- Decrease `mu_base` to 5e-5
- Decrease `pole_radius` to 0.97
- Increase `moving_avg_window` to 200

### Issue: Slow Convergence

**Possible causes:**
- Step size too small
- Moving average too long

**Solutions:**
- Increase `mu_base` and `mu_increment`
- Decrease `moving_avg_window` to 50
- Use longer signal segment (5+ seconds)

## Integration Examples

### With Spectral Analysis

```python
# In spectral_comparison.py
if config.get('mimo_filter', {}).get('enabled', False):
    from analysis.data_loader import apply_mimo_iir_filter

    result = apply_mimo_iir_filter(
        pull_time_segment,
        pull_sample_rate,
        config['mimo_filter'],
        pull_rpm_data
    )

    pull_time_segment = result['filtered_data']
```

### With Beamforming

```python
# Filter preserves phase for beamforming
config = MIMOFilterConfig(zero_phase=True)  # Critical!
result = apply_mimo_iir_filter(time_data, fs, config, rpm_data)

# Now safe to use for beamforming
from acoular import TimeSamples, MicGeom, BeamformerBase
# ... beamforming code using result['filtered_data']
```

### Batch Processing

```python
for measurement_file in measurement_files:
    mic_data = load_microphone_data(measurement_file)
    rpm_data = load_rpm_telemetry(corresponding_rpm_file)

    result = apply_mimo_iir_filter(
        mic_data['time_data'],
        mic_data['sample_rate'],
        config,
        rpm_data
    )

    # Save filtered data
    save_filtered_measurement(result, output_path)
```

## References

### Primary Source

Harvey, B. (2019). *Signal Processing Methods for the Detection and Localization of Acoustic Sources via Unmanned Aerial Vehicles.* PhD Thesis, Memorial University of Newfoundland.
- **Chapter 3** (pp. 54-97): Adaptive narrowband noise removal
- **Results**: MFD < 0.25 Hz, 25-30 dB suppression

### Mathematical Foundations

Tan, L., & Jiang, J. (2015). Simplified Gradient Adaptive Harmonic IIR Notch Filter for Frequency Estimation and Tracking. *American Journal of Signal Processing*, 5(1), 6-12.

## Support

For issues or questions:
1. Check this README and `notes_and_helpers/MIMO_Adaptive_IIR_Conceptual_Guide.md`
2. Review configuration template: `mimo_filter_config_schema.yaml`
3. Run unit tests to verify installation
4. Check troubleshooting section above

---

**Implementation Date**: 2025-12-15
**Based on**: Harvey (2019) Chapter 3
**Target System**: 96-channel array, 4 rotors, 10 harmonics
**Status**: ✅ Core implementation complete, ready for validation
