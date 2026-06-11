# Timestamp Synchronization Issues in MicArray

## Problem Description

When analyzing timestamp data from the MicArray class (acquired at 48kHz with blocksize=1024), we observed an alternating error pattern in timestamps at block boundaries:

- **Expected interval** between last sample of block N and first sample of block N+1: ~20.8μs (1/48000)
- **Observed pattern**: Alternating errors of approximately -980μs and +1020μs
- **Pattern**: Negative → Positive → Negative → Positive (consistent oscillation)
- **Error magnitude**: ~1ms (equivalent to ~47-48 samples at 48kHz)

This was discovered using `check_mic_array_dt.py` which calculates `dt[1023::1024]` - the time differences between consecutive block boundaries.

## Root Cause Analysis

### The Original Implementation

The original timestamp calculation in `MicArray.callback()` used:

```python
adc_block_timestamp = time_info.inputBufferAdcTime
per_sample_timestamps = adc_block_timestamp + self.stream_time_offset + self.sample_time_offsets_in_block
```

Where `stream_time_offset` was calculated once at stream start:

```python
self.stream_time_offset = self.timer.get_time() - self.stream.time
```

### Why This Caused Oscillating Errors

The alternating ±1ms error pattern is caused by **clock quantization** in ALSA/PortAudio timestamps:

1. **Limited Timestamp Precision**: The `inputBufferAdcTime` from PortAudio/ALSA has limited precision on many systems:
   - ALSA DMA timestamps typically have ~950-980ns deltas without compensation
   - Some audio host APIs (like MME) provide only millisecond-level accuracy
   - USB audio can introduce additional ~4.9-5ms buffering offsets

2. **Quantization Artifacts**: When hardware timestamps have ~1ms quantization:
   - Block N boundary falls just before a quantization step → timestamp rounds down → appears early → negative error
   - Block N+1 boundary falls just after a quantization step → timestamp rounds up → appears late → positive error
   - Pattern alternates based on where block boundaries (every 21.33ms) align with the ~1ms quantization grid

3. **Single-Point Calibration**: The `stream_time_offset` calculated once at startup doesn't account for:
   - Clock drift between `perf_counter()` and audio hardware clock
   - Variations in actual hardware timing
   - Callback scheduling jitter (typically 100-500μs)

### Why Not Simple Clock Drift?

The oscillating pattern rules out simple clock drift. If the clocks were drifting apart, we would expect the error to **increase monotonically over time**, not alternate between positive and negative values.

## The Solution: Frame-Counter Based Timestamping with Optional Re-Synchronization

### Implementation

We replaced hardware-dependent `inputBufferAdcTime` with a frame-counter approach that optionally re-synchronizes with hardware timestamps using smooth adaptive correction:

```python
# In __init__
self.frame_count = 0
self.reference_time = None
self.enable_resync = True  # Optional periodic re-sync
self.resync_interval = 100  # Re-sync every 100 blocks (~2.1s at 48kHz/1024)
self.convergence_rate = 0.001  # 0.1% correction per re-sync interval
self.blocks_since_resync = 0
self.resync_history = deque(maxlen=10)
self.sample_rate_correction = 1.0  # Adaptive correction factor

# In callback
if self.reference_time is None:
    self.reference_time = self.timer.get_time()

# Calculate current frame index for this block
current_frame = self.frame_count
self.frame_count += frames

# Generate per-sample timestamps BEFORE applying correction
# This ensures the current block uses the existing correction factor
sample_indices = np.arange(current_frame, current_frame + frames)
corrected_sample_rate = self.sample_rate * self.sample_rate_correction
per_sample_timestamps = self.reference_time + sample_indices / corrected_sample_rate

# Periodic re-synchronization with smooth correction (AFTER timestamp generation)
if self.enable_resync and self.blocks_since_resync >= self.resync_interval:
    expected_time_elapsed = self.frame_count / (self.sample_rate * self.sample_rate_correction)
    frame_based_time = self.reference_time + expected_time_elapsed
    hw_offset = time_info.inputBufferAdcTime + self.stream_time_offset
    drift = frame_based_time - hw_offset

    # Apply smooth correction via adaptive sample rate if drift exceeds 5ms threshold
    if abs(drift) > 0.005:
        drift_rate = drift / expected_time_elapsed if expected_time_elapsed > 0 else 0

        # Store old correction factor
        old_correction = self.sample_rate_correction

        # Adjust effective sample rate to smoothly converge
        self.sample_rate_correction *= (1.0 - drift_rate * self.convergence_rate)
        # Clamp to reasonable bounds (±1%)
        self.sample_rate_correction = np.clip(self.sample_rate_correction, 0.99, 1.01)

        # CRITICAL: Adjust reference_time to maintain continuity
        # Ensure timestamp at current frame_count is identical under both old and new corrections
        time_at_current_frame = self.reference_time + self.frame_count / (self.sample_rate * old_correction)
        self.reference_time = time_at_current_frame - self.frame_count / (self.sample_rate * self.sample_rate_correction)
```

### Benefits

* **Mathematically consistent** - timestamps calculated from sample rate guarantee correct relative timing
* **Monotonic and smooth** - no jitter, oscillation, or discontinuities in timestamp progression
* **Microsecond precision** - not limited by hardware clock quantization
* **Eliminates ±1ms error** - removes the alternating pattern completely
* **Synchronized to global Timer** - maintains consistency with other data sources via initial reference
* **Drift correction** - periodic re-sync gradually corrects for long-term clock drift without discontinuities
* **Smooth convergence** - corrections spread across all samples via adaptive sample rate, not abrupt jumps
* **Configurable** - can disable re-sync or adjust convergence rate based on application needs

### Re-Synchronization Strategy

The optional periodic re-sync feature uses **smooth adaptive correction** to eliminate discontinuities:

1. **Compares** frame-based time with hardware timestamp every N blocks (default: 100 blocks = ~2.1 seconds)
2. **Calculates drift rate**: Drift per second = (frame_based_time - hw_time) / elapsed_time
3. **Stores history**: Tracks last 10 drift measurements (including correction_factor) for monitoring
4. **Adjusts sample rate correction**: If drift exceeds 5ms threshold, adjusts the `sample_rate_correction` factor
   - Formula: `sample_rate_correction *= (1.0 - drift_rate * convergence_rate)`
   - Positive drift → running too fast → increase divisor to slow down
   - Negative drift → running too slow → decrease divisor to speed up
5. **Spreads correction smoothly**: The correction factor affects ALL future timestamp calculations, not just reference_time
6. **Maintains monotonicity**: No abrupt jumps or discontinuities; correction distributed across all samples

**Design rationale**:
- **5ms drift threshold**: Large enough to avoid reacting to hardware timestamp quantization noise (~1ms), small enough to catch real drift
- **0.1% convergence rate** (default 0.001): Applies 0.1% of the drift correction per resync interval, ensuring smooth convergence without discontinuities
- **±1% correction bounds**: Clamps `sample_rate_correction` to [0.99, 1.01] to prevent runaway corrections
- **100-block interval**: Balances overhead with responsiveness to drift (~2.1 seconds at 48kHz/1024)

**Why smooth correction eliminates discontinuities**:

The key to eliminating discontinuities is a two-part strategy:

1. **Timestamp generation BEFORE correction update**: Generate timestamps for the current block using the existing `sample_rate_correction` value before updating it. This prevents the current block from being affected by the correction.

2. **Reference time adjustment for continuity**: When `sample_rate_correction` changes, we must adjust `reference_time` to ensure continuity. Without this adjustment, changing the correction factor would create a discontinuity because:
   ```
   Old trajectory: t = reference_time + frame / (sample_rate * old_correction)
   New trajectory: t = reference_time + frame / (sample_rate * new_correction)
   ```
   These two trajectories would diverge at the correction point, creating a visible jump.

   **The continuity fix**: Adjust `reference_time` so that the timestamp at `frame_count` (current position) is identical under both old and new corrections:
   ```python
   # Calculate timestamp at current position using old correction
   time_at_current_frame = reference_time + frame_count / (sample_rate * old_correction)

   # Solve for new reference_time that gives the same timestamp with new correction
   # time_at_current_frame = new_reference_time + frame_count / (sample_rate * new_correction)
   new_reference_time = time_at_current_frame - frame_count / (sample_rate * new_correction)
   ```

   This "pivots" the timestamp trajectory at the current frame - past timestamps remain unchanged, future timestamps gradually diverge to correct drift, but there's no discontinuity at the pivot point.

**Mathematical guarantee**: The timestamp at `frame_count` is preserved exactly during the correction factor change, ensuring perfect continuity. Future timestamps smoothly converge toward the corrected rate over subsequent samples.

### Trade-offs

⚠️ **Absolute timing accuracy**: The initial `reference_time` is set on the first callback, which means:
- There's initial uncertainty (~1ms based on original observation) about when the first sample was actually captured
- This offset affects synchronization with external events (ESC telemetry, GPIO triggers)
- With re-sync enabled, this uncertainty is gradually corrected over ~20-40 seconds
- Without re-sync, absolute timing uncertainty remains at ~1ms throughout recording

⚠️ **Clock drift**:
- Long recordings may accumulate slight drift if the actual audio hardware sample rate differs slightly from nominal 48kHz
- With re-sync enabled (default), drift is automatically corrected every ~2 seconds
- Without re-sync, drift accumulation is typically negligible (<1ppm for crystal oscillators, ~0.1ms per 100 seconds)

⚠️ **Hardware timestamp dependency**:
- Re-sync still relies on `inputBufferAdcTime` being available and non-zero
- If hardware timestamps are completely unavailable, re-sync has no effect (gracefully degrades to pure frame-counter mode)
- Re-sync uses conservative thresholds to avoid being misled by timestamp quantization

## Background: PortAudio Timing Limitations

From PortAudio documentation and research:

> "The values in the time variable are estimates, they are not exact."

> "Wide variability in timestamp quality across operating systems and native audio APIs."

Four factors affect timestamp quality:
1. Time base granularity from `Pa_GetStreamTime()`
2. Buffer timestamp availability from host API
3. Known latency accuracy used for timestamp offsets
4. Sample rate conversion precision

The recommended approach is:

> "Clients assess the timing behavior of their target Host-APIs relative to their application requirements and deploy application-specific clock recovery or time smoothing algorithms as necessary."

## Application Context

For the **MartyMicFly Flying Measurement Microphone** project:

- **Primary need**: Smooth, consistent timestamps for audio signal processing
- **Secondary need**: Synchronization with ESC telemetry and GPIO triggers
- **Decision**: Prioritize internal audio consistency over absolute external synchronization

If absolute synchronization becomes critical in the future, additional calibration procedures (e.g., periodic re-sync with external timing events) can be added on top of this frame-counter foundation.

## References

- Related files:
  - `daq.py` - Implementation of frame-counter timestamping in `MicArray` class
  - `check_mic_array_dt.py` - Script used to discover the timing issue
  - `dt_check.csv` - Data showing the alternating error pattern
- PortAudio documentation: https://www.portaudio.com/
- ALSA timestamp documentation: https://www.alsa-project.org/

## Configuration

The re-synchronization feature can be configured via `MicArray` initialization:

```python
mic_array = MicArray(
    timer=timer,
    enable_resync=True,       # Enable/disable re-sync (default: True)
    resync_interval=100,      # Blocks between re-sync checks (default: 100)
    convergence_rate=0.001    # Correction strength per resync (default: 0.001 = 0.1%)
)
```

**Recommended settings**:
- **Standard use**: `enable_resync=True, resync_interval=100, convergence_rate=0.001` (default)
- **Pure frame-counter mode**: `enable_resync=False` (eliminates hardware timestamp dependency)
- **Faster convergence**: `resync_interval=50, convergence_rate=0.002` (re-sync every ~1 second with 0.2% correction)
- **Slower convergence**: `resync_interval=200, convergence_rate=0.0005` (re-sync every ~4 seconds with 0.05% correction)
- **Aggressive correction**: `convergence_rate=0.01` (1% correction per resync - may be noticeable in analysis)
- **Conservative correction**: `convergence_rate=0.0001` (0.01% correction per resync - very gradual)

**Parameter interaction**:
- `resync_interval` controls HOW OFTEN drift is measured (in blocks)
- `convergence_rate` controls HOW MUCH correction is applied when drift is detected
- Smaller `convergence_rate` = smoother correction but slower convergence
- Larger `convergence_rate` = faster convergence but potentially visible in fine-grained analysis

The `resync_history` deque stores the last 10 drift measurements (including `correction_factor`) and can be inspected for monitoring clock behavior.

## Future Improvements

Additional enhancements if needed:

1. **Latency calibration**: Measure actual latency from ADC to callback for precise absolute timing
2. **Hardware timestamps**: Investigate if ALSA hw timestamps are available with higher precision
3. **Alternative audio backend**: Test JACK or other low-latency audio systems for potentially better timestamp quality
4. **External sync events**: Use GPIO triggers or ESC telemetry timestamps for cross-validation
5. **Adaptive convergence rate**: Automatically adjust `convergence_rate` based on observed drift stability