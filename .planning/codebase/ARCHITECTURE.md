# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Modular Multi-Source Data Acquisition System with Real-Time Processing

**Key Characteristics:**
- Producer-consumer architecture with thread-safe circular buffers
- Callback-driven hardware interface via pigpio
- Unified logging layer with HDF5 persistence
- Separate analysis pipeline for post-collection processing

## Layers

**Hardware Interface Layer:**
- Purpose: Direct GPIO/serial communication with Raspberry Pi hardware
- Contains: `Timer`, `SignalMonitor`, `ESCTelemtry`, `MicArray` classes
- Location: `daq.py` (lines 20-662)
- Depends on: pigpio daemon, serial ports, USB audio
- Used by: DAQ orchestrator, HDF5Logger

**Data Acquisition Layer:**
- Purpose: Buffer and synchronize multi-source data streams
- Contains: Per-source circular buffers (deque), thread-safe access
- Location: `daq.py` (ESCTelemtry, MicArray, SignalMonitor classes)
- Depends on: Hardware Interface Layer
- Used by: HDF5Logger

**Logging Layer:**
- Purpose: Persistent storage with streaming writes
- Contains: `HDF5Logger` class
- Location: `daq.py` (lines 663-1050)
- Depends on: Data Acquisition Layer, h5py
- Used by: DAQ orchestrator

**Orchestration Layer:**
- Purpose: Coordinate all data sources and control recording
- Contains: `DAQ` class with CLI interface
- Location: `daq.py` (lines 1051-1552)
- Depends on: All other layers
- Used by: End user

**Motor Control Layer (Separate System):**
- Purpose: DShot protocol for ESC throttle control
- Contains: `DShot`, `MultiESCControler` classes
- Location: `esc_throttle_set.py` (lines 16-648)
- Depends on: pigpio only (no shared state with DAQ)
- Used by: End user (separate Python process)

**Analysis Layer:**
- Purpose: Post-collection signal processing
- Contains: Filter design, spectral analysis, beamforming
- Location: `analysis/` directory (7 modules, 4000+ lines)
- Depends on: HDF5 files from Logging Layer
- Used by: End user

## Data Flow

**Recording Flow (daq.py entry point):**

1. User launches: `python daq.py [options]`
2. DAQ instantiates: Timer, SignalMonitor, MicArray, ESCTelemtry×4, HDF5Logger
3. DAQ.start() → spawns monitoring threads:
   - MicArray.start() → sounddevice stream with per-sample timestamps
   - ESCTelemtry.monitor_thread() ×4 → serial polling with CRC validation
   - SignalMonitor.start_heartbeat() → PWM callbacks or heartbeat generation
   - HDF5Logger.start_logging() → background flush thread
4. Each source buffers data in thread-safe deque
5. Logger polls sources periodically → appends to HDF5
6. User stops (Ctrl+C or trigger) → flush final data → close HDF5

**State Management:**
- File-based: All persistent state in HDF5 files
- In-memory: Circular buffers with configurable max size
- Synchronized: Single Timer instance provides global reference

## Key Abstractions

**Timer:**
- Purpose: High-precision synchronization across all data sources
- Location: `daq.py` (lines 20-29)
- Pattern: Singleton-like (single instance passed to all components)
- Uses: `time.perf_counter()` for nanosecond-scale precision

**Circular Buffer (deque):**
- Purpose: Thread-safe, bounded data storage
- Examples: `ESCTelemtry.telemetry_data`, `MicArray.data`, `SignalMonitor.log_events`
- Pattern: Producer-consumer with lock protection
- Feature: Auto-drops oldest when full

**Data Source Interface:**
- Purpose: Uniform access to heterogeneous data streams
- Methods: `pop_all_data()`, `get_latest_sample()`, `start()`, `stop()`
- Examples: ESCTelemtry, MicArray, SignalMonitor

**@dataclass Configuration:**
- Purpose: Typed, validated configuration for analysis
- Location: `analysis/mimo_adaptive_iir_filter.py` (lines 254-310, MIMOFilterConfig)
- Pattern: Immutable config with field defaults

## Entry Points

**Data Acquisition (Primary):**
- Location: `daq.py` → `main()` at line 1458
- Triggers: `python daq.py [options]`
- Responsibilities: Parse CLI args, create DAQ, run monitoring loop

**Motor Control:**
- Location: `esc_throttle_set.py` → `run_test()` at end
- Triggers: `python esc_throttle_set.py`
- Responsibilities: DShot control, RPM playback

**Visualization:**
- Location: `visualize_data.py` → `visualize_data(filepath)`
- Triggers: Import and call with HDF5 path
- Responsibilities: Quick time-domain plots

**Analysis Suite:**
- Location: `analysis/mimo_filter_analysis.py` → `main()` at line 285
- Triggers: `python mimo_filter_analysis.py --mic-h5 ... --rpm-h5 ...`
- Responsibilities: Full MIMO filtering pipeline

## Error Handling

**Strategy:** Print warnings, continue operation where possible

**Patterns:**
- Hardware failures: Disable component, continue with remaining sources
- Audio device not found: Set `enable_mic_array = False`, log warning
- Serial port errors: Not fully handled (improvement needed)

## Cross-Cutting Concerns

**Timing:**
- All timestamps relative to single Timer instance
- MicArray uses frame-counter-based timestamps with optional drift correction
- ESC telemetry timestamps at packet parse time

**Thread Safety:**
- Each data source has own lock for buffer access
- MicArray callback has known synchronization gap (see CONCERNS.md)

**CPU Affinity:**
- daq.py pins to cores 1-3, esc_throttle_set.py pins to core 0
- Prevents pigpio/sounddevice timing interference
- Requires separate Python processes for both

---

*Architecture analysis: 2026-01-23*
*Update when major patterns change*
