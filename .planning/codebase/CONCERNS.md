# Codebase Concerns

**Analysis Date:** 2026-01-23

## Tech Debt

**Thread Safety in MicArray Callback:**
- Issue: Shared state modified without synchronization in audio callback
- Files: `daq.py` (lines 493-608, MicArray.callback method)
- Why: Performance optimization - locks add latency in audio path
- Impact: Potential race conditions on multi-core systems; timestamp inconsistencies possible
- Variables affected: `reference_time`, `sample_rate_correction`, `frame_count`, `blocks_since_resync`
- Note: `sync_lock` is defined (line 457) but never used in callback
- Fix approach: Add lock protection for critical writes, or use atomic operations

**CPU Affinity Conflicts Between Modules:**
- Issue: `daq.py` and `esc_throttle_set.py` have conflicting CPU affinity settings
- Files: `daq.py` (line 17: cores 1-3), `esc_throttle_set.py` (line 13: core 0)
- Why: Prevents pigpio/sounddevice timing interference
- Impact: Cannot run both in same Python process; second call overwrites first
- Fix approach: Document constraint clearly, or implement dynamic affinity management

**Class Name Typo:**
- Issue: Class named `ESCTelemtry` instead of `ESCTelemetry` (missing 'e')
- File: `daq.py` (line 270)
- Impact: Minor - affects readability and documentation searches
- Fix approach: Rename class (breaking change for any external users)

## Known Bugs

**Timestamp Re-sync Disabled Due to Instability:**
- Symptoms: Large jumps in drift value when re-sync enabled
- Trigger: MicArray resynchronization with hardware timestamps
- File: `daq.py` (lines 565-567, TODO comment)
- Workaround: Re-sync disabled by default; base drift stable around zero without it
- Root cause: Unknown oscillation in correction factor (documented in `notes_and_helpers/TIMESTAMP_SYNC_ISSUES.md`)
- Status: Known issue, documented, worked around

**DShot600 Timing Precision Loss:**
- Symptoms: DShot600 may have incorrect pulse widths
- Trigger: Using DShot600 speed (1.67μs bit timing)
- File: `esc_throttle_set.py` (lines 127-129)
- Code: `t0h = int(0.625) = 0` creates 0-microsecond pulse
- Workaround: Use DShot300 or DShot150 for more reliable timing
- Root cause: `int()` truncates instead of `round()`
- Fix: Change `int()` to `round()` for timing values

## Security Considerations

**No Significant Security Concerns:**
- Risk: Low - local hardware control application
- Current mitigation: Runs with user permissions
- No network services exposed
- No secrets stored in code

**File Path Handling:**
- Risk: Home directory path constructed with `getpass.getuser()` (line 1511)
- Current mitigation: Only creates files in user's home directory
- Recommendations: Validate paths before operations

## Performance Bottlenecks

**HDF5 Resize Operations:**
- Problem: Datasets resized per append operation
- File: `daq.py` (lines 879-880, 926-930)
- Measurement: Not profiled, but could be 100+ appends/second
- Cause: Each `resize()` allocates new memory
- Improvement path: Batch appends, pre-allocate larger chunks
- Note: Mitigated by `flush_interval=1.0` batching

**Serial Port Polling:**
- Problem: Blocking reads with select() timeout
- File: `daq.py` (line 366)
- Measurement: 0.1s timeout per poll
- Cause: Relies on ESC pushing telemetry; no request mechanism
- Note: TODO at line 363 suggests polling could improve sample rate

## Fragile Areas

**MicArray Callback with Drift Correction:**
- File: `daq.py` (lines 542-607)
- Why fragile: Complex math with multiple interdependent variables
- Common failures: Timestamp discontinuities, correction factor runaway
- Safe modification: Thoroughly test with synthetic data before deploying
- Test coverage: None (needs unit tests for correction math)
- Documentation: Extensive comments and `notes_and_helpers/TIMESTAMP_SYNC_ISSUES.md`

**Recording State Machine:**
- File: `daq.py` (lines 1193, 1273-1384)
- Why fragile: State transitions don't validate current state
- Common failures: Duplicate start/stop calls, partial cleanup on error
- Safe modification: Add state validation assertions
- Test coverage: None

**DShot Waveform Generation:**
- File: `esc_throttle_set.py` (lines 95-175)
- Why fragile: Microsecond-precise timing via pigpio waves
- Common failures: ESC doesn't respond if timing off
- Safe modification: Test with oscilloscope; DShot300 more reliable than DShot600
- Test coverage: Manual hardware testing only

## Scaling Limits

**Raspberry Pi Resources:**
- Current capacity: 4 ESCs, 95 mic channels, ~48kHz sample rate
- Limit: CPU bound on heavy processing; memory for large buffers
- Symptoms at limit: Audio dropouts, missed telemetry packets
- Scaling path: Reduce sample rate, fewer channels, or Pi 5

**HDF5 File Size:**
- Current capacity: Unlimited recording length
- Limit: Disk space; file access slows with very large datasets
- Symptoms at limit: Slow append, high memory during flush
- Scaling path: Split recordings, use chunked datasets (already enabled)

## Missing Critical Features

**Error Handling in Serial Port Operations:**
- Problem: Serial port init and I/O lack try/except
- File: `daq.py` (lines 358-379)
- Current workaround: Crash on port not found
- Blocks: Graceful handling of ESC disconnection
- Fix: Wrap in try/except, handle SerialException

**Input Validation in Configuration:**
- Problem: Constructor accepts parameters without range validation
- File: `daq.py` (lines 1056-1088)
- Current workaround: Rely on hardware failure to detect issues
- Blocks: Meaningful error messages for misconfiguration
- Fix: Add parameter validation in __init__

## Test Coverage Gaps

**Critical Untested Code:**
- KISS telemetry CRC calculation (`daq.py` line 255-267)
  - Risk: Silent data corruption if CRC wrong
  - Priority: High
- DShot frame generation (`esc_throttle_set.py` lines 75-111)
  - Risk: ESC communication failures
  - Priority: High
- Timing synchronization algorithms (`daq.py` lines 542-607)
  - Risk: Timestamp drift/discontinuities
  - Priority: Medium

**No Test Framework:**
- What's not tested: Everything (no tests exist)
- Risk: Regressions go unnoticed
- Priority: High
- Difficulty: Hardware dependencies require mocking

---

*Concerns audit: 2026-01-23*
*Update as issues are fixed or new ones discovered*
