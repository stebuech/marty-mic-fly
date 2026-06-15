# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Patterns

**Files:**
- snake_case.py for all modules: `daq.py`, `esc_throttle_set.py`, `data_loader.py`
- UPPERCASE.md for important docs: `CLAUDE.md`
- snake_case.yaml for configs: `example_config.yaml`

**Functions:**
- snake_case for all functions: `load_rpm_telemetry()`, `synchronize_datasets()`
- Leading underscore for private: `_pwm_callback()`, `_log_callback()`, `_heartbeat_loop()`
- Verb-first naming: `get_time()`, `start_heartbeat()`, `pop_all_data()`

**Variables:**
- snake_case for variables: `serial_conn`, `sample_rate`, `telemetry_data`
- UPPER_SNAKE_CASE for constants: `DSHOT_TIMINGS`, `DSHOT_CMD_MOTOR_STOP`
- No underscore prefix for instance variables (except private methods)

**Types:**
- PascalCase for classes: `Timer`, `SignalMonitor`, `ESCTelemtry`, `MicArray`
- PascalCase for dataclasses: `MIMOFilterConfig`
- Note: Typo in `ESCTelemtry` (should be `ESCTelemetry`)

## Code Style

**Formatting:**
- 4-space indentation (PEP 8)
- No formatter config detected (no .prettierrc, ruff.toml, etc.)
- Double blank lines between top-level classes
- Single blank line between methods

**Strings:**
- Double quotes for docstrings and string literals
- f-strings for formatted output: `f"Throttle: {throttle}%"`

**Line Length:**
- Not enforced (some lines exceed 100 chars)
- Typical: 80-120 characters

**Linting:**
- No linting configuration detected
- Recommend adding: ruff or flake8

## Import Organization

**Order:**
1. Standard library (os, sys, time, threading)
2. Third-party packages (numpy, h5py, scipy)
3. Local imports (none detected - modules are standalone)

**Style:**
```python
import getpass
import os
import serial
import struct
import time
import threading
import select
from collections import deque
import numpy as np
import h5py
from datetime import datetime
import sounddevice as sd
```

**Path Aliases:**
- None used (all absolute imports)

## Error Handling

**Patterns:**
- Try/except at component boundaries (e.g., MicArray.start)
- Warnings printed, operation continues where possible
- Hardware failures disable component, don't crash

**Example (`daq.py` lines 1209-1216):**
```python
try:
    self.mic_array.start()
    print(f"Audio monitoring started")
except RuntimeError as e:
    print(f"Warning: Audio monitoring failed to start: {e}")
    self.enable_mic_array = False
```

**What to Throw:**
- Hardware not found: RuntimeError
- Invalid configuration: ValueError (not widely used yet)

## Logging

**Framework:**
- Console output only (print statements)
- No structured logging library

**Patterns:**
- Status updates: `print(f"Starting monitoring...")`
- Errors: `print(f"Warning: {error}")` or `print(f"Error: {error}")`
- Section headers: `print("\n" + "=" * 70)`

**Improvement Needed:**
- Consider adding Python logging module

## Comments

**When to Comment:**
- Explain why, not what: `# Avoid core 0 for pigpio/sounddevice interference`
- Document business logic: `# KISS protocol: 10-byte packet with CRC8`
- Mark TODOs: `#TODO: poll/request data from esc may lead to higher sample rates`

**Docstrings (Google-style):**
```python
def __init__(self, gpio_pin, dshot_speed=600):
    """
    Initialize DShot controller

    Args:
        gpio_pin: GPIO pin number (BCM numbering)
        dshot_speed: 150, 300, or 600 (DShot protocol speed)
    """
```

**Module Docstrings:**
```python
#!/usr/bin/env python3
"""
Standalone MIMO Adaptive IIR Notch Filter Analysis

Applies MIMO adaptive IIR notch filtering to microphone array data
and generates comprehensive performance reports.

Author: MartyMicFly Project
Date: 2025-12-15
"""
```

**TODO Format:**
- `#TODO:` followed by description (note: non-standard spacing before colon)
- Some include context in following comment lines

## Function Design

**Size:**
- Some functions exceed 200 lines (refactoring opportunity)
- Complex functions like `MicArray.callback()` are 100+ lines

**Parameters:**
- Typically 3-8 parameters per function
- Use keyword arguments for optional params
- Dataclasses for complex configs (`MIMOFilterConfig`)

**Return Values:**
- Explicit returns
- `None` for empty results
- Tuples for multiple returns

## Module Design

**Exports:**
- No `__all__` declarations
- Classes at module level for import
- No barrel files (index.py)

**Structure:**
- Imports at top
- Constants after imports
- Classes in logical order
- Main block at end: `if __name__ == "__main__":`

---

*Convention analysis: 2026-01-23*
*Update when patterns change*
