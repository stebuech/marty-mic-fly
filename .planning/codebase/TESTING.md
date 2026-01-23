# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Runner:**
- Not detected - No formal test framework present
- No pytest, unittest, or similar configured

**Why No Tests:**
- Research project with hardware dependencies
- pigpio requires running daemon
- Serial communication requires physical ESCs
- Audio input requires USB microphone array

**Run Commands:**
```bash
# No test commands available
# Recommended if tests added:
uv add pytest pytest-mock
pytest tests/ -v
```

## Test File Organization

**Location:**
- No test directory exists
- No test files found (no `test_*.py` or `*_test.py`)

**Helper Scripts (Not Tests):**
- `notes_and_helpers/check_mic_array_dt.py` - Timestamp validation
- `notes_and_helpers/generate_test_audio_signals.py` - Synthetic test signals
- `esc_throttle_set.py` → `run_test()` - Integration test example

## Testable Components

**Unit Testable (no hardware):**
- `crc8_kiss()` - KISS telemetry CRC calculation (`daq.py` lines 255-267)
- `DShot._generate_frame()` - DShot frame with CRC4 (`esc_throttle_set.py` lines 75-111)
- `MIMOFilterConfig` validation (`analysis/mimo_adaptive_iir_filter.py`)
- `FrequencyInitializer` algorithms (`analysis/frequency_initialization.py`)
- `data_loader.py` functions with mock HDF5 files

**Integration Testable (with mocks):**
- ESCTelemtry packet parsing (mock serial data)
- MicArray buffer management (mock sounddevice)
- HDF5Logger file operations (temp files)
- DAQ orchestration (mock all components)

**Hardware Required:**
- DShot transmission to ESC
- GPIO interrupt timing
- Real-time audio capture
- Serial port communication

## Code Organization for Testability

**Good Patterns:**
- Separation of concerns (Timer, SignalMonitor, ESCTelemtry, MicArray as separate classes)
- Type hints in analysis modules (~70% coverage)
- Dataclass configurations enable mock injection
- Google-style docstrings with Args/Returns

**Improvement Needed:**
- Hardware dependencies injected, not created internally
- Abstract base classes for data sources
- Dependency injection for pigpio, serial, sounddevice

## Type Hints

**Coverage:**
- **Analysis modules**: High (~70%)
  - `analysis/data_loader.py` uses `Dict`, `Tuple`, `Optional`
  - `analysis/mimo_adaptive_iir_filter.py` uses `@dataclass` with type annotations
  - `analysis/frequency_initialization.py` has parameter type hints
- **Hardware modules**: Low (~20%)
  - `daq.py` - Most functions lack return type hints
  - `esc_throttle_set.py` - Minimal type hints

**Example with Types:**
```python
def load_rpm_telemetry(
    filepath: str,
    esc_ids: List[str] = None,
    time_range: Tuple[float, float] = None
) -> Dict[str, Dict[str, np.ndarray]]:
```

## Test Recommendations

**Priority 1 - Unit Tests:**
```python
# tests/test_crc.py
def test_crc8_kiss_valid_packet():
    """Test CRC8 calculation matches expected value"""
    packet = bytes([25, 0x0E, 0x10, 0x00, 0x64, 0x00, 0x00, 0x1F, 0x40])
    expected_crc = 0xAB  # Calculate expected
    assert crc8_kiss(packet) == expected_crc

# tests/test_dshot.py
def test_dshot_frame_generation():
    """Test DShot frame includes correct CRC"""
    # Mock pigpio, verify frame structure
```

**Priority 2 - Integration Tests:**
```python
# tests/test_esc_telemetry.py
def test_packet_parsing(mock_serial):
    """Test ESCTelemtry parses valid KISS packets"""
    mock_serial.read.return_value = VALID_PACKET
    esc = ESCTelemtry(port=mock_serial)
    # Verify parsed values

# tests/test_hdf5_logger.py
def test_logging_creates_file(tmp_path):
    """Test HDF5Logger creates file with expected structure"""
```

**Priority 3 - Smoke Tests:**
```python
# tests/test_smoke.py
def test_imports():
    """Verify all modules import without error"""
    import daq
    import esc_throttle_set
    from analysis import data_loader
```

## Coverage

**Requirements:**
- No coverage tracking configured
- No coverage targets defined

**Recommendation:**
```bash
# Add to dev dependencies
uv add pytest-cov

# Run with coverage
pytest tests/ --cov=daq --cov=analysis --cov-report=html
```

## Fixtures and Factories

**Existing Test Data:**
- `notes_and_helpers/generate_test_audio_signals.py` generates synthetic audio
- Real HDF5 files exist in `~/MMFDataLogs/` (not in repo)

**Recommended Pattern:**
```python
# tests/fixtures/sample_packets.py
VALID_KISS_PACKET = bytes([...])
INVALID_CRC_PACKET = bytes([...])

# tests/conftest.py
@pytest.fixture
def mock_pigpio():
    with patch('pigpio.pi') as mock:
        yield mock
```

---

*Testing analysis: 2026-01-23*
*Update when test patterns change*
