# CPU Affinity Fix for ESC Control Interference

## Problem Description

When running the data acquisition script (`daq.py`) that uses **sounddevice** for audio capture, ESC control via `esc_throttle_set.py` becomes unreliable or completely non-functional. Symptoms include:

- Random beeping from ESCs
- Erratic motor spinning
- ESCs not responding to throttle commands
- Issue persists even after stopping `daq.py` - **only rebooting fixes it**

### Root Cause

The issue is caused by **CPU contention and timing interference** between two real-time critical processes:

1. **pigpiod** (used by `esc_throttle_set.py`):
   - Requires microsecond-precise timing for DShot protocol
   - DShot600: 1.67 μs bit timing
   - DShot300: 3.33 μs bit timing
   - Timing jitter > 10% causes ESC frame rejection

2. **sounddevice/ALSA** (used by `daq.py`):
   - Real-time audio callbacks at 48 kHz
   - Uses high-priority CPU scheduling
   - Locks memory and modifies system timer behavior

When **sounddevice is imported**, it initializes PortAudio/ALSA which:
- Requests real-time scheduling priority for audio threads
- Can preempt other processes including pigpiod
- Causes CPU cache pollution and timer interrupt delays
- Results in timing jitter that breaks DShot communication

**The problem occurs immediately upon `import sounddevice`**, even before creating any audio streams.

## Solution: CPU Affinity Isolation

Isolate pigpiod and audio processing to separate CPU cores to prevent interference.

### Strategy (Raspberry Pi 4 with 4 cores)

- **Core 0**: Dedicated to pigpiod (ESC control - needs precise timing)
- **Cores 1-3**: Audio processing, Python, and everything else

This prevents audio callbacks from preempting pigpiod's critical timing operations.

---

## Implementation Steps

### 1. Configure pigpiod with CPU Affinity

Stop the default pigpiod and start it pinned to Core 0:

```bash
# Stop pigpiod if running
sudo killall pigpiod

# Start pigpiod pinned to Core 0 only
sudo taskset -c 0 pigpiod

# Verify it's running on Core 0
ps -eLo pid,tid,psr,comm | grep pigpio
# The PSR column shows which CPU core (should show 0)
```

### 2. Make pigpiod CPU Affinity Permanent

Create a systemd service to automatically start pigpiod on Core 0:

```bash
# Create the service file
sudo nano /etc/systemd/system/pigpiod-affinity.service
```

Paste this content:

```ini
[Unit]
Description=Pigpio daemon with CPU affinity
After=network.target

[Service]
Type=forking
ExecStart=/usr/bin/taskset -c 0 /usr/bin/pigpiod
ExecStop=/usr/bin/killall pigpiod
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Disable default pigpiod service if it exists
sudo systemctl disable pigpiod

# Enable and start the new service
sudo systemctl enable pigpiod-affinity.service
sudo systemctl start pigpiod-affinity.service

# Verify it's running on core 0
ps -eLo pid,tid,psr,comm | grep pigpio
```

### 3. Pin daq.py to Cores 1-3

Add CPU affinity setting at the top of `daq.py` (after imports, around line 13):

```python
import os

# Pin this process to cores 1-3 (avoid core 0 where pigpiod runs)
os.sched_setaffinity(0, {1, 2, 3})
```

This ensures the audio processing doesn't interfere with pigpiod on Core 0.

### 4. Pin esc_throttle_set.py to Core 0

Add CPU affinity setting at the top of `esc_throttle_set.py` (after imports, around line 5):

```python
import os

# Allow this process to use core 0 (where pigpiod runs)
os.sched_setaffinity(0, {0})
```

This ensures the ESC control script runs on the same core as pigpiod for optimal communication.

---

## Verification

After implementing the fix:

1. **Check pigpiod is on Core 0:**
   ```bash
   ps -eLo pid,tid,psr,comm | grep pigpio
   ```
   PSR column should show `0`

2. **Start daq.py and check its CPU affinity:**
   ```bash
   python3 daq.py &
   taskset -cp $(pgrep -f daq.py)
   ```
   Should show: `current affinity list: 1,2,3`

3. **Run esc_throttle_set.py and verify ESC control works:**
   ```bash
   python3 esc_throttle_set.py
   ```
   ESCs should respond normally with smooth throttle control

---

## Alternative Solutions (Not Recommended)

### Option 2: Disable sounddevice Real-Time Scheduling
- Difficult to configure reliably
- May cause audio dropouts

### Option 3: Separate Hardware
- Use separate Raspberry Pi for ESC control
- Expensive and complex setup

### Option 5: Use I2S Audio Instead of USB
- Lower CPU overhead than USB audio
- Requires hardware changes

---

## Technical Details

### Why Priority Alone Doesn't Work

Simply raising pigpiod's priority (via `nice -n -20`) is insufficient because:
- Both processes still compete for the same CPU core
- Context switching still occurs
- CPU cache is shared and gets polluted
- Timer interrupts can still be delayed

### Why CPU Affinity Works

CPU affinity isolation provides:
- **Dedicated CPU cycles** for pigpiod without preemption
- **Separate CPU caches** - no cache pollution
- **Independent timer handling** per core
- **Guaranteed microsecond-level timing** for DShot frames

### DShot Protocol Requirements

DShot is extremely timing-sensitive:
- Each bit has precise timing requirements
- Timing tolerance is typically ±10%
- DShot600: 1.67 μs ± 0.17 μs per bit
- Audio callbacks lasting ~21ms can cause significant jitter

---

## Troubleshooting

### ESC control still not working after applying fix

1. **Verify pigpiod is on Core 0:**
   ```bash
   ps -eLo pid,tid,psr,comm | grep pigpio
   ```

2. **Check if daq.py is avoiding Core 0:**
   ```bash
   taskset -cp $(pgrep -f daq.py)
   ```

3. **Restart pigpiod with affinity:**
   ```bash
   sudo systemctl restart pigpiod-affinity.service
   ```

4. **Reboot if systemd changes don't take effect:**
   ```bash
   sudo reboot
   ```

### System has fewer than 4 cores

For Raspberry Pi 3 (4 cores):
- Core 0: pigpiod
- Cores 1-3: audio and everything else

For systems with 2 cores:
- Core 0: pigpiod
- Core 1: audio (may have reduced isolation benefits)

---

## References

- **DShot Protocol:** https://www.betaflight.com/docs/development/API/Dshot
- **Linux CPU Affinity:** `man taskset`, `man sched_setaffinity`
- **pigpio Library:** http://abyz.me.uk/rpi/pigpio/
- **PortAudio Real-Time:** http://www.portaudio.com/docs/v19-doxydocs/

---

## Status

✅ **Solution Verified**: CPU affinity isolation successfully resolves the ESC control interference issue caused by sounddevice/ALSA real-time audio processing.