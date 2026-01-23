#!/usr/bin/env python3
import pigpio
import time
import threading
import os
import h5py
import csv
import numpy as np
from scipy.interpolate import interp1d

#TODO: Setting CPU affinity is only needed when DAQ is used in parallel, due to sounddevice/pigpio interferences. Then DAQ must live in a
# different python instance!! Also requires to start pigpiod with sudo taskset -c 0 pigpiod (set cpu affinity to cpu 0)
os.sched_setaffinity(0, {0})


class DShot:
    """
    DShot ESC controller for Raspberry Pi using pigpio
    Supports DShot150, DShot300, and DShot600

    Reference: https://www.betaflight.com/docs/development/API/Dshot

    """

    # DShot timing (in microseconds)
    DSHOT_TIMINGS = {
        150: {'bit_length': 6.67, 't0h': 2.50, 't1h': 5.00},
        300: {'bit_length': 3.33, 't0h': 1.25, 't1h': 2.50},
        600: {'bit_length': 1.67, 't0h': 0.625, 't1h': 1.25}
    }

    # DShot special commands (0-47)
    DSHOT_CMD_MOTOR_STOP = 0
    DSHOT_CMD_BEEP1 = 1
    DSHOT_CMD_BEEP2 = 2
    DSHOT_CMD_BEEP3 = 3
    DSHOT_CMD_BEEP4 = 4
    DSHOT_CMD_BEEP5 = 5
    DSHOT_CMD_ESC_INFO = 6
    DSHOT_CMD_SPIN_DIRECTION_1 = 7
    DSHOT_CMD_SPIN_DIRECTION_2 = 8
    DSHOT_CMD_3D_MODE_OFF = 9
    DSHOT_CMD_3D_MODE_ON = 10
    DSHOT_CMD_SETTINGS_REQUEST = 11
    DSHOT_CMD_SAVE_SETTINGS = 12
    DSHOT_CMD_SPIN_DIRECTION_NORMAL = 20
    DSHOT_CMD_SPIN_DIRECTION_REVERSED = 21

    # Throttle range (48-2047 for normal operation)
    DSHOT_MIN_THROTTLE = 48
    DSHOT_MAX_THROTTLE = 2047

    def __init__(self, gpio_pin, dshot_speed=600):
        """
        Initialize DShot controller

        Args:
            gpio_pin: GPIO pin number (BCM numbering)
            dshot_speed: 150, 300, or 600 (DShot protocol speed)
        """
        self.gpio_pin = gpio_pin
        self.dshot_speed = dshot_speed

        self.timing = self.DSHOT_TIMINGS[dshot_speed]

        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon")

        # Set GPIO as output
        self.pi.set_mode(self.gpio_pin, pigpio.OUTPUT)
        self.pi.write(self.gpio_pin, 0)

    def _calculate_crc(self, frame):
        """
        Calculate 4-bit CRC for DShot frame

        Args:
            frame: 12-bit frame (11 bits throttle + 1 bit telemetry)

        Returns:
            4-bit CRC value
        """
        crc = 0
        for i in range(3):
            crc ^= (frame >> (i * 4)) & 0x0F
        return crc

    def _create_dshot_frame(self, throttle, telemetry=False):
        """
        Create 16-bit DShot frame

        Args:
            throttle: Throttle value (0-2047)
            telemetry: Request telemetry (True/False)

        Returns:
            16-bit DShot frame
        """
        # Ensure throttle is in valid range
        throttle = max(0, min(2047, throttle))

        # Build 12-bit frame: 11 bits throttle + 1 bit telemetry
        frame = (throttle << 1) | (1 if telemetry else 0)

        # Calculate and append CRC
        crc = self._calculate_crc(frame)
        dshot_frame = (frame << 4) | crc

        return dshot_frame

    def _send_dshot_frame(self, frame):
        """
        Send DShot frame using pigpio waves

        Args:
            frame: 16-bit DShot frame
        """
        # Clear any existing waveforms
        self.pi.wave_clear()

        # Create waveform
        waveform = []

        # Convert timing to microseconds
        t0h = int(self.timing['t0h'])
        t1h = int(self.timing['t1h'])
        bit_length = int(self.timing['bit_length'])

        # Generate pulses for each bit (MSB first)
        for i in range(15, -1, -1):
            bit = (frame >> i) & 1

            if bit:
                # Bit 1: Long high pulse
                waveform.append(pigpio.pulse(1 << self.gpio_pin, 0, t1h))
                waveform.append(pigpio.pulse(0, 1 << self.gpio_pin, bit_length - t1h))
            else:
                # Bit 0: Short high pulse
                waveform.append(pigpio.pulse(1 << self.gpio_pin, 0, t0h))
                waveform.append(pigpio.pulse(0, 1 << self.gpio_pin, bit_length - t0h))

        # Add waveform
        self.pi.wave_add_generic(waveform)

        # Create wave
        wave_id = self.pi.wave_create()

        if wave_id >= 0:
            # Transmit wave
            self.pi.wave_send_once(wave_id)

            # Wait for transmission to complete
            while self.pi.wave_tx_busy():
                time.sleep(0.00001)

            # Delete wave
            self.pi.wave_delete(wave_id)

    def send_command(self, throttle_value, telemetry=False):
        """
        Send throttle command to ESC

        Args:
            throttle_value: 0-2047 (48-2047 for motor control, 0-47 for special commands)
            telemetry: Request telemetry data
        """
        frame = self._create_dshot_frame(throttle_value, telemetry)
        self._send_dshot_frame(frame)

    def cleanup(self):
        """
        Cleanup and stop pigpio
        """
        self.pi.stop()


class MultiESCControler:
    """
    Control multiple ESCs simultaneously with continuous command stream.

    This class maintains a background thread that continuously sends commands
    to all ESCs at 1-2 kHz, which is required for ESCs to arm and accept throttle commands.
    """

    def __init__(self, gpio_pins, dshot_speed=300, update_rate_hz=1500):
        """
        Initialize multiple ESC controllers with continuous command stream

        Args:
            gpio_pins: List of GPIO pin numbers
            dshot_speed: DShot protocol speed (150, 300, or 600)
            update_rate_hz: Command update frequency in Hz (default: 1000 = 1kHz)
        """
        self.escs = [DShot(pin, dshot_speed) for pin in gpio_pins]
        self.gpio_pins = gpio_pins
        self.update_rate_hz = update_rate_hz
        self.update_interval = 1.0 / update_rate_hz

        # Commands list - stores the current command for each ESC
        self.commands = [DShot.DSHOT_CMD_MOTOR_STOP] * len(self.escs)
        self.lock = threading.Lock()

        # Thread control
        self.running = False
        self.command_thread = None

    def command_loop_func(self):
        """
        Background thread that continuously sends commands to ESCs
        """
        while self.running:
            start_time = time.perf_counter()

            # Send current commands to all ESCs
            with self.lock:
                for esc, cmd in zip(self.escs, self.commands):
                    esc.send_command(cmd)

            # Sleep to maintain update rate
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, self.update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """
        Start the continuous command stream thread
        """
        if not self.running:
            self.running = True
            self.command_thread = threading.Thread(target=self.command_loop_func, daemon=True)
            self.command_thread.start()
            print(f"Command stream started at {self.update_rate_hz} Hz")

    def stop(self):
        """
        Stop the continuous command stream thread
        """
        if self.running:
            self.running = False
            if self.command_thread:
                self.command_thread.join(timeout=2.0)
            print("Command stream stopped")

    def arm_all(self, duration=5.0):
        """
        Arm all ESCs by sending MOTOR_STOP commands for specified duration

        Args:
            duration: How long to send arming signal (seconds)
        """
        print("Arming all ESCs...")

        # Set all commands to MOTOR_STOP
        with self.lock:
            self.commands = [DShot.DSHOT_CMD_MOTOR_STOP] * len(self.escs)

        # Start command stream if not already running
        if not self.running:
            self.start()

        # Wait for arming duration
        time.sleep(duration)
        print("All ESCs armed!")

    def disarm_all(self):
        """
        Disarm all ESCs by setting all commands to MOTOR_STOP
        """
        print("Disarming all ESCs...")
        with self.lock:
            self.commands = [DShot.DSHOT_CMD_MOTOR_STOP] * len(self.escs)

    def set_throttle(self, throttle_values):
        """
        Set individual throttle values for each ESC

        Args:
            throttle_values: List of throttle percentages (0-100) for each ESC
        """
        if len(throttle_values) != len(self.escs):
            raise ValueError("Number of throttle values must match number of ESCs")

        # Convert percentages to DShot throttle values
        with self.lock:
            for i, percent in enumerate(throttle_values):
                percent = max(0, min(100, percent))
                self.commands[i] = int(DShot.DSHOT_MIN_THROTTLE +
                                      (percent / 100.0) * (DShot.DSHOT_MAX_THROTTLE - DShot.DSHOT_MIN_THROTTLE))

    def set_all_throttle(self, throttle_percent):
        """
        Set same throttle for all ESCs

        Args:
            throttle_percent: Throttle percentage (0-100)
        """
        self.set_throttle([throttle_percent] * len(self.escs))

    def replay(
        self,
        h5file_path: str,
        throttle_mapping_path: str,
        playback_speed: float = 1.0,
        max_throttle_limit: float = 35.0,
        arm_before_replay: bool = True,
        arming_duration: float = 5.0,
        invert_spin: bool = False
    ) -> None:
        """
        Replay recorded RPM timeseries by controlling motors with interpolated throttle commands

        Reads RPM data from HDF5 file recorded with daq.py and converts to throttle
        commands using averaged calibration mapping from process_calibration.py.

        Args:
            h5file_path: Path to HDF5 recording from daq.py
            throttle_mapping_path: Path to averaged CSV calibration file (throttle_percent, rpm_mean)
            playback_speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
            max_throttle_limit: Safety limit for maximum throttle (default: 35%)
            arm_before_replay: Auto-arm ESCs before replay (default: True)
            arming_duration: Arming signal duration in seconds (default: 5.0)
            invert_spin: Reverse motor spin direction for all ESCs (default: False)

        Raises:
            FileNotFoundError: If HDF5 or CSV file not found
            ValueError: If CSV missing required columns or HDF5 missing esc_telemetry group

        Example:
            >>> multi_esc = MultiESCControler([18, 19, 20, 21], dshot_speed=300)
            >>> multi_esc.start()
            >>> multi_esc.replay(
            ...     '/path/to/telemetry_data.h5',
            ...     '/path/to/averaged_calibration.csv',
            ...     playback_speed=0.5,  # Slow motion
            ...     invert_spin=True     # Reverse direction
            ... )
        """
        print("\n" + "=" * 70)
        print("RPM REPLAY MODE")
        print("=" * 70)

        # Validate file paths
        if not os.path.exists(h5file_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5file_path}")
        if not os.path.exists(throttle_mapping_path):
            raise FileNotFoundError(f"CSV file not found: {throttle_mapping_path}")

        print(f"HDF5 recording: {h5file_path}")
        print(f"Throttle mapping: {throttle_mapping_path}")
        print(f"Playback speed: {playback_speed}x")
        print(f"Max throttle limit: {max_throttle_limit}%")
        print(f"Spin direction: {'REVERSED' if invert_spin else 'NORMAL'}")
        print("=" * 70 + "\n")

        # Load HDF5 recording
        print("Loading HDF5 recording...")
        esc_data = {}

        with h5py.File(h5file_path, 'r') as f:
            # Validate structure
            if 'esc_telemetry' not in f:
                raise ValueError("HDF5 file missing 'esc_telemetry' group")

            # Read RPM timeseries for each ESC
            for esc_num in range(1, len(self.escs) + 1):
                esc_id = f'ESC{esc_num}'

                if esc_id not in f['esc_telemetry']:
                    print(f"  ⚠ {esc_id}: Not in recording, will remain at 0% throttle")
                    esc_data[esc_id] = None
                    continue

                timestamps = f[f'esc_telemetry/{esc_id}/timestamp'][:]
                rpms = f[f'esc_telemetry/{esc_id}/rpm'][:]

                esc_data[esc_id] = {
                    'timestamps': timestamps,
                    'rpms': rpms
                }

                print(f"  ✓ {esc_id}: {len(timestamps)} samples, {timestamps[-1] - timestamps[0]:.1f}s duration")

        # Load throttle mapping CSV
        print("\nLoading throttle mapping...")

        with open(throttle_mapping_path, 'r') as csvfile:
            # Skip comment lines
            reader = csv.DictReader(line for line in csvfile if not line.startswith('#'))
            rows = list(reader)

            if len(rows) == 0:
                raise ValueError("CSV file is empty")

            # Validate required columns
            headers = rows[0].keys()
            if 'rpm_mean' not in headers or 'throttle_percent' not in headers:
                raise ValueError(
                    "CSV missing required columns: throttle_percent, rpm_mean\n"
                    "Use process_calibration.py to generate averaged calibration file"
                )

            # Load averaged calibration data
            throttles = []
            rpms = []

            for row in rows:
                throttles.append(float(row['throttle_percent']))
                rpms.append(float(row['rpm_mean']))

            throttle_map = {
                'throttle': np.array(throttles),
                'rpm': np.array(rpms)
            }

            print(f"  ✓ Loaded {len(throttles)} calibration points")
            print(f"    Throttle range: {throttles[0]:.1f}% to {throttles[-1]:.1f}%")
            print(f"    RPM range: {rpms[0]:.1f} to {rpms[-1]:.1f}")

        # Build RPM-to-throttle interpolation function (single curve for all ESCs)
        print("\nBuilding interpolation function...")

        # Create RPM -> throttle interpolator
        rpm_to_throttle_func = interp1d(
            throttle_map['rpm'],  # x: RPM values
            throttle_map['throttle'],  # y: throttle percentages
            kind='linear',
            bounds_error=False,
            fill_value=(0, max_throttle_limit)  # Clamp extrapolation
        )

        print(f"  ✓ Single calibration curve applied to all ESCs")

        # Timeline resampling
        print("\nResampling to common timeline...")

        # Find common time range across all ESCs with valid data
        valid_data = [data for data in esc_data.values() if data is not None]

        if len(valid_data) == 0:
            raise ValueError("No valid ESC data found in HDF5 file")

        t_start = max(data['timestamps'][0] for data in valid_data)
        t_end = min(data['timestamps'][-1] for data in valid_data)

        # Create uniform 50 Hz timeline
        playback_rate_hz = 50
        dt = 1.0 / playback_rate_hz
        timeline = np.arange(t_start, t_end, dt)

        print(f"  Timeline: {t_start:.2f}s to {t_end:.2f}s ({len(timeline)} points @ {playback_rate_hz} Hz)")

        # Resample each ESC's RPM to common timeline
        resampled_rpms = {}

        for esc_num in range(1, len(self.escs) + 1):
            esc_id = f'ESC{esc_num}'

            if esc_data.get(esc_id) is None:
                resampled_rpms[esc_id] = np.zeros_like(timeline)
            else:
                data = esc_data[esc_id]
                resampled_rpms[esc_id] = np.interp(
                    timeline, data['timestamps'], data['rpms']
                )

        # Set spin direction
        if invert_spin:
            print("\n" + "=" * 70)
            print("SETTING SPIN DIRECTION")
            print("=" * 70)
            print("Reversing motor spin direction...")

            if not self.running:
                print("Starting command stream...")
                self.start()

            # Send spin direction command to all ESCs
            spin_cmd = DShot.DSHOT_CMD_SPIN_DIRECTION_REVERSED
            with self.lock:
                self.commands = [spin_cmd] * len(self.escs)

            # Send command multiple times to ensure it's received
            print("Sending spin direction command (6 repetitions)...")
            for i in range(6):
                time.sleep(0.1)
                print(f"  Repetition {i+1}/6")

            # Brief pause after direction change
            time.sleep(0.5)
            print("✓ Spin direction set to REVERSED")
            print("=" * 70 + "\n")

        # Arm ESCs if requested
        if arm_before_replay:
            print("=" * 70)
            print("ARMING ESCs")
            print("=" * 70)

            if not self.running:
                print("Starting command stream...")
                self.start()

            print(f"Sending arming signal for {arming_duration}s...")
            self.arm_all(duration=arming_duration)
            print("ESCs armed and ready")
            print("=" * 70 + "\n")

        # Playback loop
        print("=" * 70)
        print("STARTING REPLAY")
        print("=" * 70)
        print(f"Duration: {t_end - t_start:.1f}s at {playback_speed}x speed = {(t_end - t_start) / playback_speed:.1f}s real time")
        print("Press Ctrl+C to stop\n")

        try:
            start_perf = time.perf_counter()
            scaled_dt = dt / playback_speed

            for i, t in enumerate(timeline):
                # Convert RPM to throttle for each ESC (using shared calibration curve)
                throttle_values = []

                for esc_num in range(len(self.escs)):
                    esc_id = f'ESC{esc_num + 1}'

                    if esc_data.get(esc_id) is None:
                        throttle_values.append(0)
                    else:
                        rpm = resampled_rpms[esc_id][i]
                        throttle = float(rpm_to_throttle_func(rpm))
                        throttle = np.clip(throttle, 0, max_throttle_limit)
                        throttle_values.append(throttle)

                # Update commands (thread-safe via self.lock in set_throttle)
                self.set_throttle(throttle_values)

                # Progress display (every 1 second)
                if i % 50 == 0:
                    elapsed_recording = t - t_start
                    elapsed_real = time.perf_counter() - start_perf
                    progress_pct = 100 * i / len(timeline)
                    print(f"Replay: {progress_pct:5.1f}% | Recording: {elapsed_recording:6.1f}s | Real time: {elapsed_real:6.1f}s")

                # Sleep to maintain timing
                target_time = start_perf + (i + 1) * scaled_dt
                sleep_time = target_time - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.1:
                    print(f"  ⚠ Warning: Playback lagging by {-sleep_time:.3f}s")

            print("\n✓ Replay completed successfully")

        except KeyboardInterrupt:
            print("\n\n⚠ Replay interrupted by user")

        except Exception as e:
            print(f"\n\n❌ Error during replay: {e}")
            raise

        finally:
            # Always stop motors
            print("\nStopping motors...")
            self.disarm_all()
            time.sleep(0.5)

            # Restore normal spin direction if it was inverted
            if invert_spin:
                print("\nRestoring normal spin direction...")
                spin_cmd = DShot.DSHOT_CMD_SPIN_DIRECTION_NORMAL
                with self.lock:
                    self.commands = [spin_cmd] * len(self.escs)

                # Send command multiple times to ensure it's received
                for _ in range(6):
                    time.sleep(0.1)

                time.sleep(0.5)
                print("✓ Spin direction restored to NORMAL")

            print("=" * 70)
            print("REPLAY FINISHED")
            print("=" * 70 + "\n")

    def cleanup(self):
        """
        Cleanup: stop thread and cleanup all ESCs
        """
        self.disarm_all()
        time.sleep(0.1)  # Give time for final stop commands
        self.stop()
        for esc in self.escs:
            esc.cleanup()

def run_test():
    # Define GPIO pins for 4 ESCs
    ESC_PINS = [18, 19, 20, 21]
    max_throttle = 20
    multi_esc = None
    try:
        # Initialize multi-ESC controller with DShot300 (600 now working for now)
        multi_esc = MultiESCControler(ESC_PINS, dshot_speed=300)

        print("Starting throttle test...")

        # Arm all ESCs
        multi_esc.arm_all()

        # Gradually increase throttle
        for throttle in range(0, max_throttle+1, 5):
            print(f"Throttle: {max(throttle,1)}%")
            multi_esc.set_all_throttle(max(throttle,1))
            time.sleep(1)

        # Hold at max_throttle% for 2 seconds
        time.sleep(2)

        # Gradually decrease throttle
        for throttle in range(max_throttle, -1, -5):
            print(f"Throttle: {max(throttle,1)}%")
            multi_esc.set_all_throttle(max(throttle,1))
            time.sleep(1)

        # Test individual control
        print("\nTesting individual ESC control...")
        multi_esc.set_throttle([5, 10, 15, 20])
        time.sleep(2)

        # Stop all motors
        multi_esc.set_all_throttle(1)
        time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if multi_esc is not None:
            multi_esc.disarm_all()
            multi_esc.cleanup()
        print("Cleanup complete")


# Example usage
if __name__ == "__main__":
    run_test()