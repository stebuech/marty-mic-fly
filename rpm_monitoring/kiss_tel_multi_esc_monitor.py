#!/usr/bin/env python3
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
import RPi.GPIO as GPIO


def crc8_kiss(data):
    """Calculate KISS CRC8 checksum
    Reference: http://ultraesc.de/downloads/KISS_telemetry_protocol.pdf"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc


class Timer:
    """High precision timer for synchronization"""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.start_time_wall = time.time()

    def get_time(self):
        """Get high-precision timestamp in seconds"""
        return time.perf_counter() - self.start_time


class TriggerMonitor:
    """Monitor trigger signal with hardware interrupts"""
    def __init__(self, trigger_pin=17, timer=None, trigger_type='rising', buffer_size=10000):

        self.trigger_pin = trigger_pin
        self.timer = timer if timer else Timer()
        self.trigger_events = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.enabled = True

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        edge = dict(rising=GPIO.RISING, falling=GPIO.FALLING).get(trigger_type, GPIO.BOTH)

        GPIO.add_event_detect(self.trigger_pin, edge, callback=self.trigger_callback)

    def trigger_callback(self, channel):
        """Called when trigger event occurs"""
        timestamp = self.timer.get_time()
        state = GPIO.input(self.trigger_pin)

        with self.lock:
            self.trigger_events.append({
                'timestamp': timestamp,
                'state': state
            })

    def pop_all_triggers(self):
        """Get and clear all triggers"""
        with self.lock:
            events = list(self.trigger_events)
            self.trigger_events.clear()
            return events

    def cleanup(self):
        GPIO.cleanup()


class KISSTelemtryMonitor:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, esc_id='ESC1', timer=None,
                 buffer_size=10000, sample_rate_window=100, pole_pair_count=14):
        """
        Initialize KISS telemetry monitor

        Args:
            port: Serial port path
            baudrate: Serial baudrate
            esc_id: ESC identifier string
            timer: Shared HighPrecisionTimer instance
            buffer_size: Maximum number of telemetry samples to keep in memory
            sample_rate_window: Number of samples to use for rate calculation
        """
        self.port = port
        self.baudrate = baudrate
        self.esc_id = esc_id
        self.timer = timer if timer else Timer()
        self.serial_conn = None
        self.buffer = bytearray()
        self.synced = False
        self.valid_packets = 0
        self.invalid_packets = 0
        self.sample_times = deque(maxlen=sample_rate_window)
        self.telemetry_data = deque(maxlen=buffer_size)
        self.pole_pair_count = pole_pair_count
        self.lock = threading.Lock()
        self.running = False

    def find_sync(self):
        """Find packet synchronization"""
        while len(self.buffer) >= 20:
            if len(self.buffer) >= 10:
                packet = self.buffer[:10]
                calculated_crc = crc8_kiss(packet[:9])

                if calculated_crc == packet[9]:
                    if len(self.buffer) >= 20:
                        next_packet = self.buffer[10:20]
                        next_crc = crc8_kiss(next_packet[:9])

                        if next_crc == next_packet[9]:
                            self.synced = True
                            return True

            self.buffer.pop(0)

        return False

    def parse_packet(self, packet):
        """Parse a 10-byte KISS packet
        Reference: http://ultraesc.de/downloads/KISS_telemetry_protocol.pdf"""
        calculated_crc = crc8_kiss(packet[:9])
        if calculated_crc != packet[9]:
            return None

        timestamp = self.timer.get_time()

        temp = packet[0]
        voltage = struct.unpack('>H', packet[1:3])[0] * 0.01
        current = struct.unpack('>H', packet[3:5])[0] * 0.01
        consumption = struct.unpack('>H', packet[5:7])[0]
        rpm = struct.unpack('>H', packet[7:9])[0] * 100 * 2 / self.pole_pair_count

        return {
            'timestamp': timestamp,
            'temperature': temp,
            'voltage': voltage,
            'current': current,
            'consumption': consumption,
            'rpm': rpm
        }

    def get_sample_rate(self):
        time_diff = self.sample_times[-1] - self.sample_times[0]
        return (len(self.sample_times) - 1) / time_diff if time_diff > 0 else 0.0

    def monitor_thread(self):
        """
        Main monitoring thread that continuously reads telemetry data from serial port

        This thread runs in the background and:
        1. Establishes serial connection to the ESC
        2. Uses select() to efficiently wait for incoming data with timeout
        3. Reads all available bytes when data arrives
        4. Processes the buffer to extract complete telemetry packets
        5. Closes the connection when stopped
        """
        # Establish serial connection
        self.serial_conn = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=None)

        self.running = True
        while self.running:
            # Wait for data with 0.1s timeout to allow checking self.running periodically
            readable, _, _ = select.select([self.serial_conn], [], [], 0.1)

            if readable:
                # Data is available - read all waiting bytes at once for efficiency
                bytes_available = self.serial_conn.in_waiting
                if bytes_available > 0:
                    data = self.serial_conn.read(bytes_available)
                    self.buffer.extend(data)
                    # Process buffer to extract complete packets
                    self.process_buffer()

        # Clean up serial connection when thread stops
        if self.serial_conn:
            self.serial_conn.close()

    def process_buffer(self):
        """Process all complete packets in buffer"""
        if not self.synced:
            self.find_sync()
            if not self.synced:
                return

        while self.synced and len(self.buffer) >= 10:
            packet = self.buffer[:10]

            telemetry = self.parse_packet(packet)
            if telemetry:
                self.valid_packets += 1
                self.sample_times.append(telemetry['timestamp'])
                self.buffer = self.buffer[10:]

                with self.lock:
                    self.telemetry_data.append(telemetry)
            else:
                self.invalid_packets += 1
                self.synced = False
                break

    def pop_all_data(self):
        """Get and clear all telemetry data"""
        with self.lock:
            data = list(self.telemetry_data)
            self.telemetry_data.clear()
            return data

    def get_latest_sample(self):
        """Get the most recent telemetry sample"""
        with self.lock:
            return self.telemetry_data[-1] if self.telemetry_data else None

    def stop(self):
        """Stop monitoring thread"""
        self.running = False


class TelemetryHDF5Logger:
    """
    HDF5 logger that continuously appends data to disk
    """

    def __init__(self, foldername=None, filename=None, flush_interval=1.0, enable_trigger=True):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'telemetry_data_{timestamp}.h5'

        self.filename = os.path.join(foldername, filename)
        self.flush_interval = flush_interval
        self.enable_trigger = enable_trigger
        self.file = h5py.File(self.filename, 'w')

        # Store metadata
        self.file.attrs['created'] = datetime.now().isoformat()
        self.file.attrs['description'] = 'ESC telemetry and trigger data for synchronization with external data'
        self.file.attrs['trigger_enabled'] = enable_trigger

        # Create groups
        self.timing_group = self.file.create_group('timing')
        self.esc_group = self.file.create_group('esc_telemetry')

        # Initialize trigger datasets only if enabled
        if enable_trigger:
            self.trigger_group = self.file.create_group('trigger')
            self.trigger_timestamps = self.trigger_group.create_dataset(
                'timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            )
            self.trigger_states = self.trigger_group.create_dataset(
                'state', shape=(0,), maxshape=(None,), dtype=np.int8, compression='gzip'
            )
        else:
            self.trigger_group = None
            self.trigger_timestamps = None
            self.trigger_states = None

        # ESC dataset references
        self.esc_datasets = {}

        # Logging thread
        self.running = False
        self.log_thread = None

        # Statistics
        self.total_trigger_events = 0
        self.total_esc_samples = {}

    def initialize_esc(self, esc_id):
        """Initialize datasets for an ESC"""
        if esc_id in self.esc_datasets:
            return

        esc_subgroup = self.esc_group.create_group(esc_id)

        # Create extendable datasets
        datasets = {
            'timestamp': esc_subgroup.create_dataset(
                'timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'temperature': esc_subgroup.create_dataset(
                'temperature', shape=(0,), maxshape=(None,), dtype=np.int8, compression='gzip'
            ),
            'voltage': esc_subgroup.create_dataset(
                'voltage', shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip'
            ),
            'current': esc_subgroup.create_dataset(
                'current', shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip'
            ),
            'consumption': esc_subgroup.create_dataset(
                'consumption', shape=(0,), maxshape=(None,), dtype=np.uint16, compression='gzip'
            ),
            'rpm': esc_subgroup.create_dataset(
                'rpm', shape=(0,), maxshape=(None,), dtype=np.uint32, compression='gzip'
            )
        }

        self.esc_datasets[esc_id] = {
            'group': esc_subgroup,
            'datasets': datasets
        }
        self.total_esc_samples[esc_id] = 0

    def append_trigger_data(self, trigger_events):
        """Append trigger events to HDF5"""
        if not self.enable_trigger or not trigger_events:
            return

        timestamps = np.array([t['timestamp'] for t in trigger_events], dtype=np.float64)
        states = np.array([t['state'] for t in trigger_events], dtype=np.int8)

        # Resize and append
        old_size = self.trigger_timestamps.shape[0]
        new_size = old_size + len(timestamps)

        self.trigger_timestamps.resize((new_size,))
        self.trigger_states.resize((new_size,))

        self.trigger_timestamps[old_size:new_size] = timestamps
        self.trigger_states[old_size:new_size] = states

        self.total_trigger_events += len(trigger_events)

    def append_esc_data(self, esc_id, data):
        """Append ESC telemetry data to HDF5"""
        if not data:
            return

        if esc_id not in self.esc_datasets:
            self.initialize_esc(esc_id)

        datasets = self.esc_datasets[esc_id]['datasets']

        # Convert to numpy arrays
        timestamps = np.array([d['timestamp'] for d in data], dtype=np.float64)
        temperatures = np.array([d['temperature'] for d in data], dtype=np.int8)
        voltages = np.array([d['voltage'] for d in data], dtype=np.float32)
        currents = np.array([d['current'] for d in data], dtype=np.float32)
        consumptions = np.array([d['consumption'] for d in data], dtype=np.uint16)
        rpms = np.array([d['rpm'] for d in data], dtype=np.uint32)

        # Resize and append each dataset
        old_size = datasets['timestamp'].shape[0]
        new_size = old_size + len(timestamps)

        for key, arr in [
            ('timestamp', timestamps),
            ('temperature', temperatures),
            ('voltage', voltages),
            ('current', currents),
            ('consumption', consumptions),
            ('rpm', rpms)
        ]:
            datasets[key].resize((new_size,))
            datasets[key][old_size:new_size] = arr

        self.total_esc_samples[esc_id] += len(data)

    def flush_data(self, escs, trigger=None):
        """Flush buffered data from all sources"""
        # Flush trigger data if enabled
        if self.enable_trigger and trigger is not None:
            self.append_trigger_data(trigger.pop_all_triggers())

        # Flush ESC data
        for esc in escs:
            self.append_esc_data(esc.esc_id, esc.pop_all_data())

        # Write to disk
        self.file.flush()

    def logging_thread_func(self, escs, trigger):
        """Background thread that periodically flushes data"""
        while self.running:
            time.sleep(self.flush_interval)
            self.flush_data(escs, trigger)

    def start_logging(self, escs, trigger, timer):
        """Start background logging thread"""
        # Save timing reference
        self.timing_group.attrs['start_time_wall'] = timer.start_time_wall
        self.timing_group.attrs['start_time_perf'] = timer.start_time

        self.running = True
        self.log_thread = threading.Thread(target=self.logging_thread_func, args=(escs, trigger))
        self.log_thread.daemon = True
        self.log_thread.start()

    def stop_logging(self, escs, trigger=None):
        """Stop logging and do final flush"""
        self.running = False

        if self.log_thread:
            self.log_thread.join(timeout=2.0)

        # Final flush
        self.flush_data(escs, trigger)

        # Update metadata
        for esc in escs:
            if esc.esc_id in self.esc_datasets:
                group = self.esc_datasets[esc.esc_id]['group']
                group.attrs['port'] = esc.port
                group.attrs['total_samples'] = self.total_esc_samples[esc.esc_id]
                group.attrs['valid_packets'] = esc.valid_packets
                group.attrs['invalid_packets'] = esc.invalid_packets

                # Calculate sample rate
                timestamps = self.esc_datasets[esc.esc_id]['datasets']['timestamp'][:]
                if len(timestamps) > 1:
                    immediate_rates = 1.0 / np.diff(timestamps)
                    group.attrs['avg_sample_rate'] = np.mean(immediate_rates)
                    group.attrs['sample_rate_std'] = np.std(immediate_rates)

        if self.enable_trigger and self.trigger_group is not None:
            self.trigger_group.attrs['total_events'] = self.total_trigger_events

    def close(self):
        """Close HDF5 file"""
        self.file.close()


class MultiESCMonitor:
    """
    Main class for monitoring multiple ESCs with optional trigger and logging
    """

    def __init__(self,
                 esc_ports=None,
                 baudrate=115200,
                 enable_trigger=True,
                 trigger_pin=17,
                 trigger_type='rising',
                 enable_logging=True,
                 log_folder=None,
                 log_filename=None,
                 flush_interval=1.0,
                 buffer_size=10000,
                 sample_rate_window=100,
                 trigger_buffer_size=10000):
        """
        Initialize multi-ESC monitor

        Args:
            esc_ports: List of ports
            baudrate: Serial baudrate for all ESCs
            enable_trigger: Enable trigger signal monitoring
            trigger_pin: GPIO pin for trigger
            trigger_type: 'rising', 'falling', or 'both'
            enable_logging: Enable data logging to HDF5
            log_filename: Output filename (None for auto-generated)
            flush_interval: How often to flush data to disk (seconds)
            buffer_size: Maximum number of telemetry samples per ESC to keep in memory
            sample_rate_window: Number of samples to use for rate calculation
            trigger_buffer_size: Maximum number of trigger events to keep in memory
        """
        # Shared timer
        self.timer = Timer()

        # Default ESC ports if not provided
        if esc_ports is None:
            esc_ports = ['/dev/ttyAMA0', '/dev/ttyAMA2', '/dev/ttyAMA3', '/dev/ttyAMA4']

        # Initialize ESC monitors
        self.escs = [KISSTelemtryMonitor(port, baudrate, f'ESC{i + 1}', self.timer, buffer_size, sample_rate_window)
                     for i, port in enumerate(esc_ports)]

        # Initialize trigger if enabled
        self.enable_trigger = enable_trigger
        self.trigger = TriggerMonitor(trigger_pin, self.timer, trigger_type, trigger_buffer_size) if self.enable_trigger else None

        # Initialize logger if enabled
        self.enable_logging = enable_logging
        self.logger = TelemetryHDF5Logger(foldername=log_folder, filename=log_filename, flush_interval=flush_interval,
                                          enable_trigger=self.enable_trigger)\
                      if self.enable_logging else None

        self.threads = []
        self.running = False

    def start(self):
        """Start all monitoring threads"""
        print("\n" + "=" * 70)
        print("MULTI-ESC TELEMETRY MONITOR")
        print("=" * 70)
        print(f"ESCs: {len(self.escs)}")
        print(f"Trigger: {self.enable_trigger}")
        print(f"Logging: {self.enable_logging}")
        print("=" * 70 + "\n")

        # Start ESC monitoring threads
        for esc in self.escs:
            thread = threading.Thread(target=esc.monitor_thread)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        # Start logging thread if enabled
        if self.enable_logging and self.logger:
            self.logger.start_logging(self.escs, self.trigger, self.timer)

        self.running = True
        print("Monitoring started")

    def stop(self):
        """Stop all threads and cleanup"""
        if not self.running:
            return

        print("\nStopping...")

        # Stop ESC threads
        for esc in self.escs:
            esc.stop()

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=1.0)

        # Stop logging if enabled
        if self.enable_logging and self.logger:
            self.logger.stop_logging(self.escs, self.trigger)
            self.logger.close()

        # Cleanup trigger
        if self.trigger:
            self.trigger.cleanup()

        self.running = False

    def get_latest_sample(self):
        return [esc.get_latest_sample() for esc in self.escs]

    def print_latest_telemetry(self):
        """Print latest telemetry data from all ESCs"""
        samples = self.get_latest_sample()
        output_parts = []
        for i, data in enumerate(samples):
            if data is not None:
                output_parts.append(
                    f"ESC {i}: {data['rpm']:4.0f} RPM | {data['voltage']:4.2f}V | {data['current']:4.2f}A | {data['temperature']:2.0f}°C"
                )
        output_line = " | ".join(output_parts)
        print(f"\r{output_line}", end='', flush=True)

    def run(self, display_interval=0.1):
        """
        Run monitoring with telemetry display
        Args:
            display_interval: How often to update telemetry display (seconds)
        """
        self.start()

        print("Press Ctrl+C to stop\n")
        time.sleep(0.1)

        try:
            while True:
                self.print_latest_telemetry()
                time.sleep(display_interval)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KISS ESC Telemetry Monitor')
    parser.add_argument('--enable-trigger', default=False, help='Enable trigger signal monitoring')
    parser.add_argument('--trigger-pin', type=int, default=17, help='GPIO pin for trigger signal (default: 17)')
    parser.add_argument('--trigger-type', choices=['rising', 'falling', 'both'],
                        default='rising', help='Trigger edge type (default: rising)')
    parser.add_argument('--data-logging', default=True, help='Enable data logging')
    parser.add_argument('--flush-interval', type=float, default=1.0, help='Data flush interval in seconds (default: 1.0)')
    parser.add_argument('--output_folder', type=str, default=None, help='Output foldername (default: ~/TelemetryLogs/)')
    parser.add_argument('--output_file', type=str, default=None, help='Output filename (default: auto-generated)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate (default: 115200)')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Telemetry buffer size per ESC (default: 10000)')
    parser.add_argument('--sample-rate-window', type=int, default=100, help='Sample rate calculation window (default: 100)')
    parser.add_argument('--trigger-buffer-size', type=int, default=10000, help='Trigger buffer size (default: 10000)')

    args = parser.parse_args()
    output_folder = f'/home/{getpass.getuser()}/TelemetryLogs/' if args.output_folder is None else args.output_folder
    # Create monitor
    monitor = MultiESCMonitor(
        enable_trigger=args.enable_trigger,
        trigger_pin=args.trigger_pin,
        trigger_type=args.trigger_type,
        enable_logging=args.data_logging,
        log_folder=output_folder,
        log_filename=args.output_file,
        flush_interval=args.flush_interval,
        baudrate=args.baudrate,
        buffer_size=args.buffer_size,
        sample_rate_window=args.sample_rate_window,
        trigger_buffer_size=args.trigger_buffer_size
    )

    # Run monitoring
    monitor.run()


if __name__ == "__main__":
    main()