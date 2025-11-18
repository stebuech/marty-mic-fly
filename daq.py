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


#TODO: Setting CPU affinity is only needed when MultiESCControler is used in parallel, due to sounddevice/pigpio interferences. Then
# MultiESCControler must live in different python instance!!
os.sched_setaffinity(0, {1, 2, 3}) # Pin this process to cores 1-3 (avoid core 0 where pigpiod runs)


class Timer:
    """High precision timer for synchronization"""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.start_time_wall = time.time()

    def get_time(self):
        """Get high-precision timestamp in seconds"""
        return time.perf_counter() - self.start_time


class GPIOTrigger:
    """Monitor trigger signal with hardware interrupts"""

    def __init__(self, trigger_pin=17, timer=None, trigger_type='rising', buffer_size=10000, mode='signal', trigger_func=None,
                 trigger_func_args=()):
        #TODO: This whole class might not work here, because of interference between sounddevice and pigpio package!!!
        import pigpio
        self.trigger_pin = trigger_pin
        self.timer = timer if timer else Timer()
        self.trigger_events = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.enabled = True
        self.mode = mode
        self.pi = pigpio.pi()

        self.pi.set_mode(self.trigger_pin, pigpio.INPUT)
        self.pi.set_pull_up_down(self.trigger_pin, pigpio.PUD_DOWN)

        edge = dict(rising=pigpio.RISING_EDGE, falling=pigpio.FALLING_EDGE).get(trigger_type, pigpio.EITHER_EDGE)

        if mode == 'signal':
            self.pi.callback(self.trigger_pin, edge, self.buffer_callback)
        elif trigger_func is not None:
            self.pi.callback(self.trigger_pin, edge, self.make_event_callback(trigger_func, trigger_func_args))

    def buffer_callback(self, gpio, level, tick):
        """Called when trigger event occurs"""
        timestamp = self.timer.get_time()

        with self.lock:
            self.trigger_events.append({
                'timestamp': timestamp,
                'state': level
            })

    def make_event_callback(self, func, func_args=()):
        def event_callback(gpio, level, tick):
            func(*func_args)
        return event_callback

    def pop_all_triggers(self):
        """Get and clear all triggers"""
        with self.lock:
            events = list(self.trigger_events)
            self.trigger_events.clear()
            return events

    def cleanup(self):
        if hasattr(self, 'pi') and self.pi.connected:
            self.pi.stop()


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


class ESCTelemtry:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, esc_id='ESC1', timer=None,
                 buffer_size=10000, sample_rate_window_len=100, pole_pair_count=14):
        """
        Initialize ESC Telemetry stream, using KISS telemetry protocol http://ultraesc.de/downloads/KISS_telemetry_protocol.pdf

        Args:
            port: Serial port path
            baudrate: Serial baudrate
            esc_id: ESC identifier string
            timer: Shared Timer instance
            buffer_size: Maximum number of telemetry samples to keep in memory
            sample_rate_window_len: Number of samples to use for rate calculation
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
        self.sample_times = deque(maxlen=sample_rate_window_len)
        self.telemetry_data = deque(maxlen=buffer_size)
        self.pole_pair_count = pole_pair_count
        self.lock = threading.Lock()
        self.running = False

    def find_sync(self):
        """Find packet synchronization. Last byte in packet equals checksum of previous 9 bytes."""
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
            #TODO: poll/request data from esc may lead to higher data/sample rates

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


class MicArray:
    """Monitor microphone array from USB device (MCHStreamer Kit from miniDSP)"""

    def __init__(self, timer=None, buffer_size=1000, channels=16, sample_rate=48000, blocksize=1024, dtype='float32',
                 enable_resync=False, resync_interval=100, convergence_rate=0.001, drift_threshold=0.00025,
                 drift_statistic='median'):
        """
        Initialize microphone array monitor for MCHStreamer device

        Args:
            timer: Shared Timer instance for synchronization
            buffer_size: Maximum number of blocks to keep in memory
            channels: Number of microphone channels (default: 16)
            sample_rate: Sample rate in Hz (default: 48000)
            blocksize: Blocksize for callback (default: 1024)
            dtype: Data type (default: 'float32')
            enable_resync: Enable periodic re-synchronization with hardware timestamps (default: True)
            resync_interval: Number of blocks between re-sync checks (default: 100)
            convergence_rate: Rate of smooth correction convergence (default: 0.001 = 0.1% per resync)
            drift_statistic: Statistic to use for drift filtering: 'mean' or 'median' (default: 'mean')
        """
        self.timer = timer if timer else Timer()
        self.channels = channels
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.dtype = dtype

        # Buffer for data blocks
        self.data = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        self.stream = None
        self.device_idx = None
        self.running = False

        self.stream_time_offset = None
        self.sync_lock = threading.Lock()

        self.total_blocks = 0
        self.overflows = 0

        # Frame counter for consistent timestamp generation
        self.frame_count = 0
        self.reference_time = None

        # Periodic re-synchronization with smooth correction
        self.enable_resync = enable_resync
        self.resync_interval = resync_interval
        self.drift_threshold = drift_threshold
        self.convergence_rate = convergence_rate
        self.drift_statistic = drift_statistic
        self.blocks_since_resync = 0
        self.timing_drift_data = deque(maxlen=round(buffer_size / blocksize + 0.5))
        self.sample_rate_correction = 1.0  # Adaptive correction factor for smooth convergence

        # Adaptive window for drift filtering (grows over time)
        self.min_drift_window = 5  # Start with 5 measurements (~10 seconds)
        self.max_drift_window = 30  # Grow to 30 measurements (~60 seconds)
        self.drift_window_size = self.min_drift_window
        self.resync_count = 0  # Track how many resyncs have occurred

        # Buffer for drift filtering
        self.drift_history = deque(maxlen=self.max_drift_window)

    def find_mchstreamer(self):
        """Find MCHStreamer device index"""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if ('MCHStreamer' in device['name'] or 'USB Audio' in device['name']) and device['max_input_channels'] >= self.channels:
                return idx
        return None

    def callback(self, indata, frames, time_info, status):
        """Callback function called by sounddevice for each block.

        Dumps block into buffer.

        Uses frame-counter based timestamping for consistent timestamps. This eliminates timing jitter from hardware clock quantization
        but trades off absolute synchronization accuracy with external events.

        Optional periodic re-synchronization compares frame-based time with hardware timestamps
        to detect and gradually correct for long-term clock drift.

        Generates per-sample timestamps based on sample rate and frame count.
        """
        if status:
            self.overflows += 1

        # Initialize reference time on first callback
        if self.reference_time is None:
            self.reference_time = self.timer.get_time()

        # Calculate current frame index for this block
        current_frame = self.frame_count

        # Generate per-sample timestamps based on frame count with adaptive correction.
        # Use current correction factor for THIS block's timestamps to ensure timestamps consistency while gradually correcting drift
        sample_indices = np.arange(current_frame, current_frame + frames)
        corrected_sample_rate = self.sample_rate * self.sample_rate_correction
        per_sample_timestamps = self.reference_time + sample_indices / corrected_sample_rate

        # Calculate expected time of first sample in frame based on current frame-counter state
        expected_time_elapsed = self.frame_count / corrected_sample_rate
        frame_based_time = self.reference_time + expected_time_elapsed

        self.frame_count += frames

        # Set adc_based_time = frame_based_time on first callback
        if self.stream_time_offset is None:
            self.stream_time_offset = frame_based_time - time_info.inputBufferAdcTime
        adc_based_time = time_info.inputBufferAdcTime + self.stream_time_offset
        drift = frame_based_time - adc_based_time

        # Add current drift to buffer
        self.drift_history.append(drift)

        # Calculate filtered drift using running statistic (mean or median) from buffer
        # This filters out quantization noise while preserving real drift signal

        # Use last N measurements based on current window size (or all if less than N exist)
        recent_drifts = list(self.drift_history)[-self.drift_window_size:]
        if len(recent_drifts) > self.drift_threshold:
            drift_mean = np.median(recent_drifts) if self.drift_statistic == 'median' else np.mean(recent_drifts)
            drift_std = np.std(recent_drifts)
        else:
            drift_mean, drift_std = 0, 0


        # Get hardware timestamp estimate (with quantization)
        hardware_time = self.timer.get_time()

        # Store complete drift data for logging (can be cleared without affecting filtering)
        self.timing_drift_data.append({
            'frame_count': self.frame_count,
            'drift': drift,
            'drift_mean': drift_mean,
            'drift_std': drift_std,
            'hw_timestamp': hardware_time,
            'adc_timestamp': adc_based_time,
            'mic_timestamp': per_sample_timestamps[0],
            'correction_factor': self.sample_rate_correction
        })


        #TODO/Note: resync might not work correctly due to occasional large jumps in drift (mean/median) value. This makes the "base" level
        # drift away when resync is enabled, which would have been stable around 0 when disabled. Additionaly, base level seems stabel
        # around 0 zero without resync anyway, so keep disabled for now!

        # Periodic re-synchronization to correct for clock drift (smooth correction).
        # Happens AFTER timestamp generation to avoid discontinuities
        if self.enable_resync and self.blocks_since_resync >= self.resync_interval:
            self.blocks_since_resync = 0
            self.resync_count += 1

            # Grow adaptive window size (every 10 resyncs, increase by 1 until max)
            if self.resync_count % 10 == 0 and self.drift_window_size < self.max_drift_window:
                self.drift_window_size += 1

            # Apply smooth correction by adjusting sample rate correction factor
            # Use FILTERED drift_mean instead of raw drift to avoid reacting to quantization noise
            if abs(drift_mean) > self.drift_threshold:
                # Calculate drift rate using filtered drift (drift per second)
                drift_rate = drift_mean / (expected_time_elapsed if expected_time_elapsed > 0 else 0)

                # Store old correction factor before changing it
                old_correction = self.sample_rate_correction

                # Adjust sample rate correction factor gradually
                # Positive drift_rate = running too fast, need to decrease sample rate
                # Negative drift_rate = running too slow, need to increase sample rate
                self.sample_rate_correction *= (1.0 - drift_rate * self.convergence_rate)

                # Clamp correction factor to reasonable bounds (0.99 to 1.01 = ±1%)
                self.sample_rate_correction = np.clip(self.sample_rate_correction, 0.99, 1.01)

                # CRITICAL: Adjust reference_time to maintain continuity at the correction point
                # When we change the correction factor, we need to ensure that the timestamp
                # at the current frame_count remains the same under both old and new corrections
                # Old: t = reference_time + frame_count / (sample_rate * old_correction)
                # New: t = new_reference_time + frame_count / (sample_rate * new_correction)
                # Setting them equal and solving for new_reference_time:
                # new_reference_time = reference_time + frame_count / (sample_rate * old_correction)
                #                                     - frame_count / (sample_rate * new_correction)
                time_to_current_frame_old = self.frame_count / (self.sample_rate * old_correction)
                time_to_current_frame_new = self.frame_count / (self.sample_rate * self.sample_rate_correction)
                self.reference_time = self.reference_time + time_to_current_frame_old - time_to_current_frame_new

        self.blocks_since_resync += 1

        block = {
            'timestamp': per_sample_timestamps,
            'data': indata.copy(),
        }

        with self.lock:
            self.data.append(block)
            self.total_blocks += 1

    def start(self):
        """Start stream"""

        self.device_idx = self.find_mchstreamer()

        self.stream = sd.InputStream(
            device=self.device_idx,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self.callback
        )

        self.stream.start()
        self.running = True

    def stop(self):
        """Stop stream"""
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def pop_all_data(self):
        """Get and clear all data"""
        with self.lock:
            data_list = list(self.data)
            self.data.clear()
            return data_list

    def get_latest_block(self):
        """Get the most recent block"""
        with self.lock:
            return self.data[-1] if self.data else None

    def pop_all_timing_drift_data(self):
        """Get and clear all timing drift data"""
        with self.lock:
            drift_data = list(self.timing_drift_data)
            self.timing_drift_data.clear()
            return drift_data


class HDF5Logger:
    """
    HDF5 logger that continuously appends data to disk
    """

    def __init__(self, foldername=None, filename=None, flush_interval=1.0, mic_array_data=False, trigger_data=False, esc_tel_data=True):
        if foldername is None:
            foldername = ''
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'telemetry_data_{timestamp}.h5'
        self.filename = os.path.join(foldername, filename)
        self.flush_interval = flush_interval
        self.mic_array_data = mic_array_data
        self.esc_tel_data = esc_tel_data
        self.trigger_data = trigger_data
        self.file = h5py.File(self.filename, 'w')

        self.file.attrs['created'] = datetime.now().isoformat()
        description = ['Mic Array Data'] * mic_array_data + ['ESC Telemetry Data'] * esc_tel_data + ['Trigger Channel Data'] * trigger_data
        self.file.attrs['description'] = ', '.join(description)
        self.file.attrs['trigger_enabled'] = trigger_data
        self.file.attrs['mic_array_enabled'] = mic_array_data
        self.file.attrs['esc_enabled'] = esc_tel_data

        self.timing_group = self.file.create_group('timing')

        if mic_array_data:
            self.mic_array_group = self.file.create_group('mic_array')
            self.mic_array_datasets = {}
            self.mic_array_initialized = False
            self.timing_drift_datasets = {}
            self.timing_drift_initialized = False
        else:
            self.mic_array_group = None
            self.mic_array_datasets = {}
            self.mic_array_initialized = False
            self.timing_drift_datasets = {}
            self.timing_drift_initialized = False

        if esc_tel_data:
            self.esc_group = self.file.create_group('esc_telemetry')
            self.esc_datasets = {}
        else:
            self.esc_group = None
            self.esc_datasets = {}

        if trigger_data:
            self.trigger_group = self.file.create_group('trigger')
            self.trigger_datasets = {}
            self.trigger_initialized = False
        else:
            self.trigger_group = None
            self.trigger_datasets = {}
            self.trigger_initialized = False

        self.running = False
        self.log_thread = None

        self.total_trigger_events = 0
        self.total_esc_samples = {}
        self.total_mic_array_blocks = 0

    def initialize_trigger(self):
        """Initialize trigger datasets"""
        if self.trigger_initialized or not self.trigger_data:
            return

        self.trigger_datasets = {
            'timestamp': self.trigger_group.create_dataset(
                'timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'state': self.trigger_group.create_dataset(
                'state', shape=(0,), maxshape=(None,), dtype=np.int8, compression='gzip'
            )
        }

        self.trigger_initialized = True

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

    def initialize_mic_array(self, mic_array_channels, sample_rate, blocksize, dtype):
        """Initialize audio datasets with per-sample timestamps"""
        if self.mic_array_initialized or not self.mic_array_data:
            return

        # Create datasets for audio data
        # Store per-sample timestamps alongside audio samples
        self.mic_array_datasets = {
            'timestamp': self.mic_array_group.create_dataset(
                'timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'audio_data': self.mic_array_group.create_dataset(
                'audio_data', shape=(0, mic_array_channels), maxshape=(None, mic_array_channels),
                dtype=dtype, compression='gzip'
            )
        }

        # Store audio metadata
        self.mic_array_group.attrs['channels'] = mic_array_channels
        self.mic_array_group.attrs['sample_rate'] = sample_rate
        self.mic_array_group.attrs['blocksize'] = blocksize
        self.mic_array_group.attrs['dtype'] = str(dtype)

        self.mic_array_initialized = True

    def initialize_timing_drift(self):
        """Initialize timing drift datasets for mic array synchronization diagnostics"""
        if self.timing_drift_initialized or not self.mic_array_data:
            return

        # Create timing_drift subgroup under mic_array
        timing_drift_group = self.mic_array_group.create_group('timing_drift')

        self.timing_drift_datasets = {
            'frame_count': timing_drift_group.create_dataset(
                'frame_count', shape=(0,), maxshape=(None,), dtype=np.int64, compression='gzip'
            ),
            'drift': timing_drift_group.create_dataset(
                'drift', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'drift_mean': timing_drift_group.create_dataset(
                'drift_mean', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'drift_std': timing_drift_group.create_dataset(
                'drift_std', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'hw_timestamp': timing_drift_group.create_dataset(
                'hw_timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'adc_timestamp': timing_drift_group.create_dataset(
                'adc_timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'mic_timestamp': timing_drift_group.create_dataset(
                'mic_timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            ),
            'correction_factor': timing_drift_group.create_dataset(
                'correction_factor', shape=(0,), maxshape=(None,), dtype=np.float64, compression='gzip'
            )
        }

        timing_drift_group.attrs['description'] = 'Clock drift diagnostics for mic array synchronization'

        self.timing_drift_initialized = True

    def append_timing_drift_data(self, timing_drift_data):
        """Append timing drift data to HDF5"""
        if not self.mic_array_data or not timing_drift_data:
            return

        if not self.timing_drift_initialized:
            self.initialize_timing_drift()

        # Convert to numpy arrays
        frame_counts = np.array([d['frame_count'] for d in timing_drift_data], dtype=np.int64)
        drifts = np.array([d['drift'] for d in timing_drift_data], dtype=np.float64)
        drift_means = np.array([d['drift_mean'] for d in timing_drift_data], dtype=np.float64)
        drift_stds = np.array([d['drift_std'] for d in timing_drift_data], dtype=np.float64)
        hw_timestamps = np.array([d['hw_timestamp'] for d in timing_drift_data], dtype=np.float64)
        adc_timestamps = np.array([d['adc_timestamp'] for d in timing_drift_data], dtype=np.float64)
        mic_timestamps = np.array([d['mic_timestamp'] for d in timing_drift_data], dtype=np.float64)
        correction_factors = np.array([d['correction_factor'] for d in timing_drift_data], dtype=np.float64)

        # Resize and append each dataset
        old_size = self.timing_drift_datasets['frame_count'].shape[0]
        new_size = old_size + len(frame_counts)

        for key, arr in [
            ('frame_count', frame_counts),
            ('drift', drifts),
            ('drift_mean', drift_means),
            ('drift_std', drift_stds),
            ('hw_timestamp', hw_timestamps),
            ('adc_timestamp', adc_timestamps),
            ('mic_timestamp', mic_timestamps),
            ('correction_factor', correction_factors)
        ]:
            self.timing_drift_datasets[key].resize((new_size,))
            self.timing_drift_datasets[key][old_size:new_size] = arr

    def append_mic_array_data(self, mic_array_blocks):
        """Append audio data blocks with per-sample timestamps to HDF5"""
        if not self.mic_array_data or not mic_array_blocks:
            return

        if not self.mic_array_initialized:
            # Initialize on first data
            first_block = mic_array_blocks[0]
            block_len, channels = first_block['data'].shape
            self.initialize_mic_array(channels, 48000, block_len, first_block['data'].dtype)

        # Concatenate all audio blocks and their per-sample timestamps
        all_mic_array_data = np.concatenate([block['data'] for block in mic_array_blocks], axis=0)
        all_sample_timestamps = np.concatenate([block['timestamp'] for block in mic_array_blocks], axis=0)

        # Append per-sample timestamps
        old_ts_size = self.mic_array_datasets['timestamp'].shape[0]
        new_ts_size = old_ts_size + len(all_sample_timestamps)
        self.mic_array_datasets['timestamp'].resize((new_ts_size,))
        self.mic_array_datasets['timestamp'][old_ts_size:new_ts_size] = all_sample_timestamps

        # Append audio data
        old_size = self.mic_array_datasets['audio_data'].shape[0]
        new_size = old_size + all_mic_array_data.shape[0]
        self.mic_array_datasets['audio_data'].resize((new_size, all_mic_array_data.shape[1]))
        self.mic_array_datasets['audio_data'][old_size:new_size, :] = all_mic_array_data

        self.total_mic_array_blocks += len(mic_array_blocks)

    def append_trigger_data(self, trigger_events):
        """Append trigger events to HDF5"""
        if not self.trigger_data or not trigger_events:
            return

        if not self.trigger_initialized:
            self.initialize_trigger()

        timestamps = np.array([t['timestamp'] for t in trigger_events], dtype=np.float64)
        states = np.array([t['state'] for t in trigger_events], dtype=np.int8)

        # Resize and append
        old_size = self.trigger_datasets['timestamp'].shape[0]
        new_size = old_size + len(timestamps)

        self.trigger_datasets['timestamp'].resize((new_size,))
        self.trigger_datasets['state'].resize((new_size,))

        self.trigger_datasets['timestamp'][old_size:new_size] = timestamps
        self.trigger_datasets['state'][old_size:new_size] = states

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

    def flush_data(self, escs, trigger=None, mic_array=None):
        """Flush buffered data from all sources"""
        # Flush trigger data if enabled
        if self.trigger_data and trigger is not None:
            self.append_trigger_data(trigger.pop_all_triggers())

        # Flush ESC data
        for esc in escs:
            self.append_esc_data(esc.esc_id, esc.pop_all_data())

        # Flush mic_array data if enabled
        if self.mic_array_data and mic_array is not None:
            self.append_mic_array_data(mic_array.pop_all_data())
            self.append_timing_drift_data(mic_array.pop_all_timing_drift_data())

        # Write to disk
        self.file.flush()

    def logging_thread_func(self, escs, trigger, mic_array):
        """Background thread that periodically flushes data"""
        while self.running:
            time.sleep(self.flush_interval)
            self.flush_data(escs, trigger, mic_array)

    def start_logging(self, escs, trigger, mic_array, timer):
        """Start background logging thread"""
        # Save timing reference
        self.timing_group.attrs['start_time_wall'] = timer.start_time_wall
        self.timing_group.attrs['start_time_perf'] = timer.start_time

        self.running = True
        self.log_thread = threading.Thread(target=self.logging_thread_func, args=(escs, trigger, mic_array))
        self.log_thread.daemon = True
        self.log_thread.start()

    def stop_logging(self, escs, trigger=None, mic_array=None):
        """Stop logging and do final flush"""
        self.running = False

        if self.log_thread:
            self.log_thread.join(timeout=2.0)

        # Final flush
        self.flush_data(escs, trigger, mic_array)

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

        if self.trigger_data and self.trigger_group is not None:
            self.trigger_group.attrs['total_events'] = self.total_trigger_events

        if self.mic_array_data and mic_array is not None and self.mic_array_group is not None:
            self.mic_array_group.attrs['total_blocks'] = self.total_mic_array_blocks
            self.mic_array_group.attrs['total_overflows'] = mic_array.overflows
            self.mic_array_group.attrs['device_name'] = str(sd.query_devices(mic_array.device_idx)['name']) if mic_array.device_idx is not None else 'Unknown'

            # Add timing drift metadata if available
            if self.timing_drift_initialized and 'timing_drift' in self.mic_array_group:
                timing_drift_group = self.mic_array_group['timing_drift']
                timing_drift_group.attrs['enable_resync'] = mic_array.enable_resync
                timing_drift_group.attrs['resync_interval'] = mic_array.resync_interval
                timing_drift_group.attrs['convergence_rate'] = mic_array.convergence_rate
                timing_drift_group.attrs['drift_threshold'] = mic_array.drift_threshold

    def close(self):
        """Close HDF5 file"""
        self.file.close()


class DAQ:
    """
    Main class for data acquisition of audio, ESC telemetry, and trigger signals.
    """

    def __init__(self,
                 # Audio
                 enable_mic_array=True,
                 mic_array_channels=16,
                 mic_array_sample_rate=48000,
                 mic_array_blocksize=1024,
                 mic_array_buffer_size=10000,
                 # ESC telemetry
                 enable_telemetry=False,
                 esc_ports=None,
                 baudrate=115200,
                 esc_buffer_size=10000,
                 sample_rate_window=100,
                 # Trigger
                 enable_trigger=False,
                 trigger_pin=17,
                 trigger_type='rising',
                 trigger_buffer_size=10000,
                 # Data logging
                 enable_logging=False,
                 log_folder=None,
                 log_filename=None,
                 flush_interval=1.0):
        """
        Initialize data monitor with optional audio, ESC telemetry, and trigger monitoring

        Args:
            enable_mic_array: Enable audio monitoring (MCHStreamer)
            mic_array_channels: Number of audio channels (default: 16)
            mic_array_sample_rate: Audio sample rate in Hz (default: 48000)
            mic_array_blocksize: Audio blocksize (default: 1024)
            mic_array_buffer_size: Maximum number of audio blocks to keep in memory
            enable_telemetry: Enable ESC telemetry monitoring
            esc_ports: List of ESC serial ports (None for default)
            baudrate: Serial baudrate for all ESCs
            esc_buffer_size: Maximum number of telemetry samples per ESC to keep in memory
            sample_rate_window: Number of samples to use for rate calculation
            enable_trigger: Enable trigger signal monitoring
            trigger_pin: GPIO pin for trigger
            trigger_type: 'rising', 'falling', or 'both'
            trigger_buffer_size: Maximum number of trigger events to keep in memory
            enable_logging: Enable data logging to HDF5
            log_folder: Output folder (None for default)
            log_filename: Output filename (None for auto-generated)
            flush_interval: How often to flush data to disk (seconds)
        """
        # Shared timer for all data sources
        self.timer = Timer()

        # MicArray monitoring
        self.enable_mic_array = enable_mic_array
        self.mic_array = MicArray(self.timer, mic_array_buffer_size, mic_array_channels,
                                  mic_array_sample_rate, mic_array_blocksize) if self.enable_mic_array else None

        # ESC telemetry monitoring
        self.enable_esc = enable_telemetry
        self.escs = []
        if self.enable_esc:
            if esc_ports is None:
                esc_ports = ['/dev/ttyAMA0', '/dev/ttyAMA2', '/dev/ttyAMA3', '/dev/ttyAMA4']
            self.escs = [ESCTelemtry(port, baudrate, f'ESC{i + 1}', self.timer,
                                     esc_buffer_size, sample_rate_window)
                         for i, port in enumerate(esc_ports)]

        # Trigger monitoring
        self.enable_trigger = enable_trigger
        self.trigger = GPIOTrigger(trigger_pin, self.timer, trigger_type, trigger_buffer_size) if self.enable_trigger else None

        # Data logging
        self.enable_logging = enable_logging
        self.logger = HDF5Logger(foldername=log_folder, filename=log_filename, flush_interval=flush_interval,
                                 mic_array_data=self.enable_mic_array, trigger_data=self.enable_trigger,
                                 esc_tel_data=self.enable_esc) \
            if self.enable_logging else None

        self.threads = []
        self.running = False

    def start(self):
        """Start all monitoring threads"""
        print("\n" + "=" * 70)
        print("DATA MONITOR")
        print("=" * 70)
        print(f"Audio: {self.enable_mic_array}")
        print(f"ESCs: {len(self.escs) if self.enable_esc else 0}")
        print(f"Trigger: {self.enable_trigger}")
        print(f"Logging: {self.enable_logging}")
        print("=" * 70 + "\n")

        # Start audio if enabled
        if self.enable_mic_array and self.mic_array:
            try:
                self.mic_array.start()
                print(f"Audio monitoring started (device: {sd.query_devices(self.mic_array.device_idx)['name']})")
            except RuntimeError as e:
                print(f"Warning: Audio monitoring failed to start: {e}")
                self.enable_mic_array = False
                self.mic_array = None

        # Start ESC monitoring threads
        if self.enable_esc:
            for esc in self.escs:
                thread = threading.Thread(target=esc.monitor_thread)
                thread.daemon = True
                thread.start()
                self.threads.append(thread)

        # Start logging thread if enabled
        if self.enable_logging and self.logger:
            self.logger.start_logging(self.escs, self.trigger, self.mic_array, self.timer)

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

        # Stop audio if enabled
        if self.mic_array:
            self.mic_array.stop()

        # Stop logging if enabled
        if self.enable_logging and self.logger:
            self.logger.stop_logging(self.escs, self.trigger, self.mic_array)
            self.logger.close()

        # Cleanup trigger
        if self.trigger:
            self.trigger.cleanup()

        self.running = False

    def get_esc_latest_sample(self):
        return [esc.get_latest_sample() for esc in self.escs]

    def run(self):
        """
        Run monitoring with telemetry display
        Args:
            display_interval: How often to update telemetry display (seconds)
        """
        self.start()

        print("Press Ctrl+C to stop\n")

        try:
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data Monitor/Recorder for 16ch MicArray, ESC Telemetry and Trigger')
    parser.add_argument('--enable-trigger', default=False, help='Enable trigger signal monitoring')
    parser.add_argument('--trigger-pin', type=int, default=17, help='GPIO pin for trigger signal (default: 17)')
    parser.add_argument('--trigger-type', choices=['rising', 'falling', 'both'],
                        default='rising', help='Trigger edge type (default: rising)')
    parser.add_argument('--enable-mic-array', action='store_true', default=True, help='Enable microphone array monitoring (MCHStreamer)')
    parser.add_argument('--mic-array-channels', type=int, default=16, help='Number of microphone channels (default: 16)')
    parser.add_argument('--mic-array-sample-rate', type=int, default=48000, help='Microphone array sample rate in Hz (default: 48000)')
    parser.add_argument('--mic-array-blocksize', type=int, default=1024, help='Microphone array blocksize (default: 1024)')
    parser.add_argument('--mic-array-buffer-size', type=int, default=100000, help='Microphone array buffer size in blocks (default: 100000)')
    parser.add_argument('--data-logging', default=True, help='Enable data logging')
    parser.add_argument('--flush-interval', type=float, default=1.0, help='Data flush interval in seconds (default: 1.0)')
    parser.add_argument('--output_folder', type=str, default=None, help='Output foldername (default: ~/MMFDataLogs/)')
    parser.add_argument('--output_file', type=str, default=None, help='Output filename (default: auto-generated)')

    parser.add_argument('--enable_telemetry', type=bool, default=True, help='Enable telemetry signal monitoring')

    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate (default: 115200)')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Telemetry buffer size per ESC (default: 10000)')
    parser.add_argument('--sample-rate-window', type=int, default=100, help='Sample rate calculation window (default: 100)')
    parser.add_argument('--trigger-buffer-size', type=int, default=10000, help='Trigger buffer size (default: 10000)')

    args = parser.parse_args()
    output_folder = f'/home/{getpass.getuser()}/MMFDataLogs/' if args.output_folder is None else args.output_folder

    daq = DAQ(
        enable_mic_array=args.enable_mic_array,
        mic_array_channels=args.mic_array_channels,
        mic_array_sample_rate=args.mic_array_sample_rate,
        mic_array_blocksize=args.mic_array_blocksize,
        mic_array_buffer_size=args.mic_array_buffer_size,
        enable_telemetry=args.enable_telemetry,
        baudrate=args.baudrate,
        esc_buffer_size=args.buffer_size,
        sample_rate_window=args.sample_rate_window,
        enable_trigger=args.enable_trigger,
        trigger_pin=args.trigger_pin,
        trigger_type=args.trigger_type,
        trigger_buffer_size=args.trigger_buffer_size,
        enable_logging=args.data_logging,
        log_folder=output_folder,
        log_filename=args.output_file,
        flush_interval=args.flush_interval
    )

    # Run it
    daq.run()


if __name__ == "__main__":
    main()