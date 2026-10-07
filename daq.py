"""Onboard data acquisition: 16-ch mic array (MCHStreamer), ESC telemetry,
rotor tacho / PDM clock events from the Pico tacho board, and trigger/gate
signals from the flight controller. Everything is streamed into one HDF5 file
per recording.

Hardware (Rev. G3 shield on the Pi 4):
  - Pico 2 tacho board on /dev/ttyAMA5 (921600 Bd): PDM_CLK divider events,
    TACHO1-6 edges and ESC telemetry TEL1-6, all timestamped in Pico ticks
    (see pico_link.py, firmware/tacho_pico).
  - Gate/trigger from the Pixhawk on GPIO17 (RC PWM, HIGH = record).
  - MCHStreamer PDM16 via USB (sounddevice / PortAudio).
  - Legacy: ESC telemetry directly on Pi UARTs (--telemetry-source uart), only
    for older boards without the Pico.

Time bases in the file:
  - `timestamp` columns: seconds since recording start in time.perf_counter
    (CLOCK_MONOTONIC), start in /timing attrs.
  - `pico_ticks` / `ticks`: Pico TIMER1 ticks (tick_hz in /pico attrs). The
    mapping Pico ticks -> audio samples is made offline from /pico/clk (one
    event every clk_div PDM clock edges = clk_div/64 samples); /pico/ping maps
    Pico ticks to Pi time.
  - mic blocks: PortAudio times (adc_time, current_time) plus the Pi time of
    each callback, for estimating the constant offset of sample 0.
"""

import os
import serial
import struct
import time
import threading
import select
from collections import deque
from datetime import datetime

import numpy as np
import h5py

from pico_link import PicoLink, crc8_kiss  # noqa: F401  (crc8_kiss re-exported for old imports)
from hw_config import DEFAULT_MOTORS, LEGACY_ESC_PORTS, PICO_BAUDRATE, PICO_PORT, TRIGGER_PIN

DEFAULT_ESC_PORTS = LEGACY_ESC_PORTS
DEFAULT_POLE_PAIRS = 12


class Timer:
    """Shared time reference: time.perf_counter() (CLOCK_MONOTONIC on Linux)."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.start_time_wall = time.time()

    def get_time(self):
        """Seconds since start"""
        return time.perf_counter() - self.start_time


class SignalMonitor:
    """GPIO signal monitor (pigpio): RC-PWM gate decoding, output mirroring,
    heartbeat generation and edge logging.

    Logged events carry the pigpio tick (µs, 32 bit, set by pigpiod when the
    edge happened) in addition to the Python-side timestamp, which lags by the
    callback delivery latency (ms range). Use tick_reference() pairs to map
    ticks to Timer seconds.
    """

    def __init__(self, listen_pin=None, output_pin=None, log_pin=None,
                 pwm_low_threshold=1400, pwm_high_threshold=1600,
                 heartbeat=False, heartbeat_interval=10.0, heartbeat_start_duration=0.1,
                 heartbeat_increment=0.1,
                 timer=None, buffer_size=10000, log_gate=False):
        """
        Args:
            listen_pin: GPIO pin to listen to RC PWM signal (optional)
            output_pin: GPIO pin to mirror state or generate heartbeat (optional)
            log_pin: GPIO pin to log state changes (optional)
            pwm_low_threshold: PWM pulse width threshold for LOW state (microseconds)
            pwm_high_threshold: PWM pulse width threshold for HIGH state (microseconds)
            heartbeat: Enable heartbeat mode when listen_pin is None
            heartbeat_interval: Time between heartbeat pulses (seconds)
            heartbeat_start_duration: Initial pulse duration (seconds)
            heartbeat_increment: Increment for each heartbeat pulse (seconds)
            timer: Shared Timer instance for logging
            buffer_size: Maximum number of log events to keep in memory
            log_gate: Also log changes of the decoded PWM state (listen_pin) as events
        """
        import pigpio

        self.listen_pin = listen_pin
        self.output_pin = output_pin
        self.log_pin = log_pin
        self.pwm_low_threshold = pwm_low_threshold
        self.pwm_high_threshold = pwm_high_threshold
        self.heartbeat = heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_start_duration = heartbeat_start_duration
        self.heartbeat_increment = heartbeat_increment
        self.timer = timer if timer else Timer()
        self.log_gate = log_gate

        self.current_state = 0
        self.last_rising_tick = None
        self.last_pulse_width = 0
        self.lock = threading.Lock()

        self.log_events = deque(maxlen=buffer_size)
        self.gate_events = deque(maxlen=buffer_size)
        self.last_log_state = 0
        self.dropped_events = 0

        self.current_heartbeat_duration = heartbeat_start_duration
        self.heartbeat_thread = None
        self.running = False
        self.measurement_started = False

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon")

        if self.listen_pin is not None:
            self.pi.set_mode(self.listen_pin, pigpio.INPUT)
            self.cb = self.pi.callback(self.listen_pin, pigpio.EITHER_EDGE, self._pwm_callback)
        else:
            self.cb = None

        if self.output_pin is not None:
            self.pi.set_mode(self.output_pin, pigpio.OUTPUT)
            self.pi.write(self.output_pin, 0)

        if self.log_pin is not None:
            self.pi.set_mode(self.log_pin, pigpio.INPUT)
            self.pi.set_pull_up_down(self.log_pin, pigpio.PUD_DOWN)
            self.last_log_state = self.pi.read(self.log_pin)
            self.log_cb = self.pi.callback(self.log_pin, pigpio.EITHER_EDGE, self._log_callback)
        else:
            self.log_cb = None

        if self.heartbeat and self.listen_pin is None and self.output_pin is not None:
            self.running = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

    def _append_event(self, dq, timestamp, state, tick):
        if len(dq) == dq.maxlen:
            self.dropped_events += 1
        dq.append({'timestamp': timestamp, 'state': state, 'tick': tick})

    def _pwm_callback(self, gpio, level, tick):
        """Measure RC PWM pulse width and update the gate state (with hysteresis)"""
        if level == 1:
            self.last_rising_tick = tick
        elif level == 0 and self.last_rising_tick is not None:
            import pigpio
            pulse_width = pigpio.tickDiff(self.last_rising_tick, tick)

            with self.lock:
                self.last_pulse_width = pulse_width
                old_state = self.current_state
                if pulse_width < self.pwm_low_threshold:
                    self.current_state = 0
                elif pulse_width > self.pwm_high_threshold:
                    self.current_state = 1

                if self.current_state != old_state:
                    if self.output_pin is not None:
                        self.pi.write(self.output_pin, self.current_state)
                    if self.log_gate:
                        # tick of the falling edge that completed the deciding pulse
                        self._append_event(self.gate_events, self.timer.get_time(),
                                           1 if self.current_state else -1, tick)

            self.last_rising_tick = None

    def _log_callback(self, gpio, level, tick):
        """Log 1 for rising and -1 for falling edges on log_pin (level 2 = watchdog, ignored)"""
        timestamp = self.timer.get_time()
        with self.lock:
            if level == 1 and self.last_log_state == 0:
                edge_value = 1
            elif level == 0 and self.last_log_state == 1:
                edge_value = -1
            else:
                return
            self.last_log_state = level
            self._append_event(self.log_events, timestamp, edge_value, tick)

    def _heartbeat_loop(self):
        """Heartbeat pulses with rising edges exactly heartbeat_interval apart,
        pulse length growing by heartbeat_increment (pulse identification)."""
        while self.running and not self.measurement_started:
            time.sleep(0.1)
        if not self.running:
            return

        next_pulse_time = time.time()
        while self.running:
            wait_time = next_pulse_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            if self.output_pin is not None and self.pi.connected:
                self.pi.write(self.output_pin, 1)
            time.sleep(self.current_heartbeat_duration)
            if self.output_pin is not None and self.pi.connected:
                self.pi.write(self.output_pin, 0)

            next_pulse_time += self.heartbeat_interval
            self.current_heartbeat_duration += self.heartbeat_increment
            if time.time() + self.current_heartbeat_duration >= next_pulse_time:
                self.current_heartbeat_duration = self.heartbeat_start_duration

    def tick_reference(self):
        """(pigpio tick, Timer seconds) sampled together, for tick -> time mapping"""
        t0 = self.timer.get_time()
        tick = self.pi.get_current_tick()
        t1 = self.timer.get_time()
        return tick, 0.5 * (t0 + t1), t1 - t0

    def get_state(self):
        with self.lock:
            return self.current_state

    def get_pulse_width(self):
        with self.lock:
            return self.last_pulse_width

    def pop_all_log_events(self):
        with self.lock:
            events = list(self.log_events)
            self.log_events.clear()
            return events

    def pop_all_gate_events(self):
        with self.lock:
            events = list(self.gate_events)
            self.gate_events.clear()
            return events

    def keep_last_gate_event(self):
        """Drop all but the newest gate event (the edge that started a recording)"""
        with self.lock:
            last = self.gate_events[-1] if self.gate_events else None
            self.gate_events.clear()
            if last is not None:
                self.gate_events.append(last)

    def start_heartbeat(self):
        self.measurement_started = True

    def stop_heartbeat(self):
        self.measurement_started = False
        self.current_heartbeat_duration = self.heartbeat_start_duration

    def close(self):
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
        if getattr(self, 'cb', None):
            self.cb.cancel()
        if getattr(self, 'log_cb', None):
            self.log_cb.cancel()
        if hasattr(self, 'pi') and self.pi.connected:
            self.pi.stop()


class ESCTelemtry:
    """KISS telemetry read directly from a Pi UART (legacy, pre-Rev. G3 boards).
    With the Pico board use PicoLink(...).escs, which has the same interface."""

    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, esc_id='ESC1', timer=None,
                 buffer_size=10000, sample_rate_window_len=100, pole_count=2 * DEFAULT_POLE_PAIRS):
        """
        Args:
            port: Serial port path
            baudrate: Serial baudrate
            esc_id: ESC identifier string
            timer: Shared Timer instance
            buffer_size: Maximum number of telemetry samples to keep in memory
            sample_rate_window_len: Number of samples to use for rate calculation
            pole_count: Number of motor magnet poles (rpm = eRPM * 2 / pole_count)
        """
        self.port = port
        self.baudrate = baudrate
        self.esc_id = esc_id
        self.timer = timer if timer else Timer()
        self.serial_conn = None
        self.buffer = bytearray()
        self.buffer_time = None
        self.synced = False
        self.valid_packets = 0
        self.invalid_packets = 0
        self.sample_times = deque(maxlen=sample_rate_window_len)
        self.telemetry_data = deque(maxlen=buffer_size)
        self.pole_count = pole_count
        self.pole_pairs = pole_count / 2
        self.lock = threading.Lock()
        self.running = False
        self.error = None

    def find_sync(self):
        """Find packet alignment: two consecutive 10-byte packets with valid CRC."""
        while len(self.buffer) >= 20:
            if (crc8_kiss(self.buffer[:9]) == self.buffer[9]
                    and crc8_kiss(self.buffer[10:19]) == self.buffer[19]):
                self.synced = True
                return True
            self.buffer.pop(0)
        return False

    def parse_packet(self, packet, timestamp=None):
        """Parse a 10-byte KISS packet, None if the CRC does not match"""
        if crc8_kiss(packet[:9]) != packet[9]:
            return None
        voltage, current, consumption, erpm_100 = struct.unpack('>HHHH', packet[1:9])
        erpm = erpm_100 * 100
        return {
            'timestamp': self.timer.get_time() if timestamp is None else timestamp,
            'temperature': packet[0],
            'voltage': voltage * 0.01,
            'current': current * 0.01,
            'consumption': consumption,
            'erpm': erpm,
            'rpm': erpm * 2 / self.pole_count,
        }

    def get_sample_rate(self):
        if len(self.sample_times) < 2:
            return 0.0
        time_diff = self.sample_times[-1] - self.sample_times[0]
        return (len(self.sample_times) - 1) / time_diff if time_diff > 0 else 0.0

    def monitor_thread(self):
        """Read the serial port until stop(); timestamps are taken when bytes arrive."""
        try:
            self.serial_conn = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=None)
            self.running = True
            while self.running:
                readable, _, _ = select.select([self.serial_conn], [], [], 0.1)
                if readable:
                    bytes_available = self.serial_conn.in_waiting
                    if bytes_available > 0:
                        t = self.timer.get_time()
                        self.buffer.extend(self.serial_conn.read(bytes_available))
                        self.process_buffer(t)
        except Exception as e:  # keep the reason, the thread would die silently otherwise
            self.error = repr(e)
            print(f"{self.esc_id} ({self.port}): telemetry thread stopped: {e}")
        finally:
            self.running = False
            if self.serial_conn:
                self.serial_conn.close()

    def process_buffer(self, timestamp=None):
        if not self.synced:
            self.find_sync()
            if not self.synced:
                return
        while self.synced and len(self.buffer) >= 10:
            telemetry = self.parse_packet(self.buffer[:10], timestamp)
            if telemetry:
                self.valid_packets += 1
                self.sample_times.append(telemetry['timestamp'])
                del self.buffer[:10]
                with self.lock:
                    self.telemetry_data.append(telemetry)
            else:
                self.invalid_packets += 1
                self.synced = False
                break

    def pop_all_data(self):
        with self.lock:
            data = list(self.telemetry_data)
            self.telemetry_data.clear()
            return data

    def get_latest_sample(self):
        with self.lock:
            return self.telemetry_data[-1] if self.telemetry_data else None

    def stop(self):
        self.running = False


class MicArray:
    """MCHStreamer (miniDSP) 16-ch PDM microphone array via sounddevice.

    The callback only copies the block and a small record; all I/O happens in
    the logger thread. Per block it records the running frame index, PortAudio's
    inputBufferAdcTime/currentTime, the Pi time of the callback and the
    under/overflow flags. A lost block (PortAudio overflow or this buffer
    running full) is visible as a flag and as a jump in first_frame.
    """

    STATUS_INPUT_UNDERFLOW = 1
    STATUS_INPUT_OVERFLOW = 2
    STATUS_BUFFER_FULL = 4

    def __init__(self, timer=None, buffer_size=1000, channels=16, sample_rate=48000, blocksize=1024,
                 dtype='float32', latency='high', device=None):
        """
        Args:
            timer: Shared Timer instance
            buffer_size: Maximum number of blocks waiting for the logger
            channels: Number of microphone channels
            sample_rate: Sample rate in Hz
            blocksize: Frames per callback
            dtype: Sample format
            latency: PortAudio input latency ('low', 'high' or seconds); 'high' gives
                the callback more slack against GIL contention from the writer
            device: Device index or name substring; None searches for the MCHStreamer
        """
        self.timer = timer if timer else Timer()
        self.channels = channels
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.dtype = dtype
        self.latency = latency
        self.device = device

        self.data = deque()
        self.max_blocks = buffer_size
        self.lock = threading.Lock()

        self.stream = None
        self.device_idx = None
        self.device_name = None
        self.running = False

        self.frame_count = 0
        self.total_blocks = 0
        self.overflows = 0
        self.dropped_blocks = 0

    def find_mchstreamer(self):
        import sounddevice as sd
        for idx, device in enumerate(sd.query_devices()):
            name = device['name']
            if self.device is not None and not isinstance(self.device, int):
                match = str(self.device) in name
            else:
                match = 'MCHStreamer' in name or 'USB Audio' in name
            if match and device['max_input_channels'] >= self.channels:
                return idx
        return None

    def callback(self, indata, frames, time_info, status):
        t_cb = self.timer.get_time()
        flags = ((self.STATUS_INPUT_UNDERFLOW if status.input_underflow else 0)
                 | (self.STATUS_INPUT_OVERFLOW if status.input_overflow else 0))
        if flags:
            self.overflows += 1
        record = (self.frame_count, frames, time_info.inputBufferAdcTime, time_info.currentTime, t_cb, flags)
        self.frame_count += frames
        with self.lock:
            if len(self.data) >= self.max_blocks:
                self.dropped_blocks += 1
                return
            self.data.append((record, indata.copy()))
            self.total_blocks += 1

    def start(self):
        import sounddevice as sd
        self.device_idx = self.device if isinstance(self.device, int) else self.find_mchstreamer()
        if self.device_idx is None:
            raise RuntimeError('MCHStreamer not found')
        self.device_name = sd.query_devices(self.device_idx)['name']
        self.frame_count = 0
        self.stream = sd.InputStream(
            device=self.device_idx,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            blocksize=self.blocksize,
            latency=self.latency,
            callback=self.callback,
        )
        self.stream.start()
        self.running = True

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def pop_all_data(self):
        """List of (record, block) tuples, see callback()"""
        with self.lock:
            data_list = list(self.data)
            self.data.clear()
            return data_list

    def get_latest_block(self):
        with self.lock:
            return self.data[-1][1] if self.data else None


class HDF5Logger:
    """Streams all sources into one HDF5 file.

    Only the logger thread touches the file while logging; the final flush in
    stop_logging() runs after that thread has been joined.

    Layout:
      /timing                       attrs start_time_wall, start_time_perf, clock
      /mic_array/audio_data         (N, channels), attrs sample_rate, channels, ...
      /mic_array/blocks/*           first_frame, frames, adc_time, current_time,
                                    callback_time, status (per callback)
      /esc_telemetry/ESCn/*         timestamp, temperature, voltage, current,
                                    consumption, erpm, rpm [, pico_ticks]
      /trigger/*                    timestamp, state, tick   (log_pin edges)
      /gate/*                       timestamp, state, tick   (decoded RC gate changes)
      /pico/clk/*                   seq, ticks
      /pico/tacho/*                 seq, ticks, state, changed (bit n = TACHO n+1)
      /pico/tel_chunks/*            ch, seq, ticks, flags, n  (raw ESC byte chunks, diagnostics)
      /pico/ping/*                  id, pi_send, pi_recv, pico_ticks
      /pico/status/*                Pico health counters, 1 Hz
    """

    def __init__(self, foldername=None, filename=None, flush_interval=1.0, mic_array_data=False,
                 trigger_data=False, esc_tel_data=True, pico_data=False, audio_compression=None):
        foldername = foldername or ''
        if foldername:
            os.makedirs(foldername, exist_ok=True)
        if filename is None:
            filename = f'telemetry_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        path = os.path.join(foldername, filename)
        root, ext = os.path.splitext(path)
        k = 1
        while os.path.exists(path):  # never overwrite an earlier recording
            path = f'{root}_{k:03d}{ext}'
            k += 1
        self.filename = path
        self.flush_interval = flush_interval
        self.mic_array_data = mic_array_data
        self.esc_tel_data = esc_tel_data
        self.trigger_data = trigger_data
        self.pico_data = pico_data
        self.audio_compression = audio_compression
        self.file = h5py.File(self.filename, 'w')

        self.file.attrs['created'] = datetime.now().isoformat()
        description = (['Mic Array Data'] * mic_array_data + ['ESC Telemetry Data'] * esc_tel_data
                       + ['Trigger Channel Data'] * trigger_data + ['Pico Tacho/Clock Data'] * pico_data)
        self.file.attrs['description'] = ', '.join(description)
        self.file.attrs['trigger_enabled'] = trigger_data
        self.file.attrs['mic_array_enabled'] = mic_array_data
        self.file.attrs['esc_enabled'] = esc_tel_data
        self.file.attrs['pico_enabled'] = pico_data
        self.file.attrs['format_version'] = 2

        self.timing_group = self.file.create_group('timing')

        self.running = False
        self.stop_event = threading.Event()
        self.log_thread = None
        self.sources = None
        self.write_error = None

        self.total_trigger_events = 0
        self.total_esc_samples = {}
        self.total_mic_array_blocks = 0
        self.mic_array_initialized = False

    # -- generic append

    def _append(self, path, columns, compression='gzip', chunks=True):
        """Append equally long columns to the datasets in group `path`."""
        n = len(next(iter(columns.values())))
        if n == 0:
            return
        group = self.file.require_group(path)
        for name, arr in columns.items():
            arr = np.asarray(arr)
            ds = group.get(name)
            if ds is None:
                ds = group.create_dataset(name, shape=(0,) + arr.shape[1:], maxshape=(None,) + arr.shape[1:],
                                          dtype=arr.dtype, chunks=chunks, compression=compression)
            old = ds.shape[0]
            ds.resize(old + n, axis=0)
            ds[old:old + n] = arr

    @staticmethod
    def _events_to_columns(events):
        return {
            'timestamp': np.array([e['timestamp'] for e in events], dtype=np.float64),
            'state': np.array([e['state'] for e in events], dtype=np.int8),
            'tick': np.array([e.get('tick', 0) for e in events], dtype=np.uint32),
        }

    # -- per source

    def initialize_mic_array(self, channels, sample_rate, blocksize, dtype, device_name=None):
        if self.mic_array_initialized:
            return
        group = self.file.require_group('mic_array')
        group.create_dataset('audio_data', shape=(0, channels), maxshape=(None, channels), dtype=dtype,
                             chunks=(max(blocksize, 4096), channels), compression=self.audio_compression)
        group.attrs['channels'] = channels
        group.attrs['sample_rate'] = sample_rate
        group.attrs['blocksize'] = blocksize
        group.attrs['dtype'] = str(dtype)
        group.attrs['device_name'] = device_name or 'Unknown'
        bg = group.create_group('blocks')
        bg.attrs['description'] = ('Per PortAudio callback: first_frame = index of the first sample of the '
                                   'block in audio_data (gaps = lost blocks), adc_time/current_time = '
                                   'PortAudio stream clock, callback_time = Timer seconds at callback entry, '
                                   'status bits 1 input underflow, 2 input overflow, 4 DAQ buffer full')
        self.mic_array_initialized = True

    def append_mic_array_data(self, items, mic_array=None):
        if not self.mic_array_data or not items:
            return
        if not self.mic_array_initialized:
            block = items[0][1]
            self.initialize_mic_array(block.shape[1], mic_array.sample_rate if mic_array else 48000,
                                      mic_array.blocksize if mic_array else len(block), block.dtype,
                                      mic_array.device_name if mic_array else None)
        records = np.array([it[0] for it in items], dtype=np.float64)
        audio = np.concatenate([it[1] for it in items], axis=0)
        ds = self.file['mic_array/audio_data']
        old = ds.shape[0]
        ds.resize(old + audio.shape[0], axis=0)
        ds[old:] = audio
        self._append('mic_array/blocks', {
            'first_frame': records[:, 0].astype(np.int64),
            'frames': records[:, 1].astype(np.int32),
            'adc_time': records[:, 2],
            'current_time': records[:, 3],
            'callback_time': records[:, 4],
            'status': records[:, 5].astype(np.uint8),
        })
        self.total_mic_array_blocks += len(items)

    def append_trigger_data(self, events, group='trigger'):
        if not events:
            return
        self._append(group, self._events_to_columns(events))
        if group == 'trigger':
            self.total_trigger_events += len(events)

    def append_esc_data(self, esc_id, data):
        if not data:
            return
        cols = {
            'timestamp': np.array([d['timestamp'] for d in data], dtype=np.float64),
            'temperature': np.array([d['temperature'] for d in data], dtype=np.int16),
            'voltage': np.array([d['voltage'] for d in data], dtype=np.float32),
            'current': np.array([d['current'] for d in data], dtype=np.float32),
            'consumption': np.array([d['consumption'] for d in data], dtype=np.uint16),
            'erpm': np.array([d['erpm'] for d in data], dtype=np.uint32),
            'rpm': np.array([d['rpm'] for d in data], dtype=np.float32),
        }
        if 'pico_ticks' in data[0]:
            cols['pico_ticks'] = np.array([d['pico_ticks'] for d in data], dtype=np.uint64)
        self._append(f'esc_telemetry/{esc_id}', cols)
        self.total_esc_samples[esc_id] = self.total_esc_samples.get(esc_id, 0) + len(data)

    def append_pico_data(self, link, t0):
        clk = link.pop_clk()
        if clk:
            a = np.array(clk, dtype=np.uint64)
            self._append('pico/clk', {'seq': a[:, 0].astype(np.uint32), 'ticks': a[:, 1]})
        tacho = link.pop_tacho()
        if tacho:
            a = np.array(tacho, dtype=np.uint64)
            self._append('pico/tacho', {'seq': a[:, 0].astype(np.uint32), 'ticks': a[:, 1],
                                        'state': a[:, 2].astype(np.uint8), 'changed': a[:, 3].astype(np.uint8)})
        tel = link.pop_tel()
        if tel:
            a = np.array(tel, dtype=np.uint64)
            self._append('pico/tel_chunks', {'ch': a[:, 0].astype(np.uint8), 'seq': a[:, 1].astype(np.uint32),
                                             'ticks': a[:, 2], 'flags': a[:, 3].astype(np.uint8),
                                             'n': a[:, 4].astype(np.uint8)})
        pings = link.pop_pings()
        if pings:
            self._append('pico/ping', {
                'id': np.array([p[0] for p in pings], dtype=np.uint32),
                'pi_send': np.array([p[1] - t0 for p in pings], dtype=np.float64),
                'pi_recv': np.array([p[2] - t0 for p in pings], dtype=np.float64),
                'pico_ticks': np.array([p[3] for p in pings], dtype=np.uint64),
            })
        status = link.pop_status()
        if status:
            cols = {'pi_time': np.array([s[0] - t0 for s in status], dtype=np.float64)}
            for key in ('t_now', 't_us'):
                cols[key] = np.array([s[1][key] for s in status], dtype=np.uint64)
            for key in ('clk_events', 'tacho_events', 'link_drops', 'rx_crc_errors', 'tacho_stalls', 'loop_max_us'):
                cols[key] = np.array([s[1][key] for s in status], dtype=np.uint32)
            for key in ('tel_bytes', 'tel_framing', 'tel_overflow'):
                cols[key] = np.array([s[1][key] for s in status], dtype=np.uint32)
            self._append('pico/status', cols)

    # -- flushing

    def flush_data(self):
        s = self.sources
        if s is None:
            return
        if self.trigger_data and s['signal_monitor'] is not None:
            self.append_trigger_data(s['signal_monitor'].pop_all_log_events())
        if s['control_monitor'] is not None:
            self.append_trigger_data(s['control_monitor'].pop_all_gate_events(), group='gate')
        if self.esc_tel_data:
            for esc in s['escs']:
                self.append_esc_data(esc.esc_id, esc.pop_all_data())
        if self.mic_array_data and s['mic_array'] is not None:
            self.append_mic_array_data(s['mic_array'].pop_all_data(), s['mic_array'])
        if self.pico_data and s['pico'] is not None:
            self.append_pico_data(s['pico'], s['timer'].start_time)
        self.file.flush()

    def logging_thread_func(self):
        while not self.stop_event.wait(self.flush_interval):
            try:
                self.flush_data()
            except Exception as e:
                self.write_error = repr(e)
                print(f"HDF5 logger error: {e}")
                return

    def start_logging(self, escs, signal_monitor, mic_array, timer, pico=None, control_monitor=None):
        """Start background logging thread"""
        self.timing_group.attrs['start_time_wall'] = timer.start_time_wall
        self.timing_group.attrs['start_time_perf'] = timer.start_time
        self.timing_group.attrs['clock'] = 'time.perf_counter (CLOCK_MONOTONIC); timestamps are seconds since start_time_perf'
        self.sources = dict(escs=escs, signal_monitor=signal_monitor, mic_array=mic_array, timer=timer,
                            pico=pico, control_monitor=control_monitor)
        for mon, name in ((signal_monitor, 'trigger'), (control_monitor, 'gate')):
            if mon is not None:
                tick, t, dt = mon.tick_reference()
                self.file.require_group(name).attrs['tick_reference_start'] = (tick, t, dt)
        self.running = True
        self.stop_event.clear()
        self.log_thread = threading.Thread(target=self.logging_thread_func, name='hdf5-logger', daemon=True)
        self.log_thread.start()

    def stop_logging(self, escs=None, signal_monitor=None, mic_array=None):
        """Stop the logger thread, final flush, write summary attributes.
        (Arguments kept for backwards compatibility; the sources from start_logging are used.)"""
        self.running = False
        self.stop_event.set()
        if self.log_thread:
            self.log_thread.join()
        s = self.sources or {}
        for mon, name in ((s.get('signal_monitor'), 'trigger'), (s.get('control_monitor'), 'gate')):
            if mon is not None and name in self.file:
                tick, t, dt = mon.tick_reference()
                self.file[name].attrs['tick_reference_end'] = (tick, t, dt)
        self.flush_data()

        for esc in s.get('escs', []):
            path = f'esc_telemetry/{esc.esc_id}'
            if path not in self.file:
                continue
            group = self.file[path]
            group.attrs['port'] = esc.port
            group.attrs['source'] = 'pico' if hasattr(esc, 'link') else 'uart'
            group.attrs['pole_pairs'] = esc.pole_pairs
            group.attrs['total_samples'] = self.total_esc_samples.get(esc.esc_id, 0)
            group.attrs['valid_packets'] = esc.valid_packets
            group.attrs['invalid_packets'] = esc.invalid_packets
            timestamps = group['timestamp'][:]
            if len(timestamps) > 1:
                rates = 1.0 / np.diff(timestamps)
                rates = rates[np.isfinite(rates)]
                if len(rates):
                    group.attrs['avg_sample_rate'] = np.mean(rates)
                    group.attrs['sample_rate_std'] = np.std(rates)

        if self.trigger_data and 'trigger' in self.file:
            self.file['trigger'].attrs['description'] = 'Edge detection events from log_pin'
            self.file['trigger'].attrs['edge_encoding'] = '1 = rising edge (0->1), -1 = falling edge (1->0)'
            self.file['trigger'].attrs['total_events'] = self.total_trigger_events
            self.file['trigger'].attrs['tick_unit'] = 'pigpio tick, microseconds, uint32 wrapping'
        if 'gate' in self.file:
            self.file['gate'].attrs['description'] = 'Decoded RC PWM gate (control pin): 1 = HIGH/record, -1 = LOW'

        mic = s.get('mic_array')
        if self.mic_array_data and mic is not None and 'mic_array' in self.file:
            g = self.file['mic_array']
            g.attrs['total_blocks'] = self.total_mic_array_blocks
            g.attrs['total_overflows'] = mic.overflows
            g.attrs['dropped_blocks'] = mic.dropped_blocks
            g.attrs['latency'] = str(mic.latency)

        pico = s.get('pico')
        if self.pico_data and pico is not None:
            g = self.file.require_group('pico')
            if pico.info:
                for k, v in pico.info.items():
                    g.attrs[k] = v
            for k, v in pico.counters().items():
                g.attrs[f'host_{k}'] = v
            g.attrs['port'] = pico.port
            g.attrs['description'] = ('Pico tacho board events in Pico ticks (tick_hz). clk: one event every '
                                      'clk_div rising PDM_CLK edges (= clk_div/64 audio samples). tacho: pin state '
                                      'after the edge, changed = XOR to the previous state, bit n = TACHO n+1.')
        if self.write_error:
            self.file.attrs['write_error'] = self.write_error

    def close(self):
        self.file.close()


class DAQ:
    """Data acquisition of audio, ESC telemetry, Pico tacho/clock events and
    trigger signals. Manual mode: start()/stop(). Gate mode (run() with
    trigger_controlled): one file per HIGH phase of the RC switch."""

    def __init__(self,
                 # Audio
                 enable_mic_array=True,
                 mic_array_channels=16,
                 mic_array_sample_rate=48000,
                 mic_array_blocksize=1024,
                 mic_array_buffer_size=2000,
                 mic_array_latency='high',
                 audio_compression=None,
                 # ESC telemetry
                 enable_telemetry=False,
                 telemetry_source='pico',
                 n_motors=DEFAULT_MOTORS,
                 esc_ports=None,
                 baudrate=115200,
                 esc_buffer_size=10000,
                 sample_rate_window=100,
                 pole_pairs=DEFAULT_POLE_PAIRS,
                 # Pico tacho board
                 enable_pico=None,
                 pico_port=PICO_PORT,
                 pico_baudrate=PICO_BAUDRATE,
                 # Signal monitoring
                 enable_signal_monitor=False,
                 listen_pin=None,
                 output_pin=None,
                 log_pin=None,
                 pwm_low_threshold=1400,
                 pwm_high_threshold=1600,
                 heartbeat=False,
                 heartbeat_interval=10.0,
                 heartbeat_start_duration=0.1,
                 heartbeat_increment=0.1,
                 signal_buffer_size=10000,
                 # Control (for start/stop via RC switch)
                 trigger_controlled=False,
                 control_pin=None,
                 # Data logging
                 enable_logging=False,
                 log_folder=None,
                 log_filename=None,
                 flush_interval=1.0):
        """
        Args (new ones; the rest as before):
            telemetry_source: 'pico' (ESC telemetry via the Pico board, Rev. G3) or
                'uart' (directly on the Pi UARTs in esc_ports, older boards)
            n_motors: ESC telemetry channels to record with the pico source (TEL1..n);
                4 = quadcopter, 6 = hexacopter
            pole_pairs: motor pole pairs, rpm = eRPM / pole_pairs
            enable_pico: log Pico tacho/clock events; default: True if telemetry_source == 'pico'
            mic_array_latency: PortAudio latency setting
            audio_compression: HDF5 filter for audio (None, 'lzf', 'gzip'); None keeps
                the writer cheap enough not to starve the audio callback
        """
        self.timer = Timer()

        self.enable_mic_array = enable_mic_array
        self.mic_array = MicArray(self.timer, mic_array_buffer_size, mic_array_channels, mic_array_sample_rate,
                                  mic_array_blocksize, latency=mic_array_latency) if enable_mic_array else None
        self.audio_compression = audio_compression

        if telemetry_source not in ('pico', 'uart'):
            raise ValueError("telemetry_source must be 'pico' or 'uart'")
        self.telemetry_source = telemetry_source
        self.enable_esc = enable_telemetry
        self.enable_pico = (telemetry_source == 'pico') if enable_pico is None else enable_pico
        self.pico = None
        if self.enable_pico or (self.enable_esc and telemetry_source == 'pico'):
            self.pico = PicoLink(pico_port, pico_baudrate, timer=self.timer, pole_pairs=pole_pairs)

        self.escs = []
        if self.enable_esc:
            if telemetry_source == 'pico':
                self.escs = self.pico.escs[:n_motors]
            else:
                esc_ports = esc_ports or DEFAULT_ESC_PORTS
                self.escs = [ESCTelemtry(port, baudrate, f'ESC{i + 1}', self.timer, esc_buffer_size,
                                         sample_rate_window, pole_count=2 * pole_pairs)
                             for i, port in enumerate(esc_ports)]

        self.enable_signal_monitor = enable_signal_monitor
        self.trigger_controlled = trigger_controlled

        # Gate monitor: decodes the RC switch PWM and logs its state changes
        self.control_monitor = SignalMonitor(
            listen_pin=control_pin,
            pwm_low_threshold=pwm_low_threshold,
            pwm_high_threshold=pwm_high_threshold,
            timer=self.timer,
            log_gate=True,
        ) if self.trigger_controlled and control_pin is not None else None

        log_pin = log_pin if self.enable_signal_monitor else None
        self.log_signal_data = self.enable_signal_monitor and log_pin is not None

        self.signal_monitor = SignalMonitor(
            listen_pin=listen_pin,
            output_pin=output_pin,
            log_pin=log_pin,
            pwm_low_threshold=pwm_low_threshold,
            pwm_high_threshold=pwm_high_threshold,
            heartbeat=heartbeat,
            heartbeat_interval=heartbeat_interval,
            heartbeat_start_duration=heartbeat_start_duration,
            heartbeat_increment=heartbeat_increment,
            timer=self.timer,
            buffer_size=signal_buffer_size
        ) if self.enable_signal_monitor else None

        self.enable_logging = enable_logging
        self.log_folder = log_folder
        self.log_filename = log_filename
        self.flush_interval = flush_interval
        self.logger = None

        self.threads = []
        self.running = False
        self.sources_open = False

        self.recording_state = 'idle'
        self.state_lock = threading.Lock()
        self.trigger_monitor_thread = None

    # -- sources that live across recordings

    def _open_sources(self):
        if self.sources_open:
            return
        if self.pico is not None:
            self.pico.start()
            info = self.pico.wait_for_info()
            if info is None:
                print(f"WARNING: no answer from the Pico on {self.pico.port} - tacho/telemetry will be empty")
            else:
                mode = ' SELFTEST BUILD' if info['selftest'] else ''
                print(f"Pico link up: fw {info['fw']}{mode}, tick {info['tick_hz'] / 1e6:.1f} MHz, "
                      f"clk_div {info['clk_div']}")
        self.sources_open = True

    def _close_sources(self):
        if self.pico is not None and self.sources_open:
            self.pico.stop()
        self.sources_open = False

    # -- recording

    def _start_recording(self):
        if self.recording_state == 'recording':
            return

        if self.enable_logging:
            self.logger = HDF5Logger(
                foldername=self.log_folder,
                filename=self.log_filename,
                flush_interval=self.flush_interval,
                mic_array_data=self.enable_mic_array,
                trigger_data=self.log_signal_data,
                esc_tel_data=self.enable_esc,
                pico_data=self.enable_pico and self.pico is not None,
                audio_compression=self.audio_compression,
            )

        # discard everything that arrived before this recording
        if self.pico is not None:
            self.pico.reset_buffers()
        if self.control_monitor is not None:
            self.control_monitor.keep_last_gate_event()

        if self.enable_mic_array and self.mic_array:
            try:
                self.mic_array.start()
                print(f"  MicArray started (device: {self.mic_array.device_name})")
            except Exception as e:
                print(f"  WARNING: MicArray failed to start: {e}")
                self.enable_mic_array = False
                self.mic_array = None
                if self.logger:
                    self.logger.mic_array_data = False

        if self.enable_esc and self.telemetry_source == 'uart':
            for esc in self.escs:
                thread = threading.Thread(target=esc.monitor_thread, daemon=True)
                thread.start()
                self.threads.append(thread)
        if self.enable_esc:
            print(f"  ESC telemetry ({self.telemetry_source}): {len(self.escs)} channels")

        if self.logger:
            self.logger.start_logging(self.escs, self.signal_monitor, self.mic_array, self.timer,
                                      pico=self.pico if self.enable_pico else None,
                                      control_monitor=self.control_monitor)
            print(f"  HDF5 logging started: {self.logger.filename}")

        if self.signal_monitor:
            self.signal_monitor.start_heartbeat()

        self.recording_state = 'recording'
        print("  Recording ACTIVE\n")

    def _stop_recording(self):
        if self.recording_state == 'idle':
            return

        for esc in self.escs:
            esc.stop()
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.threads.clear()

        if self.mic_array:
            self.mic_array.stop()
            print("  MicArray stopped")

        if self.logger:
            self.logger.stop_logging()
            self.logger.close()
            print(f"  HDF5 file saved: {self.logger.filename}")

        if self.signal_monitor:
            self.signal_monitor.stop_heartbeat()

        self.recording_state = 'idle'
        print("  Recording STOPPED\n")

    # -- public API (manual mode, also used by replay_and_record.py)

    def start(self):
        print("\n" + "=" * 70)
        print("DATA MONITOR")
        print("=" * 70)
        print(f"Audio: {self.enable_mic_array}")
        print(f"ESCs: {len(self.escs) if self.enable_esc else 0} ({self.telemetry_source})")
        print(f"Pico tacho board: {self.enable_pico}")
        print(f"Signal Monitor: {self.enable_signal_monitor}")
        print(f"Logging: {self.enable_logging}")
        print("=" * 70 + "\n")
        self._open_sources()
        with self.state_lock:
            self._start_recording()
        self.running = True

    def stop(self):
        if not self.running:
            return
        print("\nStopping...")
        with self.state_lock:
            self._stop_recording()
        self._close_sources()
        if self.signal_monitor:
            self.signal_monitor.close()
        if self.control_monitor:
            self.control_monitor.close()
        self.running = False

    def get_esc_latest_sample(self):
        return [esc.get_latest_sample() for esc in self.escs]

    def status_line(self):
        parts = []
        if self.enable_esc:
            rpm = []
            for esc in self.escs:
                s = esc.get_latest_sample()
                rpm.append(f"{s['rpm']:.0f}" if s else '-')
            parts.append('rpm ' + '/'.join(rpm))
        if self.pico is not None and self.pico.status:
            st = self.pico.status
            parts.append(f"pico clk {st['clk_events']} tacho {st['tacho_events']} drops {st['link_drops']}")
        if self.mic_array is not None and self.mic_array.running:
            parts.append(f"mic blocks {self.mic_array.total_blocks} ovf {self.mic_array.overflows} "
                         f"lost {self.mic_array.dropped_blocks}")
        if self.logger and self.logger.write_error:
            parts.append(f"WRITE ERROR {self.logger.write_error}")
        return ' | '.join(parts)

    def _trigger_monitor_thread_func(self):
        """Poll the decoded RC switch state and open/close recordings (gate mode)"""
        last_state = 0
        while self.running:
            current_state = self.control_monitor.get_state()
            if current_state == 1 and last_state == 0:
                with self.state_lock:
                    if self.recording_state == 'idle':
                        print(f"\n[RC SWITCH] HIGH at {self.timer.get_time():.3f}s - Starting recording...")
                        self._start_recording()
            elif current_state == 0 and last_state == 1:
                with self.state_lock:
                    if self.recording_state == 'recording':
                        print(f"\n[RC SWITCH] LOW at {self.timer.get_time():.3f}s - Stopping recording...")
                        self._stop_recording()
            last_state = current_state
            time.sleep(0.01)

    def run(self, status_interval=5.0):
        if self.trigger_controlled:
            self._run_trigger_controlled(status_interval)
        else:
            self._run_manual(status_interval)

    def _run_manual(self, status_interval):
        self.start()
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                time.sleep(status_interval)
                print(self.status_line())
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _run_trigger_controlled(self, status_interval):
        if not self.control_monitor:
            print("ERROR: Trigger-controlled mode requires a control pin")
            return

        print("\n" + "=" * 70)
        print("RC SWITCH CONTROLLED MODE (Gate Mode)")
        print("=" * 70)
        print(f"MicArray: {self.enable_mic_array}")
        print(f"ESCs: {len(self.escs) if self.enable_esc else 0} ({self.telemetry_source})")
        print(f"Pico tacho board: {self.enable_pico}")
        print(f"Control Pin: {self.control_monitor.listen_pin}")
        print(f"PWM Thresholds: LOW < {self.control_monitor.pwm_low_threshold} us, "
              f"HIGH > {self.control_monitor.pwm_high_threshold} us")
        print(f"Logging: {self.enable_logging}")
        print("=" * 70)
        print("\nWaiting for RC switch signal...  (Ctrl+C to exit)\n")

        self._open_sources()
        self.running = True
        self.trigger_monitor_thread = threading.Thread(target=self._trigger_monitor_thread_func, daemon=True)
        self.trigger_monitor_thread.start()

        try:
            while True:
                time.sleep(status_interval)
                if self.recording_state == 'recording':
                    print(self.status_line())
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            self.running = False
            if self.trigger_monitor_thread:
                self.trigger_monitor_thread.join(timeout=2.0)
            with self.state_lock:
                if self.recording_state == 'recording':
                    print("Stopping active recording...")
                    self._stop_recording()
            self._close_sources()
            if self.signal_monitor:
                self.signal_monitor.close()
            self.control_monitor.close()
            print("Shutdown complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Data recorder for 16ch mic array, ESC telemetry, Pico tacho board and trigger signals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    BOA = argparse.BooleanOptionalAction

    g = parser.add_argument_group('recording control')
    g.add_argument('--trigger-controlled', action=BOA, default=True,
                   help='RC PWM switch on --control-pin starts/stops recordings (HIGH = record)')
    g.add_argument('--control-pin', type=int, default=TRIGGER_PIN, help='GPIO of the RC gate signal (TRIG, J18)')
    g.add_argument('--pwm-low-threshold', type=int, default=1400, help='gate LOW below this pulse width (us)')
    g.add_argument('--pwm-high-threshold', type=int, default=1600, help='gate HIGH above this pulse width (us)')

    g = parser.add_argument_group('microphone array')
    g.add_argument('--mic-array', action=BOA, default=True, help='record the MCHStreamer')
    g.add_argument('--disable-mic-array', dest='mic_array', action='store_false', help=argparse.SUPPRESS)
    g.add_argument('--mic-array-channels', type=int, default=16)
    g.add_argument('--mic-array-sample-rate', type=int, default=48000)
    g.add_argument('--mic-array-blocksize', type=int, default=1024)
    g.add_argument('--mic-array-buffer-size', type=int, default=2000, help='blocks buffered for the writer')
    g.add_argument('--mic-array-latency', default='high', help="PortAudio latency: 'low', 'high' or seconds")
    g.add_argument('--audio-compression', choices=['none', 'lzf', 'gzip'], default='none')

    g = parser.add_argument_group('ESC telemetry and Pico tacho board')
    g.add_argument('--telemetry', action=BOA, default=True, help='record ESC telemetry')
    g.add_argument('--motors', type=int, choices=[4, 6], default=DEFAULT_MOTORS,
                   help='4 = quadcopter (TEL1-4), 6 = hexacopter (TEL1-6); pico source only')
    g.add_argument('--disable_telemetry', dest='telemetry', action='store_false', help=argparse.SUPPRESS)
    g.add_argument('--telemetry-source', choices=['pico', 'uart'], default='pico',
                   help='pico: via the tacho board (Rev. G3); uart: Pi UARTs in --esc-ports (old boards)')
    g.add_argument('--esc-ports', nargs='+', default=DEFAULT_ESC_PORTS, help='only for --telemetry-source uart')
    g.add_argument('--baudrate', type=int, default=115200, help='ESC telemetry baudrate (uart source)')
    g.add_argument('--pole-pairs', type=int, default=DEFAULT_POLE_PAIRS, help='motor pole pairs (rpm = eRPM / pole pairs)')
    g.add_argument('--buffer-size', type=int, default=10000, help='telemetry buffer per ESC (uart source)')
    g.add_argument('--pico', action=BOA, default=None,
                   help='log Pico tacho/clock events (default: on with --telemetry-source pico)')
    g.add_argument('--pico-port', default=PICO_PORT)
    g.add_argument('--pico-baudrate', type=int, default=PICO_BAUDRATE)

    g = parser.add_argument_group('signal monitor (heartbeat / edge logging, pre-Rev. G3 boards)')
    g.add_argument('--signal-monitor', action=BOA, default=False)
    g.add_argument('--listen-pin', type=int, default=None)
    g.add_argument('--output-pin', type=int, default=22)
    g.add_argument('--log-pin', type=int, default=23)
    g.add_argument('--heartbeat', action=BOA, default=False)
    g.add_argument('--heartbeat-interval', type=float, default=10.0)
    g.add_argument('--heartbeat-start-duration', type=float, default=0.1)
    g.add_argument('--heartbeat-increment', type=float, default=0.1)
    g.add_argument('--signal-buffer-size', type=int, default=10000)

    g = parser.add_argument_group('output')
    g.add_argument('--data-logging', action=BOA, default=True)
    g.add_argument('--disable-data-logging', dest='data_logging', action='store_false', help=argparse.SUPPRESS)
    g.add_argument('--flush-interval', type=float, default=1.0, help='seconds between HDF5 flushes')
    g.add_argument('--output-folder', '--output_folder', dest='output_folder', default=None,
                   help='default: ~/MMFDataLogs/')
    g.add_argument('--output-file', '--output_file', dest='output_file', default=None,
                   help='default: telemetry_data_<date>_<time>.h5; existing files are never overwritten')
    g.add_argument('--status-interval', type=float, default=5.0)

    args = parser.parse_args()
    output_folder = os.path.expanduser(args.output_folder or '~/MMFDataLogs/')

    # Keep off core 0, where pigpiod runs (sounddevice/pigpio interference, see
    # notes_and_helpers/CPU_AFFINITY_FIX.md). DShot control (esc_throttle_set.py)
    # must run in a separate Python process.
    try:
        os.sched_setaffinity(0, {1, 2, 3})
    except (AttributeError, OSError):
        pass

    latency = args.mic_array_latency
    try:
        latency = float(latency)
    except ValueError:
        pass

    daq = DAQ(
        enable_mic_array=args.mic_array,
        mic_array_channels=args.mic_array_channels,
        mic_array_sample_rate=args.mic_array_sample_rate,
        mic_array_blocksize=args.mic_array_blocksize,
        mic_array_buffer_size=args.mic_array_buffer_size,
        mic_array_latency=latency,
        audio_compression=None if args.audio_compression == 'none' else args.audio_compression,
        enable_telemetry=args.telemetry,
        telemetry_source=args.telemetry_source,
        n_motors=args.motors,
        esc_ports=args.esc_ports,
        baudrate=args.baudrate,
        esc_buffer_size=args.buffer_size,
        pole_pairs=args.pole_pairs,
        enable_pico=args.pico,
        pico_port=args.pico_port,
        pico_baudrate=args.pico_baudrate,
        enable_signal_monitor=args.signal_monitor,
        listen_pin=args.listen_pin,
        output_pin=args.output_pin,
        log_pin=args.log_pin,
        pwm_low_threshold=args.pwm_low_threshold,
        pwm_high_threshold=args.pwm_high_threshold,
        heartbeat=args.heartbeat,
        heartbeat_interval=args.heartbeat_interval,
        heartbeat_start_duration=args.heartbeat_start_duration,
        heartbeat_increment=args.heartbeat_increment,
        signal_buffer_size=args.signal_buffer_size,
        trigger_controlled=args.trigger_controlled,
        control_pin=args.control_pin,
        enable_logging=args.data_logging,
        log_folder=output_folder,
        log_filename=args.output_file,
        flush_interval=args.flush_interval,
    )
    daq.run(status_interval=args.status_interval)


if __name__ == "__main__":
    main()
