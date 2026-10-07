"""Receiver for the tacho capture board (Pico 2 on the Rev. G3 shield).

The Pico timestamps PDM-clock divider edges, optical tacho edges and ESC
telemetry bytes in one timebase (TIMER1 counting clk_sys, 150 MHz) and streams
them over /dev/ttyAMA5. This module parses that stream (protocol: see
firmware/tacho_pico/src/protocol.h) and buffers the events for the HDF5 logger
in daq.py.

Stand-alone use (bench check, prints rates and jitter):

    python pico_link.py --monitor 10
"""

import struct
import threading
import time
from collections import deque

import numpy as np

PROTO_VERSION = 1
SYNC = b'\xA5\x5A'
FRAME_OVERHEAD = 6
FRAME_MAX_PAYLOAD = 128

PKT_INFO = 0x01
PKT_STATUS = 0x02
PKT_CLK = 0x10
PKT_TACHO = 0x11
PKT_TEL = 0x12
PKT_PONG = 0x13

CMD_PING = 0x80
CMD_GET_INFO = 0x81
CMD_SET_CLKDIV = 0x82
CMD_BOOTSEL = 0x8F
BOOTSEL_MAGIC = 0xB0075E1F

TEL_FLAG_FRAMING_ERR = 0x01
TEL_FLAG_OVERFLOW = 0x02

N_TEL = 6
N_TACHO = 6


def crc16_ccitt(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE, identical to codec.c"""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


# Table-driven variant for the receive path (the bitwise loop above is the
# reference implementation and is used by the tests).
_CRC16_TABLE = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) if _c & 0x8000 else (_c << 1)
    _CRC16_TABLE.append(_c & 0xFFFF)


def _crc16_fast(data, crc=0xFFFF):
    table = _CRC16_TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ table[(crc >> 8) ^ byte]
    return crc


def crc8_kiss(data):
    """KISS telemetry CRC8 (poly 0x07), http://ultraesc.de/downloads/KISS_telemetry_protocol.pdf"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
            crc &= 0xFF
    return crc


def encode_frame(pkt_type, payload=b''):
    if len(payload) > FRAME_MAX_PAYLOAD:
        raise ValueError('payload too long')
    body = bytes([pkt_type, len(payload)]) + payload
    return SYNC + body + struct.pack('<H', _crc16_fast(body))


class FrameParser:
    """Incremental frame parser with resynchronisation on sync bytes + CRC."""

    def __init__(self):
        self.buf = bytearray()
        self.frames_ok = 0
        self.crc_errors = 0
        self.bytes_skipped = 0

    def feed(self, data):
        """Append bytes, return list of (type, payload) for every valid frame."""
        self.buf.extend(data)
        out = []
        buf = self.buf
        pos = 0
        n = len(buf)
        while True:
            start = buf.find(SYNC, pos)
            if start < 0:
                # keep a trailing 0xA5, it may be the first sync byte
                keep = 1 if n and buf[-1] == 0xA5 else 0
                self.bytes_skipped += n - pos - keep
                pos = n - keep
                break
            self.bytes_skipped += start - pos
            if n - start < 4:
                pos = start
                break
            length = buf[start + 3]
            if length > FRAME_MAX_PAYLOAD:
                self.crc_errors += 1
                pos = start + 1
                continue
            end = start + 4 + length + 2
            if end > n:
                pos = start
                break
            body = bytes(buf[start + 2:start + 4 + length])
            crc = buf[end - 2] | (buf[end - 1] << 8)
            if _crc16_fast(body) == crc:
                out.append((body[0], body[2:]))
                self.frames_ok += 1
                pos = end
            else:
                self.crc_errors += 1
                pos = start + 1
        del buf[:pos]
        return out


# --- payload decoders -------------------------------------------------------

_INFO = struct.Struct('<HIIBBIBQ16s')
_STATUS_HEAD = struct.Struct('<QQIIIIII')
_CLK = struct.Struct('<IQ')
_TACHO = struct.Struct('<IQBB')
_TEL_HEAD = struct.Struct('<BBIQB')
_PONG = struct.Struct('<IQ')


def decode_info(p):
    proto, tick_hz, clk_div, n_tacho, n_tel, tel_baud, flags, t_now, fw = _INFO.unpack_from(p)
    return dict(proto=proto, tick_hz=tick_hz, clk_div=clk_div, n_tacho=n_tacho, n_tel=n_tel,
                tel_baud=tel_baud, selftest=bool(flags & 1), t_now=t_now,
                fw=fw.split(b'\0', 1)[0].decode(errors='replace'))


def decode_status(p):
    t_now, t_us, clk_events, tacho_events, link_drops, rx_crc_errors, tacho_stalls, loop_max_us = \
        _STATUS_HEAD.unpack_from(p)
    o = _STATUS_HEAD.size
    tel_bytes = struct.unpack_from('<6I', p, o)
    tel_framing = struct.unpack_from('<6H', p, o + 24)
    tel_overflow = struct.unpack_from('<6H', p, o + 36)
    return dict(t_now=t_now, t_us=t_us, clk_events=clk_events, tacho_events=tacho_events,
                link_drops=link_drops, rx_crc_errors=rx_crc_errors, tacho_stalls=tacho_stalls,
                loop_max_us=loop_max_us, tel_bytes=tel_bytes, tel_framing=tel_framing,
                tel_overflow=tel_overflow)


def decode_tel(p):
    ch, flags, seq, t_first, n = _TEL_HEAD.unpack_from(p)
    data = bytes(p[_TEL_HEAD.size:_TEL_HEAD.size + n])
    return ch, flags, seq, t_first, data


# --- KISS telemetry reassembly ----------------------------------------------

class KissStream:
    """Reassembles 10-byte KISS frames from the Pico's per-channel chunks.

    Chunks are cut by line-idle gaps on the Pico, so a frame normally sits at
    the start of a chunk. Leftover bytes are carried into the next chunk only if
    it follows without a gap (chunk split at the 32-byte cap).
    """

    FRAME_LEN = 10

    def __init__(self, byte_ticks):
        self.byte_ticks = byte_ticks
        self.buf = bytearray()
        self.buf_t0 = 0          # Pico time of buf[0]
        self.last_seq = None
        self.last_end_t = None
        self.valid = 0
        self.invalid_bytes = 0
        self.chunks_lost = 0

    def feed(self, seq, t_first, data):
        """Return list of (t_pico_first_byte, raw 10 bytes) for valid frames."""
        contiguous = (self.last_seq is not None and seq == (self.last_seq + 1) & 0xFFFFFFFF
                      and self.last_end_t is not None
                      and t_first - self.last_end_t < 2 * self.byte_ticks)
        if self.last_seq is not None and seq != (self.last_seq + 1) & 0xFFFFFFFF:
            self.chunks_lost += (seq - self.last_seq - 1) & 0xFFFFFFFF
        if not contiguous:
            self.invalid_bytes += len(self.buf)
            self.buf.clear()
        if not self.buf:
            self.buf_t0 = t_first
        self.buf.extend(data)
        self.last_seq = seq
        self.last_end_t = t_first + (len(data) - 1) * self.byte_ticks

        frames = []
        i = 0
        buf = self.buf
        while len(buf) - i >= self.FRAME_LEN:
            if crc8_kiss(buf[i:i + 9]) == buf[i + 9]:
                frames.append((self.buf_t0 + int(i * self.byte_ticks), bytes(buf[i:i + 10])))
                self.valid += 1
                i += self.FRAME_LEN
            else:
                self.invalid_bytes += 1
                i += 1
        del buf[:i]
        self.buf_t0 += int(i * self.byte_ticks)
        return frames


def parse_kiss(raw):
    """Decode a CRC-checked 10-byte KISS frame into raw integer fields."""
    temp = raw[0]
    voltage, current, consumption, erpm_100 = struct.unpack('>HHHH', raw[1:9])
    return temp, voltage * 0.01, current * 0.01, consumption, erpm_100 * 100


# --- Pi <-> Pico time mapping -----------------------------------------------

class ClockMap:
    """Maps Pico ticks to Pi time (time.perf_counter, i.e. CLOCK_MONOTONIC).

    Fed by PING/PONG round trips. Uses the fastest third of the recent round
    trips for a linear fit, which removes most of the UART/scheduler latency.
    Only used for live display and the Pi-time convenience columns; the raw
    round trips are logged so the mapping can be redone offline.
    """

    def __init__(self, tick_hz=150_000_000, window=120):
        self.tick_hz = tick_hz
        self.samples = deque(maxlen=window)   # (pico_ticks, pi_mid, rtt)
        self.a = None                          # pi_time = a + b * (ticks - t_ref)
        self.b = 1.0 / tick_hz
        self.t_ref = 0
        self.lock = threading.Lock()

    def add(self, pico_ticks, pi_send, pi_recv):
        with self.lock:
            self.samples.append((pico_ticks, 0.5 * (pi_send + pi_recv), pi_recv - pi_send))
            s = np.array(self.samples, dtype=np.float64)
        if len(s) < 3:
            with self.lock:
                self.t_ref, self.a, self.b = int(pico_ticks), float(s[-1, 1]), 1.0 / self.tick_hz
            return
        keep = s[:, 2] <= np.quantile(s[:, 2], 1 / 3)
        if keep.sum() < 2:
            keep[:] = True
        t_ref = int(s[keep, 0][-1])
        x = s[keep, 0] - t_ref
        y = s[keep, 1]
        if np.ptp(x) > 0.5 * self.tick_hz:
            b, a = np.polyfit(x, y, 1)
        else:
            b, a = 1.0 / self.tick_hz, float(np.mean(y - x / self.tick_hz))
        with self.lock:
            self.t_ref, self.a, self.b = t_ref, float(a), float(b)

    def to_pi(self, ticks):
        with self.lock:
            if self.a is None:
                return np.nan
            return self.a + self.b * (np.asarray(ticks, dtype=np.float64) - self.t_ref)

    @property
    def valid(self):
        return self.a is not None


# --- ESC channel facade -------------------------------------------------------

class PicoESCChannel:
    """Per-ESC view on PicoLink with the interface of daq.ESCTelemtry
    (esc_id, port, valid_packets, invalid_packets, pop_all_data,
    get_latest_sample), so loggers and calibration scripts can use either."""

    def __init__(self, link, ch, esc_id, pole_pairs, timer, buffer_size):
        self.link = link
        self.ch = ch
        self.esc_id = esc_id
        self.port = f'{link.port}:TEL{ch + 1}'
        self.pole_pairs = pole_pairs
        self.timer = timer
        self.telemetry_data = deque(maxlen=buffer_size)
        self.dropped = 0
        self.lock = threading.Lock()

    @property
    def valid_packets(self):
        return self.link.kiss[self.ch].valid

    @property
    def invalid_packets(self):
        return self.link.kiss[self.ch].invalid_bytes

    def _add(self, t_pico, raw):
        temp, voltage, current, consumption, erpm = parse_kiss(raw)
        t_pi = self.link.clock.to_pi(t_pico)
        sample = {
            'timestamp': (t_pi - self.timer.start_time) if self.timer is not None else t_pi,
            'pico_ticks': t_pico,
            'temperature': temp,
            'voltage': voltage,
            'current': current,
            'consumption': consumption,
            'erpm': erpm,
            'rpm': erpm / self.pole_pairs,
        }
        with self.lock:
            if len(self.telemetry_data) == self.telemetry_data.maxlen:
                self.dropped += 1
            self.telemetry_data.append(sample)

    def pop_all_data(self):
        with self.lock:
            data = list(self.telemetry_data)
            self.telemetry_data.clear()
            return data

    def get_latest_sample(self):
        with self.lock:
            return self.telemetry_data[-1] if self.telemetry_data else None

    # ESCTelemtry compatibility: the link thread does the work
    def monitor_thread(self):
        pass

    def stop(self):
        pass


# --- link -------------------------------------------------------------------

class PicoLink:
    """Reads the Pico stream in a background thread and buffers all events.

    Buffers (deques of tuples, drained by the logger with pop_*):
      clk:    (seq, ticks)
      tacho:  (seq, ticks, state, changed)
      tel:    (ch, seq, ticks, flags, n_bytes)    raw chunk index, for diagnostics
      status: (pi_time, status dict)
      ping:   (id, pi_send, pi_recv, pico_ticks)
    ESC telemetry goes to the PicoESCChannel objects in self.escs.
    """

    def __init__(self, port='/dev/ttyAMA5', baudrate=921600, timer=None, pole_pairs=12,
                 buffer_size=200000, ping_interval=0.5, n_esc=N_TEL):
        self.port = port
        self.baudrate = baudrate
        self.timer = timer
        self.ping_interval = ping_interval
        self.serial_conn = None
        self.parser = FrameParser()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.ping_thread = None

        self.info = None
        self.info_event = threading.Event()
        self.status = None
        self.clock = ClockMap()
        self.byte_ticks = 10 / 115200 * self.clock.tick_hz
        self.kiss = [KissStream(self.byte_ticks) for _ in range(N_TEL)]
        self.escs = [PicoESCChannel(self, ch, f'ESC{ch + 1}', pole_pairs, timer, buffer_size // 10)
                     for ch in range(n_esc)]

        self.clk = deque(maxlen=buffer_size)
        self.tacho = deque(maxlen=buffer_size)
        self.tel = deque(maxlen=buffer_size)
        self.status_log = deque(maxlen=10000)
        self.pings = deque(maxlen=10000)
        self.overruns = {'clk': 0, 'tacho': 0, 'tel': 0}

        self.last_clk_seq = None
        self.last_tacho_seq = None
        self.clk_seq_gaps = 0
        self.tacho_seq_gaps = 0
        self.bytes_read = 0
        self.serial_errors = 0
        self._ping_id = 0
        self._pending_pings = {}

    # -- lifecycle

    def start(self):
        import serial
        self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.05)
        self.serial_conn.reset_input_buffer()
        self.running = True
        self.thread = threading.Thread(target=self._reader, name='pico-link', daemon=True)
        self.thread.start()
        self.ping_thread = threading.Thread(target=self._pinger, name='pico-ping', daemon=True)
        self.ping_thread.start()
        self.send(CMD_GET_INFO)

    def stop(self):
        self.running = False
        for t in (self.thread, self.ping_thread):
            if t:
                t.join(timeout=1.0)
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None

    def wait_for_info(self, timeout=2.0):
        """Block until INFO was received (sent on request and every 5 s)."""
        if not self.info_event.wait(timeout):
            self.send(CMD_GET_INFO)
            self.info_event.wait(timeout)
        return self.info

    # -- commands

    def send(self, cmd, payload=b''):
        if self.serial_conn:
            self.serial_conn.write(encode_frame(cmd, payload))

    def set_clkdiv(self, div):
        if div < 1024 or div % 64:
            raise ValueError('clk_div must be a multiple of 64 and >= 1024')
        self.info_event.clear()
        self.send(CMD_SET_CLKDIV, struct.pack('<I', div))
        return self.wait_for_info()

    def reboot_to_bootloader(self):
        self.send(CMD_BOOTSEL, struct.pack('<I', BOOTSEL_MAGIC))

    # -- threads

    def _now(self):
        return time.perf_counter()

    def _pinger(self):
        while self.running:
            with self.lock:
                self._ping_id = (self._ping_id + 1) & 0xFFFFFFFF
                pid = self._ping_id
                t_send = self._now()
                self._pending_pings[pid] = t_send
                # forget pings that never got an answer
                if len(self._pending_pings) > 20:
                    for k in sorted(self._pending_pings)[:-20]:
                        del self._pending_pings[k]
            try:
                self.send(CMD_PING, struct.pack('<I', pid))
            except Exception:
                self.serial_errors += 1
            time.sleep(self.ping_interval)

    def _reader(self):
        ser = self.serial_conn
        while self.running:
            try:
                data = ser.read(max(1, ser.in_waiting))
            except Exception:
                self.serial_errors += 1
                time.sleep(0.01)
                continue
            if not data:
                continue
            t_recv = self._now()
            self.bytes_read += len(data)
            for pkt_type, payload in self.parser.feed(data):
                try:
                    self._dispatch(pkt_type, payload, t_recv)
                except struct.error:
                    self.parser.crc_errors += 1

    @staticmethod
    def _append(dq, item, overruns, key):
        if len(dq) == dq.maxlen:
            overruns[key] += 1
        dq.append(item)

    def _dispatch(self, pkt_type, p, t_recv):
        if pkt_type == PKT_CLK:
            seq, t = _CLK.unpack_from(p)
            if self.last_clk_seq is not None and seq != (self.last_clk_seq + 1) & 0xFFFFFFFF:
                self.clk_seq_gaps += 1
            self.last_clk_seq = seq
            self._append(self.clk, (seq, t), self.overruns, 'clk')
        elif pkt_type == PKT_TACHO:
            seq, t, state, changed = _TACHO.unpack_from(p)
            if self.last_tacho_seq is not None and seq != (self.last_tacho_seq + 1) & 0xFFFFFFFF:
                self.tacho_seq_gaps += 1
            self.last_tacho_seq = seq
            self._append(self.tacho, (seq, t, state, changed), self.overruns, 'tacho')
        elif pkt_type == PKT_TEL:
            ch, flags, seq, t_first, data = decode_tel(p)
            if ch >= N_TEL:
                return
            self._append(self.tel, (ch, seq, t_first, flags, len(data)), self.overruns, 'tel')
            for t_frame, raw in self.kiss[ch].feed(seq, t_first, data):
                if ch < len(self.escs):
                    self.escs[ch]._add(t_frame, raw)
        elif pkt_type == PKT_PONG:
            pid, t_pico = _PONG.unpack_from(p)
            with self.lock:
                t_send = self._pending_pings.pop(pid, None)
            if t_send is not None:
                self.clock.add(t_pico, t_send, t_recv)
                self.pings.append((pid, t_send, t_recv, t_pico))
        elif pkt_type == PKT_STATUS:
            self.status = decode_status(p)
            self.status_log.append((t_recv, self.status))
        elif pkt_type == PKT_INFO:
            info = decode_info(p)
            if info['proto'] != PROTO_VERSION:
                print(f'WARNING: Pico protocol version {info["proto"]}, expected {PROTO_VERSION}')
            if self.info is None or info['tick_hz'] != self.clock.tick_hz:
                self.clock.tick_hz = info['tick_hz']
                self.byte_ticks = 10 / info['tel_baud'] * info['tick_hz']
                for k in self.kiss:
                    k.byte_ticks = self.byte_ticks
            self.info = info
            self.info_event.set()

    # -- buffers

    def _pop(self, dq):
        with self.lock:
            items = list(dq)
            dq.clear()
        return items

    def pop_clk(self):
        return self._pop(self.clk)

    def pop_tacho(self):
        return self._pop(self.tacho)

    def pop_tel(self):
        return self._pop(self.tel)

    def pop_status(self):
        return self._pop(self.status_log)

    def pop_pings(self):
        return self._pop(self.pings)

    def reset_buffers(self):
        """Discard buffered events (e.g. everything before a recording starts)
        and zero the overrun counters. Pings are kept for the clock mapping."""
        with self.lock:
            for dq in (self.clk, self.tacho, self.tel, self.status_log):
                dq.clear()
            for k in self.overruns:
                self.overruns[k] = 0
        for esc in self.escs:
            esc.pop_all_data()
            esc.dropped = 0

    def counters(self):
        """Host-side health counters (for HDF5 attributes)."""
        return dict(frames_ok=self.parser.frames_ok, crc_errors=self.parser.crc_errors,
                    bytes_skipped=self.parser.bytes_skipped, bytes_read=self.bytes_read,
                    clk_seq_gaps=self.clk_seq_gaps, tacho_seq_gaps=self.tacho_seq_gaps,
                    serial_errors=self.serial_errors,
                    tel_chunks_lost=[k.chunks_lost for k in self.kiss],
                    **{f'overrun_{k}': v for k, v in self.overruns.items()})


# --- bench monitor ----------------------------------------------------------

def _monitor(port, baudrate, duration):
    link = PicoLink(port, baudrate)
    link.start()
    info = link.wait_for_info()
    print('INFO:', info)
    if info is None:
        print('no INFO received - is the Pico running and the link wired?')
    t_end = time.time() + duration
    clk, tacho = [], []
    while time.time() < t_end:
        time.sleep(1.0)
        clk += link.pop_clk()
        tacho += link.pop_tacho()
        st = link.status
        esc = [f'{e.esc_id}:{e.valid_packets}' for e in link.escs]
        print(f'clk {len(clk):6d}  tacho {len(tacho):6d}  esc {" ".join(esc)}  '
              f'crc_err {link.parser.crc_errors}  '
              f'pico loop_max {st["loop_max_us"] if st else "-"} us  '
              f'drops {st["link_drops"] if st else "-"}')
    link.stop()

    tick_hz = info['tick_hz'] if info else 150e6
    if len(clk) > 2:
        t = np.array([c[1] for c in clk], dtype=np.int64)
        seq = np.array([c[0] for c in clk], dtype=np.int64)
        d = np.diff(t)
        ok = np.diff(seq) == 1
        print(f'\nCLK: {len(clk)} events, period mean {d[ok].mean():.3f} ticks '
              f'({d[ok].mean() / tick_hz * 1e3:.4f} ms), std {d[ok].std():.3f} ticks, '
              f'min {d[ok].min()} max {d[ok].max()}, seq gaps {(~ok).sum()}')
        if info:
            f_pdm = info['clk_div'] * tick_hz / d[ok].mean()
            print(f'     PDM clock {f_pdm / 1e6:.6f} MHz -> fs {f_pdm / 64:.3f} Hz (at 64x)')
    if tacho:
        t = np.array([x[1] for x in tacho], dtype=np.int64)
        state = np.array([x[2] for x in tacho])
        changed = np.array([x[3] for x in tacho])
        for ch in range(N_TACHO):
            rising = t[((changed >> ch) & 1).astype(bool) & ((state >> ch) & 1).astype(bool)]
            if len(rising) > 2:
                d = np.diff(rising)
                print(f'TACHO{ch + 1}: {len(rising)} rising edges, period {d.mean() / tick_hz * 1e3:.4f} ms '
                      f'({60 * tick_hz / d.mean():.1f} rpm at 1 pulse/rev), std {d.std():.2f} ticks')
    for e in link.escs:
        s = e.get_latest_sample()
        if s:
            print(f'{e.esc_id}: {e.valid_packets} frames, invalid bytes {e.invalid_packets}, last {s}')
    print('host counters:', link.counters())


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyAMA5')
    ap.add_argument('--baudrate', type=int, default=921600)
    ap.add_argument('--monitor', type=float, default=10.0, metavar='SECONDS')
    ap.add_argument('--bootsel', action='store_true', help='reboot the Pico into its USB bootloader')
    args = ap.parse_args()
    if args.bootsel:
        lk = PicoLink(args.port, args.baudrate)
        lk.start()
        time.sleep(0.2)
        lk.reboot_to_bootloader()
        time.sleep(0.2)
        lk.stop()
    else:
        _monitor(args.port, args.baudrate, args.monitor)
