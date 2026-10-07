"""Tests for pico_link.py (Pico tacho board protocol) and the HDF5 logger in daq.py."""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pico_link as pl  # noqa: E402

FW = REPO / 'firmware' / 'tacho_pico'


def kiss_frame(temp=30, mv=1680, ma=123, mah=5, erpm_100=450):
    body = bytes([temp]) + struct.pack('>HHHH', mv, ma, mah, erpm_100)
    return body + bytes([pl.crc8_kiss(body)])


def test_crc16_reference_value():
    assert pl.crc16_ccitt(b'123456789') == 0x29B1
    assert pl._crc16_fast(b'123456789') == 0x29B1


@pytest.mark.skipif(shutil.which('cc') is None, reason='no C compiler')
def test_python_decoder_matches_c_encoder(tmp_path):
    exe = tmp_path / 'test_codec'
    subprocess.run(['cc', '-O1', f'-I{FW / "src"}', str(FW / 'tests' / 'test_codec.c'),
                    str(FW / 'src' / 'codec.c'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)  # C unit tests incl. ts_extend wrap test
    lines = subprocess.run([str(exe), '--emit'], check=True, capture_output=True, text=True).stdout.split()
    frames = pl.FrameParser().feed(b''.join(bytes.fromhex(x) for x in lines))
    types = [t for t, _ in frames]
    assert types == [pl.PKT_CLK, pl.PKT_TACHO, pl.PKT_TEL, pl.PKT_PONG]
    assert pl._CLK.unpack(frames[0][1]) == (7, 0x123456789ABC)
    assert pl._TACHO.unpack(frames[1][1]) == (42, 0x1122334455, 0x15, 0x04)
    ch, flags, seq, t_first, data = pl.decode_tel(frames[2][1])
    assert (ch, flags, seq, t_first, len(data)) == (2, 0, 9, 5000, 10)
    assert pl._PONG.unpack(frames[3][1]) == (0xDEADBEEF, 987654321)


def test_frame_parser_resync_and_split():
    good = pl.encode_frame(pl.PKT_CLK, struct.pack('<IQ', 1, 100))
    bad = bytearray(good)
    bad[6] ^= 0xFF
    stream = b'\x00\xA5\x5A\x10' + bytes(bad) + good + b'\xA5' + good
    parser = pl.FrameParser()
    out = []
    for i in range(0, len(stream), 3):  # arbitrary read boundaries
        out += parser.feed(stream[i:i + 3])
    assert len(out) == 2
    assert all(t == pl.PKT_CLK for t, _ in out)
    assert parser.crc_errors >= 1


def test_kiss_stream_reassembly():
    ks = pl.KissStream(byte_ticks=13000)
    frames = ks.feed(0, 1_000_000, kiss_frame(erpm_100=400))
    assert len(frames) == 1 and frames[0][0] == 1_000_000
    # garbage byte in front, then a valid frame within the same chunk
    frames = ks.feed(1, 2_000_000, b'\x55' + kiss_frame(erpm_100=401))
    assert len(frames) == 1 and frames[0][0] == 2_000_000 + 13000
    assert ks.invalid_bytes == 1
    # frame split over two contiguous chunks
    f = kiss_frame(erpm_100=402)
    assert ks.feed(2, 3_000_000, f[:4]) == []
    frames = ks.feed(3, 3_000_000 + 4 * 13000, f[4:])
    assert len(frames) == 1 and pl.parse_kiss(frames[0][1])[4] == 40200
    # lost chunk is counted, partial bytes after a gap are discarded
    ks.feed(5, 9_000_000, f[:3])
    assert ks.chunks_lost == 1
    frames = ks.feed(6, 20_000_000, kiss_frame())
    assert len(frames) == 1 and ks.invalid_bytes == 4


def test_clock_map_recovers_offset_and_rate():
    rng = np.random.default_rng(0)
    tick_hz = 150e6
    cm = pl.ClockMap(tick_hz=tick_hz)
    true_rate = 1 / (tick_hz * (1 + 20e-6))  # Pico 20 ppm fast
    for k in range(100):
        ticks = int(5e9 + k * 0.5 * tick_hz)
        pi_true = 100.0 + (ticks - 5e9) * true_rate
        up, down = rng.exponential(3e-4, 2) + 1e-4
        cm.add(ticks, pi_true - up, pi_true + down)
    t_test = int(5e9 + 120 * 0.5 * tick_hz)
    assert abs(cm.to_pi(t_test) - (100.0 + (t_test - 5e9) * true_rate)) < 5e-4


class FakeLink:
    """Stands in for PicoLink in logger tests."""

    def __init__(self):
        self.port = '/dev/null'
        self.info = {'proto': 1, 'tick_hz': 150_000_000, 'clk_div': 16384, 'fw': 'test', 'selftest': True}
        self.clk = [(i, 1000 + i * 819200) for i in range(5)]
        self.tacho = [(i, 5000 + i * 2_600_000, i & 1, 1) for i in range(4)]
        self.tel = [(0, 1, 7000, 0, 10)]
        self.pings = [(1, 10.0, 10.001, 123456)]
        self.status = [(11.0, {'t_now': 1, 't_us': 2, 'clk_events': 3, 'tacho_events': 4, 'link_drops': 0,
                               'rx_crc_errors': 0, 'tacho_stalls': 0, 'loop_max_us': 40,
                               'tel_bytes': (1,) * 6, 'tel_framing': (0,) * 6, 'tel_overflow': (0,) * 6})]

    def _take(self, name):
        v = getattr(self, name)
        setattr(self, name, [])
        return v

    def pop_clk(self): return self._take('clk')
    def pop_tacho(self): return self._take('tacho')
    def pop_tel(self): return self._take('tel')
    def pop_pings(self): return self._take('pings')
    def pop_status(self): return self._take('status')
    def counters(self): return {'frames_ok': 10, 'crc_errors': 0}


class FakeESC:
    def __init__(self, link=None):
        self.esc_id = 'ESC1'
        self.port = 'x'
        self.pole_pairs = 12
        self.valid_packets = 2
        self.invalid_packets = 0
        if link is not None:
            self.link = link
        self.data = [{'timestamp': 0.1 * i, 'pico_ticks': 10 ** 10 + i, 'temperature': 30, 'voltage': 16.8,
                      'current': 1.2, 'consumption': 5, 'erpm': 42000, 'rpm': 3500.0} for i in range(3)]

    def pop_all_data(self):
        d, self.data = self.data, []
        return d


def test_hdf5_logger_layout(tmp_path):
    daq = pytest.importorskip('daq')
    link = FakeLink()
    esc = FakeESC(link)
    timer = daq.Timer()
    logger = daq.HDF5Logger(str(tmp_path), 'rec.h5', flush_interval=0.05, esc_tel_data=True, pico_data=True)
    logger.start_logging([esc], None, None, timer, pico=link)
    logger.stop_logging()
    logger.close()
    # a second logger with the same name must not overwrite the first file
    logger2 = daq.HDF5Logger(str(tmp_path), 'rec.h5')
    logger2.close()
    assert Path(logger2.filename).name == 'rec_001.h5'

    with h5py.File(tmp_path / 'rec.h5', 'r') as f:
        assert f['pico/clk/ticks'][:].tolist() == [1000 + i * 819200 for i in range(5)]
        assert f['pico/tacho/state'].dtype == np.uint8 and len(f['pico/tacho/seq']) == 4
        assert f['pico/ping/pico_ticks'][0] == 123456
        assert f['pico/status/tel_bytes'].shape == (1, 6)
        assert f['pico'].attrs['clk_div'] == 16384
        g = f['esc_telemetry/ESC1']
        assert g['pico_ticks'].dtype == np.uint64 and len(g['rpm']) == 3
        assert g.attrs['source'] == 'pico' and g.attrs['pole_pairs'] == 12
        assert 'start_time_perf' in f['timing'].attrs
