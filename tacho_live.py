#!/usr/bin/env python3
"""Live view of the optical tacho channels on the Pico board, for adjusting sensors.

Per channel: current level, raw edges, clean transitions (level stable for at
least --settle ms after the edge), edges per transition (1 = clean, more =
chatter), rpm and timing jitter from the last transitions.

    python tacho_live.py                 # Ctrl+C to quit
    python tacho_live.py --settle 0.5 --pulses-per-rev 1

Opens /dev/ttyAMA5 itself, so daq.py must not run at the same time. Switches the
Pico to both tacho edges; daq.py selects its own mode (default rising) at start.
"""

import argparse
import time
from collections import deque

import numpy as np

from pico_link import PicoLink, N_TACHO


class Channel:
    """Streaming debouncer for one tacho channel (Pico ticks)."""

    def __init__(self, settle_ticks, history=64):
        self.settle = settle_ticks
        self.level = None          # last raw level seen
        self.stable = None         # last settled level
        self.burst_start = None    # tick of the first edge of the current burst
        self.burst_edges = 0
        self.last_edge = None
        self.raw_edges = 0
        self.transitions = 0
        self.chatter = deque(maxlen=history)                 # edges per finished burst
        self.trans = {0: deque(maxlen=history), 1: deque(maxlen=history)}   # transition ticks per new level

    def edge(self, t, level):
        if self.last_edge is not None and t - self.last_edge >= self.settle:
            self._finish()
        if self.burst_start is None:
            self.burst_start = t
            self.burst_edges = 0
        self.burst_edges += 1
        self.raw_edges += 1
        self.last_edge = t
        self.level = level

    def idle(self, now):
        """Finish a burst once the line has been quiet for settle ticks."""
        if self.burst_start is not None and now - self.last_edge >= self.settle:
            self._finish()

    def _finish(self):
        self.chatter.append(self.burst_edges)
        if self.stable is not None and self.level != self.stable:
            self.trans[self.level].append(self.burst_start)
            self.transitions += 1
        self.stable = self.level
        self.burst_start = None


def stats(x, tick_hz, ppr):
    """rpm and jitter from transitions of one direction (one per pulse)"""
    if len(x) < 4:
        return None, None
    x = np.array(x, dtype=np.float64)
    p = np.diff(x)
    rpm = 60.0 * tick_hz / (np.median(p) * ppr)
    jitter_us = np.diff(x, 2).std() / np.sqrt(6) / tick_hz * 1e6
    return rpm, jitter_us


def replay(args):
    """Feed a recording through the same debouncer, one table row block per second of data."""
    import h5py
    with h5py.File(args.replay, 'r') as f:
        tick_hz = int(f['pico'].attrs['tick_hz'])
        t = f['pico/tacho/ticks'][:].astype(np.int64)
        state = f['pico/tacho/state'][:]
        changed = f['pico/tacho/changed'][:]
    if len(t) == 0:
        print('no tacho events in file')
        return
    chans = [Channel(args.settle * 1e-3 * tick_hz, args.window) for _ in range(N_TACHO)]
    sec = (t - t[0]) // tick_hz
    for s in range(int(sec[-1]) + 1):
        idx = np.where(sec == s)[0]
        for i in idx:
            for c in range(N_TACHO):
                if (changed[i] >> c) & 1:
                    chans[c].edge(int(t[i]), (int(state[i]) >> c) & 1)
        for ch in chans:
            ch.idle(int(t[0]) + (s + 1) * tick_hz)
        row = [f'{s:3d}s']
        for c in range(args.channels):
            r1, j1 = stats(chans[c].trans[1], tick_hz, args.pulses_per_rev)
            chat = f'{np.median(chans[c].chatter):.0f}' if chans[c].chatter else '-'
            row.append(f'T{c + 1} ' + ('-' if r1 is None else f'{r1:5.0f}rpm {j1:7.1f}us') + f' e/k {chat:>3}')
        print('  '.join(row))
        for ch in chans:  # statistics per second, not cumulative
            ch.trans[0].clear(); ch.trans[1].clear(); ch.chatter.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyAMA5')
    ap.add_argument('--settle', type=float, default=1.0, help='ms the level must stay constant to count as a transition')
    ap.add_argument('--pulses-per-rev', type=int, default=1)
    ap.add_argument('--channels', type=int, default=4, help='number of tacho channels shown')
    ap.add_argument('--interval', type=float, default=0.5, help='display refresh in s')
    ap.add_argument('--window', type=int, default=32, help='transitions used for rpm/jitter')
    ap.add_argument('--duration', type=float, default=None, help='stop after this many seconds')
    ap.add_argument('--no-clear', action='store_true', help='print lines instead of redrawing the screen')
    ap.add_argument('--replay', metavar='H5', help='run the same evaluation on a daq.py recording instead of live data')
    args = ap.parse_args()
    if args.replay:
        replay(args)
        return

    link = PicoLink(args.port)
    link.start()
    info = link.wait_for_info()
    if info is None:
        print(f'No answer from the Pico on {args.port}. Is daq.py running?')
        link.stop()
        return
    if link.set_tacho_edges('both') is None and info['proto'] >= 2:
        print('WARNING: could not switch the Pico to both tacho edges')
    tick_hz = info['tick_hz']
    chans = [Channel(args.settle * 1e-3 * tick_hz, args.window) for _ in range(N_TACHO)]
    now_tick = 0
    t_start = time.time()
    prev = [(0, 0)] * N_TACHO
    prev_t = time.time()

    try:
        while args.duration is None or time.time() - t_start < args.duration:
            time.sleep(args.interval)
            for seq, t, state, changed in link.pop_tacho():
                now_tick = max(now_tick, t)
                for c in range(N_TACHO):
                    if (changed >> c) & 1:
                        chans[c].edge(t, (state >> c) & 1)
                    elif chans[c].level is None:
                        chans[c].level = (state >> c) & 1
                        chans[c].stable = chans[c].level
            clk = link.pop_clk()
            if clk:
                now_tick = max(now_tick, clk[-1][1])
            if link.status:
                now_tick = max(now_tick, link.status['t_now'])
            for ch in chans:
                ch.idle(now_tick)

            dt = time.time() - prev_t
            prev_t = time.time()
            lines = []
            if not args.no_clear:
                lines.append('\033[H\033[J')
            lines.append(f'Tacho live  fw {info["fw"]}  settle {args.settle} ms  '
                         f'{args.pulses_per_rev} pulse/rev  (Ctrl+C to quit)')
            lines.append(f'{"ch":<8}{"level":>6}{"edges/s":>9}{"trans/s":>9}{"edges/kante":>13}'
                         f'{"rpm":>8}{"jitter us":>11}{"rpm(->0)":>10}{"jit(->0)":>10}')
            for c in range(args.channels):
                ch = chans[c]
                e0, t0 = prev[c]
                er = (ch.raw_edges - e0) / dt
                tr = (ch.transitions - t0) / dt
                prev[c] = (ch.raw_edges, ch.transitions)
                lvl = '?' if ch.level is None else ('HIGH' if ch.level else 'low')
                chat = f'{np.median(ch.chatter):.0f}' if ch.chatter else '-'
                r1, j1 = stats(ch.trans[1], tick_hz, args.pulses_per_rev)
                r0, j0 = stats(ch.trans[0], tick_hz, args.pulses_per_rev)
                fmt = lambda v, f: '-' if v is None else format(v, f)  # noqa: E731
                lines.append(f'TACHO{c + 1:<3}{lvl:>6}{er:9.0f}{tr:9.1f}{chat:>13}'
                             f'{fmt(r1, "8.0f"):>8}{fmt(j1, "11.1f"):>11}{fmt(r0, "10.0f"):>10}{fmt(j0, "10.1f"):>10}')
            st = link.status
            if st:
                lines.append(f'pico: link drops {st["link_drops"]}  tacho stalls {st["tacho_stalls"]}  '
                             f'crc errors {link.parser.crc_errors}')
            lines.append('rpm/jitter: from transitions to HIGH, (->0) from transitions to low; '
                         'edges/kante 1 = clean edge; level ? = no edge since start')
            print('\n'.join(lines), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()


if __name__ == '__main__':
    main()
