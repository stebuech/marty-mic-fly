// Link protocol between the tacho board (Pico 2) and the onboard Raspberry Pi.
// The Python side lives in pico_link.py; keep both in sync and bump
// PROTO_VERSION on any incompatible change.
//
// Frame:   0xA5 0x5A | type u8 | len u8 | payload[len] | crc16 (LE)
// CRC:     CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type, len, payload
// Payload: little endian, no padding.
//
// Timestamps are TIMER1 ticks (tick_hz in INFO; TIMER1 counts clk_sys, i.e.
// 150 MHz), extended to 64 bit on the Pico. All tacho, clock and telemetry
// events share this timebase.
#ifndef TACHO_PROTOCOL_H
#define TACHO_PROTOCOL_H

#include <stdint.h>

#define PROTO_VERSION 1

#define FRAME_SYNC0 0xA5
#define FRAME_SYNC1 0x5A
#define FRAME_OVERHEAD 6          // sync(2) + type + len + crc(2)
#define FRAME_MAX_PAYLOAD 128

// Pico -> Pi
enum {
    PKT_INFO   = 0x01,  // u16 proto, u32 tick_hz, u32 clk_div, u8 n_tacho, u8 n_tel,
                        // u32 tel_baud, u8 flags, u64 t_now, char fw[16]
    PKT_STATUS = 0x02,  // u64 t_now, u64 t_us (TIMER0, 1 MHz), u32 clk_events, u32 tacho_events,
                        // u32 link_drops, u32 rx_crc_errors, u32 tacho_stalls, u32 loop_max_us,
                        // u32 tel_bytes[6], u16 tel_framing[6], u16 tel_overflow[6]
    PKT_CLK    = 0x10,  // u32 seq, u64 t                  -- every clk_div PDM_CLK rising edges
    PKT_TACHO  = 0x11,  // u32 seq, u64 t, u8 state, u8 changed  -- any edge on TACHO1..6
    PKT_TEL    = 0x12,  // u8 ch, u8 flags, u32 seq, u64 t_first, u8 n, u8 data[n]
    PKT_PONG   = 0x13,  // u32 id, u64 t_rx
};

// Pi -> Pico
enum {
    CMD_PING        = 0x80,  // u32 id
    CMD_GET_INFO    = 0x81,  // -
    CMD_SET_CLKDIV  = 0x82,  // u32 div (multiple of 64, >= 1024)
    CMD_BOOTSEL     = 0x8F,  // u32 magic == BOOTSEL_MAGIC; reboots into the USB bootloader
};

#define BOOTSEL_MAGIC 0xB0075E1Fu

// INFO flags
#define INFO_FLAG_SELFTEST 0x01

// TEL flags
#define TEL_FLAG_FRAMING_ERR 0x01   // at least one framing error since the previous chunk
#define TEL_FLAG_OVERFLOW    0x02   // bytes were lost (FIFO or chunk buffer)

#define TEL_CHUNK_MAX 32

#endif
