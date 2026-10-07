// Hardware-independent helpers, unit-tested on the host (tests/test_codec.c).
#ifndef TACHO_CODEC_H
#define TACHO_CODEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "protocol.h"

uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc);

// Encode one frame into out (needs len + FRAME_OVERHEAD bytes). Returns the
// frame length, or 0 if len exceeds FRAME_MAX_PAYLOAD.
size_t frame_encode(uint8_t *out, uint8_t type, const uint8_t *payload, uint8_t len);

// Incremental decoder for incoming frames. Feed bytes one at a time;
// frame_decoder_push returns true when a complete, CRC-valid frame is
// available in dec->type / dec->payload / dec->len.
typedef struct {
    uint8_t state;
    uint8_t type;
    uint8_t len;
    uint8_t pos;
    uint8_t payload[FRAME_MAX_PAYLOAD];
    uint8_t crc_lo;
    uint32_t crc_errors;
} frame_decoder_t;

void frame_decoder_init(frame_decoder_t *dec);
bool frame_decoder_push(frame_decoder_t *dec, uint8_t byte);

// Extend a 32-bit raw timer value captured in the past to 64 bit, given a
// 64-bit reading of the same counter taken *after* the capture. Exact as long
// as the capture is younger than 2^32 ticks (28.6 s at 150 MHz). Stateless, so
// a quiet channel (stopped rotor) can never lose its epoch.
static inline uint64_t ts_extend(uint32_t captured_lo, uint64_t now64) {
    return now64 - (uint32_t)((uint32_t)now64 - captured_lo);
}

// Little-endian writers, return the advanced pointer.
static inline uint8_t *put_u8(uint8_t *p, uint8_t v) { *p++ = v; return p; }
static inline uint8_t *put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); return p + 2;
}
static inline uint8_t *put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
    return p + 4;
}
static inline uint8_t *put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
    return p + 8;
}
static inline uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Telemetry chunker: groups UART bytes of one channel into chunks separated by
// line idle gaps (> gap_ticks) or capped at TEL_CHUNK_MAX bytes. Framing of the
// ESC protocol itself (KISS, 10-byte frames) is left to the Pi.
typedef struct {
    uint8_t data[TEL_CHUNK_MAX];
    uint8_t n;
    uint8_t flags;
    uint64_t t_first;
    uint64_t t_last;
    uint32_t seq;
} tel_chunk_t;

// Returns true when the bytes collected so far form a finished chunk that the
// caller must emit (and then call tel_chunk_reset) *before* adding byte b.
bool tel_chunk_needs_flush(const tel_chunk_t *c, uint64_t t, uint64_t gap_ticks);
void tel_chunk_add(tel_chunk_t *c, uint8_t b, uint64_t t);
void tel_chunk_reset(tel_chunk_t *c);
// Encode the chunk as a PKT_TEL payload; returns payload length.
size_t tel_chunk_payload(const tel_chunk_t *c, uint8_t ch, uint8_t *out);

#endif
