#include "codec.h"

#include <string.h>

uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc) {
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

size_t frame_encode(uint8_t *out, uint8_t type, const uint8_t *payload, uint8_t len) {
    if (len > FRAME_MAX_PAYLOAD) return 0;
    out[0] = FRAME_SYNC0;
    out[1] = FRAME_SYNC1;
    out[2] = type;
    out[3] = len;
    if (len) memcpy(out + 4, payload, len);
    uint16_t crc = crc16_ccitt(out + 2, (size_t)len + 2, 0xFFFF);
    put_u16(out + 4 + len, crc);
    return (size_t)len + FRAME_OVERHEAD;
}

enum { DS_SYNC0, DS_SYNC1, DS_TYPE, DS_LEN, DS_PAYLOAD, DS_CRC0, DS_CRC1 };

void frame_decoder_init(frame_decoder_t *dec) {
    memset(dec, 0, sizeof(*dec));
    dec->state = DS_SYNC0;
}

bool frame_decoder_push(frame_decoder_t *dec, uint8_t byte) {
    switch (dec->state) {
    case DS_SYNC0:
        if (byte == FRAME_SYNC0) dec->state = DS_SYNC1;
        return false;
    case DS_SYNC1:
        dec->state = (byte == FRAME_SYNC1) ? DS_TYPE : (byte == FRAME_SYNC0 ? DS_SYNC1 : DS_SYNC0);
        return false;
    case DS_TYPE:
        dec->type = byte;
        dec->state = DS_LEN;
        return false;
    case DS_LEN:
        if (byte > FRAME_MAX_PAYLOAD) {
            dec->crc_errors++;
            dec->state = DS_SYNC0;
            return false;
        }
        dec->len = byte;
        dec->pos = 0;
        dec->state = byte ? DS_PAYLOAD : DS_CRC0;
        return false;
    case DS_PAYLOAD:
        dec->payload[dec->pos++] = byte;
        if (dec->pos == dec->len) dec->state = DS_CRC0;
        return false;
    case DS_CRC0:
        dec->crc_lo = byte;
        dec->state = DS_CRC1;
        return false;
    case DS_CRC1: {
        dec->state = DS_SYNC0;
        uint8_t hdr[2] = {dec->type, dec->len};
        uint16_t crc = crc16_ccitt(hdr, 2, 0xFFFF);
        crc = crc16_ccitt(dec->payload, dec->len, crc);
        if (crc == (uint16_t)(dec->crc_lo | (byte << 8))) return true;
        dec->crc_errors++;
        return false;
    }
    }
    dec->state = DS_SYNC0;
    return false;
}

bool tel_chunk_needs_flush(const tel_chunk_t *c, uint64_t t, uint64_t gap_ticks) {
    if (c->n == 0) return false;
    return c->n >= TEL_CHUNK_MAX || (t - c->t_last) > gap_ticks;
}

void tel_chunk_add(tel_chunk_t *c, uint8_t b, uint64_t t) {
    if (c->n >= TEL_CHUNK_MAX) {
        c->flags |= TEL_FLAG_OVERFLOW;
        return;
    }
    if (c->n == 0) c->t_first = t;
    c->data[c->n++] = b;
    c->t_last = t;
}

void tel_chunk_reset(tel_chunk_t *c) {
    c->n = 0;
    c->flags = 0;
    c->seq++;
}

size_t tel_chunk_payload(const tel_chunk_t *c, uint8_t ch, uint8_t *out) {
    uint8_t *p = out;
    p = put_u8(p, ch);
    p = put_u8(p, c->flags);
    p = put_u32(p, c->seq);
    p = put_u64(p, c->t_first);
    p = put_u8(p, c->n);
    memcpy(p, c->data, c->n);
    return (size_t)(p - out) + c->n;
}

bool tacho_batch_needs_flush(const tacho_batch_t *b, uint64_t t) {
    if (b->n == 0) return false;
    return b->n >= TACHO_BATCH_MAX || t - b->t_last > TACHO_DT_MAX;
}

void tacho_batch_add(tacho_batch_t *b, uint32_t seq, uint64_t t, uint8_t state, uint8_t changed) {
    if (b->n >= TACHO_BATCH_MAX) return;  // caller flushes first
    uint32_t dt = 0;
    if (b->n == 0) {
        uint8_t *p = put_u32(b->payload, seq);
        put_u64(p, t);
        b->t_first = t;
    } else {
        dt = (uint32_t)(t - b->t_last);
    }
    uint8_t *e = b->payload + TACHO_BATCH_HEAD + (size_t)b->n * TACHO_BATCH_ENTRY;
    e[0] = state;
    e[1] = changed;
    e[2] = (uint8_t)dt;
    e[3] = (uint8_t)(dt >> 8);
    e[4] = (uint8_t)(dt >> 16);
    b->n++;
    b->payload[12] = b->n;
    b->t_last = t;
}

size_t tacho_batch_len(const tacho_batch_t *b) {
    return b->n ? TACHO_BATCH_HEAD + (size_t)b->n * TACHO_BATCH_ENTRY : 0;
}
