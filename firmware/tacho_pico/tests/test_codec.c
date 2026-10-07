// Host unit tests for codec.c: cc -I../src test_codec.c ../src/codec.c && ./a.out
// Also writes golden frames to stdout with --emit, consumed by
// tests/test_pico_link.py to check the Python decoder against the C encoder.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec.h"

static int failures;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static void test_crc(void) {
    // CRC-16/CCITT-FALSE check value
    CHECK(crc16_ccitt((const uint8_t *)"123456789", 9, 0xFFFF) == 0x29B1);
}

static void test_ts_extend(void) {
    // Simulate a 64-bit counter crossing many 32-bit wraps with captures of
    // varying age, including long quiet periods (stopped rotor).
    uint64_t t = 0xFFFFF000ull;  // just before the first wrap
    uint64_t rng = 12345;
    for (int i = 0; i < 2000000; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t step = (rng >> 33) % 3000000ull;           // up to 20 ms
        if (i % 50000 == 0) step = 4000000000ull;            // ~27 s quiet
        t += step;
        uint64_t age = (rng >> 20) % 1000000ull;             // processing delay
        uint64_t now = t + age;
        CHECK(ts_extend((uint32_t)t, now) == t);
        if (failures > 5) return;
    }
    // exact boundary: capture at now - (2^32 - 1)
    uint64_t now = 0x500000000ull;
    CHECK(ts_extend((uint32_t)(now - 0xFFFFFFFFull), now) == now - 0xFFFFFFFFull);
    CHECK(ts_extend((uint32_t)now, now) == now);
}

static void test_frame_roundtrip(void) {
    uint8_t payload[FRAME_MAX_PAYLOAD], frame[FRAME_MAX_PAYLOAD + FRAME_OVERHEAD];
    for (int i = 0; i < FRAME_MAX_PAYLOAD; i++) payload[i] = (uint8_t)(i * 7 + 0xA5);
    frame_decoder_t dec;
    frame_decoder_init(&dec);

    // garbage, including a fake sync, must not produce a frame
    const uint8_t junk[] = {0x00, 0xA5, 0xA5, 0x5A, 0x81, 0xFF, 0x12, 0xA5};
    for (size_t i = 0; i < sizeof(junk); i++) CHECK(!frame_decoder_push(&dec, junk[i]));

    for (int len = 0; len <= FRAME_MAX_PAYLOAD; len += 13) {
        size_t n = frame_encode(frame, 0x80, payload, (uint8_t)len);
        CHECK(n == (size_t)len + FRAME_OVERHEAD);
        int got = 0;
        for (size_t i = 0; i < n; i++) got += frame_decoder_push(&dec, frame[i]);
        CHECK(got == 1);
        CHECK(dec.type == 0x80 && dec.len == len && memcmp(dec.payload, payload, len) == 0);
    }
    // corrupted frame is rejected and counted
    size_t n = frame_encode(frame, 0x81, payload, 4);
    frame[5] ^= 0x01;
    uint32_t errs = dec.crc_errors;
    int got = 0;
    for (size_t i = 0; i < n; i++) got += frame_decoder_push(&dec, frame[i]);
    CHECK(got == 0 && dec.crc_errors == errs + 1);
    CHECK(frame_encode(frame, 1, payload, FRAME_MAX_PAYLOAD + 1) == 0);
}

static void test_tel_chunk(void) {
    tel_chunk_t c = {0};
    const uint64_t gap = 100;
    uint64_t t = 1000;
    // 10 bytes back to back, then a gap, then 3 bytes
    for (int i = 0; i < 10; i++) {
        CHECK(!tel_chunk_needs_flush(&c, t, gap));
        tel_chunk_add(&c, (uint8_t)i, t);
        t += 87;
    }
    t += 500;
    CHECK(tel_chunk_needs_flush(&c, t, gap));
    uint8_t out[FRAME_MAX_PAYLOAD];
    size_t n = tel_chunk_payload(&c, 3, out);
    CHECK(n == 15 + 10 && out[0] == 3 && out[14] == 10 && out[15] == 0 && out[24] == 9);
    CHECK(c.t_first == 1000);
    tel_chunk_reset(&c);
    CHECK(c.n == 0 && c.seq == 1);
    // cap at TEL_CHUNK_MAX
    for (int i = 0; i < TEL_CHUNK_MAX; i++) tel_chunk_add(&c, 0x55, t + (uint64_t)i);
    CHECK(tel_chunk_needs_flush(&c, t + TEL_CHUNK_MAX, gap));
    tel_chunk_add(&c, 0x66, t);
    CHECK(c.n == TEL_CHUNK_MAX && (c.flags & TEL_FLAG_OVERFLOW));
}

// Emit one frame of each Pico->Pi type as hex lines for the Python test.
static void emit_hex(uint8_t type, const uint8_t *p, size_t len) {
    uint8_t frame[FRAME_MAX_PAYLOAD + FRAME_OVERHEAD];
    size_t n = frame_encode(frame, type, p, (uint8_t)len);
    for (size_t i = 0; i < n; i++) printf("%02x", frame[i]);
    printf("\n");
}

static void emit_golden(void) {
    uint8_t b[FRAME_MAX_PAYLOAD], *p;
    p = put_u32(b, 7); p = put_u64(p, 0x123456789ABCull);
    emit_hex(PKT_CLK, b, (size_t)(p - b));
    p = put_u32(b, 42); p = put_u64(p, 0x1122334455ull); p = put_u8(p, 0x15); p = put_u8(p, 0x04);
    emit_hex(PKT_TACHO, b, (size_t)(p - b));
    tel_chunk_t c = {0};
    c.seq = 9;
    const uint8_t kiss[10] = {30, 0x06, 0x90, 0x00, 0x7B, 0x00, 0x05, 0x01, 0xC2, 0x00};
    for (int i = 0; i < 10; i++) tel_chunk_add(&c, kiss[i], 5000 + (uint64_t)i * 1302);
    size_t n = tel_chunk_payload(&c, 2, b);
    emit_hex(PKT_TEL, b, n);
    p = put_u32(b, 0xDEADBEEF); p = put_u64(p, 987654321ull);
    emit_hex(PKT_PONG, b, (size_t)(p - b));
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--emit") == 0) {
        emit_golden();
        return 0;
    }
    test_crc();
    test_ts_extend();
    test_frame_roundtrip();
    test_tel_chunk();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all codec tests passed\n");
    return 0;
}
