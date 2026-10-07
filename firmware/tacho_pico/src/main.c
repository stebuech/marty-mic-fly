// Tacho capture board firmware (Raspberry Pi Pico 2 / RP2350), Rev. G3 board.
//
//   GP2       PDM_CLK (MCHStreamer, via U5)  -> clkdiv SM, event every clk_div rising edges
//   GP3..GP8  TACHO_1..6 (via U1-U4,U6,U7)   -> edge_watch SM, event on any edge
//   GP9..GP14 TEL1..6 (ESC telemetry, 115200) -> 6x PIO UART RX
//   GP0/GP1   UART0 TX/RX <-> Pi UART5 (/dev/ttyAMA5), 921600 8N1
//   GP25      onboard LED, 1 Hz blink
//
// Timestamping: PIO only detects events. A DMA pair per event source moves the
// FIFO word into a ring buffer and then copies TIMER1's raw counter (counting
// clk_sys, 150 MHz) into a timestamp ring. The CPU never touches the capture
// path; it only drains the rings, extends timestamps to 64 bit and frames
// packets for the Pi. See protocol.h for the wire format.
//
// Build with -DTACHO_SELFTEST=1 for a bench build that needs nothing connected:
// PWM on GP22 (3.0 MHz) replaces PDM_CLK, PWM on GP16..GP21 replaces the
// sensors, and a PIO UART on GP26 sends KISS frames that all six TEL receivers
// listen to.

#include <string.h>

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/pwm.h"
#include "hardware/resets.h"
#include "hardware/timer.h"
#include "hardware/uart.h"
#include "hardware/watchdog.h"
#include "pico/bootrom.h"
#include "pico/stdlib.h"

#include "capture.pio.h"
#include "codec.h"
#include "protocol.h"

#ifndef FW_VERSION
#define FW_VERSION "dev"
#endif
#ifndef TACHO_SELFTEST
#define TACHO_SELFTEST 0
#endif

// --- pins / constants -------------------------------------------------------

#define LINK_UART uart0
#define PIN_LINK_TX 0
#define PIN_LINK_RX 1
#define LINK_BAUD 921600
#define PIN_LED 25

#define N_TACHO 6
#define N_TEL 6
#define TEL_BAUD 115200
#define DEFAULT_CLK_DIV 16384  // 256 audio samples at 64x PDM oversampling

#if TACHO_SELFTEST
#define PIN_CLK 22
#define PIN_TACHO_BASE 16
static const uint tel_pins[N_TEL] = {26, 26, 26, 26, 26, 26};
#define PIN_SELFTEST_TX 26
#else
#define PIN_CLK 2
#define PIN_TACHO_BASE 3
static const uint tel_pins[N_TEL] = {9, 10, 11, 12, 13, 14};
#endif

// PIO / SM allocation: PIO0 SM0 clkdiv, SM1 edge_watch; TEL1-4 on PIO1 SM0-3,
// TEL5-6 on PIO2 SM0-1, self-test TX on PIO2 SM2.
#define PIO_CAP pio0
#define SM_CLK 0
#define SM_TACHO 1
static PIO const tel_pio[N_TEL] = {pio1, pio1, pio1, pio1, pio2, pio2};
static const uint tel_sm[N_TEL] = {0, 1, 2, 3, 0, 1};

// DMA channels (claimed at init)
static int dma_clk_fifo, dma_clk_ts, dma_tacho_fifo, dma_tacho_ts;

// Ring buffers. DMA ring wrapping needs natural alignment of the buffer size.
#define CLK_RING_BITS 12    // 4 KiB = 1024 events (5.5 s at 187.5 Hz)
#define TACHO_RING_BITS 14  // 16 KiB = 4096 events
#define CLK_RING_N (1u << (CLK_RING_BITS - 2))
#define TACHO_RING_N (1u << (TACHO_RING_BITS - 2))
static uint32_t clk_ts_ring[CLK_RING_N] __attribute__((aligned(1u << CLK_RING_BITS)));
static uint32_t tacho_ts_ring[TACHO_RING_N] __attribute__((aligned(1u << TACHO_RING_BITS)));
static uint32_t tacho_state_ring[TACHO_RING_N] __attribute__((aligned(1u << TACHO_RING_BITS)));
static uint32_t clk_fifo_sink;

// --- state ------------------------------------------------------------------

static uint32_t tick_hz;
static uint32_t clk_div = DEFAULT_CLK_DIV;
static uint clk_prog_offset;

static uint32_t clk_rd, tacho_rd;
static uint32_t clk_seq, tacho_seq;
static uint32_t tacho_prev_state;

static tel_chunk_t tel[N_TEL];
static uint64_t tel_gap_ticks;
static uint32_t tel_bytes[N_TEL];
static uint16_t tel_framing[N_TEL], tel_overflow[N_TEL];

static uint32_t link_drops, tacho_stalls, loop_max_us;
static frame_decoder_t rx_dec;

// --- timebase ---------------------------------------------------------------

static inline uint64_t now_ticks(void) { return timer_time_us_64(timer1_hw); }

static void timebase_init(void) {
    unreset_block_wait(RESETS_RESET_TIMER1_BITS);
    // Count clk_sys directly (RP2350 TIMER SOURCE register) instead of the
    // 1 MHz tick: 6.7 ns resolution, 32-bit wrap after 28.6 s.
    timer1_hw->source = TIMER_SOURCE_CLK_SYS_BITS;
    tick_hz = clock_get_hz(clk_sys);
}

// --- link TX ring -----------------------------------------------------------

#define TX_RING_N 16384u
static uint8_t tx_ring[TX_RING_N];
static uint32_t tx_wr, tx_rd;  // free-running indices

static void link_send(uint8_t type, const uint8_t *payload, uint8_t len) {
    uint8_t frame[FRAME_MAX_PAYLOAD + FRAME_OVERHEAD];
    size_t n = frame_encode(frame, type, payload, len);
    if (n == 0 || TX_RING_N - (tx_wr - tx_rd) < n) {
        link_drops++;
        return;
    }
    for (size_t i = 0; i < n; i++) tx_ring[(tx_wr + i) % TX_RING_N] = frame[i];
    tx_wr += (uint32_t)n;
}

static void link_pump_tx(void) {
    uart_hw_t *hw = uart_get_hw(LINK_UART);
    while (tx_rd != tx_wr && !(hw->fr & UART_UARTFR_TXFF_BITS)) {
        hw->dr = tx_ring[tx_rd % TX_RING_N];
        tx_rd++;
    }
}

// --- packets ----------------------------------------------------------------

static void send_info(void) {
    uint8_t buf[64], *p = buf;
    p = put_u16(p, PROTO_VERSION);
    p = put_u32(p, tick_hz);
    p = put_u32(p, clk_div);
    p = put_u8(p, N_TACHO);
    p = put_u8(p, N_TEL);
    p = put_u32(p, TEL_BAUD);
    p = put_u8(p, TACHO_SELFTEST ? INFO_FLAG_SELFTEST : 0);
    p = put_u64(p, now_ticks());
    char fw[16] = {0};
    strncpy(fw, FW_VERSION, sizeof(fw));
    memcpy(p, fw, sizeof(fw));
    p += sizeof(fw);
    link_send(PKT_INFO, buf, (uint8_t)(p - buf));
}

static void send_status(void) {
    uint8_t buf[96], *p = buf;
    p = put_u64(p, now_ticks());
    p = put_u64(p, time_us_64());
    p = put_u32(p, clk_seq);
    p = put_u32(p, tacho_seq);
    p = put_u32(p, link_drops);
    p = put_u32(p, rx_dec.crc_errors);
    p = put_u32(p, tacho_stalls);
    p = put_u32(p, loop_max_us);
    for (int i = 0; i < N_TEL; i++) p = put_u32(p, tel_bytes[i]);
    for (int i = 0; i < N_TEL; i++) p = put_u16(p, tel_framing[i]);
    for (int i = 0; i < N_TEL; i++) p = put_u16(p, tel_overflow[i]);
    link_send(PKT_STATUS, buf, (uint8_t)(p - buf));
    loop_max_us = 0;
}

// --- capture: PIO + DMA -----------------------------------------------------

// Channel `fifo` waits for the SM's RX DREQ, moves the FIFO word to fifo_dst,
// then chains to `ts`, which copies TIMER1's raw low word into ts_ring and
// chains back. Both run with transfer count 1 forever.
static void dma_pair_init(int fifo, int ts, PIO pio, uint sm, volatile void *fifo_dst,
                          uint fifo_ring_bits, uint32_t *ts_ring, uint ts_ring_bits) {
    dma_channel_config c = dma_channel_get_default_config(fifo);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, false);
    channel_config_set_write_increment(&c, fifo_ring_bits != 0);
    if (fifo_ring_bits) channel_config_set_ring(&c, true, fifo_ring_bits);
    channel_config_set_dreq(&c, pio_get_dreq(pio, sm, false));
    channel_config_set_chain_to(&c, ts);
    channel_config_set_high_priority(&c, true);
    dma_channel_configure(fifo, &c, fifo_dst, &pio->rxf[sm], 1, false);

    dma_channel_config t = dma_channel_get_default_config(ts);
    channel_config_set_transfer_data_size(&t, DMA_SIZE_32);
    channel_config_set_read_increment(&t, false);
    channel_config_set_write_increment(&t, true);
    channel_config_set_ring(&t, true, ts_ring_bits);
    channel_config_set_chain_to(&t, fifo);
    channel_config_set_high_priority(&t, true);
    dma_channel_configure(ts, &t, ts_ring, &timer1_hw->timerawl, 1, false);

    dma_channel_start(fifo);
}

static __unused void input_pin_init(PIO pio, uint pin) {
    pio_gpio_init(pio, pin);
    gpio_disable_pulls(pin);
}

static void clkdiv_start(void) {
    pio_sm_set_enabled(PIO_CAP, SM_CLK, false);
    pio_sm_clear_fifos(PIO_CAP, SM_CLK);
    pio_sm_restart(PIO_CAP, SM_CLK);
    pio_sm_exec(PIO_CAP, SM_CLK, pio_encode_jmp(clk_prog_offset));
    pio_sm_put(PIO_CAP, SM_CLK, clk_div - 1);
    pio_sm_set_enabled(PIO_CAP, SM_CLK, true);
}

static void capture_init(void) {
    // clkdiv
#if !TACHO_SELFTEST
    input_pin_init(PIO_CAP, PIN_CLK);
#endif
    clk_prog_offset = pio_add_program(PIO_CAP, &clkdiv_program);
    pio_sm_config c = clkdiv_program_get_default_config(clk_prog_offset);
    sm_config_set_in_pins(&c, PIN_CLK);
    pio_sm_set_consecutive_pindirs(PIO_CAP, SM_CLK, PIN_CLK, 1, false);
    pio_sm_init(PIO_CAP, SM_CLK, clk_prog_offset, &c);

    // edge_watch over TACHO_1..6
#if !TACHO_SELFTEST
    for (uint i = 0; i < N_TACHO; i++) input_pin_init(PIO_CAP, PIN_TACHO_BASE + i);
#endif
    uint off = pio_add_program(PIO_CAP, &edge_watch_program);
    c = edge_watch_program_get_default_config(off);
    sm_config_set_in_pins(&c, PIN_TACHO_BASE);
    sm_config_set_in_pin_count(&c, N_TACHO);
    sm_config_set_in_shift(&c, false, false, 32);
    pio_sm_set_consecutive_pindirs(PIO_CAP, SM_TACHO, PIN_TACHO_BASE, N_TACHO, false);
    pio_sm_init(PIO_CAP, SM_TACHO, off, &c);
    tacho_prev_state = (gpio_get_all() >> PIN_TACHO_BASE) & ((1u << N_TACHO) - 1);

    dma_clk_fifo = dma_claim_unused_channel(true);
    dma_clk_ts = dma_claim_unused_channel(true);
    dma_tacho_fifo = dma_claim_unused_channel(true);
    dma_tacho_ts = dma_claim_unused_channel(true);
    dma_pair_init(dma_clk_fifo, dma_clk_ts, PIO_CAP, SM_CLK, &clk_fifo_sink, 0,
                  clk_ts_ring, CLK_RING_BITS);
    dma_pair_init(dma_tacho_fifo, dma_tacho_ts, PIO_CAP, SM_TACHO, tacho_state_ring,
                  TACHO_RING_BITS, tacho_ts_ring, TACHO_RING_BITS);

    clkdiv_start();
    pio_sm_set_enabled(PIO_CAP, SM_TACHO, true);
}

static inline uint32_t ring_write_index(int ts_ch, const uint32_t *ring, uint32_t n) {
    return ((dma_hw->ch[ts_ch].write_addr - (uint32_t)(uintptr_t)ring) / 4u) & (n - 1);
}

static void service_capture(void) {
    // Read the DMA write positions first, then the clock: every entry before
    // the write position is older than `now`, which ts_extend requires.
    uint32_t clk_wr = ring_write_index(dma_clk_ts, clk_ts_ring, CLK_RING_N);
    uint32_t tacho_wr = ring_write_index(dma_tacho_ts, tacho_ts_ring, TACHO_RING_N);
    uint64_t now = now_ticks();
    uint8_t buf[16], *p;

    while (clk_rd != clk_wr) {
        uint64_t t = ts_extend(clk_ts_ring[clk_rd], now);
        clk_rd = (clk_rd + 1) & (CLK_RING_N - 1);
        p = put_u32(buf, clk_seq++);
        p = put_u64(p, t);
        link_send(PKT_CLK, buf, (uint8_t)(p - buf));
    }
    while (tacho_rd != tacho_wr) {
        uint64_t t = ts_extend(tacho_ts_ring[tacho_rd], now);
        uint32_t state = tacho_state_ring[tacho_rd] & ((1u << N_TACHO) - 1);
        tacho_rd = (tacho_rd + 1) & (TACHO_RING_N - 1);
        p = put_u32(buf, tacho_seq++);
        p = put_u64(p, t);
        p = put_u8(p, (uint8_t)state);
        p = put_u8(p, (uint8_t)(state ^ tacho_prev_state));
        tacho_prev_state = state;
        link_send(PKT_TACHO, buf, (uint8_t)(p - buf));
    }
    // edge_watch uses `push noblock`; a full FIFO drops events and sets RXSTALL.
    uint32_t stall = 1u << (PIO_FDEBUG_RXSTALL_LSB + SM_TACHO);
    if (PIO_CAP->fdebug & stall) {
        PIO_CAP->fdebug = stall;
        tacho_stalls++;
    }
}

// --- ESC telemetry ----------------------------------------------------------

static void tel_init(void) {
    uint off1 = pio_add_program(pio1, &uart_rx_program);
    uint off2 = pio_add_program(pio2, &uart_rx_program);
    float div = (float)clock_get_hz(clk_sys) / (8.0f * TEL_BAUD);
    for (int i = 0; i < N_TEL; i++) {
        PIO pio = tel_pio[i];
        uint sm = tel_sm[i], pin = tel_pins[i];
        uint off = pio == pio1 ? off1 : off2;
#if !TACHO_SELFTEST
        input_pin_init(pio, pin);
#endif
        pio_sm_set_consecutive_pindirs(pio, sm, pin, 1, false);
        pio_sm_config c = uart_rx_program_get_default_config(off);
        sm_config_set_in_pins(&c, pin);
        sm_config_set_jmp_pin(&c, pin);
        sm_config_set_in_shift(&c, true, false, 32);
        sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);
        sm_config_set_clkdiv(&c, div);
        pio_sm_init(pio, sm, off, &c);
        pio_sm_set_enabled(pio, sm, true);
    }
    // Gap that ends a chunk: 3 byte times of idle line.
    tel_gap_ticks = (uint64_t)tick_hz * 30u / TEL_BAUD;
}

static void tel_emit(int ch) {
    uint8_t buf[FRAME_MAX_PAYLOAD];
    size_t n = tel_chunk_payload(&tel[ch], (uint8_t)ch, buf);
    link_send(PKT_TEL, buf, (uint8_t)n);
    tel_chunk_reset(&tel[ch]);
}

static void service_tel(void) {
    for (int ch = 0; ch < N_TEL; ch++) {
        PIO pio = tel_pio[ch];
        uint sm = tel_sm[ch];
        uint32_t irq_bit = 1u << sm;
        if (pio->irq & irq_bit) {
            pio->irq = irq_bit;
            tel_framing[ch]++;
            tel[ch].flags |= TEL_FLAG_FRAMING_ERR;
        }
        uint32_t stall = 1u << (PIO_FDEBUG_RXSTALL_LSB + sm);
        if (pio->fdebug & stall) {
            pio->fdebug = stall;
            tel_overflow[ch]++;
            tel[ch].flags |= TEL_FLAG_OVERFLOW;
        }
        while (!pio_sm_is_rx_fifo_empty(pio, sm)) {
            uint8_t b = (uint8_t)(pio->rxf[sm] >> 24);
            uint64_t t = now_ticks();
            if (tel_chunk_needs_flush(&tel[ch], t, tel_gap_ticks)) tel_emit(ch);
            tel_chunk_add(&tel[ch], b, t);
            tel_bytes[ch]++;
        }
        if (tel_chunk_needs_flush(&tel[ch], now_ticks(), tel_gap_ticks)) tel_emit(ch);
    }
}

// --- commands from the Pi ---------------------------------------------------

static void handle_command(const frame_decoder_t *d, uint64_t t_rx) {
    uint8_t buf[16], *p;
    switch (d->type) {
    case CMD_PING:
        if (d->len < 4) return;
        p = put_u32(buf, get_u32(d->payload));
        p = put_u64(p, t_rx);
        link_send(PKT_PONG, buf, (uint8_t)(p - buf));
        break;
    case CMD_GET_INFO:
        send_info();
        break;
    case CMD_SET_CLKDIV: {
        if (d->len < 4) return;
        uint32_t div = get_u32(d->payload);
        if (div >= 1024 && div % 64 == 0) {
            clk_div = div;
            clkdiv_start();
        }
        send_info();
        break;
    }
    case CMD_BOOTSEL:
        if (d->len >= 4 && get_u32(d->payload) == BOOTSEL_MAGIC) {
            // Let the reply-less reboot happen only after pending TX is out.
            while (tx_rd != tx_wr) link_pump_tx();
            uart_tx_wait_blocking(LINK_UART);
            reset_usb_boot(0, 0);
        }
        break;
    default:
        break;
    }
}

static void service_link_rx(void) {
    uart_hw_t *hw = uart_get_hw(LINK_UART);
    while (!(hw->fr & UART_UARTFR_RXFE_BITS)) {
        uint8_t b = (uint8_t)hw->dr;
        if (frame_decoder_push(&rx_dec, b)) handle_command(&rx_dec, now_ticks());
    }
}

// --- self-test signal sources -----------------------------------------------

#if TACHO_SELFTEST
static uint selftest_tx_sm = 2;
static uint8_t st_frame[10];
static uint st_pos = sizeof(st_frame);
static uint32_t st_count;

static uint8_t crc8_kiss(const uint8_t *d, int n) {
    uint8_t crc = 0;
    for (int i = 0; i < n; i++) {
        crc ^= d[i];
        for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
    return crc;
}

static void selftest_init(void) {
    // Fake PDM_CLK: 150 MHz / 50 = 3.0 MHz on GP22 (PWM3 A)
    gpio_set_function(PIN_CLK, GPIO_FUNC_PWM);
    uint s = pwm_gpio_to_slice_num(PIN_CLK);
    pwm_set_wrap(s, 49);
    pwm_set_chan_level(s, pwm_gpio_to_channel(PIN_CLK), 25);
    pwm_set_enabled(s, true);

    // Fake tacho: GP16..21 = PWM slices 0..2, ~57/56/54 Hz, short pulses (A)
    // and long pulses (B), so every channel shows both edges.
    for (uint i = 0; i < N_TACHO; i++) gpio_set_function(PIN_TACHO_BASE + i, GPIO_FUNC_PWM);
    for (uint k = 0; k < 3; k++) {
        uint sl = pwm_gpio_to_slice_num(PIN_TACHO_BASE + 2 * k);
        pwm_set_clkdiv_int_frac4(sl, (uint8_t)(40 + k), 0);
        pwm_set_wrap(sl, 65535);
        pwm_set_both_levels(sl, 3000, 30000);
        pwm_set_enabled(sl, true);
    }

    // Fake ESC: PIO UART TX on GP26, KISS frames every 10 ms (see service)
    uint off = pio_add_program(pio2, &uart_tx_program);
    pio_sm_set_pins_with_mask(pio2, selftest_tx_sm, 1u << PIN_SELFTEST_TX, 1u << PIN_SELFTEST_TX);
    pio_sm_set_pindirs_with_mask(pio2, selftest_tx_sm, 1u << PIN_SELFTEST_TX, 1u << PIN_SELFTEST_TX);
    pio_gpio_init(pio2, PIN_SELFTEST_TX);
    pio_sm_config c = uart_tx_program_get_default_config(off);
    sm_config_set_out_shift(&c, true, false, 32);
    sm_config_set_out_pins(&c, PIN_SELFTEST_TX, 1);
    sm_config_set_sideset_pins(&c, PIN_SELFTEST_TX);
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_TX);
    sm_config_set_clkdiv(&c, (float)clock_get_hz(clk_sys) / (8.0f * TEL_BAUD));
    pio_sm_init(pio2, selftest_tx_sm, off, &c);
    pio_sm_set_enabled(pio2, selftest_tx_sm, true);
}

static void selftest_service(uint64_t now) {
    static uint64_t next;
    if (st_pos >= sizeof(st_frame) && now >= next) {
        next = now + tick_hz / 100;
        st_count++;
        st_frame[0] = 30;                                     // 30 degC
        st_frame[1] = 1680 >> 8; st_frame[2] = 1680 & 0xFF;   // 16.80 V
        st_frame[3] = 0; st_frame[4] = 123;                   // 1.23 A
        st_frame[5] = (uint8_t)(st_count >> 8); st_frame[6] = (uint8_t)st_count;  // mAh
        uint16_t erpm = (uint16_t)(400 + st_count % 100);     // eRPM/100
        st_frame[7] = (uint8_t)(erpm >> 8); st_frame[8] = (uint8_t)erpm;
        st_frame[9] = crc8_kiss(st_frame, 9);
        st_pos = 0;
    }
    while (st_pos < sizeof(st_frame) && !pio_sm_is_tx_fifo_full(pio2, selftest_tx_sm))
        pio_sm_put(pio2, selftest_tx_sm, st_frame[st_pos++]);
}
#endif

// --- main -------------------------------------------------------------------

int main(void) {
    // USB stays up only for picotool (reset interface); nothing is printed.
    stdio_init_all();

    gpio_init(PIN_LED);
    gpio_set_dir(PIN_LED, GPIO_OUT);

    uart_init(LINK_UART, LINK_BAUD);
    uart_set_format(LINK_UART, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(LINK_UART, true);
    gpio_set_function(PIN_LINK_TX, GPIO_FUNC_UART);
    gpio_set_function(PIN_LINK_RX, GPIO_FUNC_UART);
    gpio_pull_up(PIN_LINK_RX);  // idle high while the Pi UART is not set up

    frame_decoder_init(&rx_dec);
    timebase_init();
    capture_init();
    tel_init();
#if TACHO_SELFTEST
    // After tel_init: pin directions are per PIO block, and the TEL5/6
    // receivers on PIO2 would otherwise turn the PIO2 test transmitter off.
    selftest_init();
#endif

    watchdog_enable(500, true);
    send_info();

    uint64_t next_status = now_ticks() + tick_hz;
    uint64_t next_info = now_ticks() + 5ull * tick_hz;
    bool led = false;
    uint32_t loop_t0 = time_us_32();

    while (true) {
        service_capture();
        service_tel();
        service_link_rx();
#if TACHO_SELFTEST
        selftest_service(now_ticks());
#endif
        link_pump_tx();

        uint64_t now = now_ticks();
        if (now >= next_status) {
            next_status += tick_hz;
            send_status();
            led = !led;
            gpio_put(PIN_LED, led);
        }
        if (now >= next_info) {
            next_info += 5ull * tick_hz;
            send_info();
        }
        watchdog_update();

        uint32_t t1 = time_us_32();
        if (t1 - loop_t0 > loop_max_us) loop_max_us = t1 - loop_t0;
        loop_t0 = t1;
    }
}
