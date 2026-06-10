#include "okvs_util.h"

#include <stdlib.h>
#include <string.h>

/* base-256 LSD radix sort of (a,b) pairs by .b, stable, single ping-pong */
void okvs_radix_sort(okvs_pair *arr, size_t n, size_t max) {
    if (n == 0) return;
    okvs_pair *tmp = (okvs_pair *)malloc(n * sizeof(okvs_pair));
    if (!tmp) return;
    okvs_pair *src = arr, *dst = tmp;
    size_t cnt[256];
    for (size_t sh = 0; (max >> sh) > 0; sh += 8) {
        for (int i = 0; i < 256; i++) cnt[i] = 0;
        for (size_t i = 0; i < n; i++) cnt[(src[i].b >> sh) & 255]++;
        size_t s = 0;
        for (int i = 0; i < 256; i++) { size_t c = cnt[i]; cnt[i] = s; s += c; }
        for (size_t i = 0; i < n; i++) {
            size_t d = (src[i].b >> sh) & 255;
            dst[cnt[d]++] = src[i];
        }
        okvs_pair *t = src; src = dst; dst = t;
    }
    if (src != arr) memcpy(arr, src, n * sizeof(okvs_pair));
    free(tmp);
}

/* ============================= BLAKE2b ==================================== */
/* Self-contained single-shot BLAKE2b (RFC 7693), no key, no tree. The output
 * length is bound into the parameter block, matching Rust's Blake2bVar. */

static const uint64_t B2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static const uint8_t B2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3}
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#define B2B_G(a, b, c, d, x, y)                 \
    do {                                        \
        v[a] = v[a] + v[b] + (x);               \
        v[d] = ROTR64(v[d] ^ v[a], 32);         \
        v[c] = v[c] + v[d];                     \
        v[b] = ROTR64(v[b] ^ v[c], 24);         \
        v[a] = v[a] + v[b] + (y);               \
        v[d] = ROTR64(v[d] ^ v[a], 16);         \
        v[c] = v[c] + v[d];                     \
        v[b] = ROTR64(v[b] ^ v[c], 63);         \
    } while (0)

static uint64_t load64(const uint8_t *p) {
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) r |= (uint64_t)p[i] << (8 * i);
    return r;
}
static void store64(uint8_t *p, uint64_t x) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(x >> (8 * i));
}

static void b2b_compress(uint64_t h[8], const uint8_t block[128],
                         uint64_t t0, uint64_t t1, int last) {
    uint64_t v[16], m[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2B_IV[i]; }
    v[12] ^= t0;
    v[13] ^= t1;
    if (last) v[14] = ~v[14];
    for (int i = 0; i < 16; i++) m[i] = load64(block + 8 * i);

    for (int r = 0; r < 12; r++) {
        const uint8_t *s = B2B_SIGMA[r];
        B2B_G(0, 4,  8, 12, m[s[0]],  m[s[1]]);
        B2B_G(1, 5,  9, 13, m[s[2]],  m[s[3]]);
        B2B_G(2, 6, 10, 14, m[s[4]],  m[s[5]]);
        B2B_G(3, 7, 11, 15, m[s[6]],  m[s[7]]);
        B2B_G(0, 5, 10, 15, m[s[8]],  m[s[9]]);
        B2B_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2B_G(2, 7,  8, 13, m[s[12]], m[s[13]]);
        B2B_G(3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

void okvs_blake2b_n(uint8_t *out, size_t n, const uint8_t *data, size_t len) {
    /* n must be in 1..64 */
    uint64_t h[8];
    memcpy(h, B2B_IV, sizeof h);
    h[0] ^= 0x01010000ULL ^ (uint64_t)n;   /* key_len = 0 */

    uint64_t t0 = 0, t1 = 0;
    uint8_t block[128];
    size_t pos = 0;

    while (len - pos > 128) {              /* all blocks except the last */
        memcpy(block, data + pos, 128);
        pos += 128;
        t0 += 128; if (t0 < 128) t1++;
        b2b_compress(h, block, t0, t1, 0);
    }
    {
        size_t rem = len - pos;            /* final block, 0..128 bytes */
        memcpy(block, data + pos, rem);
        memset(block + rem, 0, 128 - rem);
        t0 += rem; if (t0 < rem) t1++;
        b2b_compress(h, block, t0, t1, 1);
    }

    uint8_t full[64];
    for (int i = 0; i < 8; i++) store64(full + 8 * i, h[i]);
    memcpy(out, full, n);
}

void okvs_hash(uint8_t *out, size_t to, const uint8_t *data, size_t len) {
    /* Rust: hash() uses Blake2b512 (outlen 64) and *truncates*, unlike
     * blake2b<N> which binds N into the parameter block. */
    uint8_t d64[64];
    okvs_blake2b_n(d64, 64, data, len);

    if (to <= 64) { memcpy(out, d64, to); return; }

    /* to > 64: faithful port of the Rust loop. loop_count = ceil(to/64); each
     * non-final chunk is the full 64-byte digest, the final chunk is its first
     * `last_length` bytes. Since the digest is identical every iteration, this
     * is just d64 repeated to fill `to` bytes. (Behavior preserved as-is.) */
    size_t loop_count = (to + 63) / 64;
    size_t last_length = to, off = 0;
    for (size_t i = 0; i < loop_count; i++) {
        if (i == loop_count - 1) {
            memcpy(out + off, d64, last_length);
            off += last_length;
        } else {
            memcpy(out + off, d64, 64);
            off += 64;
            last_length -= 64;
        }
    }
}