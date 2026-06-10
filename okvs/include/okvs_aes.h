#ifndef OKVS_AES_H
#define OKVS_AES_H

/*
 * AES-128 used as the band-derivation PRF for RB-OKVS encode/decode.
 *
 * Output contract, shared by all implementations:
 *   okvs_aes_prfx2(a, data, n, out) emits 32 bytes per 64-bit word data[i]:
 *     out[i*32 +  0 .. i*32 + 15] = AES_K( LE64(data[i]) || LE64(0) )
 *     out[i*32 + 16 .. i*32 + 31] = AES_K( LE64(data[i]) || LE64(1) )
 *
 * Implementations:
 *   OKVS_AES_IMPL_VAES512 : 4 AES blocks per call with VAES-512.
 *   OKVS_AES_IMPL_AESNI   : 1 AES block per call with AES-NI, no AVX-512.
 *   OKVS_AES_IMPL_SOFTWARE/default : portable software AES-128 encrypt.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef OKVS_AES_IMPL_VAES512
#define OKVS_AES_IMPL_VAES512 0
#endif
#ifndef OKVS_AES_IMPL_AESNI
#define OKVS_AES_IMPL_AESNI 0
#endif
#ifndef OKVS_AES_IMPL_SOFTWARE
#define OKVS_AES_IMPL_SOFTWARE 0
#endif

#define OKVS_AES_CTR0 0x0000000000000000ULL
#define OKVS_AES_CTR1 0x0000000000000001ULL
#define OKVS_AES_DUMM 0x4242424242424242ULL
#define WINDOW_SIZE 4

#if OKVS_AES_IMPL_VAES512 || OKVS_AES_IMPL_AESNI
#include <immintrin.h>
#endif

static inline void okvs_store_le64(uint8_t out[8], uint64_t x) {
    for (int i = 0; i < 8; ++i) out[i] = (uint8_t)(x >> (8 * i));
}

#if OKVS_AES_IMPL_VAES512

static inline __m128i okvs__kexp(__m128i k, __m128i g) {
    g = _mm_shuffle_epi32(g, _MM_SHUFFLE(3, 3, 3, 3));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, g);
}

typedef struct { __m512i rk[11]; } okvs_aes;

static inline void okvs_aes_init(okvs_aes *a, const uint8_t key[16]) {
    __m128i rk[11];
    rk[0] = _mm_loadu_si128((const __m128i *)key);
    rk[1] = okvs__kexp(rk[0], _mm_aeskeygenassist_si128(rk[0], 0x01));
    rk[2] = okvs__kexp(rk[1], _mm_aeskeygenassist_si128(rk[1], 0x02));
    rk[3] = okvs__kexp(rk[2], _mm_aeskeygenassist_si128(rk[2], 0x04));
    rk[4] = okvs__kexp(rk[3], _mm_aeskeygenassist_si128(rk[3], 0x08));
    rk[5] = okvs__kexp(rk[4], _mm_aeskeygenassist_si128(rk[4], 0x10));
    rk[6] = okvs__kexp(rk[5], _mm_aeskeygenassist_si128(rk[5], 0x20));
    rk[7] = okvs__kexp(rk[6], _mm_aeskeygenassist_si128(rk[6], 0x40));
    rk[8] = okvs__kexp(rk[7], _mm_aeskeygenassist_si128(rk[7], 0x80));
    rk[9] = okvs__kexp(rk[8], _mm_aeskeygenassist_si128(rk[8], 0x1b));
    rk[10] = okvs__kexp(rk[9], _mm_aeskeygenassist_si128(rk[9], 0x36));

    for (int i = 0; i < 11; ++i) a->rk[i] = _mm512_broadcast_i32x4(rk[i]);
}

static inline void okvs_aes_enc4(const okvs_aes *a, __m512i m, uint8_t out[64]) {
    const __m512i *rk = a->rk;
    m = _mm512_xor_si512(m, rk[0]);
    m = _mm512_aesenc_epi128(m, rk[1]); m = _mm512_aesenc_epi128(m, rk[2]);
    m = _mm512_aesenc_epi128(m, rk[3]); m = _mm512_aesenc_epi128(m, rk[4]);
    m = _mm512_aesenc_epi128(m, rk[5]); m = _mm512_aesenc_epi128(m, rk[6]);
    m = _mm512_aesenc_epi128(m, rk[7]); m = _mm512_aesenc_epi128(m, rk[8]);
    m = _mm512_aesenc_epi128(m, rk[9]);
    m = _mm512_aesenclast_epi128(m, rk[10]);
    _mm512_storeu_si512((void *)out, m);
}

static inline void okvs_aes_prfx2(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    int i = 0;
    for (; i + 1 < ndata; i += 2) {
        const uint64_t d0 = data[i];
        const uint64_t d1 = data[i + 1];
        __m512i x = _mm512_set_epi64(
            (long long)OKVS_AES_CTR1, (long long)d1,
            (long long)OKVS_AES_CTR0, (long long)d1,
            (long long)OKVS_AES_CTR1, (long long)d0,
            (long long)OKVS_AES_CTR0, (long long)d0);
        okvs_aes_enc4(a, x, out + i * 32);
    }
    if (i < ndata) {
        const uint64_t d0 = data[i];
        uint8_t tmp[64];
        __m512i x = _mm512_set_epi64(
            (long long)OKVS_AES_CTR1, (long long)OKVS_AES_DUMM,
            (long long)OKVS_AES_CTR0, (long long)OKVS_AES_DUMM,
            (long long)OKVS_AES_CTR1, (long long)d0,
            (long long)OKVS_AES_CTR0, (long long)d0);
        okvs_aes_enc4(a, x, tmp);
        memcpy(out + i * 32, tmp, 32);
    }
}

static inline void okvs_aes_prf(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    okvs_aes_prfx2(a, data, ndata, out);
}

#elif OKVS_AES_IMPL_AESNI

static inline __m128i okvs__kexp(__m128i k, __m128i g) {
    g = _mm_shuffle_epi32(g, _MM_SHUFFLE(3, 3, 3, 3));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, g);
}

typedef struct { __m128i rk[11]; } okvs_aes;

static inline void okvs_aes_init(okvs_aes *a, const uint8_t key[16]) {
    __m128i *rk = a->rk;
    rk[0] = _mm_loadu_si128((const __m128i *)key);
    rk[1] = okvs__kexp(rk[0], _mm_aeskeygenassist_si128(rk[0], 0x01));
    rk[2] = okvs__kexp(rk[1], _mm_aeskeygenassist_si128(rk[1], 0x02));
    rk[3] = okvs__kexp(rk[2], _mm_aeskeygenassist_si128(rk[2], 0x04));
    rk[4] = okvs__kexp(rk[3], _mm_aeskeygenassist_si128(rk[3], 0x08));
    rk[5] = okvs__kexp(rk[4], _mm_aeskeygenassist_si128(rk[4], 0x10));
    rk[6] = okvs__kexp(rk[5], _mm_aeskeygenassist_si128(rk[5], 0x20));
    rk[7] = okvs__kexp(rk[6], _mm_aeskeygenassist_si128(rk[6], 0x40));
    rk[8] = okvs__kexp(rk[7], _mm_aeskeygenassist_si128(rk[7], 0x80));
    rk[9] = okvs__kexp(rk[8], _mm_aeskeygenassist_si128(rk[8], 0x1b));
    rk[10] = okvs__kexp(rk[9], _mm_aeskeygenassist_si128(rk[9], 0x36));
}

static inline void okvs_aes_block(const okvs_aes *a, __m128i m, uint8_t out[16]) {
    const __m128i *rk = a->rk;
    m = _mm_xor_si128(m, rk[0]);
    m = _mm_aesenc_si128(m, rk[1]); m = _mm_aesenc_si128(m, rk[2]);
    m = _mm_aesenc_si128(m, rk[3]); m = _mm_aesenc_si128(m, rk[4]);
    m = _mm_aesenc_si128(m, rk[5]); m = _mm_aesenc_si128(m, rk[6]);
    m = _mm_aesenc_si128(m, rk[7]); m = _mm_aesenc_si128(m, rk[8]);
    m = _mm_aesenc_si128(m, rk[9]);
    m = _mm_aesenclast_si128(m, rk[10]);
    _mm_storeu_si128((__m128i *)out, m);
}

static inline void okvs_aes_prfx2(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    for (int i = 0; i < ndata; ++i) {
        __m128i x0 = _mm_set_epi64x((long long)OKVS_AES_CTR0, (long long)data[i]);
        __m128i x1 = _mm_set_epi64x((long long)OKVS_AES_CTR1, (long long)data[i]);
        okvs_aes_block(a, x0, out + i * 32);
        okvs_aes_block(a, x1, out + i * 32 + 16);
    }
}

static inline void okvs_aes_prf(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    okvs_aes_prfx2(a, data, ndata, out);
}

#else /* software AES */

typedef struct { uint8_t rk[176]; } okvs_aes;

static const uint8_t OKVS_SBOX[256] = {
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};

static inline uint8_t okvs__xt(uint8_t x) { return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b)); }

static inline void okvs_aes_init(okvs_aes *a, const uint8_t key[16]) {
    uint8_t *rk = a->rk;
    memcpy(rk, key, 16);
    uint8_t rcon = 1;
    for (int i = 16; i < 176; i += 4) {
        uint8_t t[4];
        memcpy(t, rk + i - 4, 4);
        if (i % 16 == 0) {
            uint8_t tmp = t[0];
            t[0] = (uint8_t)(OKVS_SBOX[t[1]] ^ rcon);
            t[1] = OKVS_SBOX[t[2]];
            t[2] = OKVS_SBOX[t[3]];
            t[3] = OKVS_SBOX[tmp];
            rcon = okvs__xt(rcon);
        }
        for (int j = 0; j < 4; j++) rk[i + j] = (uint8_t)(rk[i - 16 + j] ^ t[j]);
    }
}

static inline void okvs_aes_block_bytes(const okvs_aes *a, const uint8_t in[16], uint8_t out[16]) {
    const uint8_t *rk = a->rk;
    uint8_t s[16];
    for (int i = 0; i < 16; i++) s[i] = (uint8_t)(in[i] ^ rk[i]);
    for (int r = 1; r <= 10; r++) {
        uint8_t t[16];
        for (int i = 0; i < 16; i++) t[i] = OKVS_SBOX[s[i]];
        uint8_t sh[16] = {
            t[0],  t[5],  t[10], t[15],
            t[4],  t[9],  t[14], t[3],
            t[8],  t[13], t[2],  t[7],
            t[12], t[1],  t[6],  t[11]};
        if (r < 10) {
            for (int c = 0; c < 4; c++) {
                uint8_t *col = sh + 4 * c;
                uint8_t a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
                col[0] = (uint8_t)(okvs__xt(a0) ^ (okvs__xt(a1) ^ a1) ^ a2 ^ a3);
                col[1] = (uint8_t)(a0 ^ okvs__xt(a1) ^ (okvs__xt(a2) ^ a2) ^ a3);
                col[2] = (uint8_t)(a0 ^ a1 ^ okvs__xt(a2) ^ (okvs__xt(a3) ^ a3));
                col[3] = (uint8_t)((okvs__xt(a0) ^ a0) ^ a1 ^ a2 ^ okvs__xt(a3));
            }
        }
        for (int i = 0; i < 16; i++) s[i] = (uint8_t)(sh[i] ^ rk[16 * r + i]);
    }
    memcpy(out, s, 16);
}

static inline void okvs_aes_prfx2(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    for (int i = 0; i < ndata; ++i) {
        uint8_t in[16];
        okvs_store_le64(in, data[i]);
        okvs_store_le64(in + 8, OKVS_AES_CTR0);
        okvs_aes_block_bytes(a, in, out + i * 32);
        okvs_store_le64(in + 8, OKVS_AES_CTR1);
        okvs_aes_block_bytes(a, in, out + i * 32 + 16);
    }
}

static inline void okvs_aes_prf(const okvs_aes *a, const uint64_t *data, int ndata, uint8_t *out) {
    okvs_aes_prfx2(a, data, ndata, out);
}

#endif

#endif /* OKVS_AES_H */
