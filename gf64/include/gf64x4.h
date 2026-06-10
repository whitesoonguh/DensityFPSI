#ifndef GF64X4_H
#define GF64X4_H

/*
 * Batch GF(2^64) arithmetic: 4 independent field elements per ZMM register,
 * processed simultaneously with VPCLMULQDQ.
 *
 * Each 128-bit lane of a __m512i holds one element in its low 64 bits
 * (high 64 = 0). Four lanes -> four independent operations.
 *
 * REQUIRES: x86-64 with AVX-512F + AVX-512BW + VPCLMULQDQ
 *   (Intel Ice Lake-SP and later, AMD Zen 3 and later).
 * Build flags: -mavx512f -mavx512bw -mvpclmulqdq
 *
 * Same field as gf64.h: f(x) = x^64 + x^4 + x^3 + x + 1.
 * Both sides of an exchange MUST agree on this f (see gf64.h note).
 */

#include <immintrin.h>
#include <stdint.h>

#ifndef GF64_REDUCE_POLY
#define GF64_REDUCE_POLY 0x1BULL   /* low part: x^4 + x^3 + x + 1 */
#endif

#ifndef GF64X4_REDUCE_MASK
#define GF64X4_REDUCE_MASK _mm512_set_epi64(0, -1, 0, -1, 0, -1, 0, -1) /* mask for reduction; make high 64 bits zero */
#endif


/* 4 lane-wise field multiplies. Result is in the low 64 bits of each lane;
 * the high 64 bits hold scratch and must be ignored on read-out. */

/* reduction-free multiplication */
static inline __m512i gf64x4_mul_no_red(__m512i a, __m512i b) {
    return _mm512_clmulepi64_epi128(a, b, 0x00);
}

/* reduction only */
static inline __m512i gf64x4_reduce(__m512i prod) {
    const __m512i rv = _mm512_set1_epi64((long long)GF64_REDUCE_POLY);
    __m512i t  = _mm512_clmulepi64_epi128(prod, rv, 0x01);  /* hi(prod)*G */
    __m512i t2 = _mm512_clmulepi64_epi128(t,    rv, 0x01);  /* hi(t)*G */
    __m512i r  = _mm512_xor_si512(_mm512_xor_si512(prod, t), t2);
    return _mm512_and_si512(r, GF64X4_REDUCE_MASK);
}

/* High-level multiplication API */
static inline __m512i gf64x4_mul(__m512i a, __m512i b) {
    return gf64x4_reduce(gf64x4_mul_no_red(a, b));
}

/* addition in characteristic 2 is XOR */
static inline __m512i gf64x4_add(__m512i a, __m512i b) {
    return _mm512_xor_si512(a, b);
}


static inline __m512i gf64x4_zero(void) {
    return _mm512_setzero_si512();
}

static inline __m512i gf64x4_set1_u64(uint64_t x) {
    return _mm512_set1_epi64((long long)x);
}

static inline __m512i gf64x4_mul_const_no_red(uint64_t c, __m512i x) {
    return gf64x4_mul_no_red(gf64x4_set1_u64(c), x);
}

static inline __m512i gf64x4_pack4(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    return _mm512_set_epi64(0, (long long)a3, 0, (long long)a2,
                            0, (long long)a1, 0, (long long)a0);
}

/* pack 4 field elements into the low 64 bits of each lane */
static inline __m512i gf64x4_pack(const uint64_t v[4]) {
    return _mm512_set_epi64(0, (long long)v[3], 0, (long long)v[2],
                            0, (long long)v[1], 0, (long long)v[0]);
}

/* extract the 4 results from the low 64 bits of each lane */
static inline void gf64x4_unpack(__m512i z, uint64_t out[4]) {
    uint64_t buf[8];
    _mm512_storeu_si512((__m512i *)buf, z);
    out[0] = buf[0]; out[1] = buf[2]; out[2] = buf[4]; out[3] = buf[6];
}

#endif /* GF64X4_H */
