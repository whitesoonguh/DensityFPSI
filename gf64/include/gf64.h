#ifndef GF64_H
#define GF64_H

/*
 * GF(2^64) field arithmetic.
 *
 * Field: GF(2)[x] / (f),  f(x) = x^64 + x^4 + x^3 + x + 1
 * The "low part" of f is r(x) = x^4 + x^3 + x + 1 = 0x1B.
 *
 * Elements are packed into a uint64_t: bit i = coefficient of x^i.
 * Multiplication uses the PCLMULQDQ (carry-less multiply) instruction.
 *
 * Build requirement: x86-64 with -mpclmul -msse4.1 (CLMUL + _mm_extract_epi64).
 *
 * NOTE on interop: the choice of irreducible f below is a common sparse
 * pentanomial, but it is NOT a universal standard. If you exchange field
 * elements with another system/party (e.g. in MPC), both sides MUST agree
 * on the same f, otherwise representations are incompatible.
 */

#include <immintrin.h>
#include <stdint.h>

/* low part of the reduction polynomial: x^4 + x^3 + x + 1 */
#define GF64_REDUCE_POLY 0x1BULL

/* reduce a 128-bit carry-less product modulo x^64 + x^4 + x^3 + x + 1. */
static inline uint64_t gf64_reduce_words(uint64_t lo, uint64_t hi) {
    /* Since x^64 == x^4 + x^3 + x + 1, fold hi*x^64 by shifts.
     * hi*(x^4+x^3+x+1) has degree <= 67, so a second small fold handles
     * the overflow bits from hi<<{1,3,4}.  This is the same reduction that
     * the original PCG backend used, now centralized in gf64. */
    uint64_t r = lo;
    r ^= hi;
    r ^= hi << 1;
    r ^= hi << 3;
    r ^= hi << 4;

    uint64_t c = 0;
    c ^= hi >> 63;  /* carry from hi << 1 */
    c ^= hi >> 61;  /* carry from hi << 3 */
    c ^= hi >> 60;  /* carry from hi << 4 */

    r ^= c;
    r ^= c << 1;
    r ^= c << 3;
    r ^= c << 4;
    return r;
}

/* multiply two field elements */
static inline uint64_t gf64_mul(uint64_t a, uint64_t b) {
    /* 64x64 -> 127-bit carry-less product */
    __m128i prod = _mm_clmulepi64_si128(_mm_set_epi64x(0, (long long)a),
                                        _mm_set_epi64x(0, (long long)b), 0x00);
    uint64_t lo = (uint64_t)_mm_cvtsi128_si64(prod);
    uint64_t hi = (uint64_t)_mm_extract_epi64(prod, 1);
    return gf64_reduce_words(lo, hi);
}

/* reduction-free multiplication */
static inline __m128i gf64_mul_no_red(uint64_t a, uint64_t b) {
    return _mm_clmulepi64_si128(_mm_set_epi64x(0, (long long)a),
                                _mm_set_epi64x(0, (long long)b), 0x00);
}

/* reduction only */
static inline uint64_t gf64_reduce(__m128i prod) {
    uint64_t lo = (uint64_t)_mm_cvtsi128_si64(prod);
    uint64_t hi = (uint64_t)_mm_extract_epi64(prod, 1);
    return gf64_reduce_words(lo, hi);
}

/* addition/subtraction in characteristic 2 is XOR */
static inline uint64_t gf64_add(uint64_t a, uint64_t b) { return a ^ b; }
static inline uint64_t gf64_sub(uint64_t a, uint64_t b) { return a ^ b; }

static inline uint64_t gf64_square(uint64_t a) { return gf64_mul(a, a); }

/* a^e by square-and-multiply */
static inline uint64_t gf64_pow(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    while (e) { if (e & 1) r = gf64_mul(r, a); a = gf64_square(a); e >>= 1; }
    return r;
}

/* multiplicative inverse via Fermat: a^(2^64 - 2). Returns 0 for a == 0. */
static inline uint64_t gf64_inv(uint64_t a) {
    if (a == 0) return 0;
    /* Fixed exponent 111...110: 64 squarings and 63 multiplies, matching the
     * original PCG inverse schedule but using the centralized gf64 backend. */
    uint64_t r = 1;
    for (int i = 63; i >= 0; --i) {
        r = gf64_square(r);
        if (i != 0) r = gf64_mul(r, a);
    }
    return r;
}

#endif /* GF64_H */
