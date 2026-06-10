#ifndef GF64X4_POLY_H
#define GF64X4_POLY_H

/*
 * Batch of 4 polynomials over GF(2^64), multiplied simultaneously.
 *
 * Layout (SoA): a polynomial batch of length n is a __m512i array a[0..n-1],
 * where lane L (low 64 bits) of a[i] is the X^i coefficient of the L-th
 * polynomial (L = 0..3). The product batch has na + nb - 1 terms.
 *
 * Use this only when you genuinely have 4 independent products to do at
 * once; otherwise the scalar gf64_poly_mul_school is the right tool.
 *
 * The output buffer c is accumulated with XOR, so the caller MUST zero it
 * (memset) before calling.
 */

#include <immintrin.h>

/*
 * Blocked multiply -- the main public API. Writes c (it zeros c internally),
 * so c does NOT need pre-zeroing; it only needs room for na + nb - 1 elements.
 * Uses register-resident 2x2 / 4x4 / 8x8 kernels with a schoolbook fallback
 * for off-size blocks and tails.
 */
void gf64x4_poly_mul(const __m512i *a, int na,
                     const __m512i *b, int nb,
                     __m512i *c);

/* Helper function for accumulating polynomial multiplication 
* It is specifically useful for doing an product between polynomials.
*/                     
void gf64x4_poly_mul_acc(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb,
    __m512i *__restrict c);

/*
 * Schoolbook multiply. ACCUMULATES into c with XOR, so the caller MUST zero c
 * (memset) before calling. Kept as a reference / cross-check for gf64x4_poly_mul.
 */
void gf64x4_poly_mul_school(const __m512i *a, int na,
                            const __m512i *b, int nb,
                            __m512i *c);

static inline void gf64x4_poly_zero(__m512i *c, int n){
    for (int i = 0; i < n; i++) c[i] = _mm512_setzero_si512();
}

#endif /* GF64X4_POLY_H */
