#include "gf64x4_gcd.h"

#include <stddef.h>
#include <stdint.h>

/*
 * 4-way polynomial gcd-degree predicate over GF(2^64), using the
 * Bernstein-Yang / divstep-style constant-iteration update from the original
 * codebase.
 *
 * Representation:
 *   - a[i], b[i] are coefficient-i batches in the ordinary coefficient frame.
 *   - each __m512i holds 4 independent GF64 elements in the low 64 bits of its
 *     128-bit lanes; the high words are scratch/zero and ignored.
 *
 * Algorithm:
 *   The code works in the reversed frame.  Let f and g be the reversed current
 *   polynomials.  One divstep is
 *
 *       if delta > 0 and g(0) != 0:  f <- g, delta <- 1 - delta
 *       else:                       f unchanged, delta <- 1 + delta
 *       g <- (f_old(0) g_old + g_old(0) f_old) / X
 *
 *   Over characteristic two, subtraction equals addition.  The formula preserves
 *   the gcd up to a nonzero scalar.  After 2*d0+2 iterations, f contains the gcd
 *   in the reversed frame for the equal-degree protocol inputs used by S^3I.
 *   We only need the predicate deg(gcd)>0, so we test whether any coefficient of
 *   f above index 0 is nonzero.
 *
 * Important bug fix versus the previous BY sketch:
 *   gcur alternates between two scratch buffers, but the old code read g0 from a
 *   fixed pointer `g[0]`.  After the first iteration, divsteps used a stale g(0),
 *   causing false negatives on structured non-coprime inputs.  This version reads
 *   g0 from gcur[0], the same current buffer used for the remaining coefficients.
 */
void gf64x4_coprime_test(const __m512i *a, int na, const __m512i *b, int nb,
                         __m512i *work, int deg_gt0[4]) {
    if (!deg_gt0) return;
    for (int lane = 0; lane < 4; ++lane) deg_gt0[lane] = 0;
    if (!a || !b || !work || na <= 0 || nb <= 0) return;

    const __m512i zero = _mm512_setzero_si512();

    const __m512i *R0, *R1;
    int d0, d1;
    if (na >= nb) {
        R0 = a; d0 = na - 1;
        R1 = b; d1 = nb - 1;
    } else {
        R0 = b; d0 = nb - 1;
        R1 = a; d1 = na - 1;
    }

    const int Lc = d0 + 2;
    const int N  = 2 * d0 + 2;

    __m512i *f     = work;
    __m512i *gcur  = work + Lc;
    __m512i *gnext = work + 2 * Lc;

    for (int i = 0; i < Lc; ++i) {
        f[i] = (i <= d0) ? R0[d0 - i] : zero;
        const int k = d0 - i;
        gcur[i] = (i <= d0 && k >= 0 && k <= d1) ? R1[k] : zero;
        gnext[i] = zero;
    }

    int delta[4] = {1, 1, 1, 1};

    for (int s = 0; s < N; ++s) {
        const __m512i f0 = f[0];
        const __m512i g0 = gcur[0];

        uint64_t gb[8];
        _mm512_storeu_si512((__m512i *)gb, g0);

        __mmask8 swap = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const int do_swap = (delta[lane] > 0) && (gb[2 * lane] != 0);
            if (do_swap) swap |= (__mmask8)(0x3u << (2 * lane));
            delta[lane] = do_swap ? (1 - delta[lane]) : (1 + delta[lane]);
        }

        for (int i = 0; i < Lc - 1; ++i) {
            const __m512i lhs = gf64x4_mul_no_red(f0, gcur[i + 1]);
            const __m512i rhs = gf64x4_mul_no_red(g0, f[i + 1]);
            gnext[i] = gf64x4_reduce(gf64x4_add(lhs, rhs));
        }
        gnext[Lc - 1] = zero;

        for (int i = 0; i < Lc; ++i) {
            f[i] = _mm512_mask_blend_epi64(swap, f[i], gcur[i]);
        }

        __m512i *tmp = gcur;
        gcur = gnext;
        gnext = tmp;
    }

    __m512i acc = zero;
    for (int i = 1; i < Lc; ++i) acc = _mm512_or_si512(acc, f[i]);

    const __mmask8 m = _mm512_test_epi64_mask(acc, acc);
    for (int lane = 0; lane < 4; ++lane) {
        deg_gt0[lane] = (int)((m >> (2 * lane)) & 1u);
    }
}
