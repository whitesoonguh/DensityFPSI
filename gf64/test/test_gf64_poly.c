#include "gf64.h"
#include "gf64_poly.h"
#include "gf64_lin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0xdeadbeefcafef00dULL;
static uint64_t xr(void) {
    uint64_t x = st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return st = x;
}

int main(void) {
    int fails = 0;

    /* field-level sanity */
    for (int i = 0; i < 100000; i++) {
        uint64_t a = xr(), b = xr(), c = xr();
        if (gf64_mul(a, 1) != a) fails++;
        if (gf64_mul(a, b) != gf64_mul(b, a)) fails++;
        if (gf64_mul(gf64_mul(a, b), c) != gf64_mul(a, gf64_mul(b, c))) fails++;
        if (a && gf64_mul(a, gf64_inv(a)) != 1) fails++;
        if (fails) break;
    }
    printf("field core            : %s\n", fails ? "FAIL" : "ok");


    /* linear algebra / Vandermonde sanity */
    {
        const int n = 8;
        uint64_t A[n * n], Ainv[n * n], I[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) A[i * n + j] = (i == j) ? 1u : 0u;
        }
        for (int k = 0; k < 32; ++k) {
            for (int i = 0; i < n; ++i) A[i * n + i] = xr() | 1ULL;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j && (xr() & 3) == 0) A[i * n + j] = xr();
                }
            }
            if (gf64_mat_inverse(A, n, Ainv) != 0) { fails++; break; }
            for (int r = 0; r < n; ++r) {
                for (int c = 0; c < n; ++c) {
                    uint64_t acc = 0;
                    for (int k2 = 0; k2 < n; ++k2) {
                        acc ^= gf64_mul(A[r * n + k2], Ainv[k2 * n + c]);
                    }
                    I[r * n + c] = acc;
                }
            }
            int ok = 1;
            for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
                if (I[r * n + c] != (uint64_t)(r == c)) ok = 0;
            if (!ok) { fails++; break; }
        }
        printf("mat inverse           : %s\n", fails ? "FAIL" : "ok");

        uint64_t alpha[n], V[n * n], Vinv[n * n], y[n], coeff[n], y2[n];
        for (int i = 0; i < n; ++i) alpha[i] = (uint64_t)(i + 1);
        for (int i = 0; i < n; ++i) coeff[i] = xr();
        if (gf64_vandermonde_matrices(alpha, n, V, Vinv) != 0) fails++;
        gf64_matvec_rowmajor(V, coeff, n, n, y);
        gf64_matvec_rowmajor(Vinv, y, n, n, y2);
        int ok = 1;
        for (int i = 0; i < n; ++i) if (y2[i] != coeff[i]) ok = 0;
        printf("vandermonde/inverse   : %s\n", ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }

    /* polynomial cross-check at several degrees */
    int sizes[] = {4, 8, 16, 17, 31, 64, 100};
    for (int s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++) {
        int n = sizes[s], m = sizes[s], cn = n + m - 1;
        uint64_t *a = malloc((size_t)n * 8), *b = malloc((size_t)m * 8);
        for (int i = 0; i < n; i++) a[i] = xr();
        for (int i = 0; i < m; i++) b[i] = xr();

        uint64_t *cs = calloc((size_t)cn, 8), *ck = calloc((size_t)cn, 8);
        gf64_poly_mul_school(a, n, b, m, cs);
        gf64_poly_mul_kara(a, n, b, m, ck);

        int eq = memcmp(cs, ck, (size_t)cn * 8) == 0;
        int ev = 1;
        for (int t = 0; t < 1000; t++) {
            uint64_t x = xr();
            uint64_t want = gf64_mul(gf64_poly_eval(a, n, x), gf64_poly_eval(b, m, x));
            if (gf64_poly_eval(cs, cn, x) != want) { ev = 0; break; }
        }
        printf("poly n=%-4d school==kara:%s  eval:%s\n",
               n, eq ? "ok" : "FAIL", ev ? "ok" : "FAIL");
        if (!eq || !ev) fails++;
        free(a); free(b); free(cs); free(ck);
    }

    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
