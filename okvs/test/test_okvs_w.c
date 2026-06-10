#include "okvs_w.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0xcafef00dd15ea5e5ULL;
static uint64_t xr(void) { uint64_t x = st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return st = x; }

static int cmp_sz(const void *a, const void *b) {
    size_t x = *(const size_t *)a, y = *(const size_t *)b; return (x > y) - (x < y);
}

static int band_is_zero_helper(const uint64_t *band, size_t bw) {
    for (size_t t = 0; t < bw; t++) if (band[t]) return 0;
    return 1;
}
/* random w-bit band into bw words */
static void rand_band(uint64_t *band, size_t bw, size_t w) {
    for (size_t t = 0; t < bw; t++) band[t] = xr();
    size_t rem = w & 63;
    if (rem) band[bw - 1] &= (rem == 64) ? ~0ULL : ((1ULL << rem) - 1);
    if (band_is_zero_helper(band, bw)) band[0] |= 1; /* never all-zero */
}

/* build a consistent system (y from a real x*) and solve; verify all rows. */
static int roundtrip(size_t n, size_t w, double eps, uint64_t *seed_or_null) {
    size_t bw = okvs_w_words(w);
    size_t m = (size_t)(n * (1.0 + eps)) + w;

    int attempts = 0, ok = 0;
    while (attempts++ < 30 && !ok) {
        uint64_t *bands = malloc(n * bw * sizeof(uint64_t));
        size_t *sp = malloc(n * sizeof(size_t));
        uint64_t *xstar = malloc(m * sizeof(uint64_t));
        uint64_t *y = malloc(n * sizeof(uint64_t));
        uint64_t *x = malloc(m * sizeof(uint64_t));

        for (size_t i = 0; i < n; i++) sp[i] = xr() % (m - w);
        qsort(sp, n, sizeof(size_t), cmp_sz);
        for (size_t i = 0; i < n; i++) rand_band(bands + i * bw, bw, w);
        for (size_t i = 0; i < m; i++) xstar[i] = xr();
        for (size_t i = 0; i < n; i++)
            y[i] = okvs_w_inner_product(bands + i * bw, bw, xstar + sp[i]);

        uint64_t *bands2 = malloc(n * bw * sizeof(uint64_t));
        uint64_t *y2 = malloc(n * sizeof(uint64_t));
        memcpy(bands2, bands, n * bw * sizeof(uint64_t));
        memcpy(y2, y, n * sizeof(uint64_t));

        int rc = okvs_w_simple_gauss(y2, bands2, bw, sp, n, m, x, seed_or_null);
        if (rc == 0) {
            ok = 1;
            for (size_t i = 0; i < n; i++)
                if (okvs_w_inner_product(bands + i * bw, bw, x + sp[i]) != y[i]) { ok = 0; break; }
        }
        free(bands); free(sp); free(xstar); free(y); free(x); free(bands2); free(y2);
    }
    return ok;
}

/* free-var randomization: same system, two seeds -> different encodings, both
 * satisfying the equations. */
static int test_oblivious(void) {
    size_t n = 4096, w = 412, bw = okvs_w_words(w);
    double eps = 0.05; size_t m = (size_t)(n * (1.0 + eps)) + w;
    int ok = 0, attempts = 0;

    while (attempts++ < 30 && !ok) {
        uint64_t *bands = malloc(n * bw * sizeof(uint64_t));
        size_t *sp = malloc(n * sizeof(size_t));
        uint64_t *y = malloc(n * sizeof(uint64_t));
        for (size_t i = 0; i < n; i++) sp[i] = xr() % (m - w);
        qsort(sp, n, sizeof(size_t), cmp_sz);
        for (size_t i = 0; i < n; i++) rand_band(bands + i * bw, bw, w);
        for (size_t i = 0; i < n; i++) y[i] = xr();

        uint64_t *bA = malloc(n * bw * 8), *bB = malloc(n * bw * 8);
        uint64_t *yA = malloc(n * 8), *yB = malloc(n * 8);
        uint64_t *xA = malloc(m * 8), *xB = malloc(m * 8);
        memcpy(bA, bands, n * bw * 8); memcpy(yA, y, n * 8);
        memcpy(bB, bands, n * bw * 8); memcpy(yB, y, n * 8);

        uint64_t sA = 0x1111111111111111ULL, sB = 0x2222222222222222ULL;
        int rcA = okvs_w_simple_gauss(yA, bA, bw, sp, n, m, xA, &sA);
        int rcB = okvs_w_simple_gauss(yB, bB, bw, sp, n, m, xB, &sB);
        if (rcA == 0 && rcB == 0) {
            ok = 1;
            /* both satisfy the equations (against the ORIGINAL bands/y) */
            for (size_t i = 0; i < n; i++) {
                if (okvs_w_inner_product(bands + i * bw, bw, xA + sp[i]) != y[i]) ok = 0;
                if (okvs_w_inner_product(bands + i * bw, bw, xB + sp[i]) != y[i]) ok = 0;
                if (!ok) break;
            }
            /* different encodings (free vars differ) */
            int differ = memcmp(xA, xB, m * 8) != 0;
            if (!differ) ok = 0;
        }
        free(bands); free(sp); free(y);
        free(bA); free(bB); free(yA); free(yB); free(xA); free(xB);
    }
    printf("free-var randomization (double-obliv, w=412) : %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

/* vec at w=412: round-trip + lane equivalence vs scalar per coordinate */
static int test_vec(void) {
    size_t n = 2048, w = 412, bw = okvs_w_words(w), L = 19;
    double eps = 0.05; size_t m = (size_t)(n * (1.0 + eps)) + w;
    int ok = 0, lane = 1, attempts = 0;

    while (attempts++ < 30 && !ok) {
        uint64_t *bands = malloc(n * bw * 8);
        size_t *sp = malloc(n * sizeof(size_t));
        uint64_t *xstar = malloc(m * L * 8);
        uint64_t *y = malloc(n * L * 8);
        uint64_t *x = malloc(m * L * 8);
        for (size_t i = 0; i < n; i++) sp[i] = xr() % (m - w);
        qsort(sp, n, sizeof(size_t), cmp_sz);
        for (size_t i = 0; i < n; i++) rand_band(bands + i * bw, bw, w);
        for (size_t i = 0; i < m * L; i++) xstar[i] = xr();
        for (size_t i = 0; i < n; i++)
            okvs_w_inner_product_vec(bands + i * bw, bw, xstar + sp[i] * L, L, y + i * L);

        uint64_t *b2 = malloc(n * bw * 8), *y2 = malloc(n * L * 8);
        memcpy(b2, bands, n * bw * 8); memcpy(y2, y, n * L * 8);
        int rc = okvs_w_simple_gauss_vec(y2, b2, bw, sp, n, m, L, x, NULL);
        if (rc == 0) {
            ok = 1;
            uint64_t res[64];
            for (size_t i = 0; i < n && ok; i++) {
                okvs_w_inner_product_vec(bands + i * bw, bw, x + sp[i] * L, L, res);
                if (memcmp(res, y + i * L, L * 8) != 0) ok = 0;
            }
            for (size_t c = 0; c < L && lane; c++) {
                uint64_t *bc = malloc(n * bw * 8), *yc = malloc(n * 8), *xc = malloc(m * 8);
                memcpy(bc, bands, n * bw * 8);
                for (size_t i = 0; i < n; i++) yc[i] = y[i * L + c];
                int rc2 = okvs_w_simple_gauss(yc, bc, bw, sp, n, m, xc, NULL);
                if (rc2 != 0) lane = 0;
                for (size_t col = 0; col < m && lane; col++)
                    if (xc[col] != x[col * L + c]) lane = 0;
                free(bc); free(yc); free(xc);
            }
        }
        free(bands); free(sp); free(xstar); free(y); free(x); free(b2); free(y2);
    }
    printf("vec round-trip (w=412, L=19, AVX-512)        : %s\n", ok ? "ok" : "FAIL");
    printf("vec == scalar per-coordinate (lane check)    : %s\n", lane ? "ok" : "FAIL");
    return (ok && lane) ? 0 : 1;
}

int main(void) {
    int fails = 0;
    size_t ws[] = {128, 218, 412, 662};   /* incl. the 2^24 widths U256 can't hold */
    for (size_t i = 0; i < 4; i++) {
        int ok = roundtrip(8192, ws[i], 0.05, NULL);
        printf("round-trip w=%-4zu (bw=%zu words)               : %s\n",
               ws[i], okvs_w_words(ws[i]), ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    /* round-trip with random free vars too */
    {
        uint64_t seed = 0xabcdef;
        int ok = roundtrip(8192, 412, 0.05, &seed);
        printf("round-trip w=412 with random free vars       : %s\n", ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    fails += test_oblivious();
    fails += test_vec();
    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
