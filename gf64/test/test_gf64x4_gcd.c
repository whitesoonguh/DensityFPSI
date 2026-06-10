#include "gf64.h"
#include "gf64x4.h"
#include "gf64x4_gcd.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t s; } rng_t;
static uint64_t rng_next(rng_t *r) {
    uint64_t x = (r->s += 0x9e3779b97f4a7c15ULL);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static int deg(const uint64_t *a, int n) {
    for (int i = n - 1; i >= 0; --i) if (a[i]) return i;
    return -1;
}

static void mul(const uint64_t *a, int na, const uint64_t *b, int nb, uint64_t *c) {
    memset(c, 0, (size_t)(na + nb - 1) * sizeof(uint64_t));
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < nb; ++j) c[i + j] ^= gf64_mul(a[i], b[j]);
    }
}

static int gcd_degree(const uint64_t *ain, int na, const uint64_t *bin, int nb) {
    const int cap = na > nb ? na : nb;
    uint64_t *A = calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *B = calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *R = calloc((size_t)cap, sizeof(uint64_t));
    assert(A && B && R);
    memcpy(A, ain, (size_t)na * sizeof(uint64_t));
    memcpy(B, bin, (size_t)nb * sizeof(uint64_t));
    int da = deg(A, cap), db = deg(B, cap);
    while (db >= 0) {
        memset(R, 0, (size_t)cap * sizeof(uint64_t));
        memcpy(R, A, (size_t)(da + 1) * sizeof(uint64_t));
        uint64_t inv = gf64_inv(B[db]);
        int dr = deg(R, cap);
        while (dr >= db) {
            int sh = dr - db;
            uint64_t sc = gf64_mul(R[dr], inv);
            for (int i = 0; i <= db; ++i) R[i + sh] ^= gf64_mul(sc, B[i]);
            dr = deg(R, dr + 1);
        }
        uint64_t *T = A; A = B; B = R; R = T;
        da = db; db = dr;
    }
    int ret = da;
    free(A); free(B); free(R);
    return ret;
}

static __m512i pack_coeff(const uint64_t p[4]) { return gf64x4_pack(p); }

int main(void) {
    rng_t rng = {0x123456789abcdef0ULL};
    enum { D = 16, ITERS = 2000 };
    __m512i A[D], B[D], work[3 * (D + 1)];
    uint64_t as[4][D], bs[4][D];

    /* Structured non-coprime cases: each lane gets a forced common linear factor. */
    for (int it = 0; it < ITERS; ++it) {
        memset(as, 0, sizeof(as));
        memset(bs, 0, sizeof(bs));
        for (int lane = 0; lane < 4; ++lane) {
            uint64_t root = rng_next(&rng) | 1ULL;
            uint64_t factor[2] = {root, 1}; /* X + root in characteristic two */
            uint64_t q1[D - 1], q2[D - 1];
            for (int i = 0; i < D - 1; ++i) { q1[i] = rng_next(&rng); q2[i] = rng_next(&rng); }
            q1[D - 2] |= 1ULL;
            q2[D - 2] |= 1ULL;
            mul(factor, 2, q1, D - 1, as[lane]);
            mul(factor, 2, q2, D - 1, bs[lane]);
        }
        for (int i = 0; i < D; ++i) {
            uint64_t av[4] = {as[0][i], as[1][i], as[2][i], as[3][i]};
            uint64_t bv[4] = {bs[0][i], bs[1][i], bs[2][i], bs[3][i]};
            A[i] = pack_coeff(av); B[i] = pack_coeff(bv);
        }
        int got[4] = {0,0,0,0};
        gf64x4_coprime_test(A, D, B, D, work, got);
        for (int lane = 0; lane < 4; ++lane) {
            int want = gcd_degree(as[lane], D, bs[lane], D) > 0;
            if (got[lane] != want) {
                fprintf(stderr, "structured mismatch it=%d lane=%d got=%d want=%d\n", it, lane, got[lane], want);
                return 1;
            }
            assert(want == 1);
        }
    }

    /* Mixed random cases: compare against scalar Euclidean oracle. */
    for (int it = 0; it < ITERS; ++it) {
        for (int lane = 0; lane < 4; ++lane) {
            for (int i = 0; i < D; ++i) { as[lane][i] = rng_next(&rng); bs[lane][i] = rng_next(&rng); }
            as[lane][D - 1] |= 1ULL;
            bs[lane][D - 1] |= 1ULL;
            if ((it & 15) == 0) { /* occasionally force A == B */
                memcpy(bs[lane], as[lane], sizeof(as[lane]));
            }
        }
        for (int i = 0; i < D; ++i) {
            uint64_t av[4] = {as[0][i], as[1][i], as[2][i], as[3][i]};
            uint64_t bv[4] = {bs[0][i], bs[1][i], bs[2][i], bs[3][i]};
            A[i] = pack_coeff(av); B[i] = pack_coeff(bv);
        }
        int got[4] = {0,0,0,0};
        gf64x4_coprime_test(A, D, B, D, work, got);
        for (int lane = 0; lane < 4; ++lane) {
            int want = gcd_degree(as[lane], D, bs[lane], D) > 0;
            if (got[lane] != want) {
                fprintf(stderr, "random mismatch it=%d lane=%d got=%d want=%d\n", it, lane, got[lane], want);
                return 1;
            }
        }
    }

    puts("gf64x4 gcd predicate differential tests passed");
    return 0;
}
