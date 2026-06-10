#include "dmpf64/fft_additive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int dense_equal(const dmpf64_dense_poly *a, const dmpf64_dense_poly *b) {
    size_t n = a->len > b->len ? a->len : b->len;
    for (size_t i = 0; i < n; i++) {
        dmpf64_gf av = i < a->len ? a->c[i] : 0;
        dmpf64_gf bv = i < b->len ? b->c[i] : 0;
        if (av != bv) return 0;
    }
    return 1;
}

static int test_eval_interp(void) {
    uint8_t seed[16]={0}; seed[0]=0x12; seed[1]=0x34; dmpf64_rng rng; dmpf64_rng_seed(&rng, seed);
    for (size_t logn = 1; logn <= 10; logn++) {
        size_t n = (size_t)1 << logn;
        dmpf64_gf *coeff = calloc(n, sizeof(dmpf64_gf));
        dmpf64_gf *evals = calloc(n, sizeof(dmpf64_gf));
        dmpf64_gf *back = calloc(n, sizeof(dmpf64_gf));
        if (!coeff || !evals || !back) return -1;
        for (size_t i = 0; i < n; i++) coeff[i] = dmpf64_rng_gf(&rng);
        if (dmpf64_afft_eval(evals, coeff, n, logn) != 0) return -1;
        if (dmpf64_afft_interpolate(back, evals, logn) != 0) return -1;
        for (size_t i = 0; i < n; i++) {
            if (coeff[i] != back[i]) {
                fprintf(stderr, "eval/interp mismatch logn=%zu i=%zu\n", logn, i);
                return -1;
            }
        }
        free(coeff); free(evals); free(back);
    }
    return 0;
}

static int test_mul(void) {
    uint8_t seed[16]={0}; seed[0]=0x56; seed[1]=0x78; dmpf64_rng rng; dmpf64_rng_seed(&rng, seed);
    for (size_t n = 1; n <= 256; n++) {
        dmpf64_dense_poly a={0}, b={0}, naive={0}, afft={0};
        if (dmpf64_dense_random(&a, n, &rng) != 0) return -1;
        if (dmpf64_dense_random(&b, n, &rng) != 0) return -1;
        if (dmpf64_dense_mul_naive(&naive, &a, &b) != 0) return -1;
        if (dmpf64_afft_mul(&afft, &a, &b) != 0) return -1;
        if (!dense_equal(&naive, &afft)) {
            fprintf(stderr, "mul mismatch n=%zu\n", n);
            return -1;
        }
        dmpf64_dense_free(&a); dmpf64_dense_free(&b); dmpf64_dense_free(&naive); dmpf64_dense_free(&afft);
    }
    return 0;
}

static int test_cyclic(void) {
    uint8_t seed[16]={0}; seed[0]=0x99; seed[1]=0x99; dmpf64_rng rng; dmpf64_rng_seed(&rng, seed);
    for (size_t logn = 1; logn <= 10; logn++) {
        size_t n = (size_t)1 << logn;
        dmpf64_dense_poly da={0}, db={0}, prod={0};
        dmpf64_ring_elem a={0}, b={0}, afft={0}, ref={0};
        if (dmpf64_dense_random(&da, n, &rng) != 0) return -1;
        if (dmpf64_dense_random(&db, n, &rng) != 0) return -1;
        dmpf64_cyclotomic_mod mod = dmpf64_cyclotomic_new(logn);
        if (dmpf64_ring_new(&a, &da, mod) != 0) return -1;
        if (dmpf64_ring_new(&b, &db, mod) != 0) return -1;
        if (dmpf64_dense_mul_naive(&prod, &a.poly, &b.poly) != 0) return -1;
        ref.mod = mod;
        if (dmpf64_dense_new(&ref.poly, n) != 0) return -1;
        for (size_t i = 0; i < prod.len; i++) ref.poly.c[i & (n - 1)] ^= prod.c[i];
        if (dmpf64_ring_mul(&afft, &a, &b) != 0) return -1;
        if (!dmpf64_ring_eq(&ref, &afft)) {
            fprintf(stderr, "cyclic/ring-backend mismatch logn=%zu\n", logn);
            return -1;
        }
        dmpf64_dense_free(&da); dmpf64_dense_free(&db); dmpf64_dense_free(&prod);
        dmpf64_ring_free(&a); dmpf64_ring_free(&b); dmpf64_ring_free(&ref); dmpf64_ring_free(&afft);
    }
    return 0;
}

int main(void) {
    if (test_eval_interp() != 0) return 1;
    if (test_mul() != 0) return 1;
    if (test_cyclic() != 0) return 1;
    printf("test_afft: passed (%s)\n", dmpf64_afft_backend());
    return 0;
}
