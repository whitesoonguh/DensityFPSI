#include "dmpf64/ole_pcg_backend.h"

#include <stdio.h>
#include <stdlib.h>

static int verify_forced_bigstate_eval(dmpf64_backend requested) {
    uint8_t seed[16] = {1,3,3,7,9,9,2,4,6,8,10,12,14,16,18,20};
    dmpf64_rng rng;
    dmpf64_rng_seed(&rng, seed);
    dmpf64_ole_seed_regular_any s0, s1;
    if (dmpf64_ole_gen_regular_backend(&s0, &s1, 8, 2, 3, requested, 8, &rng) != 0) {
        fprintf(stderr, "gen_regular_backend failed for requested %s\n", dmpf64_backend_name(requested));
        return 1;
    }
    if (s0.backend != DMPF64_BACKEND_BIG_STATE || s1.backend != DMPF64_BACKEND_BIG_STATE) {
        fprintf(stderr, "backend was not forced to BIG_STATE\n");
        return 1;
    }
    dmpf64_ring_elem m0 = {0}, a0 = {0}, m1 = {0}, a1 = {0};
    if (dmpf64_ole_expand_regular_backend_eval(&s0, &m0, &a0) != 0 ||
        dmpf64_ole_expand_regular_backend_eval(&s1, &m1, &a1) != 0) {
        fprintf(stderr, "eval expand failed for requested %s\n", dmpf64_backend_name(requested));
        return 1;
    }
    if (!dmpf64_ole_eval_relation_ok(&m0, &a0, &m1, &a1)) {
        fprintf(stderr, "eval OLE relation failed for requested %s\n", dmpf64_backend_name(requested));
        return 1;
    }
    dmpf64_ring_free(&m0);
    dmpf64_ring_free(&m1);
    dmpf64_ring_free(&a0);
    dmpf64_ring_free(&a1);
    dmpf64_ole_seed_regular_any_free(&s0);
    dmpf64_ole_seed_regular_any_free(&s1);
    return 0;
}

int main(void) {
    if (verify_forced_bigstate_eval(DMPF64_BACKEND_DPF)) return 1;
    if (verify_forced_bigstate_eval(DMPF64_BACKEND_BIG_STATE)) return 1;
    printf("test_ole_backend: forced bigstate-eval passed\n");
    return 0;
}
