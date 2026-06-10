#include "dmpf64/ole_pcg_backend.h"

#include <stdio.h>

int main(void) {
    uint8_t seed[16] = {42};
    dmpf64_rng rng;
    dmpf64_rng_seed(&rng, seed);

    dmpf64_ole_seed_regular_any s0, s1;
    if (dmpf64_ole_gen_regular_backend(&s0, &s1, 8, 2, 3,
                                       DMPF64_BACKEND_BIG_STATE, 8, &rng) != 0) {
        fprintf(stderr, "bigstate-eval gen failed\n");
        return 1;
    }

    dmpf64_ring_elem m0 = {0}, a0 = {0}, m1 = {0}, a1 = {0};
    if (dmpf64_ole_expand_regular_backend_eval(&s0, &m0, &a0) != 0 ||
        dmpf64_ole_expand_regular_backend_eval(&s1, &m1, &a1) != 0) {
        fprintf(stderr, "bigstate-eval expand failed\n");
        return 2;
    }

    const int ok = dmpf64_ole_eval_relation_ok(&m0, &a0, &m1, &a1);
    dmpf64_ring_free(&m0);
    dmpf64_ring_free(&m1);
    dmpf64_ring_free(&a0);
    dmpf64_ring_free(&a1);
    dmpf64_ole_seed_regular_any_free(&s0);
    dmpf64_ole_seed_regular_any_free(&s1);

    if (!ok) {
        fprintf(stderr, "bigstate-eval relation failed\n");
        return 3;
    }
    printf("test_ole: bigstate-eval passed\n");
    return 0;
}
