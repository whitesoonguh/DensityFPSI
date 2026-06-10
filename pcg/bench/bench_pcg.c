#include "dmpf64/ole_pcg_backend.h"
#include "dmpf64/prg.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const size_t logn = argc > 1 ? (size_t)strtoull(argv[1], 0, 10) : 20;
    const size_t c    = argc > 2 ? (size_t)strtoull(argv[2], 0, 10) : 4;
    const size_t w    = argc > 3 ? (size_t)strtoull(argv[3], 0, 10) : 3;
    const size_t reps = argc > 4 ? (size_t)strtoull(argv[4], 0, 10) : 1;
    const size_t batch = argc > 5 ? (size_t)strtoull(argv[5], 0, 10) : 8;

    uint8_t seed[16] = {7,6,5,4,3,2,1};
    dmpf64_rng rng;
    dmpf64_rng_seed(&rng, seed);

    printf("GF(2^64) Random-OLE PCG throughput\n");
    printf("  model       : BIG_STATE + eval-domain expansion (forced)\n");
    printf("  gf64 backend: %s\n  prg backend : %s\n", dmpf64_gf_backend(), dmpf64_prg_backend());
    printf("  logN=%zu compression=%zu weight=%zu reps=%zu batch=%zu\n", logn, c, w, reps, batch);

    double keygen_total = 0.0, exp0_total = 0.0, exp1_total = 0.0;
    const size_t N = (size_t)1 << logn;
    int ok_all = 1;

    for (size_t r = 0; r < reps; ++r) {
        dmpf64_ole_seed_regular_any s0, s1;
        double t0 = now_sec();
        int rc = dmpf64_ole_gen_regular_backend(&s0, &s1, logn, c, w,
                                                DMPF64_BACKEND_BIG_STATE,
                                                batch, &rng);
        double t1 = now_sec();
        if (rc != 0) {
            fprintf(stderr, "gen_regular_backend failed\n");
            return 1;
        }

        dmpf64_ring_elem m0 = {0}, a0 = {0}, m1 = {0}, a1 = {0};
        double e0 = now_sec();
        rc = dmpf64_ole_expand_regular_backend_eval(&s0, &m0, &a0);
        double e1 = now_sec();
        rc |= dmpf64_ole_expand_regular_backend_eval(&s1, &m1, &a1);
        double e2 = now_sec();
        if (rc != 0) {
            fprintf(stderr, "eval expand failed\n");
            return 2;
        }
        const int ok = dmpf64_ole_eval_relation_ok(&m0, &a0, &m1, &a1);
        ok_all &= ok;

        keygen_total += t1 - t0;
        exp0_total += e1 - e0;
        exp1_total += e2 - e1;

        dmpf64_ring_free(&m0);
        dmpf64_ring_free(&m1);
        dmpf64_ring_free(&a0);
        dmpf64_ring_free(&a1);
        dmpf64_ole_seed_regular_any_free(&s0);
        dmpf64_ole_seed_regular_any_free(&s1);
    }

    const double keygen = keygen_total / (double)reps;
    const double exp0 = exp0_total / (double)reps;
    const double exp1 = exp1_total / (double)reps;
    const double expand_serial = exp0 + exp1;
    const double expand_parallel = exp0 > exp1 ? exp0 : exp1;

    printf("  keygen_sec_avg              : %.6f\n", keygen);
    printf("  expand_party0_sec_avg       : %.6f\n", exp0);
    printf("  expand_party1_sec_avg       : %.6f\n", exp1);
    printf("  expand_serial_sec_avg       : %.6f\n", expand_serial);
    printf("  expand_parallel_sec_avg     : %.6f\n", expand_parallel);
    printf("  expand_serial_MOLE_per_s    : %.3f\n", (double)N / expand_serial * 1e-6);
    printf("  expand_parallel_MOLE_per_s  : %.3f\n", (double)N / expand_parallel * 1e-6);
    printf("  relation                    : %s\n", ok_all ? "ok" : "FAIL");
    return ok_all ? 0 : 3;
}
