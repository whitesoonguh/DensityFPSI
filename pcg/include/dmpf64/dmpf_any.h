#ifndef DMPF64_DMPF_ANY_H
#define DMPF64_DMPF_ANY_H
#include "dpf.h"
#include "big_state.h"

typedef enum {
    DMPF64_BACKEND_DPF = 0,
    DMPF64_BACKEND_BIG_STATE = 1,
    DMPF64_BACKEND_BATCH_CODE = 2
} dmpf64_backend;

typedef struct {
    dmpf64_backend backend;
    union {
        dmpf64_dpf_dmpf_key dpf;
        dmpf64_big_state_key big;
    } u;
} dmpf64_any_key;

DMPF64_API const char *dmpf64_backend_name(dmpf64_backend b);
DMPF64_API void dmpf64_any_key_init_empty(dmpf64_any_key *k, dmpf64_backend b);
DMPF64_API void dmpf64_any_key_free(dmpf64_any_key *k);
DMPF64_API int dmpf64_any_gen(dmpf64_any_key *k0, dmpf64_any_key *k1, dmpf64_backend backend,
                              size_t input_bits, const dmpf64_kv *inputs, size_t count,
                              dmpf64_rng *rng, size_t batch_size);
DMPF64_API int dmpf64_any_eval_all(const dmpf64_any_key *k, dmpf64_gf2 **out, size_t *out_len);
DMPF64_API size_t dmpf64_any_input_length(const dmpf64_any_key *k);
DMPF64_API size_t dmpf64_any_point_count(const dmpf64_any_key *k);

#endif
