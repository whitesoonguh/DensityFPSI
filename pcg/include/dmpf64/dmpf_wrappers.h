#ifndef DMPF64_DMPF_WRAPPERS_H
#define DMPF64_DMPF_WRAPPERS_H
#include "dpf.h"
#include "big_state.h"

/* Unsupported Rust backends fail explicitly. */
typedef struct { int not_implemented; } dmpf64_batch_code_key;
DMPF64_API int dmpf64_batch_code_gen(dmpf64_batch_code_key *k0, dmpf64_batch_code_key *k1, size_t input_bits, const dmpf64_kv *inputs, size_t count, dmpf64_rng *rng);

#endif
