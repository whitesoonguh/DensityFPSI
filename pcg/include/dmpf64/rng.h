#ifndef DMPF64_RNG_H
#define DMPF64_RNG_H
#include "common.h"
#include "node.h"
#include "gf64.h"

typedef struct {
    uint64_t s[4];
} dmpf64_rng;

DMPF64_API void dmpf64_rng_seed(dmpf64_rng *rng, const uint8_t seed[16]);
DMPF64_API void dmpf64_rng_seed_from_time(dmpf64_rng *rng);
DMPF64_API uint64_t dmpf64_rng_u64(dmpf64_rng *rng);
DMPF64_API void dmpf64_rng_bytes(dmpf64_rng *rng, uint8_t *dst, size_t len);
DMPF64_API dmpf64_node dmpf64_rng_node(dmpf64_rng *rng);
DMPF64_API dmpf64_gf dmpf64_rng_gf(dmpf64_rng *rng);
DMPF64_API dmpf64_gf2 dmpf64_rng_gf2(dmpf64_rng *rng);

#endif
