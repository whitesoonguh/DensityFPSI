#include "dmpf64/rng.h"
#include <time.h>
#include <string.h>

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void dmpf64_rng_seed(dmpf64_rng *rng, const uint8_t seed[16]) {
    uint64_t x0 = 0, x1 = 0;
    memcpy(&x0, seed, 8); memcpy(&x1, seed + 8, 8);
    uint64_t x = x0 ^ (x1 + 0x9e3779b97f4a7c15ULL);
    for (int i = 0; i < 4; ++i) rng->s[i] = splitmix64(&x);
}
void dmpf64_rng_seed_from_time(dmpf64_rng *rng) {
    uint64_t t = (uint64_t)time(NULL) ^ (uintptr_t)rng;
    uint8_t seed[16]; memcpy(seed, &t, 8); t = splitmix64(&t); memcpy(seed+8, &t, 8);
    dmpf64_rng_seed(rng, seed);
}
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
uint64_t dmpf64_rng_u64(dmpf64_rng *rng) {
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0]; rng->s[3] ^= rng->s[1]; rng->s[1] ^= rng->s[2]; rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t; rng->s[3] = rotl(rng->s[3], 45);
    return result;
}
void dmpf64_rng_bytes(dmpf64_rng *rng, uint8_t *dst, size_t len) {
    while (len >= 8) { uint64_t x = dmpf64_rng_u64(rng); memcpy(dst, &x, 8); dst += 8; len -= 8; }
    if (len) { uint64_t x = dmpf64_rng_u64(rng); memcpy(dst, &x, len); }
}
dmpf64_node dmpf64_rng_node(dmpf64_rng *rng) { return (dmpf64_node){dmpf64_rng_u64(rng), dmpf64_rng_u64(rng)}; }
dmpf64_gf dmpf64_rng_gf(dmpf64_rng *rng) { return dmpf64_rng_u64(rng); }
dmpf64_gf2 dmpf64_rng_gf2(dmpf64_rng *rng) { dmpf64_gf2 z={{dmpf64_rng_u64(rng), dmpf64_rng_u64(rng)}}; return z; }
