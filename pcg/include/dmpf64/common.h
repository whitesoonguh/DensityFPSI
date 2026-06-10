#ifndef DMPF64_COMMON_H
#define DMPF64_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define DMPF64_BITS_OF_SECURITY 128u
#define DMPF64_BYTES_OF_SECURITY 16u

#ifndef DMPF64_API
#define DMPF64_API
#endif

static inline size_t dmpf64_ceil_div_size(size_t a, size_t b) { return (a + b - 1) / b; }
static inline size_t dmpf64_next_pow2(size_t x) {
    if (x <= 1) return 1;
    x--;
    for (size_t i = 1; i < sizeof(size_t) * 8; i <<= 1) x |= x >> i;
    return x + 1;
}
static inline unsigned dmpf64_log2_pow2(size_t x) {
    unsigned r = 0;
    while ((size_t)1 << r < x) r++;
    return r;
}

#endif
