#ifndef OKVS_UTIL_H
#define OKVS_UTIL_H

/*
 * Encode-side utilities for RB-OKVS, independent of the solver:
 *   - radix_sort: sort rows by band start position (the sort step of encode).
 *   - BLAKE2b / hash: derive band start (H1) and band contents (H2) from keys.
 *
 * (Split out of the old okvs.h; the U256 solver path was removed in favor of
 * the width-parameterized okvs_w.h.)
 */

#include <stddef.h>
#include <stdint.h>

/* radix sort of (a,b) pairs by .b (base-10 LSD) */
typedef struct { size_t a, b; } okvs_pair;
void okvs_radix_sort(okvs_pair *arr, size_t n, size_t max);
void okvs_radix_sort_u16(okvs_pair *arr, size_t n);

/* BLAKE2b with output length n in 1..64 (output length bound into the
 * parameter block; == Rust Blake2bVar::new(n)). */
void okvs_blake2b_n(uint8_t *out, size_t n, const uint8_t *data, size_t len);

/* hash(data, to): to<=64 -> BLAKE2b-512(data)[0..to]; to>64 -> the 64-byte
 * digest repeated to fill `to` bytes (matches the reference behavior). */
void okvs_hash(uint8_t *out, size_t to, const uint8_t *data, size_t len);

#endif /* OKVS_UTIL_H */
