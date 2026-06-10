#ifndef OKVS_ENCODE_H
#define OKVS_ENCODE_H

/*
 * RB-OKVS encode / decode on top of the okvs_w solver.
 *
 * A value is L words of GF(2^64): L=1 is a single field element; L>1 is a
 * vector or, equivalently, the coefficients of a degree-(L-1) polynomial --
 * encode/decode treat values as an XOR group (polynomial addition = coeff-wise
 * XOR), which is all RB-OKVS needs. Field/polynomial MULTIPLICATION is not part
 * of the data structure; do it in the gf64 layer before encoding or after
 * decoding (e.g. Horner evaluation of a decoded polynomial).
 *
 * Each key is hashed (BLAKE2b, counter-blocked, domain-separated by `seed`) to
 * a band start position in [0, cols-w] and a w-bit band. encode sorts rows by
 * start (radix) and solves; decode re-derives (start, band) and takes one
 * inner product. Inserted keys decode back to their value exactly.
 */

#include <stddef.h>
#include <stdint.h>

#include "okvs_w.h"
#include "okvs_util.h"

#ifdef __cplusplus
extern "C" {
#define OKVS_ALIGNAS(N) alignas(N)
#else
#define OKVS_ALIGNAS(N) _Alignas(N)
#endif

/* table size (columns) for n items, band width w, expansion eps (e.g. 0.05) */
static inline size_t okvs_w_cols(size_t n, size_t w, double eps) {
    return (size_t)((double)n * (1.0 + eps)) + w;
}

/*
 * Encode n (key, value) pairs into x_out.
 *   keys     : n records of key_len bytes, contiguous (keys[i] = keys + i*key_len)
 *   values   : n*L words, row-major (value i = values + i*L)
 *   w        : band width;  L : words per value (L>=1)
 *   cols     : table size, must be >= okvs_w_cols(n, w, eps) used at decode
 *   seed     : domain-separation bytes (may be NULL if seed_len==0)
 *   x_out    : cols*L words (output encoding)
 *   prng_state: non-NULL seeded -> free entries randomized (double-oblivious);
 *               NULL -> free entries 0 (faster, NOT oblivious)
 * Returns 0 ok, >0 (row+1) on ZeroRow (re-seed and retry), -1 on alloc failure.
 */
int okvs_w_encode(const uint8_t *keys, size_t key_len,
                  const uint64_t *values, size_t n,
                  size_t w, size_t L, size_t cols,
                  const uint8_t *seed, size_t seed_len,
                  uint64_t *x_out, uint64_t *prng_state);

/* Decode one key into out (L words). seed/w/L/cols must match the encode.
 * This rebuilds the PRF context per call -- for decoding many keys, prefer the
 * prepared-context API below. */
void okvs_w_decode(const uint8_t *key, size_t key_len,
                   size_t w, size_t L, size_t cols,
                   const uint8_t *seed, size_t seed_len,
                   const uint64_t *x_out, uint64_t *out);

/* Prepared decoder: build the PRF context once, then decode many keys cheaply
 * (no per-key key-schedule / seed hashing). `prf` is an opaque, 16-byte-aligned
 * buffer sized for either the AES-NI or software AES state. */
typedef struct {
    size_t w, bw, L, cols, poscount;
    OKVS_ALIGNAS(64) unsigned char prf[704];
} okvs_w_ctx;

void okvs_w_ctx_init(okvs_w_ctx *ctx, size_t w, size_t L, size_t cols,
                     const uint8_t *seed, size_t seed_len);

void okvs_w_decode_ctx(const okvs_w_ctx *ctx, const uint8_t *key, size_t key_len,
                       const uint64_t *x_out, uint64_t *out);
void okvs_w_decode_batch_ctx(const okvs_w_ctx *ctx, const uint8_t *keys,
                             size_t key_len, const uint64_t *x_out,
                             size_t q, uint64_t *out);
uint64_t okvs_w_decode_batch_ctx_xor(const okvs_w_ctx *ctx, const uint8_t *keys,
                                     size_t key_len, const uint64_t *x_out,
                                     size_t q);

/* No-cache locality-optimized arbitrary-query decode. This uses a temporary
 * bucket permutation by band start position, but transmits/stores no local
 * decode table. It is useful for large q when the OKVS table is much larger
 * than cache. bucket_bits=0 falls back to okvs_w_decode_batch_ctx_xor. */
uint64_t okvs_w_decode_batch_ctx_bucketed_xor(const okvs_w_ctx *ctx,
                                              const uint8_t *keys,
                                              size_t key_len,
                                              const uint64_t *x_out,
                                              size_t q,
                                              unsigned bucket_bits,
                                              int *rc_out);

/* Variant for key_len=8/xor-only benchmark: materializes keys in bucket order
 * instead of an index permutation, so the final decode pass reads keys
 * sequentially. Uses ~64 MiB more temporary memory at q=2^24. */
uint64_t okvs_w_decode_batch_ctx_bucketed_keys_xor(const okvs_w_ctx *ctx,
                                                   const uint8_t *keys,
                                                   size_t key_len,
                                                   const uint64_t *x_out,
                                                   size_t q,
                                                   unsigned bucket_bits,
                                                   int *rc_out);


/* Receiver-local decode cache for L=1 only. This is NOT transmitted.
 * Cold-decode benchmarks must count okvs_w_decode_cache_build() time plus the
 * following batch decode time. decode_chunk_bits=4 is the moderate-memory
 * default for n=2^24; 8 is a high-memory option. */
typedef struct {
    okvs_w_ctx ctx;
    size_t decode_chunk_bits;
    size_t entries_per_group;
    size_t group_count;
    uint64_t *chunk_sums;
} okvs_w_decode_cache;

int okvs_w_decode_cache_build(okvs_w_decode_cache *cache, const okvs_w_ctx *ctx,
                              const uint64_t *x_out, size_t decode_chunk_bits);
void okvs_w_decode_cache_free(okvs_w_decode_cache *cache);
uint64_t okvs_w_decode_cache_one(const okvs_w_decode_cache *cache,
                                 const uint8_t *key, size_t key_len);
void okvs_w_decode_batch_cached(const okvs_w_decode_cache *cache,
                                const uint8_t *keys, size_t key_len,
                                size_t q, uint64_t *out);
uint64_t okvs_w_decode_batch_cached_xor(const okvs_w_decode_cache *cache,
                                        const uint8_t *keys, size_t key_len,
                                        size_t q);
int okvs_w_decode_batch_cached_bucketed(const okvs_w_decode_cache *cache,
                                        const uint8_t *keys, size_t key_len,
                                        size_t q, uint64_t *out,
                                        unsigned bucket_bits);
uint64_t okvs_w_decode_batch_cached_bucketed_xor(const okvs_w_decode_cache *cache,
                                                 const uint8_t *keys, size_t key_len,
                                                 size_t q, unsigned bucket_bits,
                                                 int *rc_out);

#ifdef __cplusplus
}
#endif
#undef OKVS_ALIGNAS

#endif /* OKVS_ENCODE_H */