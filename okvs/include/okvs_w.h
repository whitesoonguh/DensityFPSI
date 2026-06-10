#ifndef OKVS_W_H
#define OKVS_W_H

/*
 * Width-parameterized RB-OKVS band solver (Bienstock-Patel-Seo-Yeo, USENIX'23;
 * Dietzfelbinger-Walzer band, ESA'19). This supersedes the fixed-256-bit
 * okvs_band.h path: the band width w is a runtime parameter, so it scales to
 * the w required for large n (e.g. n=2^24, lambda=40 needs w ~ 218..662
 * depending on epsilon, all of which exceed a 256-bit band).
 *
 * Band storage: each row is `bw = ceil(w/64)` u64 words, bits relative to the
 * row's start position (bit 0 = the start column). Bands are stored flat,
 * row r at bands[r*bw .. r*bw + bw). The band stays within w bits under
 * elimination (the eliminating row, aligned into the target's frame, lands in
 * [0, w)), so the multi-word shifted XOR never overflows -- no 2w blowup.
 *
 * The matrix is over GF(2); the only value-level op is XOR. V flavors:
 *   scalar : V = one GF(2^64) element (uint64_t).
 *   vec    : V = uint64_t[L] (AVX-512 wide XOR when available).
 *
 * Solvers MUTATE y and bands in place; start_pos must be non-decreasing.
 * Return: 0 ok, (row+1) on a zero row (Error::ZeroRow), -1 on alloc failure.
 *
 * Double-obliviousness: pass a non-NULL `prng_state` (seeded non-zero) to fill
 * the free (non-pivot) encoding entries with pseudorandom field elements, as
 * required for oblivious use (PSI/PSU). Pass NULL to fill them with 0 (faster,
 * NOT oblivious -- only for correctness testing / non-crypto use).
 */

#include <stddef.h>
#include <stdint.h>

/* words per band for a given band width w */
static inline size_t okvs_w_words(size_t w) { return (w + 63) / 64; }

/* xorshift64; *s must be seeded non-zero */
static inline uint64_t okvs_w_prng(uint64_t *s) {
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x;
}

/* ---------- scalar value: V = uint64_t ---------- */

uint64_t okvs_w_inner_product(const uint64_t *band, size_t bw, const uint64_t *x);

int okvs_w_simple_gauss(uint64_t *y, uint64_t *bands, size_t bw,
                        const size_t *start_pos, size_t rows, size_t cols,
                        uint64_t *x_out, uint64_t *prng_state);

/* ---------- array value: V = uint64_t[L], row-major ---------- */

void okvs_w_inner_product_vec(const uint64_t *band, size_t bw,
                              const uint64_t *x, size_t L, uint64_t *res);

int okvs_w_simple_gauss_vec(uint64_t *y, uint64_t *bands, size_t bw,
                            const size_t *start_pos, size_t rows, size_t cols,
                            size_t L, uint64_t *x_out, uint64_t *prng_state);

#endif /* OKVS_W_H */
