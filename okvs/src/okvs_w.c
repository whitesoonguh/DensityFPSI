#include "okvs_w.h"

#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ===================== multi-word band primitives ===================== */

/* bit idx of a band (idx < bw*64) */
static inline int band_bit(const uint64_t *band, size_t idx) {
    return (int)((band[idx >> 6] >> (idx & 63)) & 1u);
}

/* index of first set bit; bw*64 if all-zero (the ZeroRow sentinel) */
static inline size_t band_ctz(const uint64_t *band, size_t bw) {
    for (size_t t = 0; t < bw; t++)
        if (band[t]) return t * 64 + (size_t)__builtin_ctzll(band[t]);
    return bw * 64;
}

/* dst[0..bw) ^= (src >> d)[0..bw)   (logical right shift of an bw-word band).
 * Used to align the eliminating row (src) into the target row's frame and XOR.
 * The shifted src stays within [0, w) so no word beyond bw is needed. */
static inline void band_xor_shr_into(uint64_t *dst, const uint64_t *src,
                                     size_t bw, unsigned d, uint64_t mask) {
    unsigned ws = d >> 6, bs = d & 63;
#if defined(__AVX512F__)
    if (bw >= 5 && bw <= 8) {
        const __mmask8 kbw = (__mmask8)((1u << bw) - 1);
        __m512i s = _mm512_maskz_loadu_epi64(kbw, src);
        const __m512i iota = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512i idx_lo = _mm512_add_epi64(iota, _mm512_set1_epi64(ws));
        __m512i idx_hi = _mm512_add_epi64(idx_lo, _mm512_set1_epi64(1));
        __mmask8 klo = (ws >= 8) ? 0 : (__mmask8)((1u << (8 - ws)) - 1);
        __mmask8 khi = (ws >= 7) ? 0 : (__mmask8)((1u << (7 - ws)) - 1);
        __m512i lo = _mm512_maskz_permutexvar_epi64(klo, idx_lo, s);
        __m512i hi = _mm512_maskz_permutexvar_epi64(khi, idx_hi, s);
        __m512i r = _mm512_or_si512(_mm512_srlv_epi64(lo, _mm512_set1_epi64(bs)),
                                    _mm512_sllv_epi64(hi, _mm512_set1_epi64(64 - bs)));
        r = _mm512_and_si512(r, _mm512_set1_epi64((long long)mask));   /* predicate */
        __m512i dz = _mm512_maskz_loadu_epi64(kbw, dst);
        _mm512_mask_storeu_epi64(dst, kbw, _mm512_xor_si512(dz, r));
        return;
    }
#endif
    for (size_t t = 0; t < bw; t++) {
        size_t s = t + ws;
        uint64_t v = 0;
        if (s < bw) {
            v = src[s] >> bs;
            if (bs && s + 1 < bw) v |= src[s + 1] << (64 - bs);
        }
        dst[t] ^= (v & mask);
    }
}
/* ===================== scalar: V = uint64_t ===================== */

uint64_t okvs_w_inner_product(const uint64_t *band, size_t bw, const uint64_t *x) {
    /* XOR x[j] over set bits j of the band -- dominant cost of back-substitution
     * AND of decoding (one inner_product per key). */
#if defined(__AVX512F__)
    /* Dense masked XOR-reduce: each band byte is a __mmask8 selecting 8
     * consecutive x[] words; maskz-load suppresses faults on unset lanes, so we
     * sweep all w positions contiguously in bw*8 vector ops (vs ~w/2 scattered
     * scalar XORs) and reduce the 8 lanes at the end. (x86 little-endian.) */
    __m512i acc = _mm512_setzero_si512();
    const uint8_t *bb = (const uint8_t *)band;
    size_t nb = bw * 8;
    for (size_t c = 0; c < nb; c++) {
        __mmask8 m = (__mmask8)bb[c];
        if (m) acc = _mm512_xor_si512(acc, _mm512_maskz_loadu_epi64(m, x + c * 8));
    }
    uint64_t buf[8];
    _mm512_storeu_si512((void *)buf, acc);
    return buf[0] ^ buf[1] ^ buf[2] ^ buf[3] ^ buf[4] ^ buf[5] ^ buf[6] ^ buf[7];
#else
    uint64_t r = 0;
    for (size_t t = 0; t < bw; t++) {
        uint64_t b = band[t];
        size_t base = t * 64;
        while (b) {
            int j = __builtin_ctzll(b);
            r ^= x[base + (size_t)j];
            b &= b - 1;
        }
    }
    return r;
#endif
}

int okvs_w_simple_gauss(uint64_t *y, uint64_t *bands, size_t bw,
                        const size_t *start_pos, size_t rows, size_t cols,
                        uint64_t *x_out, uint64_t *prng_state) {
    size_t *pivot = (size_t *)malloc(rows * sizeof(size_t));
    if (rows && !pivot) return -1;

    for (size_t i = 0; i < rows; i++) {
        uint64_t y_i = y[i];
        const uint64_t *bi = bands + i * bw;

        size_t fo = band_ctz(bi, bw);
        if (fo == bw * 64) { free(pivot); return (int)(i + 1); } /* ZeroRow(i) */
        pivot[i] = fo + start_pos[i];

        for (size_t k = i + 1; k < rows; k++) {
            if (start_pos[k] > pivot[i]) break;
            size_t off = pivot[i] - start_pos[k];
            uint64_t *bk = bands + k * bw;
            uint64_t mask = (uint64_t)0 - (uint64_t)band_bit(bk, off);  
            band_xor_shr_into(bk, bi, bw,
                                    (unsigned)(start_pos[k] - start_pos[i]), mask);
            y[k] ^= (y_i & mask);            
        }
    }

    /* free (non-pivot) entries: random for double-obliviousness, else 0.
     * Pivot columns MUST be 0 during back-substitution, because the dot product
     * below includes the pivot bit and relies on x[pivot]=0 at that point. */
    if (prng_state) {
        for (size_t c = 0; c < cols; c++) x_out[c] = okvs_w_prng(prng_state);
        for (size_t i = 0; i < rows; i++) x_out[pivot[i]] = 0;
    } else {
        memset(x_out, 0, cols * sizeof(uint64_t));
    }

    for (size_t ii = rows; ii-- > 0;) {
        uint64_t ip = okvs_w_inner_product(bands + ii * bw, bw, x_out + start_pos[ii]);
        x_out[pivot[ii]] = ip ^ y[ii];
    }

    free(pivot);
    return 0;
}

/* ===================== array: V = uint64_t[L] ===================== */

/* acc[0..L) ^= src[0..L) (AVX-512 wide XOR when available, scalar fallback) */
static inline void vxor(uint64_t *acc, const uint64_t *src, size_t L) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= L; i += 8)
        _mm512_storeu_si512((void *)(acc + i),
            _mm512_xor_si512(_mm512_loadu_si512((const void *)(acc + i)),
                             _mm512_loadu_si512((const void *)(src + i))));
    for (; i < L; i++) acc[i] ^= src[i];
#else
    for (size_t i = 0; i < L; i++) acc[i] ^= src[i];
#endif
}

void okvs_w_inner_product_vec(const uint64_t *band, size_t bw,
                              const uint64_t *x, size_t L, uint64_t *res) {
    for (size_t c = 0; c < L; c++) res[c] = 0;
    for (size_t t = 0; t < bw; t++) {
        uint64_t b = band[t];
        size_t base = t * 64;
        while (b) {
            int j = __builtin_ctzll(b);
            vxor(res, x + (base + (size_t)j) * L, L);
            b &= b - 1;
        }
    }
}

int okvs_w_simple_gauss_vec(uint64_t *y, uint64_t *bands, size_t bw,
                            const size_t *start_pos, size_t rows, size_t cols,
                            size_t L, uint64_t *x_out, uint64_t *prng_state) {
    size_t *pivot = (size_t *)malloc(rows * sizeof(size_t));
    if (rows && !pivot) return -1;

    for (size_t i = 0; i < rows; i++) {
        const uint64_t *y_i = y + i * L;
        const uint64_t *bi = bands + i * bw;

        size_t fo = band_ctz(bi, bw);
        if (fo == bw * 64) { free(pivot); return (int)(i + 1); }
        pivot[i] = fo + start_pos[i];

        for (size_t k = i + 1; k < rows; k++) {
            if (start_pos[k] > pivot[i]) break;
            size_t off = pivot[i] - start_pos[k];
            uint64_t *bk = bands + k * bw;
            uint64_t mask = (uint64_t)0 - (uint64_t)band_bit(bk, off);
            band_xor_shr_into(bk, bi, bw, (unsigned)(start_pos[k] - start_pos[i]), mask);
            uint64_t *yk = y + k * L;
            for (size_t c = 0; c < L; c++) yk[c] ^= (y_i[c] & mask);        
        }
    }

    if (prng_state) {
        for (size_t c = 0; c < cols * L; c++) x_out[c] = okvs_w_prng(prng_state);
        for (size_t i = 0; i < rows; i++)
            memset(x_out + pivot[i] * L, 0, L * sizeof(uint64_t));
    } else {
        memset(x_out, 0, cols * L * sizeof(uint64_t));
    }

    uint64_t *tmp = (uint64_t *)malloc(L * sizeof(uint64_t));
    if (L && !tmp) { free(pivot); return -1; }

    for (size_t ii = rows; ii-- > 0;) {
        okvs_w_inner_product_vec(bands + ii * bw, bw, x_out + start_pos[ii] * L, L, tmp);
        uint64_t *dst = x_out + pivot[ii] * L;
        const uint64_t *yi = y + ii * L;
        for (size_t c = 0; c < L; c++) dst[c] = tmp[c] ^ yi[c];
    }

    free(tmp);
    free(pivot);
    return 0;
}