#include "okvs_encode.h"
#include "okvs_aes.h"

#include <stdlib.h>
#include <string.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* AES-128 PRF context keyed by a seed-derived 128-bit key (one BLAKE2b call,
 * not per okvs-key). Output is CPU-independent. */
static void aes_from_seed(okvs_aes *a, const uint8_t *seed, size_t sl) {
    uint8_t K[16];
    const uint8_t z = 0;
    okvs_blake2b_n(K, 16, sl ? seed : &z, sl);
    okvs_aes_init(a, K);
}

/* --- PRF output budget (okvs_aes.h) --------------------------------------
 * okvs_aes_prf(a, data, ndata, out) emits 32 bytes per 64-bit key word, so an
 * okvs key of key_len bytes (= key_len/8 words) gives 32*(key_len/8) bytes.
 * Layout of that output:
 *     bytes [0,4)            -> band start position (32-bit; ample for cols<=2^32)
 *     bytes [4, 4+ceil(w/8)) -> the w-bit band
 * so the per-key budget requirement is  ceil(w/8) <= 32*ndata - 4
 * (single 64-bit key: 32-4 = 28 band bytes => w <= 224, e.g. w=218 fits). */
#define OKVS_POS_BYTES 4
static inline int    prf_words(size_t key_len) { return (int)(key_len / 8); }
static inline size_t prf_out_bytes(int ndata)  { return (size_t)ndata * 32u; }
static inline size_t band_avail(int ndata)     { return prf_out_bytes(ndata) - OKVS_POS_BYTES; }

/* read 32-bit position from the PRF output */
static inline size_t read_pos(const uint8_t *prf, size_t poscount) {
    uint32_t p;
    memcpy(&p, prf, OKVS_POS_BYTES);
    return (size_t)(p % poscount);
}

/* fill a bw-word band buffer from the PRF output (bytes [4 ..]); the band needs
 * only ceil(w/8) bytes of material, the rest of the top word is zeroed/masked. */
static inline void fill_band(uint64_t *band, size_t bw, size_t w,
                             const uint8_t *prf, size_t avail) {
    size_t want = bw * 8;
    size_t copy = (want < avail) ? want : avail;   /* never read past PRF output */
    memset(band, 0, want);
    memcpy(band, prf + OKVS_POS_BYTES, copy);
    size_t r = w & 63;
    if (r) band[bw - 1] &= (1ULL << r) - 1;         /* keep exactly w bits */
}

/* derive band start position and w-bit band for one key via the AES PRF.
 * `hbuf` must hold at least prf_out_bytes(ndata) bytes. */
static void derive(const okvs_aes *aes, const uint64_t *key, int ndata,
                   size_t w, size_t bw, size_t poscount,
                   uint8_t *hbuf, size_t *pos_out, uint64_t *band_out) {
    okvs_aes_prfx2(aes, key, ndata, hbuf);
    *pos_out = read_pos(hbuf, poscount);
    fill_band(band_out, bw, w, hbuf, band_avail(ndata));
}

int okvs_w_encode(const uint8_t *keys, size_t key_len,
                  const uint64_t *values, size_t n,
                  size_t w, size_t L, size_t cols,
                  const uint8_t *seed, size_t seed_len,
                  uint64_t *x_out, uint64_t *prng_state) {
    size_t bw = okvs_w_words(w);
    size_t poscount = cols - w + 1; /* start in [0, cols-w] */

    int ndata = prf_words(key_len);                 /* okvs key = key_len/8 words */
    if (ndata < 1) return -1;                        /* need >= one full 64-bit word */
    if ((w + 7) / 8 > band_avail(ndata)) return -1;  /* band must fit PRF output   */

    size_t hbuf_sz = prf_out_bytes(ndata);

    uint64_t  *bands_s = malloc(n * bw * sizeof(uint64_t)); /* sorted bands  */
    uint64_t  *y       = malloc(n * L * sizeof(uint64_t));  /* sorted values */
    size_t    *sp      = malloc(n * sizeof(size_t));        /* sorted starts */
    okvs_pair *pr      = malloc(n * sizeof(okvs_pair));
    uint8_t   *hbuf    = malloc(hbuf_sz);
    if ((n && (!bands_s || !y || !sp || !pr)) || !hbuf) {
        free(bands_s); free(y); free(sp); free(pr); free(hbuf);
        return -1;
    }

    okvs_aes aes;
    aes_from_seed(&aes, seed, seed_len);

    /* pass 1: derive ONLY the start position of each key (first 4 PRF bytes) */
    for (size_t i = 0; i < n; i++) {
        okvs_aes_prfx2(&aes, (const uint64_t *)(keys + i * key_len), ndata, hbuf);
        pr[i].a = i;
        pr[i].b = read_pos(hbuf, poscount);
    }

    okvs_radix_sort(pr, n, poscount); /* sort rows by start position */

    /* pass 2: in sorted order, re-derive each band straight into its slot
     * (sequential writes) -- avoids gathering bands by a random permutation. */
    for (size_t j = 0; j < n; j++) {
        size_t orig = pr[j].a;
        okvs_aes_prfx2(&aes, (const uint64_t *)(keys + orig * key_len), ndata, hbuf);
        sp[j] = pr[j].b;
        fill_band(bands_s + j * bw, bw, w, hbuf, band_avail(ndata));
        memcpy(y + j * L, values + orig * L, L * sizeof(uint64_t));
    }

    int rc;
    if (L == 1)
        rc = okvs_w_simple_gauss(y, bands_s, bw, sp, n, cols, x_out, prng_state);
    else
        rc = okvs_w_simple_gauss_vec(y, bands_s, bw, sp, n, cols, L, x_out, prng_state);

    free(bands_s); free(y); free(sp); free(pr); free(hbuf);
    return rc;
}

void okvs_w_ctx_init(okvs_w_ctx *ctx, size_t w, size_t L, size_t cols,
                     const uint8_t *seed, size_t seed_len) {
    ctx->w = w;
    ctx->bw = okvs_w_words(w);
    ctx->L = L;
    ctx->cols = cols;
    ctx->poscount = cols - w + 1;
    /* NOTE: okvs_aes is now { __m512i rk[11] } = 704 bytes, 64-byte aligned.
     * Ensure okvs_w_ctx.prf is >= 704 bytes and 64-byte aligned. */
    aes_from_seed((okvs_aes *)ctx->prf, seed, seed_len);
}

void okvs_w_decode_ctx(const okvs_w_ctx *ctx, const uint8_t *key, size_t key_len,
                       const uint64_t *x_out, uint64_t *out) {
    size_t bw = ctx->bw, L = ctx->L;
    int    ndata = prf_words(key_len);
    size_t need  = prf_out_bytes(ndata);

    uint8_t hbuf[8 + 16 * 8];                       /* 136 bytes: fits ndata<=4 */
    uint8_t *hb = (need <= sizeof hbuf) ? hbuf : malloc(need);
    uint64_t band[16];
    uint64_t *bp = (bw <= 16) ? band : malloc(bw * sizeof(uint64_t));

    size_t pos;
    derive((const okvs_aes *)ctx->prf, (const uint64_t *)key, ndata, ctx->w, bw,
           ctx->poscount, hb, &pos, bp);

    if (L == 1)
        out[0] = okvs_w_inner_product(bp, bw, x_out + pos);
    else
        okvs_w_inner_product_vec(bp, bw, x_out + pos * L, L, out);

    if (hb != hbuf) free(hb);
    if (bp != band) free(bp);
}

void okvs_w_decode(const uint8_t *key, size_t key_len,
                   size_t w, size_t L, size_t cols,
                   const uint8_t *seed, size_t seed_len,
                   const uint64_t *x_out, uint64_t *out) {
    okvs_w_ctx ctx;
    okvs_w_ctx_init(&ctx, w, L, cols, seed, seed_len);
    okvs_w_decode_ctx(&ctx, key, key_len, x_out, out);
}
/* ---------------- L=1 receiver-local decode cache ---------------- */
static inline size_t ceil_div_st(size_t a, size_t b) { return (a + b - 1) / b; }

static inline uint64_t load_bits_low(const uint64_t *words, size_t bitpos) {
    size_t wi = bitpos >> 6;
    unsigned sh = (unsigned)(bitpos & 63u);
    uint64_t x = words[wi] >> sh;
    if (sh && wi + 1 < 16) x |= words[wi + 1] << (64u - sh);
    return x;
}

static inline uint64_t load_bits_from_bytes_le(const uint8_t *p, size_t bitpos) {
    const uint8_t *q = p + (bitpos >> 3);
    unsigned sh = (unsigned)(bitpos & 7u);
    uint64_t x = 0;
    memcpy(&x, q, sizeof(uint64_t));
    return x >> sh;
}

static inline uint64_t okvs_decode_cache_from_prf(const okvs_w_decode_cache *cache,
                                                  const uint8_t *hb) {
    const okvs_w_ctx *ctx = &cache->ctx;
    const size_t b = cache->decode_chunk_bits;
    const size_t E = cache->entries_per_group;
    size_t pos = read_pos(hb, ctx->poscount);
    const uint8_t *band_bytes = hb + OKVS_POS_BYTES;
    uint64_t acc = 0;
    size_t rel = 0;
    size_t abs = pos;
    while (rel < ctx->w) {
        size_t g = abs / b;
        unsigned off = (unsigned)(abs - g * b);
        size_t take = b - off;
        if (take > ctx->w - rel) take = ctx->w - rel;
        uint64_t bits = load_bits_from_bytes_le(band_bytes, rel) & ((1ULL << take) - 1ULL);
        unsigned mask = (unsigned)(bits << off);
        acc ^= cache->chunk_sums[g * E + mask];
        rel += take;
        abs += take;
    }
    return acc;
}

int okvs_w_decode_cache_build(okvs_w_decode_cache *cache, const okvs_w_ctx *ctx,
                              const uint64_t *x_out, size_t decode_chunk_bits) {
    if (!cache || !ctx || !x_out || ctx->L != 1) return -1;
    if (decode_chunk_bits != 4 && decode_chunk_bits != 8) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->ctx = *ctx;
    cache->decode_chunk_bits = decode_chunk_bits;
    cache->entries_per_group = (size_t)1u << decode_chunk_bits;
    cache->group_count = ceil_div_st(ctx->cols, decode_chunk_bits);
    size_t total = cache->group_count * cache->entries_per_group;
    cache->chunk_sums = (uint64_t*)malloc(total * sizeof(uint64_t));
    if (!cache->chunk_sums) return -1;

    const size_t b = decode_chunk_bits;
    const size_t E = cache->entries_per_group;
    for (size_t g = 0; g < cache->group_count; g++) {
        uint64_t *tab = cache->chunk_sums + g * E;
        tab[0] = 0;
        size_t base = g * b;
        for (size_t mask = 1; mask < E; mask++) {
            size_t lsb = mask & (0u - mask);
            unsigned bit = (unsigned)__builtin_ctzll((unsigned long long)lsb);
            size_t prev = mask ^ lsb;
            uint64_t v = (base + bit < ctx->cols) ? x_out[base + bit] : 0;
            tab[mask] = tab[prev] ^ v;
        }
    }
    return 0;
}

void okvs_w_decode_cache_free(okvs_w_decode_cache *cache) {
    if (!cache) return;
    free(cache->chunk_sums);
    memset(cache, 0, sizeof(*cache));
}

uint64_t okvs_w_decode_cache_one(const okvs_w_decode_cache *cache,
                                 const uint8_t *key, size_t key_len) {
    const okvs_w_ctx *ctx = &cache->ctx;
    int ndata = prf_words(key_len);
    size_t need = prf_out_bytes(ndata);
    uint8_t hbuf[160];
    uint8_t *hb = (need <= sizeof hbuf) ? hbuf : (uint8_t*)malloc(need + 8);
    okvs_aes_prfx2((const okvs_aes*)ctx->prf, (const uint64_t*)key, ndata, hb);
    uint64_t acc = okvs_decode_cache_from_prf(cache, hb);
    if (hb != hbuf) free(hb);
    return acc;
}

void okvs_w_decode_batch_cached(const okvs_w_decode_cache *cache,
                                const uint8_t *keys, size_t key_len,
                                size_t q, uint64_t *out) {
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)cache->ctx.prf;
        size_t i = 0;
        uint8_t hbuf[64];
        for (; i + 1 < q; i += 2) {
            okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
            out[i] = okvs_decode_cache_from_prf(cache, hbuf);
            out[i + 1] = okvs_decode_cache_from_prf(cache, hbuf + 32);
        }
        if (i < q) out[i] = okvs_w_decode_cache_one(cache, keys + i * 8, 8);
        return;
    }
    for (size_t i = 0; i < q; i++)
        out[i] = okvs_w_decode_cache_one(cache, keys + i * key_len, key_len);
}

uint64_t okvs_w_decode_batch_cached_xor(const okvs_w_decode_cache *cache,
                                        const uint8_t *keys, size_t key_len,
                                        size_t q) {
    uint64_t s = 0;
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)cache->ctx.prf;
        size_t i = 0;
        uint8_t hbuf[64];
        for (; i + 1 < q; i += 2) {
            okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
            s ^= okvs_decode_cache_from_prf(cache, hbuf);
            s ^= okvs_decode_cache_from_prf(cache, hbuf + 32);
        }
        if (i < q) s ^= okvs_w_decode_cache_one(cache, keys + i * 8, 8);
        return s;
    }
    for (size_t i = 0; i < q; i++)
        s ^= okvs_w_decode_cache_one(cache, keys + i * key_len, key_len);
    return s;
}


static size_t okvs_decode_pos_only(const okvs_w_decode_cache *cache,
                                   const uint8_t *key, size_t key_len) {
    const okvs_w_ctx *ctx = &cache->ctx;
    int ndata = prf_words(key_len);
    size_t need = prf_out_bytes(ndata);
    uint8_t hbuf[160];
    uint8_t *hb = (need <= sizeof hbuf) ? hbuf : (uint8_t*)malloc(need);
    okvs_aes_prfx2((const okvs_aes*)ctx->prf, (const uint64_t*)key, ndata, hb);
    size_t pos = read_pos(hb, ctx->poscount);
    if (hb != hbuf) free(hb);
    return pos;
}

static void okvs_decode_pos_batch_u64(const okvs_w_decode_cache *cache,
                                      const uint8_t *keys, size_t q,
                                      uint32_t *pos32, uint32_t *cnt,
                                      size_t B, size_t bucket_size) {
    const okvs_aes *aes = (const okvs_aes*)cache->ctx.prf;
    uint8_t hbuf[64];
    size_t i = 0;
    for (; i + 1 < q; i += 2) {
        okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
        size_t p0 = read_pos(hbuf, cache->ctx.poscount);
        size_t p1 = read_pos(hbuf + 32, cache->ctx.poscount);
        pos32[i] = (uint32_t)p0;
        pos32[i + 1] = (uint32_t)p1;
        size_t b0 = p0 / bucket_size; if (b0 >= B) b0 = B - 1;
        size_t b1 = p1 / bucket_size; if (b1 >= B) b1 = B - 1;
        cnt[b0 + 1]++; cnt[b1 + 1]++;
    }
    if (i < q) {
        size_t p = okvs_decode_pos_only(cache, keys + i * 8, 8);
        pos32[i] = (uint32_t)p;
        size_t b = p / bucket_size; if (b >= B) b = B - 1;
        cnt[b + 1]++;
    }
}

int okvs_w_decode_batch_cached_bucketed(const okvs_w_decode_cache *cache,
                                        const uint8_t *keys, size_t key_len,
                                        size_t q, uint64_t *out,
                                        unsigned bucket_bits) {
    if (!cache || !keys || !out || bucket_bits == 0 || bucket_bits > 24) return -1;
    if (q > UINT32_MAX || cache->ctx.poscount > UINT32_MAX) return -1;
    size_t B = (size_t)1u << bucket_bits;
    uint32_t *pos32 = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *order = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *cnt = (uint32_t*)calloc(B + 1, sizeof(uint32_t));
    if ((q && (!pos32 || !order)) || !cnt) { free(pos32); free(order); free(cnt); return -1; }
    size_t bucket_size = (cache->ctx.poscount + B - 1) / B;
    if (bucket_size == 0) bucket_size = 1;
    if (key_len == 8) {
        okvs_decode_pos_batch_u64(cache, keys, q, pos32, cnt, B, bucket_size);
    } else {
        for (size_t i = 0; i < q; i++) {
            size_t pos = okvs_decode_pos_only(cache, keys + i * key_len, key_len);
            pos32[i] = (uint32_t)pos;
            size_t b = pos / bucket_size;
            if (b >= B) b = B - 1;
            cnt[b + 1]++;
        }
    }
    for (size_t b = 1; b <= B; b++) cnt[b] += cnt[b - 1];
    uint32_t *cur = (uint32_t*)malloc(B * sizeof(uint32_t));
    if (!cur) { free(pos32); free(order); free(cnt); return -1; }
    memcpy(cur, cnt, B * sizeof(uint32_t));
    for (size_t i = 0; i < q; i++) {
        size_t b = ((size_t)pos32[i]) / bucket_size;
        if (b >= B) b = B - 1;
        order[cur[b]++] = (uint32_t)i;
    }
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)cache->ctx.prf;
        uint8_t hbuf[64];
        size_t k = 0;
        for (; k + 1 < q; k += 2) {
            uint64_t pair[2];
            pair[0] = *(const uint64_t*)(keys + (size_t)order[k] * 8);
            pair[1] = *(const uint64_t*)(keys + (size_t)order[k + 1] * 8);
            okvs_aes_prfx2(aes, pair, 2, hbuf);
            out[order[k]] = okvs_decode_cache_from_prf(cache, hbuf);
            out[order[k + 1]] = okvs_decode_cache_from_prf(cache, hbuf + 32);
        }
        if (k < q) {
            size_t i = order[k];
            out[i] = okvs_w_decode_cache_one(cache, keys + i * 8, 8);
        }
    } else {
        for (size_t k = 0; k < q; k++) {
            size_t i = order[k];
            out[i] = okvs_w_decode_cache_one(cache, keys + i * key_len, key_len);
        }
    }
    free(cur); free(pos32); free(order); free(cnt);
    return 0;
}

uint64_t okvs_w_decode_batch_cached_bucketed_xor(const okvs_w_decode_cache *cache,
                                                 const uint8_t *keys, size_t key_len,
                                                 size_t q, unsigned bucket_bits,
                                                 int *rc_out) {
    if (rc_out) *rc_out = 0;
    if (!cache || !keys || bucket_bits == 0 || bucket_bits > 24) { if (rc_out) *rc_out = -1; return 0; }
    if (q > UINT32_MAX || cache->ctx.poscount > UINT32_MAX) { if (rc_out) *rc_out = -1; return 0; }
    size_t B = (size_t)1u << bucket_bits;
    uint32_t *pos32 = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *order = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *cnt = (uint32_t*)calloc(B + 1, sizeof(uint32_t));
    if ((q && (!pos32 || !order)) || !cnt) { free(pos32); free(order); free(cnt); if (rc_out) *rc_out = -1; return 0; }
    size_t bucket_size = (cache->ctx.poscount + B - 1) / B;
    if (bucket_size == 0) bucket_size = 1;
    if (key_len == 8) {
        okvs_decode_pos_batch_u64(cache, keys, q, pos32, cnt, B, bucket_size);
    } else {
        for (size_t i = 0; i < q; i++) {
            size_t pos = okvs_decode_pos_only(cache, keys + i * key_len, key_len);
            pos32[i] = (uint32_t)pos;
            size_t b = pos / bucket_size; if (b >= B) b = B - 1;
            cnt[b + 1]++;
        }
    }
    for (size_t b = 1; b <= B; b++) cnt[b] += cnt[b - 1];
    uint32_t *cur = (uint32_t*)malloc(B * sizeof(uint32_t));
    if (!cur) { free(pos32); free(order); free(cnt); if (rc_out) *rc_out = -1; return 0; }
    memcpy(cur, cnt, B * sizeof(uint32_t));
    for (size_t i = 0; i < q; i++) {
        size_t b = ((size_t)pos32[i]) / bucket_size;
        if (b >= B) b = B - 1;
        order[cur[b]++] = (uint32_t)i;
    }
    uint64_t acc = 0;
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)cache->ctx.prf;
        uint8_t hbuf[64];
        size_t k = 0;
        for (; k + 1 < q; k += 2) {
            uint64_t pair[2];
            pair[0] = *(const uint64_t*)(keys + (size_t)order[k] * 8);
            pair[1] = *(const uint64_t*)(keys + (size_t)order[k + 1] * 8);
            okvs_aes_prfx2(aes, pair, 2, hbuf);
            acc ^= okvs_decode_cache_from_prf(cache, hbuf);
            acc ^= okvs_decode_cache_from_prf(cache, hbuf + 32);
        }
        if (k < q) {
            size_t i = order[k];
            acc ^= okvs_w_decode_cache_one(cache, keys + i * 8, 8);
        }
    } else {
        for (size_t k = 0; k < q; k++) {
            size_t i = order[k];
            acc ^= okvs_w_decode_cache_one(cache, keys + i * key_len, key_len);
        }
    }
    free(cur); free(pos32); free(order); free(cnt);
    return acc;
}

/* ---------------- L=1 no-cache batch decode ---------------- */
#if defined(__AVX512F__)
static inline uint64_t okvs_xor_reduce_m512(__m512i acc) {
    uint64_t buf[8];
    _mm512_storeu_si512((void *)buf, acc);
    return buf[0] ^ buf[1] ^ buf[2] ^ buf[3] ^ buf[4] ^ buf[5] ^ buf[6] ^ buf[7];
}
#endif

/* Direct L=1 decode from PRF bytes. This avoids materializing a bw-word band
 * on the stack and scans exactly ceil(w/8) band bytes rather than bw*8 bytes.
 * For the target L=1, w=218 regime this removes 4 dummy byte chunks and one
 * memset/memcpy per query. */
static inline uint64_t okvs_inner_product_prfbytes_l1(const okvs_w_ctx *ctx,
                                                      const uint8_t *hb,
                                                      const uint64_t *x_out) {
    size_t pos = read_pos(hb, ctx->poscount);
    const uint8_t *bb = hb + OKVS_POS_BYTES;
    size_t nbytes = (ctx->w + 7) >> 3;
    unsigned tail = (unsigned)(ctx->w & 7u);
#if defined(__AVX512F__)
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m = (unsigned)bb[c];
        if (tail && c + 1 == nbytes) m &= (1u << tail) - 1u;
        if (m) acc = _mm512_xor_si512(acc, _mm512_maskz_loadu_epi64((__mmask8)m, x_out + pos + c * 8));
    }
    return okvs_xor_reduce_m512(acc);
#else
    uint64_t r = 0;
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m = (unsigned)bb[c];
        if (tail && c + 1 == nbytes) m &= (1u << tail) - 1u;
        while (m) { unsigned j = (unsigned)__builtin_ctz(m); r ^= x_out[pos + c * 8 + j]; m &= m - 1; }
    }
    return r;
#endif
}

static inline void okvs_inner_product_prfbytes2_l1(const okvs_w_ctx *ctx,
                                                   const uint8_t *hb0,
                                                   const uint8_t *hb1,
                                                   const uint64_t *x_out,
                                                   uint64_t *r0, uint64_t *r1) {
    size_t pos0 = read_pos(hb0, ctx->poscount);
    size_t pos1 = read_pos(hb1, ctx->poscount);
    const uint8_t *bb0 = hb0 + OKVS_POS_BYTES;
    const uint8_t *bb1 = hb1 + OKVS_POS_BYTES;
    size_t nbytes = (ctx->w + 7) >> 3;
    unsigned tail = (unsigned)(ctx->w & 7u);
#if defined(__AVX512F__)
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m0 = (unsigned)bb0[c];
        unsigned m1 = (unsigned)bb1[c];
        if (tail && c + 1 == nbytes) { unsigned tm = (1u << tail) - 1u; m0 &= tm; m1 &= tm; }
        if (m0) acc0 = _mm512_xor_si512(acc0, _mm512_maskz_loadu_epi64((__mmask8)m0, x_out + pos0 + c * 8));
        if (m1) acc1 = _mm512_xor_si512(acc1, _mm512_maskz_loadu_epi64((__mmask8)m1, x_out + pos1 + c * 8));
    }
    *r0 = okvs_xor_reduce_m512(acc0);
    *r1 = okvs_xor_reduce_m512(acc1);
#else
    *r0 = okvs_inner_product_prfbytes_l1(ctx, hb0, x_out);
    *r1 = okvs_inner_product_prfbytes_l1(ctx, hb1, x_out);
#endif
}

static inline uint64_t okvs_inner_product_prfbytes_pos_l1(const okvs_w_ctx *ctx,
                                                          const uint8_t *hb,
                                                          size_t pos,
                                                          const uint64_t *x_out) {
    const uint8_t *bb = hb + OKVS_POS_BYTES;
    size_t nbytes = (ctx->w + 7) >> 3;
    unsigned tail = (unsigned)(ctx->w & 7u);
#if defined(__AVX512F__)
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m = (unsigned)bb[c];
        if (tail && c + 1 == nbytes) m &= (1u << tail) - 1u;
        if (m) acc = _mm512_xor_si512(acc, _mm512_maskz_loadu_epi64((__mmask8)m, x_out + pos + c * 8));
    }
    return okvs_xor_reduce_m512(acc);
#else
    uint64_t r = 0;
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m = (unsigned)bb[c];
        if (tail && c + 1 == nbytes) m &= (1u << tail) - 1u;
        while (m) { unsigned j = (unsigned)__builtin_ctz(m); r ^= x_out[pos + c * 8 + j]; m &= m - 1; }
    }
    return r;
#endif
}

static inline void okvs_inner_product_prfbytes2_pos_l1(const okvs_w_ctx *ctx,
                                                       const uint8_t *hb0,
                                                       const uint8_t *hb1,
                                                       size_t pos0, size_t pos1,
                                                       const uint64_t *x_out,
                                                       uint64_t *r0, uint64_t *r1) {
    const uint8_t *bb0 = hb0 + OKVS_POS_BYTES;
    const uint8_t *bb1 = hb1 + OKVS_POS_BYTES;
    size_t nbytes = (ctx->w + 7) >> 3;
    unsigned tail = (unsigned)(ctx->w & 7u);
#if defined(__AVX512F__)
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    for (size_t c = 0; c < nbytes; c++) {
        unsigned m0 = (unsigned)bb0[c];
        unsigned m1 = (unsigned)bb1[c];
        if (tail && c + 1 == nbytes) { unsigned tm = (1u << tail) - 1u; m0 &= tm; m1 &= tm; }
        if (m0) acc0 = _mm512_xor_si512(acc0, _mm512_maskz_loadu_epi64((__mmask8)m0, x_out + pos0 + c * 8));
        if (m1) acc1 = _mm512_xor_si512(acc1, _mm512_maskz_loadu_epi64((__mmask8)m1, x_out + pos1 + c * 8));
    }
    *r0 = okvs_xor_reduce_m512(acc0);
    *r1 = okvs_xor_reduce_m512(acc1);
#else
    *r0 = okvs_inner_product_prfbytes_pos_l1(ctx, hb0, pos0, x_out);
    *r1 = okvs_inner_product_prfbytes_pos_l1(ctx, hb1, pos1, x_out);
#endif
}

static inline uint64_t okvs_decode_ctx_from_prf_l1(const okvs_w_ctx *ctx,
                                                   const uint8_t *hb,
                                                   int ndata,
                                                   const uint64_t *x_out) {
    if (ndata == 1) return okvs_inner_product_prfbytes_l1(ctx, hb, x_out);
    uint64_t band[16];
    size_t pos = read_pos(hb, ctx->poscount);
    fill_band(band, ctx->bw, ctx->w, hb, band_avail(ndata));
    return okvs_w_inner_product(band, ctx->bw, x_out + pos);
}

void okvs_w_decode_batch_ctx(const okvs_w_ctx *ctx, const uint8_t *keys,
                             size_t key_len, const uint64_t *x_out,
                             size_t q, uint64_t *out) {
    if (ctx->L != 1) {
        for (size_t i = 0; i < q; i++) okvs_w_decode_ctx(ctx, keys + i * key_len, key_len, x_out, out + i);
        return;
    }
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)ctx->prf;
        uint8_t hbuf[64];
        size_t i = 0;
        for (; i + 1 < q; i += 2) {
            okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
            okvs_inner_product_prfbytes2_l1(ctx, hbuf, hbuf + 32, x_out, out + i, out + i + 1);
        }
        if (i < q) okvs_w_decode_ctx(ctx, keys + i * 8, 8, x_out, out + i);
        return;
    }
    for (size_t i = 0; i < q; i++) okvs_w_decode_ctx(ctx, keys + i * key_len, key_len, x_out, out + i);
}

uint64_t okvs_w_decode_batch_ctx_xor(const okvs_w_ctx *ctx, const uint8_t *keys,
                                     size_t key_len, const uint64_t *x_out,
                                     size_t q) {
    uint64_t acc = 0;
    if (ctx->L == 1 && key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)ctx->prf;
        uint8_t hbuf[64];
        size_t i = 0;
        for (; i + 1 < q; i += 2) {
            okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
            uint64_t r0, r1;
            okvs_inner_product_prfbytes2_l1(ctx, hbuf, hbuf + 32, x_out, &r0, &r1);
            acc ^= r0 ^ r1;
        }
        if (i < q) { uint64_t v; okvs_w_decode_ctx(ctx, keys + i * 8, 8, x_out, &v); acc ^= v; }
        return acc;
    }
    for (size_t i = 0; i < q; i++) { uint64_t v; okvs_w_decode_ctx(ctx, keys + i * key_len, key_len, x_out, &v); acc ^= v; }
    return acc;
}

static size_t okvs_decode_pos_only_ctx(const okvs_w_ctx *ctx,
                                       const uint8_t *key, size_t key_len) {
    int ndata = prf_words(key_len);
    uint8_t hbuf[64];
    okvs_aes_prfx2((const okvs_aes*)ctx->prf, (const uint64_t*)key, ndata, hbuf);
    return read_pos(hbuf, ctx->poscount);
}

static void okvs_decode_pos_batch_ctx_u64(const okvs_w_ctx *ctx,
                                          const uint8_t *keys, size_t q,
                                          uint32_t *pos32, uint32_t *cnt,
                                          size_t B, size_t bucket_size) {
    const okvs_aes *aes = (const okvs_aes*)ctx->prf;
    uint8_t hbuf[64];
    size_t i = 0;
    for (; i + 1 < q; i += 2) {
        okvs_aes_prfx2(aes, (const uint64_t*)(keys + i * 8), 2, hbuf);
        uint32_t p0 = (uint32_t)read_pos(hbuf, ctx->poscount);
        uint32_t p1 = (uint32_t)read_pos(hbuf + 32, ctx->poscount);
        pos32[i] = p0;
        pos32[i + 1] = p1;
        size_t b0 = (size_t)p0 / bucket_size; if (b0 >= B) b0 = B - 1;
        size_t b1 = (size_t)p1 / bucket_size; if (b1 >= B) b1 = B - 1;
        cnt[b0 + 1]++;
        cnt[b1 + 1]++;
    }
    if (i < q) {
        size_t p = okvs_decode_pos_only_ctx(ctx, keys + i * 8, 8);
        pos32[i] = (uint32_t)p;
        size_t b = p / bucket_size; if (b >= B) b = B - 1;
        cnt[b + 1]++;
    }
}

uint64_t okvs_w_decode_batch_ctx_bucketed_xor(const okvs_w_ctx *ctx,
                                              const uint8_t *keys,
                                              size_t key_len,
                                              const uint64_t *x_out,
                                              size_t q,
                                              unsigned bucket_bits,
                                              int *rc_out) {
    if (rc_out) *rc_out = 0;
    if (!bucket_bits) return okvs_w_decode_batch_ctx_xor(ctx, keys, key_len, x_out, q);
    if (ctx->L != 1 || q > UINT32_MAX || ctx->poscount > UINT32_MAX || bucket_bits > 24) {
        if (rc_out) *rc_out = -1;
        return 0;
    }
    size_t B = (size_t)1u << bucket_bits;
    uint32_t *pos32 = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *order = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *cnt = (uint32_t*)calloc(B + 1, sizeof(uint32_t));
    if ((q && (!pos32 || !order)) || !cnt) {
        free(pos32); free(order); free(cnt);
        if (rc_out) *rc_out = -1;
        return 0;
    }
    size_t bucket_size = (ctx->poscount + B - 1) / B;
    if (!bucket_size) bucket_size = 1;
    if (key_len == 8) {
        okvs_decode_pos_batch_ctx_u64(ctx, keys, q, pos32, cnt, B, bucket_size);
    } else {
        for (size_t i = 0; i < q; i++) {
            size_t p = okvs_decode_pos_only_ctx(ctx, keys + i * key_len, key_len);
            pos32[i] = (uint32_t)p;
            size_t b = p / bucket_size; if (b >= B) b = B - 1;
            cnt[b + 1]++;
        }
    }
    for (size_t b = 1; b <= B; b++) cnt[b] += cnt[b - 1];
    uint32_t *cur = (uint32_t*)malloc(B * sizeof(uint32_t));
    if (!cur) { free(pos32); free(order); free(cnt); if (rc_out) *rc_out = -1; return 0; }
    memcpy(cur, cnt, B * sizeof(uint32_t));
    for (size_t i = 0; i < q; i++) {
        size_t b = (size_t)pos32[i] / bucket_size;
        if (b >= B) b = B - 1;
        order[cur[b]++] = (uint32_t)i;
    }
    uint64_t acc = 0;
    if (key_len == 8) {
        const okvs_aes *aes = (const okvs_aes*)ctx->prf;
        uint8_t hbuf[64];
        size_t k = 0;
        for (; k + 1 < q; k += 2) {
            uint64_t pair[2];
            pair[0] = *(const uint64_t*)(keys + (size_t)order[k] * 8);
            pair[1] = *(const uint64_t*)(keys + (size_t)order[k + 1] * 8);
            okvs_aes_prfx2(aes, pair, 2, hbuf);
            uint64_t r0, r1;
            okvs_inner_product_prfbytes2_l1(ctx, hbuf, hbuf + 32, x_out, &r0, &r1);
            acc ^= r0 ^ r1;
        }
        if (k < q) {
            uint64_t key = *(const uint64_t*)(keys + (size_t)order[k] * 8);
            okvs_aes_prfx2(aes, &key, 1, hbuf);
            acc ^= okvs_inner_product_prfbytes_l1(ctx, hbuf, x_out);
        }
    } else {
        for (size_t k = 0; k < q; k++) {
            size_t i = order[k];
            uint64_t v;
            okvs_w_decode_ctx(ctx, keys + i * key_len, key_len, x_out, &v);
            acc ^= v;
        }
    }
    free(cur); free(pos32); free(order); free(cnt);
    return acc;
}

uint64_t okvs_w_decode_batch_ctx_bucketed_keys_xor(const okvs_w_ctx *ctx,
                                                   const uint8_t *keys,
                                                   size_t key_len,
                                                   const uint64_t *x_out,
                                                   size_t q,
                                                   unsigned bucket_bits,
                                                   int *rc_out) {
    if (rc_out) *rc_out = 0;
    if (!bucket_bits) return okvs_w_decode_batch_ctx_xor(ctx, keys, key_len, x_out, q);
    if (ctx->L != 1 || key_len != 8 || q > UINT32_MAX || ctx->poscount > UINT32_MAX || bucket_bits > 24) {
        if (rc_out) *rc_out = -1;
        return 0;
    }
    size_t B = (size_t)1u << bucket_bits;
    uint32_t *pos32 = (uint32_t*)malloc(q * sizeof(uint32_t));
    uint32_t *cnt = (uint32_t*)calloc(B + 1, sizeof(uint32_t));
    uint64_t *bkeys = (uint64_t*)malloc(q * sizeof(uint64_t));
    if ((q && (!pos32 || !bkeys)) || !cnt) {
        free(pos32); free(cnt); free(bkeys);
        if (rc_out) *rc_out = -1;
        return 0;
    }
    size_t bucket_size = (ctx->poscount + B - 1) / B;
    if (!bucket_size) bucket_size = 1;
    okvs_decode_pos_batch_ctx_u64(ctx, keys, q, pos32, cnt, B, bucket_size);
    for (size_t b = 1; b <= B; b++) cnt[b] += cnt[b - 1];
    uint32_t *cur = (uint32_t*)malloc(B * sizeof(uint32_t));
    if (!cur) { free(pos32); free(cnt); free(bkeys); if (rc_out) *rc_out = -1; return 0; }
    memcpy(cur, cnt, B * sizeof(uint32_t));
    const uint64_t *k64 = (const uint64_t*)keys;
    for (size_t i = 0; i < q; i++) {
        size_t b = (size_t)pos32[i] / bucket_size;
        if (b >= B) b = B - 1;
        bkeys[cur[b]++] = k64[i];
    }
    const okvs_aes *aes = (const okvs_aes*)ctx->prf;
    uint8_t hbuf[64];
    uint64_t acc = 0;
    size_t k = 0;
    for (; k + 1 < q; k += 2) {
        okvs_aes_prfx2(aes, bkeys + k, 2, hbuf);
        uint64_t r0, r1;
        okvs_inner_product_prfbytes2_l1(ctx, hbuf, hbuf + 32, x_out, &r0, &r1);
        acc ^= r0 ^ r1;
    }
    if (k < q) {
        okvs_aes_prfx2(aes, bkeys + k, 1, hbuf);
        acc ^= okvs_inner_product_prfbytes_l1(ctx, hbuf, x_out);
    }
    free(cur); free(pos32); free(cnt); free(bkeys);
    return acc;
}
