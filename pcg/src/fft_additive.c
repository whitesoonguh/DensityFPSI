#include "dmpf64/fft_additive.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

#define DMPF64_AFFT_NAIVE_THRESHOLD 32u

typedef struct {
    size_t max_k;
    dmpf64_gf **s;     /* s[k][t] is coeff of X^(2^t) in S_k, 0 <= t <= k. */
    dmpf64_gf *basis;  /* normalized Cantor-style basis: S_k(basis[k]) = 1. */
    dmpf64_gf *c;      /* c[k] = S_k(beta_k); normalized to 1 in the fast plan. */
    dmpf64_gf *cinv;   /* inverse of c[k], used by interpolation. */
    uint8_t support_count[64];
    uint8_t support_pos[64][64]; /* nonzero S_k positions below the leading term. */
} dmpf64_afft_plan;

static inline dmpf64_gf gf_mul_const(dmpf64_gf a, dmpf64_gf b) {
    if (a == 0 || b == 0) return 0;
    if (a == 1) return b;
    if (b == 1) return a;
    return dmpf64_gf_mul(a, b);
}

#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
static inline __m512i gf64_reduce128_low64x4(__m512i p) {
    /* p contains four 128-bit carryless products, one per 128-bit lane.
       The reduced 64-bit result is returned in the low 64-bit slot of each
       128-bit lane; the high slots are ignored by the caller. */
    __m512i hi = _mm512_shuffle_epi32(p, _MM_SHUFFLE(3,2,3,2));
    __m512i r = p;
    r = _mm512_xor_si512(r, hi);
    r = _mm512_xor_si512(r, _mm512_slli_epi64(hi, 1));
    r = _mm512_xor_si512(r, _mm512_slli_epi64(hi, 3));
    r = _mm512_xor_si512(r, _mm512_slli_epi64(hi, 4));
    __m512i c = _mm512_xor_si512(_mm512_srli_epi64(hi, 63),
                 _mm512_xor_si512(_mm512_srli_epi64(hi, 61), _mm512_srli_epi64(hi, 60)));
    r = _mm512_xor_si512(r, c);
    r = _mm512_xor_si512(r, _mm512_slli_epi64(c, 1));
    r = _mm512_xor_si512(r, _mm512_slli_epi64(c, 3));
    r = _mm512_xor_si512(r, _mm512_slli_epi64(c, 4));
    return r;
}

static inline __m512i gf64_mul_vec8(__m512i a, __m512i b) {
    __m512i pe = _mm512_clmulepi64_epi128(a, b, 0x00); /* lanes 0,2,4,6 */
    __m512i po = _mm512_clmulepi64_epi128(a, b, 0x11); /* lanes 1,3,5,7 */
    __m512i re = gf64_reduce128_low64x4(pe);
    __m512i ro = gf64_reduce128_low64x4(po);
    __m512i res = _mm512_maskz_mov_epi64(0x55, re);
    res = _mm512_mask_mov_epi64(res, 0xAA, _mm512_bslli_epi128(ro, 8));
    return res;
}
#endif

static double afft_now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

static void plan_free(dmpf64_afft_plan *p) {
    if (!p) return;
    if (p->s) {
        for (size_t i = 0; i <= p->max_k; i++) free(p->s[i]);
        free(p->s);
    }
    free(p->basis);
    free(p->c);
    free(p->cinv);
    memset(p, 0, sizeof(*p));
}

static dmpf64_gf s_eval(const dmpf64_afft_plan *p, size_t k, dmpf64_gf x) {
    if (k == 0) return x;
    dmpf64_gf xp = x;
    dmpf64_gf acc = 0;
    for (size_t t = 0; t <= k; t++) {
        dmpf64_gf c = p->s[k][t];
        if (c) acc ^= gf_mul_const(c, xp);
        if (t != k) xp = dmpf64_gf_square(xp);
    }
    return acc;
}

static int solve_linearized_preimage(const dmpf64_afft_plan *p, size_t k,
                                      dmpf64_gf target, dmpf64_gf *x_out) {
    uint64_t rows[64];
    uint8_t rhs[64];
    memset(rows, 0, sizeof(rows));
    memset(rhs, 0, sizeof(rhs));

    for (unsigned col = 0; col < 64; col++) {
        dmpf64_gf y = s_eval(p, k, (dmpf64_gf)1ULL << col);
        for (unsigned row = 0; row < 64; row++) {
            if ((y >> row) & 1ULL) rows[row] |= (uint64_t)1ULL << col;
        }
    }
    for (unsigned row = 0; row < 64; row++) rhs[row] = (uint8_t)((target >> row) & 1ULL);

    unsigned r = 0;
    unsigned piv_col[64];
    for (unsigned col = 0; col < 64; col++) {
        unsigned piv = 64;
        for (unsigned i = r; i < 64; i++) {
            if ((rows[i] >> col) & 1ULL) { piv = i; break; }
        }
        if (piv == 64) continue;
        uint64_t tr = rows[r]; rows[r] = rows[piv]; rows[piv] = tr;
        uint8_t tb = rhs[r]; rhs[r] = rhs[piv]; rhs[piv] = tb;
        for (unsigned i = 0; i < 64; i++) {
            if (i != r && ((rows[i] >> col) & 1ULL)) {
                rows[i] ^= rows[r];
                rhs[i] ^= rhs[r];
            }
        }
        piv_col[r] = col;
        r++;
    }
    for (unsigned i = r; i < 64; i++) {
        if (rows[i] == 0 && rhs[i]) return -1;
    }
    dmpf64_gf x = 0;
    for (unsigned i = 0; i < r; i++) {
        if (rhs[i]) x |= (dmpf64_gf)1ULL << piv_col[i];
    }
    *x_out = x;
    return 0;
}

static int plan_init(dmpf64_afft_plan *p, size_t max_k) {
    if (max_k > 63) return -1;
    memset(p, 0, sizeof(*p));
    p->max_k = max_k;
    p->s = (dmpf64_gf **)calloc(max_k + 1, sizeof(*p->s));
    p->basis = (dmpf64_gf *)calloc(max_k + 1, sizeof(*p->basis));
    p->c = (dmpf64_gf *)calloc(max_k + 1, sizeof(*p->c));
    p->cinv = (dmpf64_gf *)calloc(max_k + 1, sizeof(*p->cinv));
    if (!p->s || !p->basis || !p->c || !p->cinv) { plan_free(p); return -1; }

    p->s[0] = (dmpf64_gf *)calloc(1, sizeof(dmpf64_gf));
    if (!p->s[0]) { plan_free(p); return -1; }
    p->s[0][0] = 1; /* S_0(X)=X. */
    p->support_count[0] = 0;

    for (size_t k = 0; k < max_k; k++) {
        dmpf64_gf beta = 0;
        if (solve_linearized_preimage(p, k, 1, &beta) != 0 || beta == 0) {
            plan_free(p); return -1;
        }
        dmpf64_gf c = s_eval(p, k, beta);
        if (c != 1) { plan_free(p); return -1; }
        p->basis[k] = beta;
        p->c[k] = 1;
        p->cinv[k] = 1;

        p->s[k + 1] = (dmpf64_gf *)calloc(k + 2, sizeof(dmpf64_gf));
        if (!p->s[k + 1]) { plan_free(p); return -1; }
        /* Normalized Cantor-style recurrence: S_{k+1}=S_k^2+S_k. */
        p->s[k + 1][0] = p->s[k][0];
        for (size_t t = 1; t <= k; t++) {
            p->s[k + 1][t] = dmpf64_gf_square(p->s[k][t - 1]) ^ p->s[k][t];
        }
        p->s[k + 1][k + 1] = dmpf64_gf_square(p->s[k][k]);
        uint8_t cnt = 0;
        for (size_t t = 0; t <= k; t++) {
            if (p->s[k + 1][t]) p->support_pos[k + 1][cnt++] = (uint8_t)t;
        }
        p->support_count[k + 1] = cnt;
    }
    if (max_k > 0) { p->c[max_k] = 1; p->cinv[max_k] = 1; }
    return 0;
}

static void poly_div_s_scratch(dmpf64_gf *q, dmpf64_gf *r,
                               const dmpf64_gf *f, size_t k,
                               const dmpf64_afft_plan *plan,
                               dmpf64_gf *tmp) {
    /* Divide len=2^(k+1) polynomial f by S_k, deg(S_k)=2^k.
       q and r have length h=2^k. tmp has length 2h.
       In the normalized recurrence S_{k+1}=S_k^2+S_k, S_k is linearized
       and very sparse; use its support list rather than scanning all k terms. */
    size_t h = (size_t)1 << k;
    size_t n = h << 1;
    memcpy(tmp, f, n * sizeof(dmpf64_gf));
    memset(q, 0, h * sizeof(dmpf64_gf));
    const uint8_t cnt = plan->support_count[k];
    const uint8_t *poslist = plan->support_pos[k];

    for (size_t pos = n; pos-- > h;) {
        dmpf64_gf coeff = tmp[pos];
        if (!coeff) continue;
        size_t qi = pos - h;
        q[qi] ^= coeff;
        for (uint8_t u = 0; u < cnt; u++) {
            uint8_t t = poslist[u];
            dmpf64_gf sc = plan->s[k][t];
            size_t dst = qi + ((size_t)1 << t);
            tmp[dst] ^= (sc == 1) ? coeff : gf_mul_const(coeff, sc);
        }
    }
    memcpy(r, tmp, h * sizeof(dmpf64_gf));
}

static int eval_rec_fast(dmpf64_gf *out, const dmpf64_gf *f,
                         size_t k, dmpf64_gf shift,
                         const dmpf64_afft_plan *plan,
                         dmpf64_gf *scratch) {
    if (k == 0) { out[0] = f[0]; return 0; }
    size_t h = (size_t)1 << (k - 1);
    size_t n = h << 1;

    dmpf64_gf *tmp = scratch;
    dmpf64_gf *q = tmp + n;
    dmpf64_gf *r = q + h;
    dmpf64_gf *g0 = r + h;
    dmpf64_gf *g1 = g0 + h;
    dmpf64_gf *rec = g1 + h;

    poly_div_s_scratch(q, r, f, k - 1, plan, tmp);

    dmpf64_gf s = s_eval(plan, k - 1, shift);
    dmpf64_gf c = plan->c[k - 1];
    for (size_t i = 0; i < h; i++) {
        dmpf64_gf sq = gf_mul_const(s, q[i]);
        dmpf64_gf scq = gf_mul_const(s ^ c, q[i]);
        g0[i] = r[i] ^ sq;
        g1[i] = r[i] ^ scq;
    }
    int rc = eval_rec_fast(out, g0, k - 1, shift, plan, rec);
    if (rc == 0) rc = eval_rec_fast(out + h, g1, k - 1, shift ^ plan->basis[k - 1], plan, rec);
    return rc;
}

static void add_q_times_s(dmpf64_gf *out, const dmpf64_gf *q,
                          size_t k, const dmpf64_afft_plan *plan) {
    size_t h = (size_t)1 << k;
    const uint8_t cnt = plan->support_count[k];
    const uint8_t *poslist = plan->support_pos[k];
    for (size_t i = 0; i < h; i++) {
        dmpf64_gf qi = q[i];
        if (!qi) continue;
        for (uint8_t u = 0; u < cnt; u++) {
            uint8_t t = poslist[u];
            dmpf64_gf sc = plan->s[k][t];
            out[i + ((size_t)1 << t)] ^= (sc == 1) ? qi : gf_mul_const(qi, sc);
        }
        out[i + h] ^= qi; /* leading term X^(2^k). */
    }
}

static int interp_rec_fast(dmpf64_gf *out, const dmpf64_gf *vals,
                           size_t k, dmpf64_gf shift,
                           const dmpf64_afft_plan *plan,
                           dmpf64_gf *scratch) {
    if (k == 0) { out[0] = vals[0]; return 0; }
    size_t h = (size_t)1 << (k - 1);
    dmpf64_gf *g0 = scratch;
    dmpf64_gf *g1 = g0 + h;
    dmpf64_gf *q = g1 + h;
    dmpf64_gf *r = q + h;
    dmpf64_gf *rec = r + h;

    int rc = interp_rec_fast(g0, vals, k - 1, shift, plan, rec);
    if (rc == 0) rc = interp_rec_fast(g1, vals + h, k - 1, shift ^ plan->basis[k - 1], plan, rec);
    if (rc != 0) return rc;

    dmpf64_gf s = s_eval(plan, k - 1, shift);
    dmpf64_gf cinv = plan->cinv[k - 1];
    if (cinv == 0) return -1;

    for (size_t i = 0; i < h; i++) {
        q[i] = gf_mul_const(g0[i] ^ g1[i], cinv);
        r[i] = g0[i] ^ gf_mul_const(s, q[i]);
    }
    memset(out, 0, (h << 1) * sizeof(dmpf64_gf));
    memcpy(out, r, h * sizeof(dmpf64_gf));
    add_q_times_s(out, q, k - 1, plan);
    return 0;
}


static size_t eval_scratch_len(size_t n) { return 6 * n + 16; }
static size_t interp_scratch_len(size_t n) { return 4 * n + 16; }

int dmpf64_afft_eval(dmpf64_gf *out, const dmpf64_gf *coeffs,
                     size_t len, size_t log_n) {
    if (log_n > 63) return -1;
    size_t n = (size_t)1 << log_n;
    dmpf64_gf *buf = (dmpf64_gf *)calloc(n, sizeof(dmpf64_gf));
    if (!buf) return -1;
    if (len > n) len = n;
    memcpy(buf, coeffs, len * sizeof(dmpf64_gf));
    dmpf64_afft_plan plan;
    if (plan_init(&plan, log_n) != 0) { free(buf); return -1; }
    dmpf64_gf *scratch = (dmpf64_gf *)calloc(eval_scratch_len(n), sizeof(dmpf64_gf));
    if (!scratch) { plan_free(&plan); free(buf); return -1; }
    int rc = eval_rec_fast(out, buf, log_n, 0, &plan, scratch);
    free(scratch);
    plan_free(&plan);
    free(buf);
    return rc;
}

int dmpf64_afft_interpolate(dmpf64_gf *out, const dmpf64_gf *values,
                            size_t log_n) {
    if (log_n > 63) return -1;
    size_t n = (size_t)1 << log_n;
    dmpf64_afft_plan plan;
    if (plan_init(&plan, log_n) != 0) return -1;
    dmpf64_gf *scratch = (dmpf64_gf *)calloc(interp_scratch_len(n), sizeof(dmpf64_gf));
    if (!scratch) { plan_free(&plan); return -1; }
    int rc = interp_rec_fast(out, values, log_n, 0, &plan, scratch);
    free(scratch);
    plan_free(&plan);
    return rc;
}

static size_t ceil_log2_size(size_t x) {
    size_t k = 0, n = 1;
    while (n < x) { n <<= 1; k++; }
    return k;
}

int dmpf64_afft_mul(dmpf64_dense_poly *out,
                    const dmpf64_dense_poly *a,
                    const dmpf64_dense_poly *b) {
    if (a->len == 0 || b->len == 0) return dmpf64_dense_new(out, 0);
    size_t out_len = a->len + b->len - 1;
    if (out_len <= DMPF64_AFFT_NAIVE_THRESHOLD) return dmpf64_dense_mul_naive(out, a, b);
    size_t log_m = ceil_log2_size(out_len);
    size_t m = (size_t)1 << log_m;

    dmpf64_gf *fa = (dmpf64_gf *)calloc(m * 4, sizeof(dmpf64_gf));
    if (!fa) return -1;
    dmpf64_gf *fb = fa + m;
    dmpf64_gf *ea = fb + m;
    dmpf64_gf *eb = ea + m;
    memcpy(fa, a->c, a->len * sizeof(dmpf64_gf));
    memcpy(fb, b->c, b->len * sizeof(dmpf64_gf));

    dmpf64_afft_plan plan;
    if (plan_init(&plan, log_m) != 0) { free(fa); return -1; }
    size_t esz = eval_scratch_len(m);
    size_t isz = interp_scratch_len(m);
    size_t wsz = esz > isz ? esz : isz;
    dmpf64_gf *scratch = (dmpf64_gf *)calloc(wsz, sizeof(dmpf64_gf));
    if (!scratch) { plan_free(&plan); free(fa); return -1; }

    int rc = eval_rec_fast(ea, fa, log_m, 0, &plan, scratch);
    if (rc == 0) rc = eval_rec_fast(eb, fb, log_m, 0, &plan, scratch);
    if (rc == 0) {
        for (size_t i = 0; i < m; i++) ea[i] = dmpf64_gf_mul(ea[i], eb[i]);
        rc = dmpf64_dense_new(out, out_len);
        if (rc == 0) {
            dmpf64_gf *coeff = eb; /* reuse */
            memset(coeff, 0, m * sizeof(dmpf64_gf));
            rc = interp_rec_fast(coeff, ea, log_m, 0, &plan, scratch);
            if (rc == 0) memcpy(out->c, coeff, out_len * sizeof(dmpf64_gf));
            else dmpf64_dense_free(out);
        }
    }
    free(scratch);
    plan_free(&plan);
    free(fa);
    return rc;
}

int dmpf64_afft_cyclic_mul(dmpf64_ring_elem *out,
                           const dmpf64_ring_elem *a,
                           const dmpf64_ring_elem *b) {
    if (a->mod.degree != b->mod.degree) return -1;
    size_t n = a->mod.degree;
    if (n <= 16) return dmpf64_ring_mul(out, a, b);

    /* Compute the ordinary product with an AFFT of length 2N, then fold
       coefficients modulo x^N+1.  In characteristic two this is cyclic xor.
       This direct path avoids materializing a temporary dense polynomial of
       length 2N-1 before reduction. */
    size_t m = n << 1;
    size_t log_m = 0;
    for (size_t t = m; t > 1; t >>= 1) log_m++;

    dmpf64_gf *fa = (dmpf64_gf *)calloc(m * 4, sizeof(dmpf64_gf));
    if (!fa) return -1;
    dmpf64_gf *fb = fa + m;
    dmpf64_gf *ea = fb + m;
    dmpf64_gf *eb = ea + m;
    memcpy(fa, a->poly.c, n * sizeof(dmpf64_gf));
    memcpy(fb, b->poly.c, n * sizeof(dmpf64_gf));

    dmpf64_afft_plan plan;
    if (plan_init(&plan, log_m) != 0) { free(fa); return -1; }
    size_t esz = eval_scratch_len(m);
    size_t isz = interp_scratch_len(m);
    size_t wsz = esz > isz ? esz : isz;
    dmpf64_gf *scratch = (dmpf64_gf *)calloc(wsz, sizeof(dmpf64_gf));
    if (!scratch) { plan_free(&plan); free(fa); return -1; }

    int rc = eval_rec_fast(ea, fa, log_m, 0, &plan, scratch);
    if (rc == 0) rc = eval_rec_fast(eb, fb, log_m, 0, &plan, scratch);
    if (rc == 0) {
        for (size_t i = 0; i < m; i++) ea[i] = dmpf64_gf_mul(ea[i], eb[i]);
        memset(eb, 0, m * sizeof(dmpf64_gf));
        rc = interp_rec_fast(eb, ea, log_m, 0, &plan, scratch);
    }
    if (rc == 0) {
        out->mod = a->mod;
        rc = dmpf64_dense_new(&out->poly, n);
        if (rc == 0) {
            for (size_t i = 0; i < (m - 1); i++) out->poly.c[i & (n - 1)] ^= eb[i];
        }
    }
    free(scratch);
    plan_free(&plan);
    free(fa);
    return rc;
}



typedef struct {
    size_t log_m;
    size_t m;
    int initialized;
    dmpf64_afft_plan plan;
    dmpf64_gf *work;    /* 4*m: fa, fb, ea, eb */
    size_t work_len;
    dmpf64_gf *scratch;
    size_t scratch_len;
    dmpf64_gf *twiddles; /* flat stage twiddle table; total length m-1. */
    size_t twiddle_len;
    size_t twiddle_off[65];
} dmpf64_afft_cyclic_cache;

static dmpf64_afft_cyclic_cache g_cyclic_cache = {0};



static int support_all_one(const dmpf64_afft_plan *plan, size_t sidx) {
    const uint8_t cnt = plan->support_count[sidx];
    const uint8_t *poslist = plan->support_pos[sidx];
    for (uint8_t u = 0; u < cnt; u++) if (plan->s[sidx][poslist[u]] != 1) return 0;
    return 1;
}


static void mono_to_novel_inplace(dmpf64_gf *a, size_t k, const dmpf64_afft_plan *plan) {
    if (k == 0) return;
    size_t h = (size_t)1 << (k - 1);
    size_t n = h << 1;
    const size_t sidx = k - 1;
    const uint8_t cnt = plan->support_count[sidx];
    const uint8_t *poslist = plan->support_pos[sidx];
#if defined(__AVX512F__)
    if (h >= 16 && support_all_one(plan, sidx)) {
        size_t end = h;
        while (end >= 8) {
            size_t i = end - 8;
            __m512i coeff = _mm512_loadu_si512((const void *)(a + h + i));
            /* The largest non-leading shift is at most h/2, so for h>=16
               these stores cannot corrupt another coefficient inside this
               already-loaded 8-lane chunk. */
            for (uint8_t u = 0; u < cnt; u++) {
                size_t dst = i + ((size_t)1 << poslist[u]);
                __m512i old = _mm512_loadu_si512((const void *)(a + dst));
                _mm512_storeu_si512((void *)(a + dst), _mm512_xor_si512(old, coeff));
            }
            if (i == 0) break;
            end = i;
        }
        /* h is a power of two and h>=16, so no scalar tail. */
    } else
#endif
    {
        for (size_t pos = n; pos-- > h;) {
            dmpf64_gf coeff = a[pos];
            if (!coeff) continue;
            size_t qi = pos - h;
            for (uint8_t u = 0; u < cnt; u++) {
                uint8_t t = poslist[u];
                dmpf64_gf sc = plan->s[sidx][t];
                size_t dst = qi + ((size_t)1 << t);
                a[dst] ^= (sc == 1) ? coeff : gf_mul_const(coeff, sc);
            }
        }
    }
    mono_to_novel_inplace(a, k - 1, plan);
    mono_to_novel_inplace(a + h, k - 1, plan);
}

static void novel_to_mono_inplace(dmpf64_gf *a, size_t k, const dmpf64_afft_plan *plan) {
    if (k == 0) return;
    size_t h = (size_t)1 << (k - 1);
    novel_to_mono_inplace(a, k - 1, plan);
    novel_to_mono_inplace(a + h, k - 1, plan);
    const size_t sidx = k - 1;
    const uint8_t cnt = plan->support_count[sidx];
    const uint8_t *poslist = plan->support_pos[sidx];
#if defined(__AVX512F__)
    if (h >= 16 && support_all_one(plan, sidx)) {
        for (size_t i = 0; i < h; i += 8) {
            __m512i qi = _mm512_loadu_si512((const void *)(a + h + i));
            /* Load q before stores: some destinations may lie in the high half
               behind the current q pointer, exactly matching the scalar
               ascending triangular order. */
            for (uint8_t u = 0; u < cnt; u++) {
                size_t dst = i + ((size_t)1 << poslist[u]);
                __m512i old = _mm512_loadu_si512((const void *)(a + dst));
                _mm512_storeu_si512((void *)(a + dst), _mm512_xor_si512(old, qi));
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < h; i++) {
            dmpf64_gf qi = a[h + i];
            if (!qi) continue;
            for (uint8_t u = 0; u < cnt; u++) {
                uint8_t t = poslist[u];
                dmpf64_gf sc = plan->s[sidx][t];
                size_t dst = i + ((size_t)1 << t);
                a[dst] ^= (sc == 1) ? qi : gf_mul_const(qi, sc);
            }
        }
    }
}

/* Convert a zero-padded monomial prefix to novel basis.  If only the first
   active_len coefficients are nonzero and active_len <= 2^(k-1), the top
   quotient is exactly zero, so the high novel half remains zero and the
   conversion can recurse only into the low half.  This is the common ring/OLE
   case: length-N inputs are transformed with length-2N or length-4N AFFTs. */
static void mono_to_novel_prefix_inplace(dmpf64_gf *a, size_t k, size_t active_len,
                                         const dmpf64_afft_plan *plan) {
    if (k == 0 || active_len == 0) return;
    size_t n = (size_t)1 << k;
    if (active_len >= n) {
        mono_to_novel_inplace(a, k, plan);
        return;
    }
    size_t h = n >> 1;
    if (active_len <= h) {
        mono_to_novel_prefix_inplace(a, k - 1, active_len, plan);
        return;
    }
    /* Rare fallback for partially active high half.  It preserves correctness;
       the optimized code paths below call this only with active_len <= h. */
    mono_to_novel_inplace(a, k, plan);
}

static int cyclic_cache_build_twiddles(dmpf64_afft_cyclic_cache *c) {
    const size_t K = c->log_m;
    const size_t total = c->m ? c->m - 1 : 0;
    size_t tw_bytes = total * sizeof(dmpf64_gf);
    if ((tw_bytes & 63u) == 0) c->twiddles = (dmpf64_gf *)aligned_alloc(64, tw_bytes);
    if (!c->twiddles) c->twiddles = (dmpf64_gf *)malloc(tw_bytes);
    if (total && !c->twiddles) return -1;
    c->twiddle_len = total;
    size_t off = 0;
    for (size_t l = K; l >= 1; l--) {
        size_t blocks = (size_t)1 << (K - l);
        c->twiddle_off[l] = off;
        dmpf64_gf *tw = c->twiddles + off;
        tw[0] = 0;
        for (size_t r = 0; r < K - l; r++) {
            size_t step = (size_t)1 << r;
            dmpf64_gf gamma = s_eval(&c->plan, l - 1, c->plan.basis[l + r]);
            for (size_t j = 0; j < step; j++) tw[step + j] = tw[j] ^ gamma;
        }
        off += blocks;
        if (l == 1) break;
    }
    return off == total ? 0 : -1;
}

static inline void novel_butterfly_eval_pair(dmpf64_gf *lo, dmpf64_gf *hi, size_t h, dmpf64_gf s) {
    size_t i = 0;
    if (s == 0) {
#if defined(__AVX512F__)
        for (; i + 8 <= h; i += 8) {
            __m512i a = _mm512_loadu_si512((const void *)(lo + i));
            __m512i b = _mm512_loadu_si512((const void *)(hi + i));
            _mm512_storeu_si512((void *)(hi + i), _mm512_xor_si512(a, b));
        }
#endif
        for (; i < h; i++) hi[i] = lo[i] ^ hi[i];
        return;
    }
    if (s == 1) {
#if defined(__AVX512F__)
        for (; i + 8 <= h; i += 8) {
            __m512i a = _mm512_loadu_si512((const void *)(lo + i));
            __m512i b = _mm512_loadu_si512((const void *)(hi + i));
            _mm512_storeu_si512((void *)(lo + i), _mm512_xor_si512(a, b));
            _mm512_storeu_si512((void *)(hi + i), a);
        }
#endif
        for (; i < h; i++) {
            dmpf64_gf a = lo[i], b = hi[i];
            lo[i] = a ^ b;
            hi[i] = a;
        }
        return;
    }
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
    __m512i ss = _mm512_set1_epi64((long long)s);
    for (; i + 8 <= h; i += 8) {
        __m512i a = _mm512_loadu_si512((const void *)(lo + i));
        __m512i b = _mm512_loadu_si512((const void *)(hi + i));
        __m512i sb = gf64_mul_vec8(ss, b);
        __m512i u = _mm512_xor_si512(a, sb);
        __m512i v = _mm512_xor_si512(u, b); /* since S(beta)=1 */
        _mm512_storeu_si512((void *)(lo + i), u);
        _mm512_storeu_si512((void *)(hi + i), v);
    }
#endif
    for (; i < h; i++) {
        dmpf64_gf a = lo[i], b = hi[i];
        dmpf64_gf u = a ^ dmpf64_gf_mul(s, b);
        lo[i] = u;
        hi[i] = u ^ b;
    }
}

static inline void novel_butterfly_interp_pair(dmpf64_gf *lo, dmpf64_gf *hi, size_t h, dmpf64_gf s) {
    size_t i = 0;
    if (s == 0) {
#if defined(__AVX512F__)
        for (; i + 8 <= h; i += 8) {
            __m512i u = _mm512_loadu_si512((const void *)(lo + i));
            __m512i v = _mm512_loadu_si512((const void *)(hi + i));
            _mm512_storeu_si512((void *)(hi + i), _mm512_xor_si512(u, v));
        }
#endif
        for (; i < h; i++) hi[i] = lo[i] ^ hi[i];
        return;
    }
    if (s == 1) {
#if defined(__AVX512F__)
        for (; i + 8 <= h; i += 8) {
            __m512i u = _mm512_loadu_si512((const void *)(lo + i));
            __m512i v = _mm512_loadu_si512((const void *)(hi + i));
            __m512i b = _mm512_xor_si512(u, v);
            _mm512_storeu_si512((void *)(lo + i), v);
            _mm512_storeu_si512((void *)(hi + i), b);
        }
#endif
        for (; i < h; i++) {
            dmpf64_gf u = lo[i], v = hi[i];
            dmpf64_gf b = u ^ v;
            lo[i] = v;
            hi[i] = b;
        }
        return;
    }
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
    __m512i ss = _mm512_set1_epi64((long long)s);
    for (; i + 8 <= h; i += 8) {
        __m512i u = _mm512_loadu_si512((const void *)(lo + i));
        __m512i v = _mm512_loadu_si512((const void *)(hi + i));
        __m512i b = _mm512_xor_si512(u, v);
        __m512i sb = gf64_mul_vec8(ss, b);
        __m512i a = _mm512_xor_si512(u, sb);
        _mm512_storeu_si512((void *)(lo + i), a);
        _mm512_storeu_si512((void *)(hi + i), b);
    }
#endif
    for (; i < h; i++) {
        dmpf64_gf u = lo[i], v = hi[i];
        dmpf64_gf b = u ^ v;
        lo[i] = u ^ dmpf64_gf_mul(s, b);
        hi[i] = b;
    }
}


#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
static inline __m512i idx512_epi64_8(long long a0,long long a1,long long a2,long long a3,long long a4,long long a5,long long a6,long long a7) {
    return _mm512_set_epi64(a7,a6,a5,a4,a3,a2,a1,a0);
}

static void novel_eval_stage_low_vpclmul(dmpf64_gf *data, size_t blocks, size_t h, const dmpf64_gf *tw) {
    if (h == 1) {
        const __m512i ilo = idx512_epi64_8(0,2,4,6,8,10,12,14);
        const __m512i ihi = idx512_epi64_8(1,3,5,7,9,11,13,15);
        size_t b = 0;
        for (; b + 8 <= blocks; b += 8) {
            dmpf64_gf *base = data + (b << 1);
            __m512i lo = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i hi = _mm512_i64gather_epi64(ihi, base, 8);
            __m512i sv = _mm512_loadu_si512((const void *)(tw + b));
            __m512i u = _mm512_xor_si512(lo, gf64_mul_vec8(sv, hi));
            __m512i v = _mm512_xor_si512(u, hi);
            _mm512_i64scatter_epi64(base, ilo, u, 8);
            _mm512_i64scatter_epi64(base, ihi, v, 8);
        }
        for (; b < blocks; b++) novel_butterfly_eval_pair(data + 2*b, data + 2*b + 1, 1, tw[b]);
    } else if (h == 2) {
        const __m512i ilo = idx512_epi64_8(0,1,4,5,8,9,12,13);
        const __m512i ihi = idx512_epi64_8(2,3,6,7,10,11,14,15);
        size_t b = 0;
        for (; b + 4 <= blocks; b += 4) {
            dmpf64_gf *base = data + (b << 2);
            __m512i lo = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i hi = _mm512_i64gather_epi64(ihi, base, 8);
            dmpf64_gf s0=tw[b], s1=tw[b+1], s2=tw[b+2], s3=tw[b+3];
            __m512i sv = _mm512_set_epi64((long long)s3,(long long)s3,(long long)s2,(long long)s2,(long long)s1,(long long)s1,(long long)s0,(long long)s0);
            __m512i u = _mm512_xor_si512(lo, gf64_mul_vec8(sv, hi));
            __m512i v = _mm512_xor_si512(u, hi);
            _mm512_i64scatter_epi64(base, ilo, u, 8);
            _mm512_i64scatter_epi64(base, ihi, v, 8);
        }
        for (; b < blocks; b++) novel_butterfly_eval_pair(data + 4*b, data + 4*b + 2, 2, tw[b]);
    } else if (h == 4) {
        const __m512i ilo = idx512_epi64_8(0,1,2,3,8,9,10,11);
        const __m512i ihi = idx512_epi64_8(4,5,6,7,12,13,14,15);
        size_t b = 0;
        for (; b + 2 <= blocks; b += 2) {
            dmpf64_gf *base = data + (b << 3);
            __m512i lo = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i hi = _mm512_i64gather_epi64(ihi, base, 8);
            dmpf64_gf s0=tw[b], s1=tw[b+1];
            __m512i sv = _mm512_set_epi64((long long)s1,(long long)s1,(long long)s1,(long long)s1,(long long)s0,(long long)s0,(long long)s0,(long long)s0);
            __m512i u = _mm512_xor_si512(lo, gf64_mul_vec8(sv, hi));
            __m512i v = _mm512_xor_si512(u, hi);
            _mm512_i64scatter_epi64(base, ilo, u, 8);
            _mm512_i64scatter_epi64(base, ihi, v, 8);
        }
        for (; b < blocks; b++) novel_butterfly_eval_pair(data + 8*b, data + 8*b + 4, 4, tw[b]);
    }
}

static void novel_interp_stage_low_vpclmul(dmpf64_gf *data, size_t blocks, size_t h, const dmpf64_gf *tw) {
    if (h == 1) {
        const __m512i ilo = idx512_epi64_8(0,2,4,6,8,10,12,14);
        const __m512i ihi = idx512_epi64_8(1,3,5,7,9,11,13,15);
        size_t b = 0;
        for (; b + 8 <= blocks; b += 8) {
            dmpf64_gf *base = data + (b << 1);
            __m512i u = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i v = _mm512_i64gather_epi64(ihi, base, 8);
            __m512i bv = _mm512_xor_si512(u, v);
            __m512i sv = _mm512_loadu_si512((const void *)(tw + b));
            __m512i a = _mm512_xor_si512(u, gf64_mul_vec8(sv, bv));
            _mm512_i64scatter_epi64(base, ilo, a, 8);
            _mm512_i64scatter_epi64(base, ihi, bv, 8);
        }
        for (; b < blocks; b++) novel_butterfly_interp_pair(data + 2*b, data + 2*b + 1, 1, tw[b]);
    } else if (h == 2) {
        const __m512i ilo = idx512_epi64_8(0,1,4,5,8,9,12,13);
        const __m512i ihi = idx512_epi64_8(2,3,6,7,10,11,14,15);
        size_t b = 0;
        for (; b + 4 <= blocks; b += 4) {
            dmpf64_gf *base = data + (b << 2);
            __m512i u = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i v = _mm512_i64gather_epi64(ihi, base, 8);
            __m512i bv = _mm512_xor_si512(u, v);
            dmpf64_gf s0=tw[b], s1=tw[b+1], s2=tw[b+2], s3=tw[b+3];
            __m512i sv = _mm512_set_epi64((long long)s3,(long long)s3,(long long)s2,(long long)s2,(long long)s1,(long long)s1,(long long)s0,(long long)s0);
            __m512i a = _mm512_xor_si512(u, gf64_mul_vec8(sv, bv));
            _mm512_i64scatter_epi64(base, ilo, a, 8);
            _mm512_i64scatter_epi64(base, ihi, bv, 8);
        }
        for (; b < blocks; b++) novel_butterfly_interp_pair(data + 4*b, data + 4*b + 2, 2, tw[b]);
    } else if (h == 4) {
        const __m512i ilo = idx512_epi64_8(0,1,2,3,8,9,10,11);
        const __m512i ihi = idx512_epi64_8(4,5,6,7,12,13,14,15);
        size_t b = 0;
        for (; b + 2 <= blocks; b += 2) {
            dmpf64_gf *base = data + (b << 3);
            __m512i u = _mm512_i64gather_epi64(ilo, base, 8);
            __m512i v = _mm512_i64gather_epi64(ihi, base, 8);
            __m512i bv = _mm512_xor_si512(u, v);
            dmpf64_gf s0=tw[b], s1=tw[b+1];
            __m512i sv = _mm512_set_epi64((long long)s1,(long long)s1,(long long)s1,(long long)s1,(long long)s0,(long long)s0,(long long)s0,(long long)s0);
            __m512i a = _mm512_xor_si512(u, gf64_mul_vec8(sv, bv));
            _mm512_i64scatter_epi64(base, ilo, a, 8);
            _mm512_i64scatter_epi64(base, ihi, bv, 8);
        }
        for (; b < blocks; b++) novel_butterfly_interp_pair(data + 8*b, data + 8*b + 4, 4, tw[b]);
    }
}
#endif

static void novel_eval_inplace_cached(dmpf64_afft_cyclic_cache *c, dmpf64_gf *data) {
    const size_t K = c->log_m;
    for (size_t l = K; l >= 1; l--) {
        size_t h = (size_t)1 << (l - 1);
        size_t bs = h << 1;
        size_t blocks = (size_t)1 << (K - l);
        dmpf64_gf *tw = c->twiddles + c->twiddle_off[l];
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
        if (h == 1 || h == 2 || h == 4) {
            novel_eval_stage_low_vpclmul(data, blocks, h, tw);
        } else
#endif
        {
            for (size_t b = 0; b < blocks; b++) {
                size_t off = b * bs;
                novel_butterfly_eval_pair(data + off, data + off + h, h, tw[b]);
            }
        }
        if (l == 1) break;
    }
}

static void novel_interp_inplace_cached(dmpf64_afft_cyclic_cache *c, dmpf64_gf *data) {
    const size_t K = c->log_m;
    for (size_t l = 1; l <= K; l++) {
        size_t h = (size_t)1 << (l - 1);
        size_t bs = h << 1;
        size_t blocks = (size_t)1 << (K - l);
        dmpf64_gf *tw = c->twiddles + c->twiddle_off[l];
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
        if (h == 1 || h == 2 || h == 4) {
            novel_interp_stage_low_vpclmul(data, blocks, h, tw);
        } else
#endif
        {
            for (size_t b = 0; b < blocks; b++) {
                size_t off = b * bs;
                novel_butterfly_interp_pair(data + off, data + off + h, h, tw[b]);
            }
        }
    }
}


static int cyclic_cache_prepare_work(size_t log_m, size_t min_work_len);

void dmpf64_afft_clear_cache(void) {
    if (g_cyclic_cache.initialized) plan_free(&g_cyclic_cache.plan);
    free(g_cyclic_cache.work);
    free(g_cyclic_cache.scratch);
    free(g_cyclic_cache.twiddles);
    memset(&g_cyclic_cache, 0, sizeof(g_cyclic_cache));
}

int dmpf64_afft_prepare_eval_domain(size_t log_n) {
    if (log_n >= 63) return -1;
    size_t n = (size_t)1 << log_n;
    /* Sparse eval and monomial eval paths need an N-word work buffer and
       the stage twiddle table for V_log_n. */
    return cyclic_cache_prepare_work(log_n, n);
}

static int cyclic_cache_prepare_work(size_t log_m, size_t min_work_len) {
    size_t m = (size_t)1 << log_m;
    /* Cached cyclic multiplication now uses in-place monomial<->novel
       conversion and in-place novel AFFT, so no large transform scratch is
       needed.  Keep a tiny allocation only to preserve cache invariants. */
    size_t scratch_len = 1;
    if (g_cyclic_cache.initialized && g_cyclic_cache.log_m == log_m &&
        g_cyclic_cache.work_len >= min_work_len && g_cyclic_cache.scratch_len >= scratch_len) {
        return 0;
    }
    dmpf64_afft_clear_cache();
    g_cyclic_cache.log_m = log_m;
    g_cyclic_cache.m = m;
    if (plan_init(&g_cyclic_cache.plan, log_m) != 0) {
        memset(&g_cyclic_cache, 0, sizeof(g_cyclic_cache));
        return -1;
    }
    g_cyclic_cache.initialized = 1;
    size_t bytes = min_work_len * sizeof(dmpf64_gf);
    if ((bytes & 63u) == 0) g_cyclic_cache.work = (dmpf64_gf *)aligned_alloc(64, bytes);
    if (!g_cyclic_cache.work) g_cyclic_cache.work = (dmpf64_gf *)malloc(bytes);
    size_t scratch_bytes = scratch_len * sizeof(dmpf64_gf);
    if ((scratch_bytes & 63u) == 0) g_cyclic_cache.scratch = (dmpf64_gf *)aligned_alloc(64, scratch_bytes);
    if (!g_cyclic_cache.scratch) g_cyclic_cache.scratch = (dmpf64_gf *)malloc(scratch_bytes);
    if (!g_cyclic_cache.work || !g_cyclic_cache.scratch) {
        dmpf64_afft_clear_cache();
        return -1;
    }
    g_cyclic_cache.work_len = min_work_len;
    g_cyclic_cache.scratch_len = scratch_len;
    if (cyclic_cache_build_twiddles(&g_cyclic_cache) != 0) {
        dmpf64_afft_clear_cache();
        return -1;
    }
    return 0;
}

static int cyclic_cache_prepare(size_t log_m) {
    size_t m = (size_t)1 << log_m;
    return cyclic_cache_prepare_work(log_m, 2 * m);
}

static int cyclic_cached_coeffs(const dmpf64_ring_elem *a,
                                const dmpf64_ring_elem *b,
                                dmpf64_gf **coeff_out,
                                size_t *m_out) {
    if (a->mod.degree != b->mod.degree) return -1;
    size_t n = a->mod.degree;
    size_t m = n << 1;
    size_t log_m = 0;
    for (size_t t = m; t > 1; t >>= 1) log_m++;
    if (cyclic_cache_prepare(log_m) != 0) return -1;

    dmpf64_gf *fa = g_cyclic_cache.work;       /* monomial/novel/eval/product/result */
    dmpf64_gf *fb = fa + m;                    /* monomial/novel/eval b */
    memset(fa, 0, 2 * m * sizeof(dmpf64_gf));
    memcpy(fa, a->poly.c, n * sizeof(dmpf64_gf));
    memcpy(fb, b->poly.c, n * sizeof(dmpf64_gf));

    const char *trace = getenv("DMPF64_AFFT_TRACE");
    double t0 = trace ? afft_now_sec() : 0.0, t1, t2, t3, t4, t5, t6;
    int rc = 0;
    mono_to_novel_prefix_inplace(fa, log_m, n, &g_cyclic_cache.plan);
    t1 = trace ? afft_now_sec() : 0.0;
    mono_to_novel_prefix_inplace(fb, log_m, n, &g_cyclic_cache.plan);
    t2 = trace ? afft_now_sec() : 0.0;
    novel_eval_inplace_cached(&g_cyclic_cache, fa);
    t3 = trace ? afft_now_sec() : 0.0;
    novel_eval_inplace_cached(&g_cyclic_cache, fb);
    t4 = trace ? afft_now_sec() : 0.0;
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
    size_t i = 0;
    for (; i + 8 <= m; i += 8) {
        __m512i va = _mm512_loadu_si512((const void *)(fa + i));
        __m512i vb = _mm512_loadu_si512((const void *)(fb + i));
        _mm512_storeu_si512((void *)(fa + i), gf64_mul_vec8(va, vb));
    }
    for (; i < m; i++) fa[i] = dmpf64_gf_mul(fa[i], fb[i]);
#else
    for (size_t i = 0; i < m; i++) fa[i] = dmpf64_gf_mul(fa[i], fb[i]);
#endif
    t5 = trace ? afft_now_sec() : 0.0;
    novel_interp_inplace_cached(&g_cyclic_cache, fa);
    t6 = trace ? afft_now_sec() : 0.0;
    novel_to_mono_inplace(fa, log_m, &g_cyclic_cache.plan);
    if (trace) {
        double t7 = afft_now_sec();
        fprintf(stderr, "AFFT_TRACE logm=%zu m=%zu mono2nov_a=%.6f mono2nov_b=%.6f eval_a=%.6f eval_b=%.6f point=%.6f interp=%.6f nov2mono=%.6f\n",
                log_m, m, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6);
    }
    if (rc != 0) return rc;
    *coeff_out = fa;
    *m_out = m;
    return 0;
}



static int dense_alloc_uninit(dmpf64_dense_poly *p, size_t len) {
    p->len = 0;
    p->c = NULL;
    if (len == 0) return 0;
    size_t bytes = len * sizeof(dmpf64_gf);
    if ((bytes & 63u) == 0) p->c = (dmpf64_gf *)aligned_alloc(64, bytes);
    if (!p->c) p->c = (dmpf64_gf *)malloc(bytes);
    if (!p->c) return -1;
    p->len = len;
    return 0;
}

static void cyclic_fold_to(dmpf64_gf *out, const dmpf64_gf *coeff, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= n - 1; i += 8) {
        __m512i lo = _mm512_loadu_si512((const void *)(coeff + i));
        __m512i hi = _mm512_loadu_si512((const void *)(coeff + n + i));
        _mm512_storeu_si512((void *)(out + i), _mm512_xor_si512(lo, hi));
    }
#endif
    for (; i + 1 < n; i++) out[i] = coeff[i] ^ coeff[n + i];
    out[n - 1] = coeff[n - 1];
}

static void cyclic_fold_addto(dmpf64_gf *acc, const dmpf64_gf *coeff, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= n - 1; i += 8) {
        __m512i a = _mm512_loadu_si512((const void *)(acc + i));
        __m512i lo = _mm512_loadu_si512((const void *)(coeff + i));
        __m512i hi = _mm512_loadu_si512((const void *)(coeff + n + i));
        _mm512_storeu_si512((void *)(acc + i), _mm512_xor_si512(a, _mm512_xor_si512(lo, hi)));
    }
#endif
    for (; i + 1 < n; i++) acc[i] ^= coeff[i] ^ coeff[n + i];
    acc[n - 1] ^= coeff[n - 1];
}



static void xor_shift_vec(dmpf64_gf *out, const dmpf64_gf *h, size_t off, size_t count) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= count; i += 8) {
        __m512i a = _mm512_loadu_si512((const void *)(out + off + i));
        __m512i b = _mm512_loadu_si512((const void *)(h + i));
        _mm512_storeu_si512((void *)(out + off + i), _mm512_xor_si512(a, b));
    }
#endif
    for (; i < count; i++) out[off + i] ^= h[i];
}

static int reduce_subspace_len2_ones_inplace(dmpf64_gf *out, size_t log_n,
                                                dmpf64_gf *coeffs,
                                                const dmpf64_afft_plan *plan) {
    size_t n = (size_t)1 << log_n;
    const uint8_t cnt = plan->support_count[log_n];
    const uint8_t *poslist = plan->support_pos[log_n];
    if (cnt == 0) {
        memcpy(out, coeffs, n * sizeof(*out));
        return 0;
    }
    size_t shifts[64];
    size_t max_shift = 0;
    for (uint8_t u = 0; u < cnt; u++) {
        shifts[u] = (size_t)1 << poslist[u];
        if (shifts[u] > max_shift) max_shift = shifts[u];
    }

    /* Reduce f_low + X^N f_high modulo the normalized subspace polynomial
       S_log_n(X)=X^N + sum_u X^{shifts[u]}.  Only the last max_shift high
       coefficients can cascade back into the high half; after resolving that
       tail, the remaining high half contributes independent xor-shift streams,
       which are AVX512-vectorized by xor_shift_vec(). */
    memcpy(out, coeffs, n * sizeof(*out));
    dmpf64_gf *h = coeffs + n;
    size_t tail = max_shift < n ? max_shift : n;
    size_t k = n;
    while (k-- > n - tail) {
        dmpf64_gf c = h[k];
        if (!c) continue;
        for (uint8_t u = 0; u < cnt; u++) {
            size_t d = k + shifts[u];
            if (d < n) out[d] ^= c;
            else h[d - n] ^= c;
        }
    }
    const size_t bulk = n - tail;
    for (uint8_t u = 0; u < cnt; u++) xor_shift_vec(out, h, shifts[u], bulk);
    return 0;
}

int dmpf64_afft_reduce_subspace_len2_inplace_cached(dmpf64_gf *out, size_t log_n,
                                                    dmpf64_gf *coeffs) {
    if (!out || !coeffs) return -1;
    size_t n = (size_t)1 << log_n;
    if (cyclic_cache_prepare_work(log_n, 1) != 0) return -1;
    if (support_all_one(&g_cyclic_cache.plan, log_n)) {
        return reduce_subspace_len2_ones_inplace(out, log_n, coeffs, &g_cyclic_cache.plan);
    }
    return dmpf64_afft_reduce_subspace_cached(out, log_n, coeffs, 2*n);
}

int dmpf64_afft_reduce_subspace_cached(dmpf64_gf *out, size_t log_n,
                                          const dmpf64_gf *coeffs, size_t len) {
    if (!out || (!coeffs && len)) return -1;
    size_t n = (size_t)1 << log_n;
    size_t max_len = n << 1;
    if (len > max_len) len = max_len;
    if (cyclic_cache_prepare_work(log_n, max_len) != 0) return -1;
    dmpf64_gf *tmp = g_cyclic_cache.work;
    memset(tmp, 0, max_len * sizeof(dmpf64_gf));
    if (len) memcpy(tmp, coeffs, len * sizeof(dmpf64_gf));
    const dmpf64_afft_plan *plan = &g_cyclic_cache.plan;
    const uint8_t cnt = plan->support_count[log_n];
    const uint8_t *poslist = plan->support_pos[log_n];
    for (size_t pos = max_len; pos-- > n;) {
        dmpf64_gf coeff = tmp[pos];
        if (!coeff) continue;
        tmp[pos] = 0;
        size_t base = pos - n;
        for (uint8_t u = 0; u < cnt; u++) {
            uint8_t t = poslist[u];
            dmpf64_gf sc = plan->s[log_n][t];
            size_t dst = base + ((size_t)1 << t);
            tmp[dst] ^= (sc == 1) ? coeff : gf_mul_const(coeff, sc);
        }
    }
    memcpy(out, tmp, n * sizeof(dmpf64_gf));
    return 0;
}


int dmpf64_afft_eval_subspace_cached(dmpf64_gf *out, size_t log_n,
                                     const dmpf64_gf *coeffs, size_t len) {
    if (!out || (!coeffs && len)) return -1;
    size_t n = (size_t)1 << log_n;
    size_t m = n << 1;
    if (len > m) len = m;
    if (cyclic_cache_prepare_work(log_n + 1, m) != 0) return -1;
    dmpf64_gf *buf = g_cyclic_cache.work;
    memset(buf, 0, m * sizeof(dmpf64_gf));
    if (len) memcpy(buf, coeffs, len * sizeof(dmpf64_gf));
    /* In the novel basis for V_{log_n+1}, coefficients split as
       f = r + q*S_log_n.  On the subspace V_log_n, S_log_n vanishes, so
       the lower half is exactly the reduced polynomial in novel basis. */
    mono_to_novel_prefix_inplace(buf, log_n + 1, len, &g_cyclic_cache.plan);
    dmpf64_afft_cyclic_cache sub = g_cyclic_cache;
    sub.log_m = log_n;
    sub.m = n;
    novel_eval_inplace_cached(&sub, buf);
    memcpy(out, buf, n * sizeof(dmpf64_gf));
    return 0;
}

int dmpf64_afft_eval_mono_cached(dmpf64_gf *out, size_t log_n,
                                    const dmpf64_gf *coeffs, size_t active_len) {
    if (!out || (!coeffs && active_len)) return -1;
    size_t n = (size_t)1 << log_n;
    if (active_len > n) active_len = n;
    if (cyclic_cache_prepare_work(log_n, n) != 0) return -1;
    dmpf64_gf *buf = g_cyclic_cache.work;
    memset(buf, 0, n * sizeof(dmpf64_gf));
    if (active_len) memcpy(buf, coeffs, active_len * sizeof(dmpf64_gf));
    mono_to_novel_prefix_inplace(buf, log_n, active_len, &g_cyclic_cache.plan);
    novel_eval_inplace_cached(&g_cyclic_cache, buf);
    memcpy(out, buf, n * sizeof(dmpf64_gf));
    return 0;
}



int dmpf64_afft_eval_mono_inplace_cached(dmpf64_gf *buf, size_t log_n,
                                         size_t active_len) {
    if (!buf) return -1;
    size_t n = (size_t)1 << log_n;
    if (active_len > n) active_len = n;
    if (cyclic_cache_prepare_work(log_n, 1) != 0) return -1;
    mono_to_novel_prefix_inplace(buf, log_n, active_len, &g_cyclic_cache.plan);
    novel_eval_inplace_cached(&g_cyclic_cache, buf);
    return 0;
}

int dmpf64_afft_eval_sparse_cached(dmpf64_gf *out, size_t log_n,
                                   const dmpf64_sparse_poly *p) {
    if (!out || !p) return -1;
    size_t n = (size_t)1 << log_n;
    /* Transform directly in the caller's output buffer.  The previous version
       transformed in the global cache work buffer and copied N GF64 words out;
       at N=2^24 that extra 128 MiB copy is visible in E2E setup time. */
    if (cyclic_cache_prepare_work(log_n, 1) != 0) return -1;
    memset(out, 0, n * sizeof(dmpf64_gf));
    size_t active_len = 0;
    for (size_t i = 0; i < p->len; i++) {
        size_t idx = p->terms[i].idx;
        if (idx >= n) idx &= (n - 1);
        out[idx] ^= p->terms[i].coeff;
        if (idx + 1 > active_len) active_len = idx + 1;
    }
    mono_to_novel_prefix_inplace(out, log_n, active_len, &g_cyclic_cache.plan);
    novel_eval_inplace_cached(&g_cyclic_cache, out);
    return 0;
}

static void copy_mono_to_eval_buffer(dmpf64_gf *dst, size_t m,
                                     const dmpf64_ring_elem *src) {
    memset(dst, 0, m * sizeof(dmpf64_gf));
    memcpy(dst, src->poly.c, src->mod.degree * sizeof(dmpf64_gf));
    mono_to_novel_prefix_inplace(dst, g_cyclic_cache.log_m, src->mod.degree, &g_cyclic_cache.plan);
    novel_eval_inplace_cached(&g_cyclic_cache, dst);
}

static void fold4_to(dmpf64_gf *out, const dmpf64_gf *coeff, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8) {
        __m512i c0 = _mm512_loadu_si512((const void *)(coeff + i));
        __m512i c1 = _mm512_loadu_si512((const void *)(coeff + n + i));
        __m512i c2 = _mm512_loadu_si512((const void *)(coeff + 2*n + i));
        __m512i c3 = _mm512_loadu_si512((const void *)(coeff + 3*n + i));
        __m512i v = _mm512_xor_si512(_mm512_xor_si512(c0, c1), _mm512_xor_si512(c2, c3));
        _mm512_storeu_si512((void *)(out + i), v);
    }
#endif
    for (; i < n; i++) out[i] = coeff[i] ^ coeff[n + i] ^ coeff[2*n + i] ^ coeff[3*n + i];
}

int dmpf64_afft_cyclic_sum_product3_generated(dmpf64_ring_elem *out,
                                                         dmpf64_cyclotomic_mod mod,
                                                         size_t count,
                                                         const dmpf64_ring_elem * const *a,
                                                         const dmpf64_ring_elem * const *b,
                                                         dmpf64_afft_fill_mono_fn fill_c,
                                                         void *ctx) {
    const size_t n = mod.degree;
    if (!fill_c) return -1;
    if (count == 0) {
        out->mod = mod;
        return dmpf64_dense_new(&out->poly, n);
    }
    for (size_t i = 0; i < count; i++) {
        if (!a[i] || !b[i]) return -1;
        if (a[i]->mod.degree != n || b[i]->mod.degree != n) return -1;
    }
    if (n <= 512) {
        out->mod = mod;
        if (dmpf64_dense_new(&out->poly, n) != 0) return -1;
        dmpf64_ring_elem ctmp = {0};
        ctmp.mod = mod;
        for (size_t i = 0; i < count; i++) {
            dmpf64_ring_elem ab = {0};
            if (dmpf64_dense_new(&ctmp.poly, n) != 0) { dmpf64_ring_free(out); return -1; }
            int rc = fill_c(ctx, i, ctmp.poly.c, n);
            if (rc == 0) rc = dmpf64_afft_cyclic_mul_cached(&ab, a[i], b[i]);
            if (rc == 0) rc = dmpf64_ring_mul_addto(out, &ab, &ctmp);
            dmpf64_ring_free(&ab);
            dmpf64_ring_free(&ctmp);
            if (rc != 0) { dmpf64_ring_free(out); return rc; }
        }
        return 0;
    }

    const size_t m = n << 2;
    size_t log_m = 0;
    for (size_t t = m; t > 1; t >>= 1) log_m++;

    size_t unique_cap = 2 * count;
    if (unique_cap == 0) unique_cap = 1;
    const dmpf64_ring_elem **uniq = (const dmpf64_ring_elem **)calloc(unique_cap, sizeof(*uniq));
    size_t *amap = (size_t *)calloc(count ? count : 1, sizeof(*amap));
    size_t *bmap = (size_t *)calloc(count ? count : 1, sizeof(*bmap));
    if (!uniq || !amap || !bmap) { free(uniq); free(amap); free(bmap); return -1; }
    size_t ucount = 0;
    for (size_t z = 0; z < count; z++) {
        size_t u = 0;
        for (; u < ucount; u++) if (uniq[u] == a[z]) break;
        if (u == ucount) uniq[ucount++] = a[z];
        amap[z] = u;
        u = 0;
        for (; u < ucount; u++) if (uniq[u] == b[z]) break;
        if (u == ucount) uniq[ucount++] = b[z];
        bmap[z] = u;
    }

    if (cyclic_cache_prepare_work(log_m, (2 + ucount) * m) != 0) {
        free(uniq); free(amap); free(bmap);
        return -1;
    }

    dmpf64_gf *acc = g_cyclic_cache.work;
    dmpf64_gf *ec  = acc + m;
    dmpf64_gf *uev = ec + m;
    memset(acc, 0, m * sizeof(dmpf64_gf));

    const char *trace = getenv("DMPF64_AFFT_TRACE");
    double t0 = trace ? afft_now_sec() : 0.0;
    double tf = 0.0, tdense = 0.0, tp = 0.0, ti = 0.0, tfo = 0.0, tfill = 0.0;

    double x0 = trace ? afft_now_sec() : 0.0;
    for (size_t u = 0; u < ucount; u++) copy_mono_to_eval_buffer(uev + u * m, m, uniq[u]);
    if (trace) tdense = afft_now_sec() - x0;

    for (size_t z = 0; z < count; z++) {
        x0 = trace ? afft_now_sec() : 0.0;
        memset(ec, 0, m * sizeof(dmpf64_gf));
        int rc = fill_c(ctx, z, ec, n);
        if (rc != 0) { free(uniq); free(amap); free(bmap); return rc; }
        if (trace) tfill += afft_now_sec() - x0;

        x0 = trace ? afft_now_sec() : 0.0;
        mono_to_novel_prefix_inplace(ec, log_m, n, &g_cyclic_cache.plan);
        novel_eval_inplace_cached(&g_cyclic_cache, ec);
        if (trace) tf += afft_now_sec() - x0;

        const dmpf64_gf *ea = uev + amap[z] * m;
        const dmpf64_gf *eb = uev + bmap[z] * m;
        x0 = trace ? afft_now_sec() : 0.0;
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
        size_t i = 0;
        for (; i + 8 <= m; i += 8) {
            __m512i va = _mm512_loadu_si512((const void *)(ea + i));
            __m512i vb = _mm512_loadu_si512((const void *)(eb + i));
            __m512i vc = _mm512_loadu_si512((const void *)(ec + i));
            __m512i prod = gf64_mul_vec8(gf64_mul_vec8(va, vb), vc);
            __m512i old = _mm512_loadu_si512((const void *)(acc + i));
            _mm512_storeu_si512((void *)(acc + i), _mm512_xor_si512(old, prod));
        }
        for (; i < m; i++) acc[i] ^= dmpf64_gf_mul(dmpf64_gf_mul(ea[i], eb[i]), ec[i]);
#else
        for (size_t i = 0; i < m; i++) acc[i] ^= dmpf64_gf_mul(dmpf64_gf_mul(ea[i], eb[i]), ec[i]);
#endif
        if (trace) tp += afft_now_sec() - x0;
    }
    free(uniq); free(amap); free(bmap);

    x0 = trace ? afft_now_sec() : 0.0;
    novel_interp_inplace_cached(&g_cyclic_cache, acc);
    novel_to_mono_inplace(acc, log_m, &g_cyclic_cache.plan);
    if (trace) ti = afft_now_sec() - x0;

    out->mod = mod;
    int rc = dense_alloc_uninit(&out->poly, n);
    if (rc == 0) {
        x0 = trace ? afft_now_sec() : 0.0;
        fold4_to(out->poly.c, acc, n);
        if (trace) tfo = afft_now_sec() - x0;
    }
    if (trace) {
        double total = afft_now_sec() - t0;
        fprintf(stderr, "AFFT3GEN_TRACE logm=%zu m=%zu count=%zu dense_fwd=%.6f share_fill=%.6f share_fwd=%.6f pointacc=%.6f inv+nov2mono=%.6f fold=%.6f total=%.6f\n",
                log_m, m, count, tdense, tfill, tf, tp, ti, tfo, total);
    }
    return rc;
}

int dmpf64_afft_cyclic_sum_product3(dmpf64_ring_elem *out,
                                    dmpf64_cyclotomic_mod mod,
                                    size_t count,
                                    const dmpf64_ring_elem * const *a,
                                    const dmpf64_ring_elem * const *b,
                                    const dmpf64_ring_elem * const *c) {
    const size_t n = mod.degree;
    if (count == 0) {
        out->mod = mod;
        return dmpf64_dense_new(&out->poly, n);
    }
    for (size_t i = 0; i < count; i++) {
        if (!a[i] || !b[i] || !c[i]) return -1;
        if (a[i]->mod.degree != n || b[i]->mod.degree != n || c[i]->mod.degree != n) return -1;
    }
    if (n <= 512) {
        out->mod = mod;
        if (dmpf64_dense_new(&out->poly, n) != 0) return -1;
        for (size_t i = 0; i < count; i++) {
            dmpf64_ring_elem ab = {0};
            int rc = dmpf64_afft_cyclic_mul_cached(&ab, a[i], b[i]);
            if (rc == 0) rc = dmpf64_ring_mul_addto(out, &ab, c[i]);
            dmpf64_ring_free(&ab);
            if (rc != 0) { dmpf64_ring_free(out); return rc; }
        }
        return 0;
    }

    /* For reduce(A*B*C), deg(A*B*C) < 3N, so a length-4N additive FFT is
       sufficient.  Accumulate in eval domain and invert/fold exactly once.
       The dense random ring elements are reused across the c^2 OLE pairs, so
       pretransform unique a/b pointers and only stream-transform the DMPF
       share element c_i. */
    const size_t m = n << 2;
    size_t log_m = 0;
    for (size_t t = m; t > 1; t >>= 1) log_m++;

    size_t unique_cap = 2 * count;
    if (unique_cap == 0) unique_cap = 1;
    const dmpf64_ring_elem **uniq = (const dmpf64_ring_elem **)calloc(unique_cap, sizeof(*uniq));
    size_t *amap = (size_t *)calloc(count ? count : 1, sizeof(*amap));
    size_t *bmap = (size_t *)calloc(count ? count : 1, sizeof(*bmap));
    if (!uniq || !amap || !bmap) { free(uniq); free(amap); free(bmap); return -1; }
    size_t ucount = 0;
    for (size_t z = 0; z < count; z++) {
        size_t u = 0;
        for (; u < ucount; u++) if (uniq[u] == a[z]) break;
        if (u == ucount) uniq[ucount++] = a[z];
        amap[z] = u;
        u = 0;
        for (; u < ucount; u++) if (uniq[u] == b[z]) break;
        if (u == ucount) uniq[ucount++] = b[z];
        bmap[z] = u;
    }

    if (cyclic_cache_prepare_work(log_m, (2 + ucount) * m) != 0) {
        free(uniq); free(amap); free(bmap);
        return -1;
    }

    dmpf64_gf *acc = g_cyclic_cache.work;
    dmpf64_gf *ec  = acc + m;
    dmpf64_gf *uev = ec + m;
    memset(acc, 0, m * sizeof(dmpf64_gf));

    const char *trace = getenv("DMPF64_AFFT_TRACE");
    double t0 = trace ? afft_now_sec() : 0.0;
    double tf = 0.0, tdense = 0.0, tp = 0.0, ti = 0.0, tfo = 0.0;

    double x0 = trace ? afft_now_sec() : 0.0;
    for (size_t u = 0; u < ucount; u++) copy_mono_to_eval_buffer(uev + u * m, m, uniq[u]);
    if (trace) tdense = afft_now_sec() - x0;

    for (size_t z = 0; z < count; z++) {
        x0 = trace ? afft_now_sec() : 0.0;
        copy_mono_to_eval_buffer(ec, m, c[z]);
        if (trace) tf += afft_now_sec() - x0;

        const dmpf64_gf *ea = uev + amap[z] * m;
        const dmpf64_gf *eb = uev + bmap[z] * m;
        x0 = trace ? afft_now_sec() : 0.0;
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
        size_t i = 0;
        for (; i + 8 <= m; i += 8) {
            __m512i va = _mm512_loadu_si512((const void *)(ea + i));
            __m512i vb = _mm512_loadu_si512((const void *)(eb + i));
            __m512i vc = _mm512_loadu_si512((const void *)(ec + i));
            __m512i prod = gf64_mul_vec8(gf64_mul_vec8(va, vb), vc);
            __m512i old = _mm512_loadu_si512((const void *)(acc + i));
            _mm512_storeu_si512((void *)(acc + i), _mm512_xor_si512(old, prod));
        }
        for (; i < m; i++) acc[i] ^= dmpf64_gf_mul(dmpf64_gf_mul(ea[i], eb[i]), ec[i]);
#else
        for (size_t i = 0; i < m; i++) acc[i] ^= dmpf64_gf_mul(dmpf64_gf_mul(ea[i], eb[i]), ec[i]);
#endif
        if (trace) tp += afft_now_sec() - x0;
    }
    free(uniq); free(amap); free(bmap);

    x0 = trace ? afft_now_sec() : 0.0;
    novel_interp_inplace_cached(&g_cyclic_cache, acc);
    novel_to_mono_inplace(acc, log_m, &g_cyclic_cache.plan);
    if (trace) ti = afft_now_sec() - x0;

    out->mod = mod;
    int rc = dense_alloc_uninit(&out->poly, n);
    if (rc == 0) {
        x0 = trace ? afft_now_sec() : 0.0;
        fold4_to(out->poly.c, acc, n);
        if (trace) tfo = afft_now_sec() - x0;
    }
    if (trace) {
        double total = afft_now_sec() - t0;
        fprintf(stderr, "AFFT3_TRACE logm=%zu m=%zu count=%zu dense_fwd=%.6f share_fwd=%.6f pointacc=%.6f inv+nov2mono=%.6f fold=%.6f total=%.6f\n",
                log_m, m, count, tdense, tf, tp, ti, tfo, total);
    }
    return rc;
}

int dmpf64_afft_cyclic_mul_cached(dmpf64_ring_elem *out,
                                  const dmpf64_ring_elem *a,
                                  const dmpf64_ring_elem *b) {
    if (a->mod.degree != b->mod.degree) return -1;
    size_t n = a->mod.degree;
    if (n <= 16) return dmpf64_afft_cyclic_mul(out, a, b);
    dmpf64_gf *coeff = NULL;
    size_t m = 0;
    int rc = cyclic_cached_coeffs(a, b, &coeff, &m);
    if (rc != 0) return rc;
    out->mod = a->mod;
    rc = dense_alloc_uninit(&out->poly, n);
    if (rc == 0) {
        (void)m;
        cyclic_fold_to(out->poly.c, coeff, n);
    }
    return rc;
}

int dmpf64_afft_cyclic_mul_addto_cached(dmpf64_ring_elem *acc,
                                        const dmpf64_ring_elem *a,
                                        const dmpf64_ring_elem *b) {
    if (a->mod.degree != b->mod.degree || acc->mod.degree != a->mod.degree) return -1;
    size_t n = a->mod.degree;
    if (n <= 16) {
        dmpf64_ring_elem tmp = {0};
        int rc = dmpf64_afft_cyclic_mul(&tmp, a, b);
        if (rc == 0) {
            for (size_t i = 0; i < n; i++) acc->poly.c[i] ^= tmp.poly.c[i];
        }
        dmpf64_ring_free(&tmp);
        return rc;
    }
    dmpf64_gf *coeff = NULL;
    size_t m = 0;
    int rc = cyclic_cached_coeffs(a, b, &coeff, &m);
    if (rc != 0) return rc;
    (void)m;
    cyclic_fold_addto(acc->poly.c, coeff, n);
    return 0;
}

const char *dmpf64_afft_backend(void) {
    return "normalized-cantor-novel-prefix-conversion-vpclmul-zero-one-xorshift-reduce";
}
