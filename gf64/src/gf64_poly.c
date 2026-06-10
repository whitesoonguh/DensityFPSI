#include "gf64_poly.h"
#include "gf64.h"

#include <stdlib.h>

void gf64_poly_mul_school(const uint64_t *a, int na,
                          const uint64_t *b, int nb,
                          uint64_t *c) {
    for (int i = 0; i < na; i++) {
        uint64_t ai = a[i];
        if (!ai) continue;                 /* skip zero coeffs */
        for (int j = 0; j < nb; j++)
            c[i + j] ^= gf64_mul(ai, b[j]);
    }
}

/* threshold below which Karatsuba defers to schoolbook */
#define GF64_POLY_KARA_CUTOFF 16

void gf64_poly_mul_kara(const uint64_t *a, int na,
                        const uint64_t *b, int nb,
                        uint64_t *c) {
    if (na <= GF64_POLY_KARA_CUTOFF || nb <= GF64_POLY_KARA_CUTOFF) {
        gf64_poly_mul_school(a, na, b, nb, c);
        return;
    }

    int k = (na > nb ? na : nb) / 2;
    if (k >= na || k >= nb) {              /* lopsided: just do schoolbook */
        gf64_poly_mul_school(a, na, b, nb, c);
        return;
    }

    int n0 = k, n1 = na - k, m0 = k, m1 = nb - k;
    const uint64_t *a0 = a, *a1 = a + k, *b0 = b, *b1 = b + k;

    int top = (n1 > m1 ? n1 : m1);
    /* char 2: (a0 + a1), (b0 + b1) are just XORs, no sign handling */
    uint64_t *as = calloc((size_t)top, sizeof(uint64_t));
    uint64_t *bs = calloc((size_t)top, sizeof(uint64_t));
    for (int i = 0; i < n0; i++) as[i] ^= a0[i];
    for (int i = 0; i < n1; i++) as[i] ^= a1[i];
    for (int i = 0; i < m0; i++) bs[i] ^= b0[i];
    for (int i = 0; i < m1; i++) bs[i] ^= b1[i];

    uint64_t *z0 = calloc((size_t)(n0 + m0 - 1 > 0 ? n0 + m0 - 1 : 1), sizeof(uint64_t));
    uint64_t *z2 = calloc((size_t)(n1 + m1 - 1), sizeof(uint64_t));
    uint64_t *z1 = calloc((size_t)(2 * top - 1), sizeof(uint64_t));

    gf64_poly_mul_kara(a0, n0, b0, m0, z0);
    gf64_poly_mul_kara(a1, n1, b1, m1, z2);
    gf64_poly_mul_kara(as, top, bs, top, z1);

    /* z1 <- z1 - z0 - z2  (XOR in char 2) */
    for (int i = 0; i < n0 + m0 - 1; i++) z1[i] ^= z0[i];
    for (int i = 0; i < n1 + m1 - 1; i++) z1[i] ^= z2[i];

    /* c = z0 + z1 * X^k + z2 * X^{2k} */
    for (int i = 0; i < n0 + m0 - 1; i++) c[i]         ^= z0[i];
    for (int i = 0; i < 2 * top - 1;  i++) c[i + k]     ^= z1[i];
    for (int i = 0; i < n1 + m1 - 1; i++) c[i + 2 * k] ^= z2[i];

    free(as); free(bs); free(z0); free(z1); free(z2);
}

uint64_t gf64_poly_eval(const uint64_t *a, int n, uint64_t x) {
    uint64_t acc = 0;
    for (int i = n - 1; i >= 0; i--)
        acc = gf64_add(gf64_mul(acc, x), a[i]);
    return acc;
}


// uint64_t gf64_poly_deg(const uint64_t *a, int n) {
//     while (n > 0 && a[n - 1] == 0)
//         n--;
//     return n - 1;
// }

// // Make a monic polynomial in-place, return degree, or -1 if zero poly.
// int gf64_poly_make_monic(uint64_t *a, int n)
// {
//     int d = gf64_poly_deg(a, n);
//     if (d < 0)
//         return -1;

//     uint64_t lc = a[d];
//     uint64_t inv = gf64_inv(lc);

//     for (int i = 0; i <= d; i++)
//         a[i] = gf64_mul(a[i], inv);

//     return d;
// }

// void gf64_poly_divrem(
//     const uint64_t *a, int na,
//     const uint64_t *b, int nb,
//     uint64_t *q,
//     uint64_t *r)
// {
//     int da = gf64_poly_deg(a, na);
//     int db = gf64_poly_deg(b, nb);

//     memset(q, 0, (na > 0 ? na : 1) * sizeof(uint64_t));
//     memcpy(r, a, na * sizeof(uint64_t));

//     if (db < 0)
//         return;

//     uint64_t inv_lc = gf64_inv(b[db]);

//     while (da >= db) {
//         int k = da - db;

//         uint64_t lead = r[da];
//         uint64_t f = gf64_mul(lead, inv_lc);

//         q[k] ^= f;

//         for (int j = 0; j <= db; j++) {
//             r[j + k] ^= gf64_mul(f, b[j]);
//         }

//         while (da >= 0 && r[da] == 0)
//             da--;
//     }
// }

// void gf64_poly_mod(
//     const uint64_t *a, int na,
//     const uint64_t *b, int nb,
//     uint64_t *r)
// {
//     uint64_t *q = (uint64_t *)calloc((na > 0 ? na : 1), sizeof(uint64_t));
//     gf64_poly_divrem(a, na, b, nb, q, r);
//     free(q);
// }

// int gf64_poly_gcd(
//     const uint64_t *a, int na,
//     const uint64_t *b, int nb,
//     uint64_t *g)
// {
//     int da = gf64_poly_deg(a, na);
//     int db = gf64_poly_deg(b, nb);

//     if (da < 0 && db < 0)
//         return -1;

//     int cap = (na > nb ? na : nb);

//     uint64_t *A = (uint64_t *)calloc(cap, sizeof(uint64_t));
//     uint64_t *B = (uint64_t *)calloc(cap, sizeof(uint64_t));
//     uint64_t *R = (uint64_t *)calloc(cap, sizeof(uint64_t));
//     uint64_t *Q = (uint64_t *)calloc(cap, sizeof(uint64_t));

//     memcpy(A, a, na * sizeof(uint64_t));
//     memcpy(B, b, nb * sizeof(uint64_t));

//     int degA = da;
//     int degB = db;

//     while (degB >= 0) {
//         memset(R, 0, cap * sizeof(uint64_t));
//         memset(Q, 0, cap * sizeof(uint64_t));

//         gf64_poly_divrem(A, degA + 1, B, degB + 1, Q, R);

//         int degR = gf64_poly_deg(R, cap);

//         uint64_t *tmp = A;
//         A = B;
//         B = R;
//         R = tmp;

//         degA = degB;
//         degB = degR;
//     }

//     int dg = gf64_poly_make_monic(A, degA + 1);

//     memset(g, 0, cap * sizeof(uint64_t));
//     memcpy(g, A, (dg + 1) * sizeof(uint64_t));

//     free(A);
//     free(B);
//     free(R);
//     free(Q);
//     return dg;
// }

// Random Sampling 


// Batch Evaluation (with Precomputation)



// Batch Interpolation (with Precomputation)



// Batch Random Sampling on Evaluation Domain (with Precomputation)


#include <string.h>

int gf64_poly_trim_len(const uint64_t *a, int n) {
    while (n > 0 && a[n - 1] == 0) --n;
    return n;
}

int gf64_poly_degree(const uint64_t *a, int n) {
    return gf64_poly_trim_len(a, n) - 1;
}

int gf64_poly_make_monic(uint64_t *a, int n) {
    int d = gf64_poly_degree(a, n);
    if (d < 0) return -1;
    uint64_t lc = a[d];
    if (lc == 1) return d;
    uint64_t inv = gf64_inv(lc);
    for (int i = 0; i <= d; ++i) a[i] = gf64_mul(a[i], inv);
    return gf64_poly_degree(a, d + 1);
}

int gf64_poly_mod(uint64_t *r, int rcap,
                  const uint64_t *a, int na,
                  const uint64_t *m, int nm) {
    int md = gf64_poly_degree(m, nm);
    if (md < 0 || rcap <= 0) return -1;
    memset(r, 0, (size_t)rcap * sizeof(uint64_t));
    int copy = na < rcap ? na : rcap;
    if (copy > 0) memcpy(r, a, (size_t)copy * sizeof(uint64_t));
    int rd = gf64_poly_degree(r, rcap);
    uint64_t inv_lc = gf64_inv(m[md]);
    while (rd >= md) {
        uint64_t scale = gf64_mul(r[rd], inv_lc);
        int shift = rd - md;
        for (int i = 0; i <= md; ++i) r[i + shift] ^= gf64_mul(scale, m[i]);
        rd = gf64_poly_degree(r, rd + 1);
    }
    return rd + 1;
}

int gf64_poly_div_exact(uint64_t *q, int qcap,
                        const uint64_t *a, int na,
                        const uint64_t *d, int nd) {
    int dd = gf64_poly_degree(d, nd);
    int ad = gf64_poly_degree(a, na);
    if (dd < 0 || qcap <= 0) return -1;
    memset(q, 0, (size_t)qcap * sizeof(uint64_t));
    if (ad < dd) return (ad < 0) ? 0 : -1;
    uint64_t *r = (uint64_t *)calloc((size_t)(ad + 1), sizeof(uint64_t));
    if (!r) return -1;
    memcpy(r, a, (size_t)(ad + 1) * sizeof(uint64_t));
    uint64_t inv_lc = gf64_inv(d[dd]);
    int rd = ad;
    while (rd >= dd) {
        uint64_t scale = gf64_mul(r[rd], inv_lc);
        int shift = rd - dd;
        if (shift >= qcap) { free(r); return -1; }
        q[shift] ^= scale;
        for (int i = 0; i <= dd; ++i) r[i + shift] ^= gf64_mul(scale, d[i]);
        rd = gf64_poly_degree(r, rd + 1);
    }
    int exact = (rd < 0);
    free(r);
    if (!exact) return -1;
    return gf64_poly_trim_len(q, qcap);
}

int gf64_poly_gcd(uint64_t *g, int gcap,
                  const uint64_t *a, int na,
                  const uint64_t *b, int nb) {
    int cap = na > nb ? na : nb;
    if (cap <= 0 || gcap <= 0) return -1;
    uint64_t *A = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *B = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *R = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    if (!A || !B || !R) { free(A); free(B); free(R); return -1; }
    int ca = na < cap ? na : cap;
    int cb = nb < cap ? nb : cap;
    if (ca > 0) memcpy(A, a, (size_t)ca * sizeof(uint64_t));
    if (cb > 0) memcpy(B, b, (size_t)cb * sizeof(uint64_t));
    int degA = gf64_poly_degree(A, cap);
    int degB = gf64_poly_degree(B, cap);
    while (degB >= 0) {
        gf64_poly_mod(R, cap, A, degA + 1, B, degB + 1);
        uint64_t *T = A; A = B; B = R; R = T;
        degA = degB;
        degB = gf64_poly_degree(B, cap);
    }
    memset(g, 0, (size_t)gcap * sizeof(uint64_t));
    if (degA < 0) { free(A); free(B); free(R); return -1; }
    if (degA + 1 > gcap) { free(A); free(B); free(R); return -1; }
    memcpy(g, A, (size_t)(degA + 1) * sizeof(uint64_t));
    int gd = gf64_poly_make_monic(g, degA + 1);
    free(A); free(B); free(R);
    return gd;
}

int gf64_poly_square_mod(uint64_t *out, int outcap,
                         const uint64_t *a, int na,
                         const uint64_t *m, int nm) {
    int ad = gf64_poly_degree(a, na);
    if (ad < 0) { if (outcap > 0) memset(out, 0, (size_t)outcap * sizeof(uint64_t)); return 0; }
    int sn = 2 * ad + 1;
    uint64_t *sq = (uint64_t *)calloc((size_t)sn, sizeof(uint64_t));
    if (!sq) return -1;
    for (int i = 0; i <= ad; ++i) sq[2 * i] = gf64_square(a[i]);
    int ret = gf64_poly_mod(out, outcap, sq, sn, m, nm);
    free(sq);
    return ret;
}

int gf64_poly_trace_mod(uint64_t *out, int outcap,
                        const uint64_t *a, int na,
                        const uint64_t *m, int nm) {
    if (outcap <= 0) return -1;
    uint64_t *z = (uint64_t *)calloc((size_t)outcap, sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)calloc((size_t)outcap, sizeof(uint64_t));
    if (!z || !tmp) { free(z); free(tmp); return -1; }
    int zn = gf64_poly_mod(z, outcap, a, na, m, nm);
    if (zn < 0) { free(z); free(tmp); return -1; }
    memset(out, 0, (size_t)outcap * sizeof(uint64_t));
    for (int i = 0; i < outcap; ++i) out[i] = z[i];
    for (int i = 1; i < 64; ++i) {
        int tn = gf64_poly_square_mod(tmp, outcap, z, outcap, m, nm);
        if (tn < 0) { free(z); free(tmp); return -1; }
        memcpy(z, tmp, (size_t)outcap * sizeof(uint64_t));
        for (int j = 0; j < outcap; ++j) out[j] ^= z[j];
    }
    memcpy(tmp, out, (size_t)outcap * sizeof(uint64_t));
    int ret = gf64_poly_mod(out, outcap, tmp, outcap, m, nm);
    free(z); free(tmp);
    return ret;
}

static uint64_t gf64_poly_splitmix(uint64_t *st) {
    uint64_t x = (*st += 0x9e3779b97f4a7c15ULL);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static int gf64_u64_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int gf64_poly_split_rec(uint64_t *roots, int *nroots, int rootcap,
                               const uint64_t *f_in, int nf, uint64_t *rng) {
    int d = gf64_poly_degree(f_in, nf);
    if (d <= 0) return 0;
    if (d == 1) {
        if (*nroots >= rootcap) return -1;
        roots[(*nroots)++] = f_in[0];
        return 0;
    }
    int cap = d + 1;
    uint64_t *f = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *a = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *tr = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *g = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *q = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    if (!f || !a || !tr || !g || !q) { free(f); free(a); free(tr); free(g); free(q); return -1; }
    memcpy(f, f_in, (size_t)cap * sizeof(uint64_t));
    gf64_poly_make_monic(f, cap);
    for (int attempt = 0; attempt < 256; ++attempt) {
        memset(a, 0, (size_t)cap * sizeof(uint64_t));
        for (int i = 0; i < d; ++i) a[i] = gf64_poly_splitmix(rng);
        if (gf64_poly_degree(a, d) < 0) continue;
        if (gf64_poly_trace_mod(tr, cap, a, d, f, cap) < 0) continue;
        int gd = gf64_poly_gcd(g, cap, f, cap, tr, cap);
        if (gd > 0 && gd < d) {
            int qn = gf64_poly_div_exact(q, cap, f, cap, g, gd + 1);
            if (qn < 0) { free(f); free(a); free(tr); free(g); free(q); return -1; }
            int r1 = gf64_poly_split_rec(roots, nroots, rootcap, g, gd + 1, rng);
            int r2 = gf64_poly_split_rec(roots, nroots, rootcap, q, qn, rng);
            free(f); free(a); free(tr); free(g); free(q);
            return (r1 || r2) ? -1 : 0;
        }
    }
    free(f); free(a); free(tr); free(g); free(q);
    return -1;
}

int gf64_poly_roots_split(uint64_t *roots, int rootcap,
                          const uint64_t *f, int nf,
                          uint64_t seed) {
    if (rootcap <= 0) return -1;
    int cap = gf64_poly_trim_len(f, nf);
    if (cap <= 1) return 0;
    uint64_t *mono = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    if (!mono) return -1;
    memcpy(mono, f, (size_t)cap * sizeof(uint64_t));
    gf64_poly_make_monic(mono, cap);
    int nroots = 0;
    uint64_t rng = seed;
    int ret = gf64_poly_split_rec(roots, &nroots, rootcap, mono, cap, &rng);
    free(mono);
    if (ret != 0) return -1;
    qsort(roots, (size_t)nroots, sizeof(uint64_t), gf64_u64_cmp);
    int w = 0;
    for (int i = 0; i < nroots; ++i) {
        if (w == 0 || roots[i] != roots[w - 1]) roots[w++] = roots[i];
    }
    return w;
}
