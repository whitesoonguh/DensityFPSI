#include "gf64x4_poly.h"
#include "gf64x4.h"

void gf64x4_poly_mul_school(const __m512i *a, int na,
                            const __m512i *b, int nb,
                            __m512i *c) {
    #pragma unroll                                
    for (int i = 0; i < na; i++) {
        __m512i ai = a[i];
        #pragma unroll 
        for (int j = 0; j < nb; j++)
            c[i + j] = gf64x4_add(c[i + j], gf64x4_mul(ai, b[j]));
    }
}

// Smaller Kernels
static inline void gf64x4_poly_mul_2x2_acc(
    const __m512i *__restrict a,
    const __m512i *__restrict b,
    __m512i *__restrict c) 
{
    __m512i c0 = c[0], c1 = c[1], c2 = c[2];
    const __m512i a0 = a[0], a1 = a[1];
    const __m512i b0 = b[0], b1 = b[1];

    c0 = gf64x4_add(c0, gf64x4_mul_no_red(a0, b0));
    c1 = gf64x4_add(c1, gf64x4_mul_no_red(a0, b1));
    c1 = gf64x4_add(c1, gf64x4_mul_no_red(a1, b0));
    c2 = gf64x4_add(c2, gf64x4_mul_no_red(a1, b1));
    c[0] = c0; c[1] = c1; c[2] = c2;
}

static inline void gf64x4_poly_mul_4x4_acc(
    const __m512i *__restrict a,
    const __m512i *__restrict b,
    __m512i *__restrict c)
{
    __m512i c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    __m512i c4 = c[4], c5 = c[5], c6 = c[6];

    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

    c0 = gf64x4_add(c0, gf64x4_mul_no_red(a0, b0));

    c1 = gf64x4_add(c1, gf64x4_mul_no_red(a0, b1));
    c1 = gf64x4_add(c1, gf64x4_mul_no_red(a1, b0));

    c2 = gf64x4_add(c2, gf64x4_mul_no_red(a0, b2));
    c2 = gf64x4_add(c2, gf64x4_mul_no_red(a1, b1));
    c2 = gf64x4_add(c2, gf64x4_mul_no_red(a2, b0));

    c3 = gf64x4_add(c3, gf64x4_mul_no_red(a0, b3));
    c3 = gf64x4_add(c3, gf64x4_mul_no_red(a1, b2));
    c3 = gf64x4_add(c3, gf64x4_mul_no_red(a2, b1));
    c3 = gf64x4_add(c3, gf64x4_mul_no_red(a3, b0));

    c4 = gf64x4_add(c4, gf64x4_mul_no_red(a1, b3));
    c4 = gf64x4_add(c4, gf64x4_mul_no_red(a2, b2));
    c4 = gf64x4_add(c4, gf64x4_mul_no_red(a3, b1));

    c5 = gf64x4_add(c5, gf64x4_mul_no_red(a2, b3));
    c5 = gf64x4_add(c5, gf64x4_mul_no_red(a3, b2));

    c6 = gf64x4_add(c6, gf64x4_mul_no_red(a3, b3));

    c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
    c[4] = c4; c[5] = c5; c[6] = c6;
}

static inline void gf64x4_poly_mul_8x8_acc(
    const __m512i *__restrict a,
    const __m512i *__restrict b,
    __m512i *__restrict c)
{
    __m512i c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3], c4 = c[4];
    __m512i c5 = c[5], c6 = c[6], c7 = c[7], c8 = c[8], c9 = c[9];
    __m512i c10 = c[10], c11 = c[11], c12 = c[12], c13 = c[13], c14 = c[14];

    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];

    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];

    c0  = gf64x4_add(c0,  gf64x4_mul_no_red(a0, b0));

    c1  = gf64x4_add(c1,  gf64x4_mul_no_red(a0, b1));
    c1  = gf64x4_add(c1,  gf64x4_mul_no_red(a1, b0));

    c2  = gf64x4_add(c2,  gf64x4_mul_no_red(a0, b2));
    c2  = gf64x4_add(c2,  gf64x4_mul_no_red(a1, b1));
    c2  = gf64x4_add(c2,  gf64x4_mul_no_red(a2, b0));

    c3  = gf64x4_add(c3,  gf64x4_mul_no_red(a0, b3));
    c3  = gf64x4_add(c3,  gf64x4_mul_no_red(a1, b2));
    c3  = gf64x4_add(c3,  gf64x4_mul_no_red(a2, b1));
    c3  = gf64x4_add(c3,  gf64x4_mul_no_red(a3, b0));

    c4  = gf64x4_add(c4,  gf64x4_mul_no_red(a0, b4));
    c4  = gf64x4_add(c4,  gf64x4_mul_no_red(a1, b3));
    c4  = gf64x4_add(c4,  gf64x4_mul_no_red(a2, b2));
    c4  = gf64x4_add(c4,  gf64x4_mul_no_red(a3, b1));
    c4  = gf64x4_add(c4,  gf64x4_mul_no_red(a4, b0));

    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a0, b5));
    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a1, b4));
    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a2, b3));
    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a3, b2));
    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a4, b1));
    c5  = gf64x4_add(c5,  gf64x4_mul_no_red(a5, b0));

    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a0, b6));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a1, b5));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a2, b4));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a3, b3));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a4, b2));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a5, b1));
    c6  = gf64x4_add(c6,  gf64x4_mul_no_red(a6, b0));

    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a0, b7));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a1, b6));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a2, b5));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a3, b4));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a4, b3));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a5, b2));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a6, b1));
    c7  = gf64x4_add(c7,  gf64x4_mul_no_red(a7, b0));

    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a1, b7));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a2, b6));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a3, b5));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a4, b4));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a5, b3));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a6, b2));
    c8  = gf64x4_add(c8,  gf64x4_mul_no_red(a7, b1));

    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a2, b7));
    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a3, b6));
    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a4, b5));
    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a5, b4));
    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a6, b3));
    c9  = gf64x4_add(c9,  gf64x4_mul_no_red(a7, b2));

    c10 = gf64x4_add(c10, gf64x4_mul_no_red(a3, b7));
    c10 = gf64x4_add(c10, gf64x4_mul_no_red(a4, b6));
    c10 = gf64x4_add(c10, gf64x4_mul_no_red(a5, b5));
    c10 = gf64x4_add(c10, gf64x4_mul_no_red(a6, b4));
    c10 = gf64x4_add(c10, gf64x4_mul_no_red(a7, b3));

    c11 = gf64x4_add(c11, gf64x4_mul_no_red(a4, b7));
    c11 = gf64x4_add(c11, gf64x4_mul_no_red(a5, b6));
    c11 = gf64x4_add(c11, gf64x4_mul_no_red(a6, b5));
    c11 = gf64x4_add(c11, gf64x4_mul_no_red(a7, b4));

    c12 = gf64x4_add(c12, gf64x4_mul_no_red(a5, b7));
    c12 = gf64x4_add(c12, gf64x4_mul_no_red(a6, b6));
    c12 = gf64x4_add(c12, gf64x4_mul_no_red(a7, b5));

    c13 = gf64x4_add(c13, gf64x4_mul_no_red(a6, b7));
    c13 = gf64x4_add(c13, gf64x4_mul_no_red(a7, b6));

    c14 = gf64x4_add(c14, gf64x4_mul_no_red(a7, b7));

    c[0] = c0;   c[1] = c1;   c[2] = c2;   c[3] = c3;   c[4] = c4;
    c[5] = c5;   c[6] = c6;   c[7] = c7;   c[8] = c8;   c[9] = c9;
    c[10] = c10; c[11] = c11; c[12] = c12; c[13] = c13; c[14] = c14;
}

// static inline void gf64x4_poly_zero(__m512i *c, int nc)
// {
//     const __m512i z = _mm512_setzero_si512();
//     for (int i = 0; i < nc; i++) c[i] = z;
// }

/* fallback: arbitrary small tail */
static inline void gf64x4_poly_mul_school_acc(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb,
    __m512i *__restrict c)
{
    for (int i = 0; i < na; i++) {
        const __m512i ai = a[i];
        for (int j = 0; j < nb; j++) {
            c[i + j] = gf64x4_add(c[i + j], gf64x4_mul_no_red(ai, b[j]));
        }
    }
}

/* accumulate kernel dispatcher */
static inline void gf64x4_poly_mul_block_acc(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb,
    __m512i *__restrict c)
{
    if (na == 8 && nb == 8) {
        gf64x4_poly_mul_8x8_acc(a, b, c);
    } else if (na == 4 && nb == 4) {
        gf64x4_poly_mul_4x4_acc(a, b, c);
    } else if (na == 2 && nb == 2) {
        gf64x4_poly_mul_2x2_acc(a, b, c);
    } else {
        gf64x4_poly_mul_school_acc(a, na, b, nb, c);
    }
}

/* public API: c = a * b */
void gf64x4_poly_mul_acc(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb,
    __m512i *__restrict c)
{
    const int nc = na + nb - 1;
    for (int ib = 0; ib < na; ib += 8) {
        int la = na - ib;
        if (la > 8) la = 8;

        for (int jb = 0; jb < nb; jb += 8) {
            int lb = nb - jb;
            if (lb > 8) lb = 8;

            gf64x4_poly_mul_block_acc(
                a + ib, la,
                b + jb, lb,
                c + ib + jb
            );
        }
    }
}

void gf64x4_poly_mul(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb,
    __m512i *__restrict c)
{
    const int nc = na + nb - 1;
    gf64x4_poly_zero(c, nc);
    gf64x4_poly_mul_acc(a, na, b, nb, c);
    for (int i = 0; i < nc; i++) {
        c[i] = gf64x4_reduce(c[i]);
    }
}
