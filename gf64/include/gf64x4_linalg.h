#ifndef GF64X4_LIN_H
#define GF64X4_LIN_H

#include "gf64.h"
#include "gf64x4.h"
#include "gf64x4_poly.h"

/* Linear algebra operations for GF(2^64) vectors/polynomials */
/* Compute the inner product of two polynomial vectors */
void gf64x4_poly_inner_prod(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb, int n,
    __m512i *__restrict c
);

/* Compute matrix-matrix product A*B = C. A: M X K, B: K X N 
* Remark: A: row-major, B: row-major, C: row-major
*/
void gf64x4_matmat_prod(
    const int64_t *__restrict A,
    const __m512i *__restrict B,
    int M, int K, int N, __m512i *__restrict C
);



/* Multipoint evaluation using Vandermonde Matrix */
void gf64x4_poly_coeff_to_eval(
    const int64_t* V, int nv, 
    const __m512i* a, int na,
    __m512i* y
);

/* Interpolation using Vandermonde Matrix */
void gf64x4_poly_eval_to_coeff(
    const int64_t* Vinv, int nv,
    const __m512i* a, int deg,
    __m512i* y
);

/* Random Point Sampling */
void gf64x4_poly_random_sample(
    const int64_t *__restrict Vcr, int nv,
    const __m512i *__restrict a, int deg,
    __m512i *__restrict y
);

#endif