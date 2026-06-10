#include "gf64x4_linalg.h"
#include "gf64_lin.h"
#include <string.h>

/* Linear algebra operations for GF(2^64) vectors/polynomials */
/* Compute the inner product of two vectors */
static inline __m512i gf64x4_inner_prod(    
    const int64_t *__restrict a,        
    const __m512i *__restrict b, int n
) {
    __m512i acc = _mm512_setzero_si512();
    __m512i av;
    #pragma unroll
    for (int i = 0; i < n; i++) {
        av = _mm512_set1_epi64(a[i]);
        acc = gf64x4_add(acc, gf64x4_mul_no_red(av, b[i]));
    }
    return gf64x4_reduce(acc);
}

/* Compute the inner product of two polynomial vectors */
void gf64x4_poly_inner_prod(
    const __m512i *__restrict a, int na,
    const __m512i *__restrict b, int nb, int n,
    __m512i *__restrict c
) {
    const int nc = na + nb - 1;
    gf64x4_poly_zero(c, nc);

    for (int i = 0; i < n; i++) {
        gf64x4_poly_mul_acc(a + i * na, na, b + i * nb, nb, c);
    }
    for (int i = 0; i < nc; i++) {
        c[i] = gf64x4_reduce(c[i]);
    }    
}

/* Compute vector-matrix product v*M = y. v: 1 x K, M: K x N */
static inline void gf64x4_vecmat_prod(
    const int64_t *__restrict v,        
    const __m512i *__restrict M,
    int K, int N, __m512i *__restrict y
) {
    for (int n = 0; n < N; n++) y[n] = _mm512_setzero_si512();
    for (int k = 0; k < K; k++) {
        const __m512i av = _mm512_set1_epi64(v[k]);
        const __m512i *Mr = M + (size_t)k * N;
        for (int n = 0; n < N; n++) {
            y[n] = gf64x4_add(y[n], gf64x4_mul_no_red(av, Mr[n]));
        }
    }
    for (int n = 0; n < N; n++) y[n] = gf64x4_reduce(y[n]);
}

/* Compute matrix-vector product M*x=y. M = N * K, x = K * 1 */
static inline void gf64x4_matvec_prod(
    const int64_t *__restrict M,
    const __m512i *__restrict x, 
    int N, int K,
    __m512i *__restrict y
) {
    for (int n = 0; n < N; n++) y[n] = _mm512_setzero_si512();
    for (int k = 0; k < K; k++) {
        const __m512i xk = x[k];
        for (int n = 0; n < N; n++) {
            y[n]=gf64x4_add(
                y[n], gf64x4_mul_no_red(_mm512_set1_epi64(M[n * K + k]), xk)
            );
        }
    }
    for (int n = 0; n < N; n++) y[n] = gf64x4_reduce(y[n]);
}

/* Compute matrix-matrix product A*B = C. A: M X K, B: K X N 
* Remark: A: row-major, B: row-major, C: row-major
*/
void gf64x4_matmat_prod(
    const int64_t *__restrict A,
    const __m512i *__restrict B,
    int M, int K, int N, __m512i *__restrict C
) {
    for (int r = 0; r < M; r++) {
        gf64x4_vecmat_prod(A + (size_t)r * K, B, K, N, C + (size_t)r * N);
    }
}

/* Multipoint evaluation using Vandermonde Matrix */
void gf64x4_poly_coeff_to_eval(
    const int64_t* V, int nv, 
    const __m512i* a, int na,
    __m512i* y
) {
    gf64x4_matvec_prod(V,a,nv,na,y);
}

/* Interpolation using Vandermonde Matrix */
void gf64x4_poly_eval_to_coeff(
    const int64_t* Vinv, int nv,
    const __m512i* a, int deg,
    __m512i* y
) {
    gf64x4_matvec_prod(Vinv,a,deg,nv,y);
}

/* Random Point Sampling */
void gf64x4_poly_random_sample(
    const int64_t *__restrict Vcr, int nv,
    const __m512i *__restrict a, int deg,
    __m512i *__restrict y
) {
    for (int i = 0; i < deg; i++)
        y[i] = a[i];                              /* top block is the identity */
 
    gf64x4_matvec_prod(Vcr, a, nv - deg, deg, y + deg);
}
