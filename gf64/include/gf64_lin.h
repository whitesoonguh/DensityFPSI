#ifndef GF64_LIN_H
#define GF64_LIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t gf64_vec_dot(const uint64_t *a, const uint64_t *b, int n);
void gf64_matvec_rowmajor(const uint64_t *A, const uint64_t *x,
                          int rows, int cols, uint64_t *y);
int gf64_mat_inverse(const uint64_t *A, int n, uint64_t *Ainv);
int gf64_vandermonde_matrices(const uint64_t *alpha, int n,
                              uint64_t *V, uint64_t *Vinv);

#ifdef __cplusplus
}
#endif

#endif /* GF64_LIN_H */
