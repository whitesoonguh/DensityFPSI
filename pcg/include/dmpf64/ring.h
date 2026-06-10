#ifndef DMPF64_RING_H
#define DMPF64_RING_H
#include "common.h"
#include "polynomial.h"

typedef struct { size_t log_degree; size_t degree; } dmpf64_cyclotomic_mod;
typedef struct { dmpf64_dense_poly poly; dmpf64_cyclotomic_mod mod; } dmpf64_ring_elem;
typedef struct { dmpf64_sparse_poly poly; dmpf64_cyclotomic_mod mod; } dmpf64_sparse_ring_elem;

static inline dmpf64_cyclotomic_mod dmpf64_cyclotomic_new(size_t log_degree) { return (dmpf64_cyclotomic_mod){log_degree, (size_t)1 << log_degree}; }
DMPF64_API int dmpf64_mod_dense(dmpf64_dense_poly *out, const dmpf64_dense_poly *in, dmpf64_cyclotomic_mod mod);
DMPF64_API void dmpf64_ring_free(dmpf64_ring_elem *e);
DMPF64_API void dmpf64_sparse_ring_free(dmpf64_sparse_ring_elem *e);
DMPF64_API int dmpf64_ring_new(dmpf64_ring_elem *e, const dmpf64_dense_poly *p, dmpf64_cyclotomic_mod mod);
DMPF64_API int dmpf64_sparse_ring_new(dmpf64_sparse_ring_elem *e, const dmpf64_sparse_poly *p, dmpf64_cyclotomic_mod mod);
DMPF64_API int dmpf64_ring_add(dmpf64_ring_elem *out, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_ring_add_inplace(dmpf64_ring_elem *acc, const dmpf64_ring_elem *x);
DMPF64_API int dmpf64_ring_mul(dmpf64_ring_elem *out, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_ring_mul_addto(dmpf64_ring_elem *acc, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_sparse_ring_mul_dense(dmpf64_ring_elem *out, const dmpf64_sparse_ring_elem *a, const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_sparse_ring_mul_dense_addto(dmpf64_ring_elem *acc, const dmpf64_sparse_ring_elem *a, const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_ring_eq(const dmpf64_ring_elem *a, const dmpf64_ring_elem *b);

#endif
