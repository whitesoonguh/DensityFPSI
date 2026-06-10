#ifndef DMPF64_POLYNOMIAL_H
#define DMPF64_POLYNOMIAL_H
#include "common.h"
#include "gf64.h"
#include "rng.h"
#include "dpf.h"

typedef struct { size_t idx; dmpf64_gf coeff; } dmpf64_sparse_term;
typedef struct { dmpf64_sparse_term *terms; size_t len; size_t cap; } dmpf64_sparse_poly;
typedef struct { dmpf64_gf *c; size_t len; } dmpf64_dense_poly;

DMPF64_API void dmpf64_dense_free(dmpf64_dense_poly *p);
DMPF64_API int dmpf64_dense_new(dmpf64_dense_poly *p, size_t len);
DMPF64_API int dmpf64_dense_random(dmpf64_dense_poly *p, size_t len, dmpf64_rng *rng);
DMPF64_API bool dmpf64_dense_eq_trim(const dmpf64_dense_poly *a, const dmpf64_dense_poly *b);
DMPF64_API int dmpf64_dense_add(dmpf64_dense_poly *out, const dmpf64_dense_poly *a, const dmpf64_dense_poly *b);
DMPF64_API int dmpf64_dense_mul_naive(dmpf64_dense_poly *out, const dmpf64_dense_poly *a, const dmpf64_dense_poly *b);
DMPF64_API dmpf64_gf dmpf64_dense_eval(const dmpf64_dense_poly *p, dmpf64_gf x);

DMPF64_API void dmpf64_sparse_free(dmpf64_sparse_poly *p);
DMPF64_API int dmpf64_sparse_random(dmpf64_sparse_poly *p, size_t degree, size_t weight, dmpf64_rng *rng);
DMPF64_API int dmpf64_sparse_random_regular(dmpf64_sparse_poly *p, size_t degree, size_t weight, dmpf64_rng *rng);
DMPF64_API int dmpf64_sparse_mul(dmpf64_sparse_poly *out, const dmpf64_sparse_poly *a, const dmpf64_sparse_poly *b);
DMPF64_API int dmpf64_sparse_to_dense(dmpf64_dense_poly *out, const dmpf64_sparse_poly *p);

DMPF64_API int dmpf64_share_polynomial(dmpf64_dpf_dmpf_key *k0, dmpf64_dpf_dmpf_key *k1, const dmpf64_sparse_poly *p, size_t points, size_t log_degree, dmpf64_rng *rng);
DMPF64_API int dmpf64_from_share(dmpf64_dense_poly *out, const dmpf64_dpf_dmpf_key *k);
DMPF64_API int dmpf64_share_regular_polynomial_mul(dmpf64_dpf_dmpf_key **blocks0, dmpf64_dpf_dmpf_key **blocks1, size_t *block_count, size_t *block_size, const dmpf64_sparse_poly *p, size_t sqrt_points, size_t log_degree, dmpf64_rng *rng);
DMPF64_API int dmpf64_from_share_regular_error(dmpf64_dense_poly *out, const dmpf64_dpf_dmpf_key *blocks, size_t block_count, size_t block_size, size_t output_degree);

#endif
