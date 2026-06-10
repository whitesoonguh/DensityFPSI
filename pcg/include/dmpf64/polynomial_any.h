#ifndef DMPF64_POLYNOMIAL_ANY_H
#define DMPF64_POLYNOMIAL_ANY_H
#include "polynomial.h"
#include "dmpf_any.h"

typedef struct {
    size_t block_size;
    size_t block_count;
    dmpf64_any_key *blocks;
} dmpf64_any_regular_share;

DMPF64_API int dmpf64_share_polynomial_any(dmpf64_any_key *k0, dmpf64_any_key *k1, dmpf64_backend backend,
                                           const dmpf64_sparse_poly *p, size_t points,
                                           size_t log_degree, dmpf64_rng *rng, size_t batch_size);
DMPF64_API int dmpf64_from_share_any(dmpf64_dense_poly *out, const dmpf64_any_key *k);
DMPF64_API int dmpf64_share_regular_polynomial_mul_any(dmpf64_any_key **blocks0, dmpf64_any_key **blocks1,
                                                       size_t *block_count, size_t *block_size,
                                                       dmpf64_backend backend, const dmpf64_sparse_poly *p,
                                                       size_t sqrt_points, size_t log_degree,
                                                       dmpf64_rng *rng, size_t batch_size);
DMPF64_API int dmpf64_from_share_regular_error_any(dmpf64_dense_poly *out, const dmpf64_any_key *blocks,
                                                   size_t block_count, size_t block_size, size_t output_degree);
DMPF64_API int dmpf64_from_share_regular_error_mod_any(dmpf64_dense_poly *out, const dmpf64_any_key *blocks,
                                                       size_t block_count, size_t block_size, size_t mod_degree);
DMPF64_API int dmpf64_from_share_regular_error_mod_into_any(dmpf64_gf *out, const dmpf64_any_key *blocks,
                                                            size_t block_count, size_t block_size, size_t mod_degree);
/* Write a contiguous regular-error share without modular wrap.  Unlike the
 * _mod_into variant this overwrites out[0..output_degree), so callers do not
 * need to pre-zero a full coefficient buffer when blocks tile the output. */
DMPF64_API int dmpf64_from_share_regular_error_write_any(dmpf64_gf *out, const dmpf64_any_key *blocks,
                                                         size_t block_count, size_t block_size, size_t output_degree);
/* XOR a contiguous regular-error share into out[0..output_degree).  The caller
 * owns zeroing/initializing out.  This is used by eval-domain OLE expansion to
 * exploit linearity and combine symmetric tensor-product shares before the
 * expensive reduce+AFFT stage. */
DMPF64_API int dmpf64_from_share_regular_error_xor_write_any(dmpf64_gf *out, const dmpf64_any_key *blocks,
                                                             size_t block_count, size_t block_size, size_t output_degree);
DMPF64_API void dmpf64_any_regular_share_free(dmpf64_any_regular_share *s);

#endif
