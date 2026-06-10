#ifndef DMPF64_OLE_PCG_BACKEND_H
#define DMPF64_OLE_PCG_BACKEND_H
#include "common.h"
#include "rng.h"
#include "ring.h"
#include "polynomial_any.h"
#include "ole_pcg.h"

typedef struct {
    uint8_t my_vec_seed[16];
    uint8_t peer_vec_seed[16];
    dmpf64_role role;
    dmpf64_backend backend;
    size_t batch_size;
    size_t compression_factor;
    dmpf64_sparse_ring_elem *sparse_polynomials;
    dmpf64_any_regular_share *tensor_product_fss;
    dmpf64_cyclotomic_mod mod;
} dmpf64_ole_seed_regular_any;

DMPF64_API void dmpf64_ole_seed_regular_any_free(dmpf64_ole_seed_regular_any *s);
DMPF64_API int dmpf64_ole_gen_regular_backend(dmpf64_ole_seed_regular_any *first,
                                              dmpf64_ole_seed_regular_any *second,
                                              size_t log_degree, size_t compression_factor,
                                              size_t weight, dmpf64_backend backend,
                                              size_t batch_size, dmpf64_rng *rng);
DMPF64_API int dmpf64_ole_expand_regular_backend(const dmpf64_ole_seed_regular_any *s,
                                                 dmpf64_ring_elem *mul_share,
                                                 dmpf64_ring_elem *add_share);
/* Evaluation-domain OLE expansion.  The output arrays still use dmpf64_ring_elem
 * as containers of length-N GF64 vectors, but poly.c[i] is the i-th additive
 * FFT evaluation coordinate, not a monomial coefficient.  Verification is
 * pointwise: mul0[i] * mul1[i] == add0[i] ^ add1[i]. */
DMPF64_API int dmpf64_ole_expand_regular_backend_eval(const dmpf64_ole_seed_regular_any *s,
                                                      dmpf64_ring_elem *mul_share_eval,
                                                      dmpf64_ring_elem *add_share_eval);
DMPF64_API int dmpf64_ole_eval_relation_ok(const dmpf64_ring_elem *m0, const dmpf64_ring_elem *a0,
                                           const dmpf64_ring_elem *m1, const dmpf64_ring_elem *a1);

#endif
