#ifndef DMPF64_OLE_PCG_H
#define DMPF64_OLE_PCG_H
#include "common.h"
#include "rng.h"
#include "ring.h"

typedef enum { DMPF64_ROLE_FIRST = 0, DMPF64_ROLE_SECOND = 1 } dmpf64_role;

typedef struct {
    uint8_t my_vec_seed[16];
    uint8_t peer_vec_seed[16];
    dmpf64_role role;
    size_t compression_factor;
    dmpf64_sparse_ring_elem *sparse_polynomials;
    dmpf64_dpf_dmpf_key *tensor_product_fss;
    dmpf64_cyclotomic_mod mod;
} dmpf64_ole_seed;

typedef struct {
    size_t block_size;
    size_t block_count;
    dmpf64_dpf_dmpf_key *blocks;
} dmpf64_regular_share;

typedef struct {
    uint8_t my_vec_seed[16];
    uint8_t peer_vec_seed[16];
    dmpf64_role role;
    size_t compression_factor;
    dmpf64_sparse_ring_elem *sparse_polynomials;
    dmpf64_regular_share *tensor_product_fss;
    dmpf64_cyclotomic_mod mod;
} dmpf64_ole_seed_regular;

DMPF64_API void dmpf64_ole_seed_free(dmpf64_ole_seed *s);
DMPF64_API void dmpf64_ole_seed_regular_free(dmpf64_ole_seed_regular *s);
DMPF64_API int dmpf64_ole_gen(dmpf64_ole_seed *first, dmpf64_ole_seed *second, size_t log_degree, size_t compression_factor, size_t weight, dmpf64_rng *rng);
DMPF64_API int dmpf64_ole_gen_regular(dmpf64_ole_seed_regular *first, dmpf64_ole_seed_regular *second, size_t log_degree, size_t compression_factor, size_t weight, dmpf64_rng *rng);
DMPF64_API int dmpf64_ole_expand(const dmpf64_ole_seed *s, dmpf64_ring_elem *mul_share, dmpf64_ring_elem *add_share);
DMPF64_API int dmpf64_ole_expand_regular(const dmpf64_ole_seed_regular *s, dmpf64_ring_elem *mul_share, dmpf64_ring_elem *add_share);

#endif
