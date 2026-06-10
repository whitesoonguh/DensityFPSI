#ifndef DMPF64_BIG_STATE_H
#define DMPF64_BIG_STATE_H
#include "common.h"
#include "node.h"
#include "gf64.h"
#include "rng.h"
#include "dpf.h"

/* GF(2^64)^2-specialized port of src/big_state.rs.  The public API mirrors
 * the DMPF key interface used by the rest of this C port. */

typedef struct {
    size_t t;
    dmpf64_node *bits;
    size_t nodes_per_point_per_direction;
} dmpf64_signs_cw;

typedef struct {
    size_t seed_count;
    dmpf64_node *seeds;
    dmpf64_signs_cw signs;
    size_t batch_size;
} dmpf64_big_cw;

typedef struct {
    size_t t;
    dmpf64_gf2 *o;
    size_t batch_size;
} dmpf64_conv_cw;

typedef struct {
    bool sign;
    dmpf64_node root;
    dmpf64_big_cw *cws;
    size_t input_bits;
    dmpf64_conv_cw conv_cw;
} dmpf64_big_state_key;

typedef struct {
    dmpf64_node *sign;
    dmpf64_node *new_sign;
    size_t t;
    size_t nodes;
} dmpf64_big_state_session;

DMPF64_API void dmpf64_big_state_key_init_empty(dmpf64_big_state_key *k);
DMPF64_API void dmpf64_big_state_key_free(dmpf64_big_state_key *k);
DMPF64_API int dmpf64_big_state_gen(dmpf64_big_state_key *k0, dmpf64_big_state_key *k1,
                                    size_t input_bits, const dmpf64_kv *inputs,
                                    size_t count, dmpf64_rng *rng, size_t batch_size);
DMPF64_API int dmpf64_big_state_session_init(dmpf64_big_state_session *s, size_t t);
DMPF64_API void dmpf64_big_state_session_free(dmpf64_big_state_session *s);
DMPF64_API dmpf64_gf2 dmpf64_big_state_eval(const dmpf64_big_state_key *k, dmpf64_node x);
DMPF64_API dmpf64_gf2 dmpf64_big_state_eval_with_session(const dmpf64_big_state_key *k, dmpf64_node x, dmpf64_big_state_session *s);
DMPF64_API int dmpf64_big_state_eval_all(const dmpf64_big_state_key *k, dmpf64_gf2 **out, size_t *out_len);
DMPF64_API int dmpf64_big_state_eval_all_into_coeff_mod(const dmpf64_big_state_key *k, dmpf64_gf *out, size_t coeff_start, size_t coeff_count, size_t mod_degree);
/* Write a contiguous coefficient slice without modular wrap and without XOR.
 * Single-thread cache-backed fast path used by eval-domain OLE reconstruction. */
DMPF64_API int dmpf64_big_state_eval_all_write_coeff(const dmpf64_big_state_key *k, dmpf64_gf *out, size_t coeff_count);
/* XOR a contiguous coefficient slice without modular wrap.  Unlike
 * eval_all_into_coeff_mod(), this keeps the prefix-pruned tree traversal used
 * by write_coeff instead of expanding the full leaf space for wraparound. */
DMPF64_API int dmpf64_big_state_eval_all_xor_coeff(const dmpf64_big_state_key *k, dmpf64_gf *out, size_t coeff_count);
DMPF64_API size_t dmpf64_big_state_input_length(const dmpf64_big_state_key *k);
DMPF64_API size_t dmpf64_big_state_point_count(const dmpf64_big_state_key *k);

#endif
