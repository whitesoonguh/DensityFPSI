#ifndef DMPF64_DPF_H
#define DMPF64_DPF_H
#include "common.h"
#include "node.h"
#include "gf64.h"
#include "rng.h"

typedef struct {
    dmpf64_node node;
} dmpf64_correction_word;

typedef struct {
    dmpf64_node root;
    bool root_bit;
    dmpf64_correction_word *cws;
    size_t input_bits;
    dmpf64_gf2 last_cw;
} dmpf64_dpf_key;

typedef struct {
    dmpf64_dpf_key *dpf_keys;
    size_t count;
} dmpf64_dpf_dmpf_key;

typedef struct {
    dmpf64_node *cur_seeds;
    dmpf64_node *next_seeds;
    bool *cur_signs;
    bool *next_signs;
    size_t cap;
} dmpf64_dpf_session;

typedef struct { dmpf64_node x; dmpf64_gf2 y; } dmpf64_kv;

DMPF64_API void dmpf64_dpf_key_init_empty(dmpf64_dpf_key *k);
DMPF64_API void dmpf64_dpf_key_free(dmpf64_dpf_key *k);
DMPF64_API int dmpf64_dpf_key_clone(dmpf64_dpf_key *dst, const dmpf64_dpf_key *src);
DMPF64_API int dmpf64_dpf_gen(dmpf64_dpf_key *k0, dmpf64_dpf_key *k1, dmpf64_node alpha, size_t input_bits, dmpf64_gf2 beta, dmpf64_rng *rng);
DMPF64_API dmpf64_gf2 dmpf64_dpf_eval(const dmpf64_dpf_key *k, dmpf64_node x);
DMPF64_API int dmpf64_dpf_eval_all(const dmpf64_dpf_key *k, dmpf64_gf2 *out /* length 1<<input_bits */);
DMPF64_API int dmpf64_dpf_eval_all_add(const dmpf64_dpf_key *k, dmpf64_dpf_session *sess, dmpf64_gf2 *out);

DMPF64_API int dmpf64_dpf_session_init(dmpf64_dpf_session *s, size_t input_bits);
DMPF64_API void dmpf64_dpf_session_free(dmpf64_dpf_session *s);

DMPF64_API void dmpf64_dpf_dmpf_key_init_empty(dmpf64_dpf_dmpf_key *k);
DMPF64_API void dmpf64_dpf_dmpf_key_free(dmpf64_dpf_dmpf_key *k);
DMPF64_API int dmpf64_dpf_dmpf_gen(dmpf64_dpf_dmpf_key *k0, dmpf64_dpf_dmpf_key *k1, size_t input_bits, const dmpf64_kv *inputs, size_t count, dmpf64_rng *rng);
DMPF64_API dmpf64_gf2 dmpf64_dpf_dmpf_eval(const dmpf64_dpf_dmpf_key *k, dmpf64_node x);
DMPF64_API int dmpf64_dpf_dmpf_eval_all(const dmpf64_dpf_dmpf_key *k, dmpf64_gf2 **out, size_t *out_len);
DMPF64_API size_t dmpf64_dpf_dmpf_input_length(const dmpf64_dpf_dmpf_key *k);
DMPF64_API size_t dmpf64_dpf_dmpf_point_count(const dmpf64_dpf_dmpf_key *k);

#endif
