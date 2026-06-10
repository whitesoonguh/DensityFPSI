#ifndef DMPF64_NODE_H
#define DMPF64_NODE_H
#include "common.h"

typedef struct { uint64_t lo, hi; } dmpf64_node;

static inline dmpf64_node dmpf64_node_zero(void) { return (dmpf64_node){0,0}; }
static inline dmpf64_node dmpf64_node_u64(uint64_t x) { return (dmpf64_node){x,0}; }
static inline bool dmpf64_node_eq(dmpf64_node a, dmpf64_node b) { return a.lo == b.lo && a.hi == b.hi; }
static inline dmpf64_node dmpf64_node_xor(dmpf64_node a, dmpf64_node b) { return (dmpf64_node){a.lo ^ b.lo, a.hi ^ b.hi}; }
static inline void dmpf64_node_xor_inplace(dmpf64_node *a, dmpf64_node b) { a->lo ^= b.lo; a->hi ^= b.hi; }
static inline bool dmpf64_node_is_zero(dmpf64_node a) { return (a.lo | a.hi) == 0; }
static inline bool dmpf64_node_get_bit_msb(dmpf64_node v, size_t idx) {
    /* Rust: (u128 >> (127 - idx)) & 1 */
    size_t bit = 127u - idx;
    return bit < 64 ? ((v.lo >> bit) & 1u) : ((v.hi >> (bit - 64u)) & 1u);
}
static inline bool dmpf64_node_get_bit_lsb(dmpf64_node v, size_t idx) {
    return idx < 64 ? ((v.lo >> idx) & 1u) : ((v.hi >> (idx - 64u)) & 1u);
}
static inline void dmpf64_node_set_bit_lsb(dmpf64_node *v, size_t idx, bool bit) {
    if (idx < 64) {
        uint64_t m = UINT64_C(1) << idx;
        v->lo = (v->lo & ~m) ^ ((uint64_t)bit << idx);
    } else {
        idx -= 64;
        uint64_t m = UINT64_C(1) << idx;
        v->hi = (v->hi & ~m) ^ ((uint64_t)bit << idx);
    }
}
static inline void dmpf64_node_set_bit_msb(dmpf64_node *v, size_t idx, bool bit) {
    dmpf64_node_set_bit_lsb(v, 127u - idx, bit);
}
static inline void dmpf64_node_mask_lsbs(dmpf64_node *v, size_t bits) {
    if (bits >= 128) return;
    if (bits == 0) { v->lo = v->hi = 0; return; }
    if (bits < 64) { v->lo &= (UINT64_C(1) << bits) - 1; v->hi = 0; }
    else if (bits == 64) { v->hi = 0; }
    else { v->hi &= (UINT64_C(1) << (bits - 64)) - 1; }
}
static inline void dmpf64_node_pop_first_two_bits(dmpf64_node *v, bool *b0, bool *b1) {
    *b0 = (v->lo & 1u) != 0;
    *b1 = (v->lo & 2u) != 0;
    v->lo &= ~UINT64_C(3);
}
static inline void dmpf64_node_push_first_two_bits(dmpf64_node *v, bool b0, bool b1) {
    v->lo ^= (uint64_t)b0;
    v->lo ^= ((uint64_t)b1) << 1;
}
static inline dmpf64_node dmpf64_encode_index(uint64_t idx, size_t bits) {
    if (bits == 0) return dmpf64_node_zero();
    if (bits >= 64) {
        /* only small domains are used by the tests/bench helpers */
        return (dmpf64_node){0, idx << (128 - bits - 64)};
    }
    return (dmpf64_node){0, idx << (64 - bits)};
}

DMPF64_API int dmpf64_node_cmp(dmpf64_node a, dmpf64_node b);

#endif
