#ifndef DMPF64_GF64_H
#define DMPF64_GF64_H
#include "common.h"
#include "node.h"

/* GF(2^64) with p(x)=x^64+x^4+x^3+x+1. Addition/subtraction/negation are xor. */
typedef uint64_t dmpf64_gf;
typedef struct { dmpf64_gf v[2]; } dmpf64_gf2;

static inline dmpf64_gf dmpf64_gf_zero(void) { return 0; }
static inline dmpf64_gf dmpf64_gf_one(void) { return 1; }
static inline dmpf64_gf dmpf64_gf_add(dmpf64_gf a, dmpf64_gf b) { return a ^ b; }
static inline dmpf64_gf dmpf64_gf_sub(dmpf64_gf a, dmpf64_gf b) { return a ^ b; }
static inline bool dmpf64_gf_is_zero(dmpf64_gf a) { return a == 0; }

DMPF64_API dmpf64_gf dmpf64_gf_mul(dmpf64_gf a, dmpf64_gf b);
DMPF64_API dmpf64_gf dmpf64_gf_square(dmpf64_gf a);
DMPF64_API dmpf64_gf dmpf64_gf_inv(dmpf64_gf a);
DMPF64_API const char *dmpf64_gf_backend(void);

static inline dmpf64_gf2 dmpf64_gf2_zero(void) { dmpf64_gf2 z = {{0,0}}; return z; }
static inline dmpf64_gf2 dmpf64_gf2_from_bool(bool b) { dmpf64_gf2 z = {{(uint64_t)b,(uint64_t)b}}; return z; }
static inline dmpf64_gf2 dmpf64_gf2_from_node(dmpf64_node n) { dmpf64_gf2 z = {{n.hi, n.lo}}; return z; }
static inline bool dmpf64_gf2_eq(dmpf64_gf2 a, dmpf64_gf2 b) { return a.v[0] == b.v[0] && a.v[1] == b.v[1]; }
static inline bool dmpf64_gf2_is_zero(dmpf64_gf2 a) { return (a.v[0] | a.v[1]) == 0; }
static inline dmpf64_gf2 dmpf64_gf2_add(dmpf64_gf2 a, dmpf64_gf2 b) { dmpf64_gf2 z={{a.v[0]^b.v[0], a.v[1]^b.v[1]}}; return z; }
static inline dmpf64_gf2 dmpf64_gf2_sub(dmpf64_gf2 a, dmpf64_gf2 b) { return dmpf64_gf2_add(a,b); }
static inline dmpf64_gf2 dmpf64_gf2_neg(dmpf64_gf2 a) { return a; }
static inline dmpf64_gf2 dmpf64_gf2_mul(dmpf64_gf2 a, dmpf64_gf2 b) { dmpf64_gf2 z={{dmpf64_gf_mul(a.v[0],b.v[0]), dmpf64_gf_mul(a.v[1],b.v[1])}}; return z; }
static inline dmpf64_gf2 dmpf64_gf2_mul_bool(dmpf64_gf2 a, bool b) { if (!b) return dmpf64_gf2_zero(); return a; }
DMPF64_API dmpf64_gf2 dmpf64_gf2_inv(dmpf64_gf2 a);

#endif
