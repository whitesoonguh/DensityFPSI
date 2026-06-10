#ifndef GF64_POLY_H
#define GF64_POLY_H

/*
 * Polynomials over GF(2^64).
 *
 * A polynomial of length n is a uint64_t array a[0..n-1] where a[i] is the
 * GF(2^64) coefficient of X^i. The product of an n-term and an m-term
 * polynomial has n + m - 1 terms.
 *
 * IMPORTANT: the output buffer `c` is accumulated with XOR, so the caller
 * MUST zero-initialize it (calloc / memset) before calling these functions.
 */

#include <stdint.h>

/*
 * Schoolbook multiplication. O(n*m).
 * Recommended for small degree (up to a few dozen): no branches, no heap,
 * cache-friendly. This is the default you want for typical MPC use.
 *
 *   c must have room for (na + nb - 1) elements, zeroed.
 */
void gf64_poly_mul_school(const uint64_t *a, int na,
                          const uint64_t *b, int nb,
                          uint64_t *c);

/*
 * Karatsuba multiplication, recursive, falling back to schoolbook below
 * GF64_POLY_KARA_CUTOFF. Only worth it for larger degree. This reference
 * version allocates scratch on the heap per recursion, so for small inputs
 * the schoolbook routine is faster; benchmark before switching.
 *
 *   c must have room for (na + nb - 1) elements, zeroed.
 */
void gf64_poly_mul_kara(const uint64_t *a, int na,
                        const uint64_t *b, int nb,
                        uint64_t *c);

/* Horner evaluation: returns a(x). */
uint64_t gf64_poly_eval(const uint64_t *a, int n, uint64_t x);

/* Return the degree of a polynomial, or -1 for the zero polynomial. */
int gf64_poly_degree(const uint64_t *a, int n);

/* Return the coefficient length after trimming trailing zero coefficients. */
int gf64_poly_trim_len(const uint64_t *a, int n);

/* Make a nonzero polynomial monic in place. Returns its degree, or -1 for zero. */
int gf64_poly_make_monic(uint64_t *a, int n);

/* r <- a mod m. r must have room for rcap coefficients. Returns trimmed length. */
int gf64_poly_mod(uint64_t *r, int rcap,
                  const uint64_t *a, int na,
                  const uint64_t *m, int nm);

/* q <- a / d for exact division. Returns trimmed quotient length, or -1 if not exact. */
int gf64_poly_div_exact(uint64_t *q, int qcap,
                        const uint64_t *a, int na,
                        const uint64_t *d, int nd);

/* g <- monic gcd(a,b). Returns degree, or -1 if gcd is zero. */
int gf64_poly_gcd(uint64_t *g, int gcap,
                  const uint64_t *a, int na,
                  const uint64_t *b, int nb);

/* out <- a(X)^2 mod m(X). Returns trimmed output length. */
int gf64_poly_square_mod(uint64_t *out, int outcap,
                         const uint64_t *a, int na,
                         const uint64_t *m, int nm);

/* out <- Tr(a) mod m, where Tr(a)=a+a^2+...+a^(2^63). */
int gf64_poly_trace_mod(uint64_t *out, int outcap,
                        const uint64_t *a, int na,
                        const uint64_t *m, int nm);

/* Split a monic square-free polynomial known to be a product of linear factors.
 * roots receives unique roots in sorted order. Returns root count, or -1 on failure.
 */
int gf64_poly_roots_split(uint64_t *roots, int rootcap,
                          const uint64_t *f, int nf,
                          uint64_t seed);



#endif /* GF64_POLY_H */
