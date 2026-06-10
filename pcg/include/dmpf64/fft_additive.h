#ifndef DMPF64_FFT_ADDITIVE_H
#define DMPF64_FFT_ADDITIVE_H
#include "common.h"
#include "polynomial.h"
#include "ring.h"

/* Additive FFT over GF(2^64).
 *
 * Public eval/interpolate accept monomial-basis coefficients and use the
 * correctness-first additive-subspace implementation.  The cached cyclic ring
 * multiplication path uses a faster normalized Cantor-style basis and an
 * internal novel-basis transform: evaluation/interpolation split by low/high
 * halves and no longer divide by subspace polynomials in the transform itself.
 *
 * Remaining limitation: monomial <-> novel conversion is still recursive, so
 * the cached ring path is not yet the final fully optimized Gao-Mateer/LCH
 * basis-conversion kernel.
 */

DMPF64_API int dmpf64_afft_eval(dmpf64_gf *out, const dmpf64_gf *coeffs,
                                size_t len, size_t log_n);
DMPF64_API int dmpf64_afft_interpolate(dmpf64_gf *out, const dmpf64_gf *values,
                                       size_t log_n);
DMPF64_API int dmpf64_afft_mul(dmpf64_dense_poly *out,
                               const dmpf64_dense_poly *a,
                               const dmpf64_dense_poly *b);
DMPF64_API int dmpf64_afft_cyclic_mul(dmpf64_ring_elem *out,
                                      const dmpf64_ring_elem *a,
                                      const dmpf64_ring_elem *b);
/* Single-threaded cached-plan entry point used by ring.c.  It reuses the
 * additive FFT plan, work buffers, and scratch arena for repeated calls at the
 * same ring degree.  Not thread-safe by design; this port targets single-core
 * PCG expansion. */
DMPF64_API int dmpf64_afft_cyclic_mul_cached(dmpf64_ring_elem *out,
                                             const dmpf64_ring_elem *a,
                                             const dmpf64_ring_elem *b);
DMPF64_API int dmpf64_afft_cyclic_mul_addto_cached(dmpf64_ring_elem *acc,
                                                   const dmpf64_ring_elem *a,
                                                   const dmpf64_ring_elem *b);
DMPF64_API void dmpf64_afft_clear_cache(void);
/* Prepare public AFFT plan/workspace for eval-domain OLE at size 2^log_n.
 * This is single-threaded static preprocessing; it depends only on public N. */
DMPF64_API int dmpf64_afft_prepare_eval_domain(size_t log_n);
/* Cached forward additive FFT from monomial coefficients into evaluation domain.
 * Single-threaded and not thread-safe, sharing the internal AFFT plan cache with
 * the ring kernels. */
DMPF64_API int dmpf64_afft_eval_mono_cached(dmpf64_gf *out, size_t log_n,
                                            const dmpf64_gf *coeffs, size_t active_len);
DMPF64_API int dmpf64_afft_eval_mono_inplace_cached(dmpf64_gf *buf, size_t log_n,
                                                    size_t active_len);
DMPF64_API int dmpf64_afft_eval_sparse_cached(dmpf64_gf *out, size_t log_n,
                                              const dmpf64_sparse_poly *p);
/* Reduce a monomial polynomial modulo the additive subspace vanishing
 * polynomial S_log_n and write N coefficients.  This is the correct quotient
 * reduction for evaluation-domain OLE over the additive subgroup. */
DMPF64_API int dmpf64_afft_reduce_subspace_cached(dmpf64_gf *out, size_t log_n,
                                                  const dmpf64_gf *coeffs, size_t len);
/* Specialized len=2N reduction that may mutate coeffs[N..2N). */
DMPF64_API int dmpf64_afft_reduce_subspace_len2_inplace_cached(dmpf64_gf *out, size_t log_n,
                                                               dmpf64_gf *coeffs);
/* Directly evaluate a polynomial of length up to 2N on the N-point additive
 * subspace.  This is equivalent to reduce_subspace_cached()+eval_mono_cached()
 * but avoids materializing the reduced monomial polynomial. */
DMPF64_API int dmpf64_afft_eval_subspace_cached(dmpf64_gf *out, size_t log_n,
                                                const dmpf64_gf *coeffs, size_t len);

/* Sum of reduced triple products using a single additive-FFT eval-domain
 * accumulator:
 *   out = XOR_i reduce_mod_xN_plus_1(a_i * b_i * c_i).
 *
 * This is intended for OLE-PCG expansion, where the old path performed
 * reduce(a_i*b_i) and then multiplied by c_i with a second ring_mul.
 * The function is single-threaded and uses the same cached plan/workspace as
 * the cyclic ring kernel. */
DMPF64_API int dmpf64_afft_cyclic_sum_product3(dmpf64_ring_elem *out,
                                               dmpf64_cyclotomic_mod mod,
                                               size_t count,
                                               const dmpf64_ring_elem * const *a,
                                               const dmpf64_ring_elem * const *b,
                                               const dmpf64_ring_elem * const *c);

/* Streaming variant: for each product term, c_i is generated directly into
 * a caller-provided monomial coefficient buffer instead of being materialized
 * as a dmpf64_ring_elem first.  fill_c(ctx, i, dst, n) must XOR/write the
 * length-n monomial coefficients of c_i into dst[0..n).  The buffer is already
 * zeroed by this function and has transform workspace after n that the callback
 * must not touch. */
typedef int (*dmpf64_afft_fill_mono_fn)(void *ctx, size_t i, dmpf64_gf *dst, size_t n);
DMPF64_API int dmpf64_afft_cyclic_sum_product3_generated(dmpf64_ring_elem *out,
                                                         dmpf64_cyclotomic_mod mod,
                                                         size_t count,
                                                         const dmpf64_ring_elem * const *a,
                                                         const dmpf64_ring_elem * const *b,
                                                         dmpf64_afft_fill_mono_fn fill_c,
                                                         void *ctx);

DMPF64_API const char *dmpf64_afft_backend(void);

#endif
