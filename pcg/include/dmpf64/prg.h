#ifndef DMPF64_PRG_H
#define DMPF64_PRG_H
#include "common.h"
#include "node.h"

DMPF64_API void dmpf64_prg_init(void);
DMPF64_API const char *dmpf64_prg_backend(void);
DMPF64_API void dmpf64_double_prg(dmpf64_node in, dmpf64_node out[2]);
DMPF64_API void dmpf64_double_prg_many(const dmpf64_node *in, size_t n, dmpf64_node *out);
DMPF64_API void dmpf64_many_prg(dmpf64_node in, uint16_t first_child, size_t count, dmpf64_node *out);
DMPF64_API void dmpf64_many_many_prg(const dmpf64_node *in, size_t n, uint16_t first_child, size_t count_per_input, dmpf64_node *out);
/* Fused single-thread helper for BigState eval_all. seed_out receives child labels
 * 0/1 exactly as dmpf64_double_prg_many; sign_out receives labels 2/3 exactly as
 * dmpf64_many_many_prg(..., first_child=2, count_per_input=2). */
DMPF64_API void dmpf64_bigstate_prg_many(const dmpf64_node *in, size_t n, dmpf64_node *seed_out, dmpf64_node *sign_out);

#endif
