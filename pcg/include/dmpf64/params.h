#ifndef DMPF64_PARAMS_H
#define DMPF64_PARAMS_H
#include "common.h"
#include "dmpf_any.h"

/* Convenience presets for the GF(2^64) regular OLE eval-domain path.
 * Security strength is not estimated by this library; treat these as parameter
 * labels and verify them with an external Ring-LPN estimator for your exact
 * assumption/model. */
typedef struct {
    size_t log_degree;
    size_t compression_factor;
    size_t weight;
    size_t batch_size;
    dmpf64_backend backend;
    const char *name;
} dmpf64_ole_param_set;

static inline dmpf64_ole_param_set dmpf64_params_ring_lpn_128_fast(void) {
    dmpf64_ole_param_set p = {24, 1, 66, 8, DMPF64_BACKEND_BIG_STATE,
                              "ring-lpn-128-fast-evaldomain-experimental"};
    return p;
}

static inline dmpf64_ole_param_set dmpf64_params_primefield_like_66_2(void) {
    dmpf64_ole_param_set p = {24, 2, 66, 8, DMPF64_BACKEND_BIG_STATE,
                              "primefield-like-66-2"};
    return p;
}

#endif
