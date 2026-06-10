#include "dmpf64/gf64.h"
#include <gf64.h>

/*
 * PCG keeps its public dmpf64_gf_* API, but the field arithmetic backend is
 * centralized in the top-level gf64 module.  This removes the old duplicate
 * carry-less-multiply/reduction implementation while preserving the PCG ABI.
 */

dmpf64_gf dmpf64_gf_mul(dmpf64_gf a, dmpf64_gf b) {
    return gf64_mul(a, b);
}

dmpf64_gf dmpf64_gf_square(dmpf64_gf a) {
    return gf64_square(a);
}

dmpf64_gf dmpf64_gf_inv(dmpf64_gf a) {
    return gf64_inv(a);
}

dmpf64_gf2 dmpf64_gf2_inv(dmpf64_gf2 a) {
    dmpf64_gf2 z = {{dmpf64_gf_inv(a.v[0]), dmpf64_gf_inv(a.v[1])}};
    return z;
}

const char *dmpf64_gf_backend(void) {
    return "gf64/pclmul-shiftred";
}
