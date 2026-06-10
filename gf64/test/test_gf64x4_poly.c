#include "gf64.h"
#include "gf64_poly.h"
#include "gf64x4.h"
#include "gf64x4_poly.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0x55aa55aa1234ULL;
static uint64_t xr(void) {
    uint64_t x = st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return st = x;
}

int main(void) {
    int fails = 0;
    /* sizes hit the 2x2 (n=2), 4x4 (n=4), 8x8 (n=8) register kernels, the
     * multi-block path (16,32,64) and schoolbook tails (1,3,5,7,9,15,17,31). */
    int sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64};

    for (int s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++) {
        int n = sizes[s], cn = 2 * n - 1;

        uint64_t A[4][64], B[4][64];
        for (int l = 0; l < 4; l++)
            for (int i = 0; i < n; i++) { A[l][i] = xr(); B[l][i] = xr(); }

        /* scalar reference: 4 independent products */
        uint64_t ref[4][127];
        for (int l = 0; l < 4; l++) {
            memset(ref[l], 0, (size_t)cn * 8);
            gf64_poly_mul_school(A[l], n, B[l], n, ref[l]);
        }

        __m512i *av = aligned_alloc(64, (size_t)n * sizeof(__m512i));
        __m512i *bv = aligned_alloc(64, (size_t)n * sizeof(__m512i));
        __m512i *cm = aligned_alloc(64, (size_t)cn * sizeof(__m512i));  /* blocked */
        __m512i *cs = aligned_alloc(64, (size_t)cn * sizeof(__m512i));  /* school  */
        for (int i = 0; i < n; i++) {
            uint64_t pa[4] = {A[0][i], A[1][i], A[2][i], A[3][i]};
            uint64_t pb[4] = {B[0][i], B[1][i], B[2][i], B[3][i]};
            av[i] = gf64x4_pack(pa);
            bv[i] = gf64x4_pack(pb);
        }

        gf64x4_poly_mul(av, n, bv, n, cm);                 /* zeros internally */
        memset(cs, 0, (size_t)cn * sizeof(__m512i));
        gf64x4_poly_mul_school(av, n, bv, n, cs);          /* needs zeroed c   */

        int ok_ref = 1, ok_sch = 1;
        for (int k = 0; k < cn && (ok_ref || ok_sch); k++) {
            uint64_t gm[4], gs[4];
            gf64x4_unpack(cm[k], gm);
            gf64x4_unpack(cs[k], gs);
            for (int l = 0; l < 4; l++) {
                if (gm[l] != ref[l][k]) ok_ref = 0;
                if (gm[l] != gs[l])     ok_sch = 0;
            }
        }
        printf("n=%-3d  blocked vs scalar-x4: %s   blocked vs school: %s\n",
               n, ok_ref ? "ok" : "FAIL", ok_sch ? "ok" : "FAIL");
        if (!ok_ref || !ok_sch) fails++;

        free(av); free(bv); free(cm); free(cs);
    }

    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
