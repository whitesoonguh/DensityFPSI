#include "gf64_lin.h"
#include "gf64.h"

#include <stdlib.h>
#include <string.h>

uint64_t gf64_vec_dot(const uint64_t *a, const uint64_t *b, int n) {
    uint64_t acc = 0;
    for (int i = 0; i < n; ++i) if (a[i] && b[i]) acc ^= gf64_mul(a[i], b[i]);
    return acc;
}

void gf64_matvec_rowmajor(const uint64_t *A, const uint64_t *x,
                          int rows, int cols, uint64_t *y) {
    for (int r = 0; r < rows; ++r) y[r] = gf64_vec_dot(A + (size_t)r * cols, x, cols);
}

int gf64_mat_inverse(const uint64_t *A, int n, uint64_t *Ainv) {
    if (n <= 0) return -1;
    uint64_t *M = (uint64_t *)malloc((size_t)n * n * sizeof(uint64_t));
    if (!M) return -1;
    memcpy(M, A, (size_t)n * n * sizeof(uint64_t));
    memset(Ainv, 0, (size_t)n * n * sizeof(uint64_t));
    for (int i = 0; i < n; ++i) Ainv[(size_t)i * n + i] = 1;
    for (int col = 0; col < n; ++col) {
        int piv = col;
        while (piv < n && M[(size_t)piv * n + col] == 0) ++piv;
        if (piv == n) { free(M); return -1; }
        if (piv != col) {
            for (int k = 0; k < n; ++k) {
                uint64_t t = M[(size_t)col * n + k];
                M[(size_t)col * n + k] = M[(size_t)piv * n + k];
                M[(size_t)piv * n + k] = t;
                t = Ainv[(size_t)col * n + k];
                Ainv[(size_t)col * n + k] = Ainv[(size_t)piv * n + k];
                Ainv[(size_t)piv * n + k] = t;
            }
        }
        uint64_t inv_piv = gf64_inv(M[(size_t)col * n + col]);
        for (int k = 0; k < n; ++k) {
            M[(size_t)col * n + k] = gf64_mul(M[(size_t)col * n + k], inv_piv);
            Ainv[(size_t)col * n + k] = gf64_mul(Ainv[(size_t)col * n + k], inv_piv);
        }
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            uint64_t factor = M[(size_t)row * n + col];
            if (!factor) continue;
            for (int k = 0; k < n; ++k) {
                M[(size_t)row * n + k] ^= gf64_mul(factor, M[(size_t)col * n + k]);
                Ainv[(size_t)row * n + k] ^= gf64_mul(factor, Ainv[(size_t)col * n + k]);
            }
        }
    }
    free(M);
    return 0;
}

int gf64_vandermonde_matrices(const uint64_t *alpha, int n,
                              uint64_t *V, uint64_t *Vinv) {
    if (n <= 0) return -1;
    for (int i = 0; i < n; ++i) {
        uint64_t xpow = 1;
        for (int k = 0; k < n; ++k) {
            V[(size_t)i * n + k] = xpow;
            xpow = gf64_mul(xpow, alpha[i]);
        }
    }
    return gf64_mat_inverse(V, n, Vinv);
}
