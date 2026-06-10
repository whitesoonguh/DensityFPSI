#include "okvs_encode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0xfeedfaceULL;
static uint64_t xr(void) { uint64_t x = st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return st = x; }

/* encode n random (key,value) pairs, decode every key, check the value. */
static int roundtrip(size_t n, size_t w, size_t L, int oblivious) {
    size_t key_len = 16;
    double eps = 0.05;
    size_t cols = okvs_w_cols(n, w, eps);

    uint8_t *keys = malloc(n * key_len);
    uint64_t *vals = malloc(n * L * sizeof(uint64_t));
    uint64_t *x = malloc(cols * L * sizeof(uint64_t));
    uint8_t seed[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    for (size_t i = 0; i < n * key_len; i++) keys[i] = (uint8_t)xr();
    for (size_t i = 0; i < n * L; i++) vals[i] = xr();

    int rc, ok = 0, tries = 0;
    uint64_t prng = 0xabcdef1234567890ULL;
    while (tries++ < 8 && !ok) {
        rc = okvs_w_encode(keys, key_len, vals, n, w, L, cols, seed, sizeof seed,
                           x, oblivious ? &prng : NULL);
        if (rc == 0) {
            ok = 1;
            uint64_t out[8];
            for (size_t i = 0; i < n; i++) {
                okvs_w_decode(keys + i * key_len, key_len, w, L, cols,
                              seed, sizeof seed, x, out);
                if (memcmp(out, vals + i * L, L * sizeof(uint64_t)) != 0) { ok = 0; break; }
            }
        } else {
            /* ZeroRow (astronomically unlikely at these w): re-seed */
            seed[0]++;
        }
    }
    free(keys); free(vals); free(x);
    return ok;
}

int main(void) {
    int fails = 0;
    struct { size_t n, w, L; int ob; const char *tag; } T[] = {
        {1u << 12, 218, 1, 0, "L=1 scalar,  zeros"},
        {1u << 12, 218, 1, 1, "L=1 scalar,  oblivious"},
        {1u << 12, 412, 4, 0, "L=4 poly,    zeros"},
        {1u << 12, 412, 4, 1, "L=4 poly,    oblivious"},
        {1u << 14, 412, 1, 1, "L=1 scalar,  n=2^14 obliv"},
    };
    for (size_t i = 0; i < sizeof(T) / sizeof(T[0]); i++) {
        int ok = roundtrip(T[i].n, T[i].w, T[i].L, T[i].ob);
        printf("encode/decode  %-26s : %s\n", T[i].tag, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}