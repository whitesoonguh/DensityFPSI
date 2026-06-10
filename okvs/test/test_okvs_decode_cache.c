#include "okvs_encode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0xdec0de1234567890ULL;
static uint64_t xr(void) { uint64_t x = st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return st = x; }

int main(void) {
    const size_t n = 1u << 14;
    const size_t key_len = 8;
    const size_t L = 1;
    const size_t w = 218;
    const double eps = 0.1;
    const size_t cols = okvs_w_cols(n, w, eps);
    uint8_t seed[8] = {3,1,4,1,5,9,2,6};
    uint64_t *keys = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint64_t *vals = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint64_t *x = (uint64_t*)malloc(cols * sizeof(uint64_t));
    uint64_t *out = (uint64_t*)malloc(n * sizeof(uint64_t));
    if (!keys || !vals || !x || !out) return 2;
    for (size_t i = 0; i < n; i++) { keys[i] = xr(); vals[i] = xr(); }
    uint64_t prng = 0xabcdef9876543210ULL;
    int rc = okvs_w_encode((const uint8_t*)keys, key_len, vals, n, w, L, cols,
                           seed, sizeof seed, x, &prng);
    if (rc) { printf("encode failed rc=%d\n", rc); return 1; }
    okvs_w_ctx ctx;
    okvs_w_ctx_init(&ctx, w, L, cols, seed, sizeof seed);
    okvs_w_decode_cache cache;
    rc = okvs_w_decode_cache_build(&cache, &ctx, x, 4);
    if (rc) { printf("cache build failed rc=%d\n", rc); return 1; }
    okvs_w_decode_batch_cached(&cache, (const uint8_t*)keys, key_len, n, out);
    int ok = 1;
    for (size_t i = 0; i < n; i++) if (out[i] != vals[i]) { ok = 0; break; }
    printf("decode cache chunk_bits=4 roundtrip : %s\n", ok ? "ok" : "FAIL");
    uint64_t sink1 = 0, sink2 = 0;
    for (size_t i = 0; i < n; i++) sink1 ^= out[i];
    int brc = okvs_w_decode_batch_cached_bucketed(&cache, (const uint8_t*)keys, key_len, n, out, 8);
    for (size_t i = 0; i < n; i++) sink2 ^= out[i];
    ok = ok && brc == 0 && sink1 == sink2;
    printf("decode cache bucketed checksum      : %s\n", (brc == 0 && sink1 == sink2) ? "ok" : "FAIL");
    uint64_t direct_xor = okvs_w_decode_batch_ctx_xor(&ctx, (const uint8_t*)keys, key_len, x, n);
    int nbc = 0;
    uint64_t bucket_xor = okvs_w_decode_batch_ctx_bucketed_keys_xor(&ctx, (const uint8_t*)keys, key_len, x, n, 8, &nbc);
    ok = ok && nbc == 0 && direct_xor == bucket_xor;
    printf("no-cache bucketed checksum         : %s\n", (nbc == 0 && direct_xor == bucket_xor) ? "ok" : "FAIL");
    okvs_w_decode_cache_free(&cache);
    free(keys); free(vals); free(x); free(out);
    return ok ? 0 : 1;
}
