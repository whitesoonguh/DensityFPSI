#include "okvs_util.h"

#include <stdio.h>
#include <string.h>

static int hexeq(const uint8_t *got, const char *hex, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned b;
        sscanf(hex + 2 * i, "%2x", &b);
        if (got[i] != (uint8_t)b) return 0;
    }
    return 1;
}

static const unsigned char in_empty[] = {0};
static const unsigned char in_abc[] = {97, 98, 99};
static const unsigned char in_range200[] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
    128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
    160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
    192,193,194,195,196,197,198,199};

static int test_blake2b(void) {
    struct { const unsigned char *d; size_t len; size_t N; const char *hex; } V[] = {
        {in_empty, 0, 16, "cae66941d9efbd404e4d88758ea67670"},
        {in_empty, 0, 32, "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"},
        {in_empty, 0, 64, "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"},
        {in_abc, 3, 16, "cf4ab791c62b8d2b2109c90275287816"},
        {in_abc, 3, 32, "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319"},
        {in_abc, 3, 48, "6f56a82c8e7ef526dfe182eb5212f7db9df1317e57815dbda46083fc30f54ee6c66ba83be64b302d7cba6ce15bb556f4"},
        {in_abc, 3, 64, "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"},
        {in_range200, 200, 32, "63c3d97a9f8894d5e043a707b0fee7f7ec4c049a23bbf1079df20b4165f9e22d"},
        {in_range200, 200, 64, "fb3c1f0f56a56f8e316fdf5d853c8c872c39635d083634c3904fc3ac07d1b578e85ff0e480e92d44ade33b62e893ee32343e79ddf6ef292e89b582d312502314"},
    };
    int ok = 1;
    uint8_t out[64];
    for (size_t i = 0; i < sizeof(V) / sizeof(V[0]); i++) {
        okvs_blake2b_n(out, V[i].N, V[i].d, V[i].len);
        if (!hexeq(out, V[i].hex, V[i].N)) ok = 0;
    }
    printf("blake2b_n vs hashlib (N=16/32/48/64)       : %s\n", ok ? "ok" : "FAIL");

    uint8_t d64[64], h[200];
    okvs_blake2b_n(d64, 64, in_abc, 3);
    okvs_hash(h, 40, in_abc, 3);
    int h_ok = memcmp(h, d64, 40) == 0;
    okvs_hash(h, 150, in_abc, 3);
    h_ok = h_ok && memcmp(h, d64, 64) == 0 && memcmp(h + 64, d64, 64) == 0
                && memcmp(h + 128, d64, 150 - 128) == 0;
    printf("hash() truncate(<=64) + repeat(>64)        : %s\n", h_ok ? "ok" : "FAIL");
    return (ok && h_ok) ? 0 : 1;
}

static int test_radix(void) {
    okvs_pair arr[] = {{20, 0}, {10, 2}, {0, 1}};
    okvs_radix_sort(arr, 3, 2);
    int ok = arr[0].b == 0 && arr[1].b == 1 && arr[2].b == 2
          && arr[0].a == 20 && arr[1].a == 0 && arr[2].a == 10;
    printf("radix_sort (by .b, pair stable)            : %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int fails = 0;
    fails += test_blake2b();
    fails += test_radix();
    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
