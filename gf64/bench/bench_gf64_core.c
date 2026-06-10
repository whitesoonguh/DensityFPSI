#include "gf64.h"
#include "gf64_poly.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef GF64_BENCH_HAVE_X4
#include "gf64x4.h"
#include "gf64x4_gcd.h"
#include "gf64x4_linalg.h"
#include "gf64x4_poly.h"
#endif

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }
static uint64_t rng_state = 0x123456789abcdef0ULL;
static uint64_t xr(void){ uint64_t x=rng_state; x^=x<<13; x^=x>>7; x^=x<<17; return rng_state=x; }

static void bench_add(size_t n){
    uint64_t a=xr(), b=xr(), acc=0;
    double t0=now_s();
    for(size_t i=0;i<n;i++){ a^=i; b^=acc; acc ^= gf64_add(a,b); }
    double s=now_s()-t0;
    printf("add_xor\n  ops=%zu sec=%.6f ops/s=%.3f M/s checksum=%016llx\n", n, s, (double)n/s/1e6, (unsigned long long)acc);
}

static void bench_mul(size_t n){
    uint64_t a=xr(), b=xr(), acc=0;
    double t0=now_s();
    for(size_t i=0;i<n;i++){ a^=(uint64_t)i*0x9e3779b97f4a7c15ULL; b^=acc; acc ^= gf64_mul(a,b); }
    double s=now_s()-t0;
    printf("mul_clmul\n  ops=%zu sec=%.6f ops/s=%.3f M/s checksum=%016llx\n", n, s, (double)n/s/1e6, (unsigned long long)acc);
}

static void bench_poly(size_t reps, int deg){
    int na=deg+1, nb=deg+1, nc=na+nb-1;
    uint64_t *a=(uint64_t*)malloc((size_t)na*8), *b=(uint64_t*)malloc((size_t)nb*8), *c=(uint64_t*)malloc((size_t)nc*8);
    for(int i=0;i<na;i++) a[i]=xr();
    for(int i=0;i<nb;i++) b[i]=xr();
    uint64_t acc=0;
    double t0=now_s();
    for(size_t r=0;r<reps;r++){
        a[0]^=(uint64_t)r^acc;
        memset(c,0,(size_t)nc*8);
        gf64_poly_mul_school(a,na,b,nb,c);
        acc^=c[0]^c[nc-1];
    }
    double s=now_s()-t0;
    printf("poly_mul_school\n  degree=%d reps=%zu sec=%.6f mul/s=%.3f M/s checksum=%016llx\n", deg, reps, s, (double)reps/s/1e6, (unsigned long long)acc);
    free(a); free(b); free(c);
}

#ifdef GF64_BENCH_HAVE_X4
static inline uint64_t low64(__m512i v){ return (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v)); }
static void fill512(__m512i*p,int n){ for(int i=0;i<n;i++){ uint64_t v[8]; for(int k=0;k<8;k++) v[k]=xr(); p[i]=_mm512_loadu_si512((void*)v);} }
static void fill64(int64_t*p,int n){ for(int i=0;i<n;i++) p[i]=(int64_t)xr(); }

static void bench_gcd_x4(long batches){
    enum { D=32 };
    __m512i A[D], B[D], work[3*(D+1)];
    int deg_gt0[4];
    fill512(A,D); fill512(B,D);
    uint64_t acc=0;
    double t0=now_s();
    for(long it=0; it<batches; ++it){
        A[0]=_mm512_xor_si512(A[0], _mm512_set1_epi64((long long)(it^acc)));
        gf64x4_coprime_test(A,D,B,D,work,deg_gt0);
        acc += (uint64_t)(deg_gt0[0] | (deg_gt0[1]<<1) | (deg_gt0[2]<<2) | (deg_gt0[3]<<3));
    }
    double s=now_s()-t0;
    printf("gcd_coprime_x4\n  gcds=%ld sec=%.6f gcd/s=%.3f M/s checksum=%016llx\n", batches*4, s, (double)(batches*4)/s/1e6, (unsigned long long)acc);
}

static void bench_linalg_x4(long reps){
    enum { M=16, K=32, N=1 };
    static int64_t A[M*K];
    static __m512i B[K*N] __attribute__((aligned(64)));
    static __m512i C[M*N] __attribute__((aligned(64)));
    fill64(A,M*K); fill512(B,K*N);
    uint64_t acc=0;
    double t0=now_s();
    for(long it=0; it<reps; ++it){
        A[0]^=(int64_t)(it^acc);
        gf64x4_matmat_prod(A,B,M,K,N,C);
        acc^=low64(C[0]);
    }
    double s=now_s()-t0;
    printf("linalg_matmat_x4\n  calls=%ld sec=%.6f calls/s=%.3f M/s checksum=%016llx\n", reps, s, (double)reps/s/1e6, (unsigned long long)acc);
}
#endif

int main(int argc, char** argv){
    size_t n = argc > 1 ? (size_t)strtoull(argv[1],0,0) : (1u<<22);
    size_t poly_reps = argc > 2 ? (size_t)strtoull(argv[2],0,0) : (1u<<16);
    int poly_deg = argc > 3 ? atoi(argv[3]) : 16;
    printf("GF64 core benchmark\n");
    bench_add(n);
    bench_mul(n);
    bench_poly(poly_reps, poly_deg);
#ifdef GF64_BENCH_HAVE_X4
    bench_gcd_x4((long)(argc > 4 ? strtoull(argv[4],0,0) : (1u<<14)));
    bench_linalg_x4((long)(argc > 5 ? strtoull(argv[5],0,0) : (1u<<16)));
#else
    printf("gcd_coprime_x4\n  skipped: GF64x4/AVX512 target disabled\n");
    printf("linalg_matmat_x4\n  skipped: GF64x4/AVX512 target disabled\n");
#endif
    return 0;
}
