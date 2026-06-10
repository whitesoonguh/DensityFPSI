#include "okvs_encode.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }
static uint64_t st=0x72626f6b7673ULL;
static uint64_t xr(void){ uint64_t x=st; x^=x<<13; x^=x>>7; x^=x<<17; return st=x; }

int main(int argc, char** argv){
    size_t n = argc > 1 ? (size_t)strtoull(argv[1],0,0) : ((size_t)1<<20);
    size_t q = argc > 2 ? (size_t)strtoull(argv[2],0,0) : n;
    size_t w = argc > 3 ? (size_t)strtoull(argv[3],0,0) : 218;
    double eps = argc > 4 ? strtod(argv[4],0) : 0.10;
    const size_t key_len=8, L=1;
    uint8_t seed[8] = {9,8,7,6,5,4,3,2};
    size_t cols = okvs_w_cols(n,w,eps);
    uint64_t *keys=(uint64_t*)malloc(n*sizeof(uint64_t));
    uint64_t *queries=(uint64_t*)malloc(q*sizeof(uint64_t));
    uint64_t *vals=(uint64_t*)malloc(n*sizeof(uint64_t));
    uint64_t *table=(uint64_t*)malloc(cols*sizeof(uint64_t));
    uint64_t *out=(uint64_t*)malloc(q*sizeof(uint64_t));
    if(!keys||!queries||!vals||!table||!out){ fprintf(stderr,"allocation failed\n"); return 2; }
    for(size_t i=0;i<n;i++){ keys[i]=xr(); vals[i]=xr(); }
    for(size_t i=0;i<q;i++) queries[i]=keys[i%n];
    uint64_t free_seed=0xC0FFEEULL;
    double t0=now_s();
    int rc=okvs_w_encode((const uint8_t*)keys,key_len,vals,n,w,L,cols,seed,sizeof seed,table,&free_seed);
    double enc=now_s()-t0;
    if(rc){ fprintf(stderr,"okvs encode failed rc=%d\n",rc); return 3; }
    okvs_w_ctx ctx; okvs_w_ctx_init(&ctx,w,L,cols,seed,sizeof seed);
    t0=now_s();
    okvs_w_decode_batch_ctx(&ctx,(const uint8_t*)queries,key_len,table,q,out);
    double dec=now_s()-t0;
    uint64_t chk=0; int ok=1;
    for(size_t i=0;i<q;i++){ chk^=out[i]; if(out[i]!=vals[i%n]) ok=0; }
    printf("OKVS encode/decode benchmark\n");
    printf("  n=%zu q=%zu w=%zu eps=%.3f cols=%zu\n",n,q,w,eps,cols);
    printf("  encode_sec=%.6f encode_items/s=%.3f M/s\n",enc,(double)n/enc/1e6);
    printf("  decode_sec=%.6f decode_queries/s=%.3f M/s\n",dec,(double)q/dec/1e6);
    printf("  table_bytes=%zu verified=%s checksum=%016llx\n",cols*sizeof(uint64_t), ok?"yes":"no", (unsigned long long)chk);
    free(keys); free(queries); free(vals); free(table); free(out);
    return ok?0:4;
}
