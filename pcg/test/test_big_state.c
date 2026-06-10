#include "dmpf64/big_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int contains(const dmpf64_kv *kvs, size_t n, dmpf64_node x, dmpf64_gf2 *y){
    for(size_t i=0;i<n;i++) if(dmpf64_node_eq(kvs[i].x,x)){*y=kvs[i].y; return 1;}
    return 0;
}
int main(void){
    uint8_t seed[16]={9,8,7,6,5,4,3,2,1,0,11,12,13,14,15,16};
    dmpf64_rng rng; dmpf64_rng_seed(&rng,seed);
    const size_t bits=8, npoints=10;
    dmpf64_kv kvs[npoints];
    int used[256]={0};
    for(size_t i=0;i<npoints;i++){
        uint64_t idx;
        do { idx=dmpf64_rng_u64(&rng)&((1u<<bits)-1u); } while(used[idx]);
        used[idx]=1;
        kvs[i].x=dmpf64_encode_index(idx,bits);
        kvs[i].y=dmpf64_rng_gf2(&rng);
    }
    dmpf64_big_state_key k0,k1;
    if(dmpf64_big_state_gen(&k0,&k1,bits,kvs,npoints,&rng,4)!=0){fprintf(stderr,"gen failed\n"); return 1;}
    dmpf64_gf2 *e0=NULL,*e1=NULL; size_t l0=0,l1=0;
    if(dmpf64_big_state_eval_all(&k0,&e0,&l0)!=0 || dmpf64_big_state_eval_all(&k1,&e1,&l1)!=0 || l0!=(1u<<bits)||l1!=l0){fprintf(stderr,"eval_all failed\n"); return 1;}
    for(size_t i=0;i<l0;i++){
        dmpf64_node x=dmpf64_encode_index(i,bits);
        dmpf64_gf2 a=dmpf64_big_state_eval(&k0,x), b=dmpf64_big_state_eval(&k1,x);
        if(!dmpf64_gf2_eq(a,e0[i]) || !dmpf64_gf2_eq(b,e1[i])){fprintf(stderr,"eval mismatch at %zu\n",i); return 1;}
        dmpf64_gf2 sum=dmpf64_gf2_add(a,b), want=dmpf64_gf2_zero();
        if(contains(kvs,npoints,x,&want)){
            if(!dmpf64_gf2_eq(sum,want)){fprintf(stderr,"target mismatch at %zu got %016llx/%016llx want %016llx/%016llx\n",i,(unsigned long long)sum.v[0],(unsigned long long)sum.v[1],(unsigned long long)want.v[0],(unsigned long long)want.v[1]); return 1;}
        }else if(!dmpf64_gf2_is_zero(sum)){fprintf(stderr,"zero mismatch at %zu\n",i); return 1;}
    }
    free(e0); free(e1); dmpf64_big_state_key_free(&k0); dmpf64_big_state_key_free(&k1);
    printf("test_big_state: passed\n"); return 0;
}
