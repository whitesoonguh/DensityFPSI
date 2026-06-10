#include "dmpf64/dpf.h"
#include <stdio.h>
#include <stdlib.h>

static int test_single_dpf(void){
    uint8_t seed[16]={1,2,3}; dmpf64_rng rng; dmpf64_rng_seed(&rng,seed);
    const size_t DEPTH=10; uint64_t point=dmpf64_rng_u64(&rng)&((1ULL<<DEPTH)-1); dmpf64_node alpha=dmpf64_encode_index(point,DEPTH); dmpf64_gf2 beta=dmpf64_rng_gf2(&rng);
    dmpf64_dpf_key k0,k1; if(dmpf64_dpf_gen(&k0,&k1,alpha,DEPTH,beta,&rng)!=0)return 1;
    dmpf64_gf2 *a=calloc(1ULL<<DEPTH,sizeof(*a)), *b=calloc(1ULL<<DEPTH,sizeof(*b)); dmpf64_dpf_eval_all(&k0,a); dmpf64_dpf_eval_all(&k1,b);
    for(uint64_t i=0;i<(1ULL<<DEPTH);i++){ dmpf64_node x=dmpf64_encode_index(i,DEPTH); dmpf64_gf2 e0=dmpf64_dpf_eval(&k0,x), e1=dmpf64_dpf_eval(&k1,x); if(!dmpf64_gf2_eq(e0,a[i])||!dmpf64_gf2_eq(e1,b[i])){fprintf(stderr,"eval mismatch\n");return 2;} dmpf64_gf2 s=dmpf64_gf2_add(e0,e1); if(i==point){ if(!dmpf64_gf2_eq(s,beta)){fprintf(stderr,"point mismatch\n");return 3;} } else if(!dmpf64_gf2_is_zero(s)){fprintf(stderr,"zero mismatch at %lu\n",i);return 4;} }
    free(a);free(b);dmpf64_dpf_key_free(&k0);dmpf64_dpf_key_free(&k1); return 0;
}
static int test_dmpf(void){
    uint8_t seed[16]={9,8,7}; dmpf64_rng rng; dmpf64_rng_seed(&rng,seed); const size_t DEPTH=9, P=13; dmpf64_kv kv[P];
    for(size_t i=0;i<P;i++){ kv[i].x=dmpf64_encode_index((uint64_t)(i*3+1),DEPTH); kv[i].y=dmpf64_rng_gf2(&rng); }
    dmpf64_dpf_dmpf_key k0,k1; if(dmpf64_dpf_dmpf_gen(&k0,&k1,DEPTH,kv,P,&rng)!=0)return 1;
    dmpf64_gf2 *a=NULL,*b=NULL; size_t n=0,m=0; dmpf64_dpf_dmpf_eval_all(&k0,&a,&n); dmpf64_dpf_dmpf_eval_all(&k1,&b,&m); if(n!=m)return 2;
    for(uint64_t i=0;i<n;i++){ dmpf64_node x=dmpf64_encode_index(i,DEPTH); dmpf64_gf2 s=dmpf64_gf2_add(dmpf64_dpf_dmpf_eval(&k0,x),dmpf64_dpf_dmpf_eval(&k1,x)); dmpf64_gf2 expected=dmpf64_gf2_zero(); for(size_t j=0;j<P;j++) if(dmpf64_node_eq(kv[j].x,x)) expected=kv[j].y; if(!dmpf64_gf2_eq(s,expected)){fprintf(stderr,"dmpf mismatch %lu\n",i);return 3;} dmpf64_gf2 ss=dmpf64_gf2_add(a[i],b[i]); if(!dmpf64_gf2_eq(ss,expected)){fprintf(stderr,"dmpf all mismatch\n");return 4;} }
    free(a);free(b); dmpf64_dpf_dmpf_key_free(&k0); dmpf64_dpf_dmpf_key_free(&k1); return 0;
}
int main(void){int r=test_single_dpf(); if(r)return r; r=test_dmpf(); if(r)return r; puts("test_dpf ok"); return 0;}
