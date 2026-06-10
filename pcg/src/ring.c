#include "dmpf64/ring.h"
#include "dmpf64/fft_additive.h"
#include <stdlib.h>
#include <string.h>

int dmpf64_mod_dense(dmpf64_dense_poly *out, const dmpf64_dense_poly *in, dmpf64_cyclotomic_mod mod){
    if(dmpf64_dense_new(out,mod.degree)!=0)return -1;
    for(size_t i=0;i<in->len;i++) out->c[i & (mod.degree-1)] ^= in->c[i];
    return 0;
}
void dmpf64_ring_free(dmpf64_ring_elem *e){ if(!e)return; dmpf64_dense_free(&e->poly); }
void dmpf64_sparse_ring_free(dmpf64_sparse_ring_elem *e){ if(!e)return; dmpf64_sparse_free(&e->poly); }
int dmpf64_ring_new(dmpf64_ring_elem *e, const dmpf64_dense_poly *p, dmpf64_cyclotomic_mod mod){ e->mod=mod; return dmpf64_mod_dense(&e->poly,p,mod); }
int dmpf64_sparse_ring_new(dmpf64_sparse_ring_elem *e, const dmpf64_sparse_poly *p, dmpf64_cyclotomic_mod mod){ e->mod=mod; memset(&e->poly,0,sizeof(e->poly)); e->poly.terms=malloc(p->len*sizeof(*p->terms)); if(p->len && !e->poly.terms)return -1; memcpy(e->poly.terms,p->terms,p->len*sizeof(*p->terms)); e->poly.len=e->poly.cap=p->len; return 0; }
int dmpf64_ring_add(dmpf64_ring_elem *out, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b){ if(a->mod.degree!=b->mod.degree)return -1; out->mod=a->mod; if(dmpf64_dense_new(&out->poly,a->mod.degree)!=0)return -1; for(size_t i=0;i<a->mod.degree;i++) out->poly.c[i]=a->poly.c[i]^b->poly.c[i]; return 0; }
int dmpf64_ring_add_inplace(dmpf64_ring_elem *acc, const dmpf64_ring_elem *x){ if(acc->mod.degree!=x->mod.degree)return -1; for(size_t i=0;i<acc->mod.degree;i++) acc->poly.c[i]^=x->poly.c[i]; return 0; }

static int ring_mul_cyclic_naive(dmpf64_ring_elem *out, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b){
    if(a->mod.degree!=b->mod.degree)return -1;
    size_t N=a->mod.degree;
    out->mod=a->mod;
    if(dmpf64_dense_new(&out->poly,N)!=0)return -1;
    /* In characteristic two, x^N+1 == x^N-1, so reduction is cyclic xor. */
    for(size_t i=0;i<N;i++){
        dmpf64_gf ai=a->poly.c[i];
        if(!ai) continue;
        size_t split = N - i;
        for(size_t j=0;j<split;j++) out->poly.c[i+j] ^= dmpf64_gf_mul(ai,b->poly.c[j]);
        for(size_t j=split;j<N;j++) out->poly.c[j-split] ^= dmpf64_gf_mul(ai,b->poly.c[j]);
    }
    return 0;
}
int dmpf64_ring_mul(dmpf64_ring_elem *out, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b){
    if(a->mod.degree!=b->mod.degree)return -1;
    if(a->mod.degree >= 1024) return dmpf64_afft_cyclic_mul_cached(out, a, b);
    return ring_mul_cyclic_naive(out, a, b);
}
int dmpf64_ring_mul_addto(dmpf64_ring_elem *acc, const dmpf64_ring_elem *a, const dmpf64_ring_elem *b){
    if(a->mod.degree!=b->mod.degree || acc->mod.degree!=a->mod.degree)return -1;
    if(a->mod.degree >= 1024) return dmpf64_afft_cyclic_mul_addto_cached(acc, a, b);
    dmpf64_ring_elem tmp={0};
    int rc=ring_mul_cyclic_naive(&tmp,a,b);
    if(rc==0) rc=dmpf64_ring_add_inplace(acc,&tmp);
    dmpf64_ring_free(&tmp);
    return rc;
}

int dmpf64_sparse_ring_mul_dense_addto(dmpf64_ring_elem *acc, const dmpf64_sparse_ring_elem *a, const dmpf64_ring_elem *b){
    if(a->mod.degree!=b->mod.degree || acc->mod.degree!=b->mod.degree)return -1;
    size_t N=a->mod.degree;
    for(size_t ti=0;ti<a->poly.len;ti++){
        size_t idx=a->poly.terms[ti].idx & (N-1);
        dmpf64_gf coeff=a->poly.terms[ti].coeff;
        if(!coeff) continue;
        size_t split = N - idx;
        dmpf64_gf *dst0 = acc->poly.c + idx;
        const dmpf64_gf *src0 = b->poly.c;
        for(size_t j=0;j<split;j++) dst0[j] ^= dmpf64_gf_mul(coeff, src0[j]);
        dmpf64_gf *dst1 = acc->poly.c;
        const dmpf64_gf *src1 = b->poly.c + split;
        for(size_t j=split;j<N;j++) dst1[j-split] ^= dmpf64_gf_mul(coeff, src1[j-split]);
    }
    return 0;
}
int dmpf64_sparse_ring_mul_dense(dmpf64_ring_elem *out, const dmpf64_sparse_ring_elem *a, const dmpf64_ring_elem *b){
    if(a->mod.degree!=b->mod.degree)return -1;
    out->mod=a->mod;
    if(dmpf64_dense_new(&out->poly,a->mod.degree)!=0)return -1;
    return dmpf64_sparse_ring_mul_dense_addto(out,a,b);
}
int dmpf64_ring_eq(const dmpf64_ring_elem *a, const dmpf64_ring_elem *b){ if(a->mod.degree!=b->mod.degree)return 0; for(size_t i=0;i<a->mod.degree;i++)if(a->poly.c[i]!=b->poly.c[i])return 0; return 1; }
