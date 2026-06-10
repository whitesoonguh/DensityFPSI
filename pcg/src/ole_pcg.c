#include "dmpf64/ole_pcg.h"
#include "dmpf64/fft_additive.h"
#include <stdlib.h>
#include <string.h>

static int polynomials_from_seed(const uint8_t seed[16], dmpf64_cyclotomic_mod mod, size_t count, dmpf64_ring_elem **outp){
    dmpf64_rng rng; dmpf64_rng_seed(&rng, seed);
    dmpf64_ring_elem *out=calloc(count?count:1,sizeof(*out)); if(!out)return -1;
    for(size_t i=0;i<count;i++){
        out[i].mod=mod;
        if(dmpf64_dense_new(&out[i].poly,mod.degree)!=0){ for(size_t j=0;j<i;j++) dmpf64_ring_free(&out[j]); free(out); return -1; }
        for(size_t j=0;j<mod.degree;j++) out[i].poly.c[j]=dmpf64_rng_gf(&rng);
    }
    *outp=out; return 0;
}
static int sparse_polys_seed(const uint8_t seed[16], bool regular, dmpf64_cyclotomic_mod mod, size_t weight, size_t count, dmpf64_sparse_ring_elem **outp){
    dmpf64_rng rng; dmpf64_rng_seed(&rng, seed); dmpf64_sparse_ring_elem *out=calloc(count,sizeof(*out)); if(!out)return -1;
    for(size_t i=0;i<count;i++){ dmpf64_sparse_poly p={0}; int rc= regular?dmpf64_sparse_random_regular(&p,mod.degree,weight,&rng):dmpf64_sparse_random(&p,mod.degree,weight,&rng); if(rc!=0){free(out);return -1;} if(dmpf64_sparse_ring_new(&out[i],&p,mod)!=0){dmpf64_sparse_free(&p);free(out);return -1;} dmpf64_sparse_free(&p); }
    *outp=out; return 0;
}

void dmpf64_ole_seed_free(dmpf64_ole_seed *s){ if(!s)return; for(size_t i=0;i<s->compression_factor;i++)dmpf64_sparse_ring_free(&s->sparse_polynomials[i]); free(s->sparse_polynomials); size_t n=s->compression_factor*s->compression_factor; for(size_t i=0;i<n;i++)dmpf64_dpf_dmpf_key_free(&s->tensor_product_fss[i]); free(s->tensor_product_fss); memset(s,0,sizeof(*s)); }
void dmpf64_ole_seed_regular_free(dmpf64_ole_seed_regular *s){ if(!s)return; for(size_t i=0;i<s->compression_factor;i++)dmpf64_sparse_ring_free(&s->sparse_polynomials[i]); free(s->sparse_polynomials); size_t n=s->compression_factor*s->compression_factor; for(size_t i=0;i<n;i++){ for(size_t b=0;b<s->tensor_product_fss[i].block_count;b++)dmpf64_dpf_dmpf_key_free(&s->tensor_product_fss[i].blocks[b]); free(s->tensor_product_fss[i].blocks);} free(s->tensor_product_fss); memset(s,0,sizeof(*s)); }

int dmpf64_ole_gen(dmpf64_ole_seed *first, dmpf64_ole_seed *second, size_t log_degree, size_t compression_factor, size_t weight, dmpf64_rng *rng){
    memset(first,0,sizeof(*first)); memset(second,0,sizeof(*second)); dmpf64_cyclotomic_mod mod=dmpf64_cyclotomic_new(log_degree);
    uint8_t poly_seed[16], first_sparse_seed[16], second_sparse_seed[16]; dmpf64_rng_bytes(rng,poly_seed,16); dmpf64_rng_bytes(rng,first_sparse_seed,16); dmpf64_rng_bytes(rng,second_sparse_seed,16);
    dmpf64_sparse_ring_elem *sp0=NULL,*sp1=NULL; if(sparse_polys_seed(first_sparse_seed,false,mod,weight,compression_factor,&sp0)!=0)return -1; if(sparse_polys_seed(second_sparse_seed,false,mod,weight,compression_factor,&sp1)!=0)return -1;
    size_t total=compression_factor*compression_factor; dmpf64_dpf_dmpf_key *keys0=calloc(total,sizeof(*keys0)), *keys1=calloc(total,sizeof(*keys1)); if(!keys0||!keys1)return -1;
    for(size_t i=0;i<compression_factor;i++) for(size_t j=0;j<compression_factor;j++){ dmpf64_sparse_poly prod={0}; if(dmpf64_sparse_mul(&prod,&sp0[i].poly,&sp1[j].poly)!=0)return -1; if(dmpf64_share_polynomial(&keys0[i*compression_factor+j],&keys1[i*compression_factor+j],&prod,weight*weight,log_degree+1,rng)!=0){dmpf64_sparse_free(&prod);return -1;} dmpf64_sparse_free(&prod); }
    memcpy(first->my_vec_seed,poly_seed,16); memcpy(first->peer_vec_seed,poly_seed,16); first->role=DMPF64_ROLE_FIRST; first->compression_factor=compression_factor; first->sparse_polynomials=sp0; first->tensor_product_fss=keys0; first->mod=mod;
    memcpy(second->my_vec_seed,poly_seed,16); memcpy(second->peer_vec_seed,poly_seed,16); second->role=DMPF64_ROLE_SECOND; second->compression_factor=compression_factor; second->sparse_polynomials=sp1; second->tensor_product_fss=keys1; second->mod=mod;
    return 0;
}

int dmpf64_ole_gen_regular(dmpf64_ole_seed_regular *first, dmpf64_ole_seed_regular *second, size_t log_degree, size_t compression_factor, size_t weight, dmpf64_rng *rng){
    memset(first,0,sizeof(*first)); memset(second,0,sizeof(*second)); dmpf64_cyclotomic_mod mod=dmpf64_cyclotomic_new(log_degree); uint8_t poly_seed[16], s0seed[16], s1seed[16]; dmpf64_rng_bytes(rng,poly_seed,16); dmpf64_rng_bytes(rng,s0seed,16); dmpf64_rng_bytes(rng,s1seed,16);
    dmpf64_sparse_ring_elem *sp0=NULL,*sp1=NULL; if(sparse_polys_seed(s0seed,true,mod,weight,compression_factor,&sp0)!=0)return -1; if(sparse_polys_seed(s1seed,true,mod,weight,compression_factor,&sp1)!=0)return -1; size_t total=compression_factor*compression_factor; dmpf64_regular_share *rs0=calloc(total,sizeof(*rs0)), *rs1=calloc(total,sizeof(*rs1)); if(!rs0||!rs1)return -1;
    for(size_t i=0;i<compression_factor;i++) for(size_t j=0;j<compression_factor;j++){ dmpf64_sparse_poly prod={0}; if(dmpf64_sparse_mul(&prod,&sp0[i].poly,&sp1[j].poly)!=0)return -1; dmpf64_dpf_dmpf_key *b0=NULL,*b1=NULL; size_t bc=0,bs=0; if(dmpf64_share_regular_polynomial_mul(&b0,&b1,&bc,&bs,&prod,weight,log_degree+1,rng)!=0){dmpf64_sparse_free(&prod);return -1;} rs0[i*compression_factor+j]=(dmpf64_regular_share){bs,bc,b0}; rs1[i*compression_factor+j]=(dmpf64_regular_share){bs,bc,b1}; dmpf64_sparse_free(&prod); }
    memcpy(first->my_vec_seed,poly_seed,16); memcpy(first->peer_vec_seed,poly_seed,16); first->role=DMPF64_ROLE_FIRST; first->compression_factor=compression_factor; first->sparse_polynomials=sp0; first->tensor_product_fss=rs0; first->mod=mod;
    memcpy(second->my_vec_seed,poly_seed,16); memcpy(second->peer_vec_seed,poly_seed,16); second->role=DMPF64_ROLE_SECOND; second->compression_factor=compression_factor; second->sparse_polynomials=sp1; second->tensor_product_fss=rs1; second->mod=mod; return 0;
}

static int zero_ring(dmpf64_ring_elem *e, dmpf64_cyclotomic_mod mod){ e->mod=mod; return dmpf64_dense_new(&e->poly,mod.degree); }


static int ole_expand_add_share_eval3_nonregular(const dmpf64_ole_seed *s,
                                                 const dmpf64_ring_elem *my,
                                                 const dmpf64_ring_elem *peer,
                                                 dmpf64_ring_elem *add_share) {
    const size_t cf = s->compression_factor;
    const size_t count = cf * cf;
    const dmpf64_ring_elem **avec = calloc(count ? count : 1, sizeof(*avec));
    const dmpf64_ring_elem **bvec = calloc(count ? count : 1, sizeof(*bvec));
    const dmpf64_ring_elem **cvec = calloc(count ? count : 1, sizeof(*cvec));
    dmpf64_ring_elem *shares = calloc(count ? count : 1, sizeof(*shares));
    if (!avec || !bvec || !cvec || !shares) { free(avec); free(bvec); free(cvec); free(shares); return -1; }

    int rc = 0;
    for (size_t i = 0; rc == 0 && i < cf; i++) for (size_t j = 0; rc == 0 && j < cf; j++) {
        size_t z = i * cf + j;
        if (s->role == DMPF64_ROLE_FIRST) { avec[z] = &my[i]; bvec[z] = &peer[j]; }
        else { avec[z] = &peer[i]; bvec[z] = &my[j]; }
        dmpf64_dense_poly dense_share = {0};
        dmpf64_dense_poly modded = {0};
        rc = dmpf64_from_share(&dense_share, &s->tensor_product_fss[z]);
        if (rc == 0) rc = dmpf64_mod_dense(&modded, &dense_share, s->mod);
        if (rc == 0) { shares[z].poly = modded; shares[z].mod = s->mod; cvec[z] = &shares[z]; }
        else dmpf64_dense_free(&modded);
        dmpf64_dense_free(&dense_share);
    }
    if (rc == 0) rc = dmpf64_afft_cyclic_sum_product3(add_share, s->mod, count, avec, bvec, cvec);
    for (size_t z = 0; z < count; z++) dmpf64_ring_free(&shares[z]);
    free(shares); free(avec); free(bvec); free(cvec);
    return rc;
}

static int ole_expand_add_share_eval3_regular(const dmpf64_ole_seed_regular *s,
                                              const dmpf64_ring_elem *my,
                                              const dmpf64_ring_elem *peer,
                                              dmpf64_ring_elem *add_share) {
    const size_t cf = s->compression_factor;
    const size_t count = cf * cf;
    const dmpf64_ring_elem **avec = calloc(count ? count : 1, sizeof(*avec));
    const dmpf64_ring_elem **bvec = calloc(count ? count : 1, sizeof(*bvec));
    const dmpf64_ring_elem **cvec = calloc(count ? count : 1, sizeof(*cvec));
    dmpf64_ring_elem *shares = calloc(count ? count : 1, sizeof(*shares));
    if (!avec || !bvec || !cvec || !shares) { free(avec); free(bvec); free(cvec); free(shares); return -1; }

    int rc = 0;
    for (size_t i = 0; rc == 0 && i < cf; i++) for (size_t j = 0; rc == 0 && j < cf; j++) {
        size_t z = i * cf + j;
        if (s->role == DMPF64_ROLE_FIRST) { avec[z] = &my[i]; bvec[z] = &peer[j]; }
        else { avec[z] = &peer[i]; bvec[z] = &my[j]; }
        dmpf64_regular_share *rs = &s->tensor_product_fss[z];
        dmpf64_dense_poly dense_share = {0};
        dmpf64_dense_poly modded = {0};
        rc = dmpf64_from_share_regular_error(&dense_share, rs->blocks, rs->block_count, rs->block_size, s->mod.degree * 2);
        if (rc == 0) rc = dmpf64_mod_dense(&modded, &dense_share, s->mod);
        if (rc == 0) { shares[z].poly = modded; shares[z].mod = s->mod; cvec[z] = &shares[z]; }
        else dmpf64_dense_free(&modded);
        dmpf64_dense_free(&dense_share);
    }
    if (rc == 0) rc = dmpf64_afft_cyclic_sum_product3(add_share, s->mod, count, avec, bvec, cvec);
    for (size_t z = 0; z < count; z++) dmpf64_ring_free(&shares[z]);
    free(shares); free(avec); free(bvec); free(cvec);
    return rc;
}

int dmpf64_ole_expand(const dmpf64_ole_seed *s, dmpf64_ring_elem *mul_share, dmpf64_ring_elem *add_share){
    dmpf64_ring_elem *my=NULL,*peer=NULL; if(polynomials_from_seed(s->my_vec_seed,s->mod,s->compression_factor,&my)!=0)return -1; if(polynomials_from_seed(s->peer_vec_seed,s->mod,s->compression_factor,&peer)!=0)return -1;
    int rc = 0;
    if(zero_ring(mul_share,s->mod)!=0){ rc=-1; goto cleanup; }
    for(size_t i=0; rc==0 && i<s->compression_factor;i++){ rc=dmpf64_sparse_ring_mul_dense_addto(mul_share,&s->sparse_polynomials[i],&my[i]); }
    if(rc!=0) goto cleanup;
    if(s->mod.degree >= 1024) {
        rc = ole_expand_add_share_eval3_nonregular(s, my, peer, add_share);
    } else {
        if(zero_ring(add_share,s->mod)!=0){ rc=-1; goto cleanup; }
        for(size_t i=0;rc==0 && i<s->compression_factor;i++) for(size_t j=0;rc==0 && j<s->compression_factor;j++){ const dmpf64_ring_elem *a,*b; if(s->role==DMPF64_ROLE_FIRST){a=&my[i]; b=&peer[j];} else {a=&peer[i]; b=&my[j];} dmpf64_ring_elem dense_prod={0}; if(dmpf64_ring_mul(&dense_prod,a,b)!=0){rc=-1;break;} dmpf64_dense_poly dense_share={0}; if(dmpf64_from_share(&dense_share,&s->tensor_product_fss[i*s->compression_factor+j])!=0){dmpf64_ring_free(&dense_prod);rc=-1;break;} dmpf64_dense_poly modded={0}; if(dmpf64_mod_dense(&modded,&dense_share,s->mod)!=0){dmpf64_ring_free(&dense_prod);dmpf64_dense_free(&dense_share);rc=-1;break;} dmpf64_ring_elem share_elem={.poly=modded,.mod=s->mod}; rc=dmpf64_ring_mul_addto(add_share,&dense_prod,&share_elem); dmpf64_ring_free(&dense_prod); dmpf64_dense_free(&dense_share); dmpf64_ring_free(&share_elem); }
    }
cleanup:
    if(my){for(size_t i=0;i<s->compression_factor;i++)dmpf64_ring_free(&my[i]);free(my);} if(peer){for(size_t i=0;i<s->compression_factor;i++)dmpf64_ring_free(&peer[i]);free(peer);} return rc;
}

int dmpf64_ole_expand_regular(const dmpf64_ole_seed_regular *s, dmpf64_ring_elem *mul_share, dmpf64_ring_elem *add_share){
    dmpf64_ring_elem *my=NULL,*peer=NULL; if(polynomials_from_seed(s->my_vec_seed,s->mod,s->compression_factor,&my)!=0)return -1; if(polynomials_from_seed(s->peer_vec_seed,s->mod,s->compression_factor,&peer)!=0)return -1;
    int rc = 0;
    if(zero_ring(mul_share,s->mod)!=0){ rc=-1; goto cleanup; }
    for(size_t i=0; rc==0 && i<s->compression_factor;i++){ rc=dmpf64_sparse_ring_mul_dense_addto(mul_share,&s->sparse_polynomials[i],&my[i]); }
    if(rc!=0) goto cleanup;
    if(s->mod.degree >= 1024) {
        rc = ole_expand_add_share_eval3_regular(s, my, peer, add_share);
    } else {
        if(zero_ring(add_share,s->mod)!=0){ rc=-1; goto cleanup; }
        for(size_t i=0;rc==0 && i<s->compression_factor;i++) for(size_t j=0;rc==0 && j<s->compression_factor;j++){ const dmpf64_ring_elem *a,*b; if(s->role==DMPF64_ROLE_FIRST){a=&my[i]; b=&peer[j];} else {a=&peer[i]; b=&my[j];} dmpf64_ring_elem dense_prod={0}; if(dmpf64_ring_mul(&dense_prod,a,b)!=0){rc=-1;break;} dmpf64_regular_share *rs=&s->tensor_product_fss[i*s->compression_factor+j]; dmpf64_dense_poly dense_share={0}; if(dmpf64_from_share_regular_error(&dense_share,rs->blocks,rs->block_count,rs->block_size,s->mod.degree*2)!=0){dmpf64_ring_free(&dense_prod);rc=-1;break;} dmpf64_dense_poly modded={0}; if(dmpf64_mod_dense(&modded,&dense_share,s->mod)!=0){dmpf64_ring_free(&dense_prod);dmpf64_dense_free(&dense_share);rc=-1;break;} dmpf64_ring_elem share_elem={.poly=modded,.mod=s->mod}; rc=dmpf64_ring_mul_addto(add_share,&dense_prod,&share_elem); dmpf64_ring_free(&dense_prod); dmpf64_dense_free(&dense_share); dmpf64_ring_free(&share_elem); }
    }
cleanup:
    if(my){for(size_t i=0;i<s->compression_factor;i++)dmpf64_ring_free(&my[i]);free(my);} if(peer){for(size_t i=0;i<s->compression_factor;i++)dmpf64_ring_free(&peer[i]);free(peer);} return rc;
}
