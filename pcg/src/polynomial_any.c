#include "dmpf64/polynomial_any.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int sparse_push_local(dmpf64_sparse_poly *p, size_t idx, dmpf64_gf c){
    if(p->len==p->cap){ size_t nc=p->cap?p->cap*2:16; void *q=realloc(p->terms,nc*sizeof(*p->terms)); if(!q)return -1; p->terms=(dmpf64_sparse_term*)q; p->cap=nc; }
    p->terms[p->len++]=(dmpf64_sparse_term){idx,c}; return 0;
}
static int term_cmp_local(const void*a,const void*b){const dmpf64_sparse_term*x=a,*y=b; return x->idx<y->idx?-1:x->idx>y->idx?1:0;}
static void sparse_sort_combine_local(dmpf64_sparse_poly *p){
    if(p->len==0)return; qsort(p->terms,p->len,sizeof(*p->terms),term_cmp_local); size_t w=0;
    for(size_t r=0;r<p->len;r++){ if(p->terms[r].coeff==0)continue; if(w&&p->terms[w-1].idx==p->terms[r].idx) p->terms[w-1].coeff^=p->terms[r].coeff; else p->terms[w++]=p->terms[r]; if(w&&p->terms[w-1].coeff==0)w--; }
    p->len=w;
}

int dmpf64_share_polynomial_any(dmpf64_any_key *k0, dmpf64_any_key *k1, dmpf64_backend backend, const dmpf64_sparse_poly *p, size_t points, size_t log_degree, dmpf64_rng *rng, size_t batch_size){
    if(points < p->len || log_degree == 0) return -1;
    size_t lower=1; size_t input_length=log_degree-lower; size_t max_points=(size_t)1<<input_length; if(points>max_points) points=max_points;
    dmpf64_kv *inputs=(dmpf64_kv*)calloc(points?points:1,sizeof(*inputs)); if(!inputs) return -1;
    size_t len=0;
    for(size_t i=0;i<p->len;i++){
        size_t idx=p->terms[i].idx; uint64_t cur=(uint64_t)(idx>>lower); size_t internal=idx&1u;
        dmpf64_node x=dmpf64_encode_index(cur,input_length); size_t j;
        for(j=0;j<len;j++) if(dmpf64_node_eq(inputs[j].x,x)) break;
        if(j==len){ inputs[len].x=x; inputs[len].y=dmpf64_gf2_zero(); len++; }
        inputs[j].y.v[internal]=p->terms[i].coeff;
    }
    while(len<points){
        uint64_t idx=dmpf64_rng_u64(rng)%((uint64_t)1<<log_degree); dmpf64_node x=dmpf64_encode_index(idx>>lower,input_length);
        bool exists=false; for(size_t j=0;j<len;j++) if(dmpf64_node_eq(inputs[j].x,x)){exists=true;break;}
        if(exists) continue; inputs[len].x=x; inputs[len].y=dmpf64_gf2_zero(); len++;
    }
    int rc=dmpf64_any_gen(k0,k1,backend,input_length,inputs,len,rng,batch_size); free(inputs); return rc;
}
int dmpf64_from_share_any(dmpf64_dense_poly *out, const dmpf64_any_key *k){
    dmpf64_gf2 *evals=NULL; size_t n=0; if(dmpf64_any_eval_all(k,&evals,&n)!=0) return -1;
    if(dmpf64_dense_new(out,n*2)!=0){free(evals);return -1;}
    for(size_t i=0;i<n;i++){ out->c[2*i]=evals[i].v[0]; out->c[2*i+1]=evals[i].v[1]; }
    free(evals); return 0;
}
int dmpf64_share_regular_polynomial_mul_any(dmpf64_any_key **blocks0, dmpf64_any_key **blocks1, size_t *block_count, size_t *block_size, dmpf64_backend backend, const dmpf64_sparse_poly *p, size_t sqrt_points, size_t log_degree, dmpf64_rng *rng, size_t batch_size){
    size_t total_degree=(size_t)1<<log_degree; size_t t=sqrt_points;
    /* Performance heuristic for the Ring-LPN-128 regular OLE preset: the
       original weight=66 decomposition is slightly faster with a 40-way half
       partition on this GF64/additive-FFT backend. The env override below is
       kept for benchmarking alternative decompositions. */
    if (sqrt_points >= 48 && sqrt_points <= 96) t = 40;
    const char *t_env = getenv("DMPF64_REGULAR_T");
    if(t_env && t_env[0]){ size_t tv=(size_t)strtoull(t_env,NULL,10); if(tv>0) t=tv; }
    *block_size=dmpf64_ceil_div_size(total_degree/2,t); size_t bdp=dmpf64_next_pow2(*block_size); size_t log_sub=dmpf64_log2_pow2(bdp); *block_count=dmpf64_ceil_div_size(total_degree,*block_size);
    dmpf64_any_key *b0=calloc(*block_count?*block_count:1,sizeof(*b0)), *b1=calloc(*block_count?*block_count:1,sizeof(*b1)); if(!b0||!b1){free(b0);free(b1);return -1;}
    size_t coeff_idx=0;
    for(size_t blk=0; blk<*block_count; blk++){
        size_t min_deg=blk*(*block_size), max_deg=(blk+1)*(*block_size); size_t mirror=blk<(*block_count-blk)?blk:(*block_count-blk); size_t expected=2*mirror+1;
        dmpf64_sparse_poly tmp={0};
        while(coeff_idx<p->len && p->terms[coeff_idx].idx<max_deg){ if(sparse_push_local(&tmp,p->terms[coeff_idx].idx-min_deg,p->terms[coeff_idx].coeff)!=0){dmpf64_sparse_free(&tmp);return -1;} coeff_idx++; }
        size_t points_for_block = expected;
        if(points_for_block < tmp.len) points_for_block = tmp.len;
        for(size_t i=0; tmp.len<points_for_block && i<*block_size; i++){
            bool ex=false; for(size_t j=0;j<tmp.len;j++) if(tmp.terms[j].idx==i){ex=true;break;}
            if(!ex && sparse_push_local(&tmp,i,0)!=0){dmpf64_sparse_free(&tmp);return -1;}
        }
        sparse_sort_combine_local(&tmp);
        if(dmpf64_share_polynomial_any(&b0[blk],&b1[blk],backend,&tmp,points_for_block,log_sub,rng,batch_size)!=0){dmpf64_sparse_free(&tmp); for(size_t x=0;x<=blk;x++){dmpf64_any_key_free(&b0[x]);dmpf64_any_key_free(&b1[x]);} free(b0);free(b1);return -1;}
        dmpf64_sparse_free(&tmp);
    }
    *blocks0=b0; *blocks1=b1; return 0;
}
int dmpf64_from_share_regular_error_any(dmpf64_dense_poly *out, const dmpf64_any_key *blocks, size_t block_count, size_t block_size, size_t output_degree){
    if(dmpf64_dense_new(out,output_degree)!=0) return -1;
    for(size_t b=0;b<block_count;b++){
        dmpf64_dense_poly p={0}; if(dmpf64_from_share_any(&p,&blocks[b])!=0){dmpf64_dense_free(out);return -1;}
        size_t start=b*block_size; for(size_t i=0;i<block_size && start+i<output_degree && i<p.len;i++) out->c[start+i]=p.c[i];
        dmpf64_dense_free(&p);
    }
    return 0;
}

int dmpf64_from_share_regular_error_mod_into_any(dmpf64_gf *out, const dmpf64_any_key *blocks, size_t block_count, size_t block_size, size_t mod_degree){
    for(size_t b=0;b<block_count;b++){
        size_t start=b*block_size;
        if(blocks[b].backend==DMPF64_BACKEND_BIG_STATE){
            if(dmpf64_big_state_eval_all_into_coeff_mod(&blocks[b].u.big,out,start,block_size,mod_degree)!=0){ return -1; }
            continue;
        }
        dmpf64_gf2 *evals=NULL; size_t n=0;
        if(dmpf64_any_eval_all(&blocks[b],&evals,&n)!=0){ return -1; }
        size_t coeffs = block_size < 2*n ? block_size : 2*n;
        for(size_t i=0;i<coeffs;i++){
            dmpf64_gf c = evals[i>>1].v[i&1u];
            out[(start+i)&(mod_degree-1)] ^= c;
        }
        free(evals);
    }
    return 0;
}


int dmpf64_from_share_regular_error_write_any(dmpf64_gf *out, const dmpf64_any_key *blocks,
                                               size_t block_count, size_t block_size, size_t output_degree){
    size_t written = 0;
    for(size_t b=0; b<block_count && written<output_degree; b++){
        size_t start = b * block_size;
        size_t coeffs = block_size;
        if(start >= output_degree) break;
        if(coeffs > output_degree - start) coeffs = output_degree - start;
        if(blocks[b].backend==DMPF64_BACKEND_BIG_STATE){
            if(dmpf64_big_state_eval_all_write_coeff(&blocks[b].u.big, out + start, coeffs)!=0) return -1;
        } else {
            dmpf64_gf2 *evals=NULL; size_t n=0;
            if(dmpf64_any_eval_all(&blocks[b],&evals,&n)!=0) return -1;
            size_t maxc = coeffs < 2*n ? coeffs : 2*n;
            for(size_t i=0;i<maxc;i++) out[start+i]=evals[i>>1].v[i&1u];
            if(maxc < coeffs) memset(out+start+maxc,0,(coeffs-maxc)*sizeof(*out));
            free(evals);
        }
        written = start + coeffs;
    }
    if(written < output_degree) memset(out+written,0,(output_degree-written)*sizeof(*out));
    return 0;
}


int dmpf64_from_share_regular_error_xor_write_any(dmpf64_gf *out, const dmpf64_any_key *blocks,
                                                   size_t block_count, size_t block_size, size_t output_degree){
    if(!out && output_degree) return -1;
    if(output_degree == 0) return 0;
    for(size_t b=0; b<block_count; b++){
        size_t start = b * block_size;
        if(start >= output_degree) break;
        size_t coeffs = block_size;
        if(coeffs > output_degree - start) coeffs = output_degree - start;
        if(blocks[b].backend==DMPF64_BACKEND_BIG_STATE){
            if(dmpf64_big_state_eval_all_xor_coeff(&blocks[b].u.big, out + start, coeffs)!=0) return -1;
            continue;
        }
        dmpf64_gf2 *evals=NULL; size_t n=0;
        if(dmpf64_any_eval_all(&blocks[b],&evals,&n)!=0) return -1;
        size_t maxc = coeffs < 2*n ? coeffs : 2*n;
        for(size_t i=0;i<maxc;i++) out[start+i] ^= evals[i>>1].v[i&1u];
        free(evals);
    }
    return 0;
}

int dmpf64_from_share_regular_error_mod_any(dmpf64_dense_poly *out, const dmpf64_any_key *blocks, size_t block_count, size_t block_size, size_t mod_degree){
    if(dmpf64_dense_new(out,mod_degree)!=0) return -1;
    int rc = dmpf64_from_share_regular_error_mod_into_any(out->c, blocks, block_count, block_size, mod_degree);
    if(rc != 0) dmpf64_dense_free(out);
    return rc;
}

void dmpf64_any_regular_share_free(dmpf64_any_regular_share *s){
    if(!s) return; for(size_t i=0;i<s->block_count;i++) dmpf64_any_key_free(&s->blocks[i]); free(s->blocks); memset(s,0,sizeof(*s));
}
