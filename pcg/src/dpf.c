#include "dmpf64/dpf.h"
#include "dmpf64/prg.h"
#include <stdlib.h>
#include <string.h>

static int kv_cmp(const void *a, const void *b) {
    const dmpf64_kv *aa=(const dmpf64_kv*)a, *bb=(const dmpf64_kv*)b;
    return dmpf64_node_cmp(aa->x, bb->x);
}

void dmpf64_dpf_key_init_empty(dmpf64_dpf_key *k) { memset(k, 0, sizeof(*k)); }
void dmpf64_dpf_key_free(dmpf64_dpf_key *k) { if (!k) return; free(k->cws); memset(k,0,sizeof(*k)); }
int dmpf64_dpf_key_clone(dmpf64_dpf_key *dst, const dmpf64_dpf_key *src) {
    dmpf64_dpf_key_init_empty(dst);
    *dst = *src;
    dst->cws = NULL;
    if (src->input_bits) {
        dst->cws = (dmpf64_correction_word*)malloc(src->input_bits * sizeof(*dst->cws));
        if (!dst->cws) return -1;
        memcpy(dst->cws, src->cws, src->input_bits * sizeof(*dst->cws));
    }
    return 0;
}

static dmpf64_correction_word cw_new(dmpf64_node node, bool left_bit, bool right_bit) {
    dmpf64_node_push_first_two_bits(&node, left_bit, right_bit);
    return (dmpf64_correction_word){node};
}

int dmpf64_dpf_gen(dmpf64_dpf_key *k0, dmpf64_dpf_key *k1, dmpf64_node alpha, size_t input_bits, dmpf64_gf2 beta, dmpf64_rng *rng) {
    dmpf64_dpf_key_init_empty(k0); dmpf64_dpf_key_init_empty(k1);
    dmpf64_correction_word *cws = NULL;
    if (input_bits) {
        cws = (dmpf64_correction_word*)calloc(input_bits, sizeof(*cws));
        if (!cws) return -1;
    }
    bool t0=false, t1=true;
    dmpf64_node seed0 = dmpf64_rng_node(rng), seed1 = dmpf64_rng_node(rng);
    dmpf64_node root0=seed0, root1=seed1;
    for (size_t i=0;i<input_bits;i++) {
        dmpf64_node ch0[2], ch1[2];
        dmpf64_double_prg(seed0, ch0); dmpf64_double_prg(seed1, ch1);
        dmpf64_node sl0=ch0[0], sr0=ch0[1], sl1=ch1[0], sr1=ch1[1];
        bool path = dmpf64_node_get_bit_msb(alpha, i);
        bool tl0,tmp, tl1, tr0, tr1;
        dmpf64_node_pop_first_two_bits(&sl0, &tl0, &tmp);
        dmpf64_node_pop_first_two_bits(&sl1, &tl1, &tmp);
        dmpf64_node_pop_first_two_bits(&sr0, &tr0, &tmp);
        dmpf64_node_pop_first_two_bits(&sr1, &tr1, &tmp);
        bool diff_left = !(tl0 ^ tl1 ^ path);
        bool diff_right = tr0 ^ tr1 ^ path;
        dmpf64_node cw_node = !path ? dmpf64_node_xor(sr0, sr1) : dmpf64_node_xor(sl0, sl1);
        seed0 = path ? sr0 : sl0;
        seed1 = path ? sr1 : sl1;
        if (t0) dmpf64_node_xor_inplace(&seed0, cw_node);
        if (t1) dmpf64_node_xor_inplace(&seed1, cw_node);
        if (path) { t0 = tr0 ^ (t0 & diff_right); t1 = tr1 ^ (t1 & diff_right); }
        else      { t0 = tl0 ^ (t0 & diff_left);  t1 = tl1 ^ (t1 & diff_left); }
        cws[i] = cw_new(cw_node, diff_left, diff_right);
    }
    dmpf64_gf2 conv0 = dmpf64_gf2_from_node(seed0);
    dmpf64_gf2 conv1 = dmpf64_gf2_from_node(seed1);
    dmpf64_gf2 last = dmpf64_gf2_sub(dmpf64_gf2_sub(conv0, conv1), beta);
    if (t0) last = dmpf64_gf2_neg(last);
    k0->root = root0; k0->root_bit=false; k0->input_bits=input_bits; k0->cws=cws; k0->last_cw=last;
    if (dmpf64_dpf_key_clone(k1, k0) != 0) { dmpf64_dpf_key_free(k0); return -1; }
    k1->root = root1; k1->root_bit=true;
    return 0;
}

dmpf64_gf2 dmpf64_dpf_eval(const dmpf64_dpf_key *k, dmpf64_node x) {
    bool t = k->root_bit;
    dmpf64_node s = k->root;
    for (size_t i=0;i<k->input_bits;i++) {
        bool path = dmpf64_node_get_bit_msb(x, i);
        dmpf64_node seeds[2]; dmpf64_double_prg(s, seeds);
        s = seeds[path ? 1 : 0];
        bool new_t,tmp; dmpf64_node_pop_first_two_bits(&s, &new_t, &tmp);
        if (t) {
            dmpf64_node cw = k->cws[i].node;
            bool lb, rb; dmpf64_node_pop_first_two_bits(&cw, &lb, &rb);
            dmpf64_node_xor_inplace(&s, cw);
            new_t ^= (lb & !path) ^ (rb & path);
        }
        t = new_t;
    }
    dmpf64_gf2 out = dmpf64_gf2_from_node(s);
    if (t) out = dmpf64_gf2_add(out, k->last_cw);
    if (k->root_bit) out = dmpf64_gf2_neg(out);
    return out;
}

int dmpf64_dpf_session_init(dmpf64_dpf_session *sess, size_t input_bits) {
    memset(sess,0,sizeof(*sess));
    if (input_bits > 27) input_bits = 1;
    sess->cap = (size_t)1 << input_bits;
    sess->cur_seeds = (dmpf64_node*)malloc(sess->cap * sizeof(dmpf64_node));
    sess->next_seeds = (dmpf64_node*)malloc(sess->cap * sizeof(dmpf64_node));
    sess->cur_signs = (bool*)malloc(sess->cap * sizeof(bool));
    sess->next_signs = (bool*)malloc(sess->cap * sizeof(bool));
    if (!sess->cur_seeds || !sess->next_seeds || !sess->cur_signs || !sess->next_signs) { dmpf64_dpf_session_free(sess); return -1; }
    return 0;
}
void dmpf64_dpf_session_free(dmpf64_dpf_session *sess) {
    if (!sess) return;
    free(sess->cur_seeds); free(sess->next_seeds); free(sess->cur_signs); free(sess->next_signs); memset(sess,0,sizeof(*sess));
}

int dmpf64_dpf_eval_all_add(const dmpf64_dpf_key *k, dmpf64_dpf_session *sess, dmpf64_gf2 *out) {
    size_t n = (size_t)1 << k->input_bits;
    if (sess->cap < n) return -1;
    dmpf64_node *cur=sess->cur_seeds, *next=sess->next_seeds;
    bool *cs=sess->cur_signs, *ns=sess->next_signs;
    cur[0]=k->root; cs[0]=k->root_bit;
    for (size_t depth=0; depth<k->input_bits; depth++) {
        size_t cur_n=(size_t)1<<depth;
        dmpf64_double_prg_many(cur, cur_n, next);
        dmpf64_node cur_cw = k->cws[depth].node;
        bool cw_l,cw_r; dmpf64_node_pop_first_two_bits(&cur_cw, &cw_l, &cw_r);
        for (size_t j=0;j<cur_n;j++) {
            dmpf64_node seed_l=next[2*j], seed_r=next[2*j+1];
            bool tl,tr,tmp; dmpf64_node_pop_first_two_bits(&seed_l,&tl,&tmp); dmpf64_node_pop_first_two_bits(&seed_r,&tr,&tmp);
            if (cs[j]) { dmpf64_node_xor_inplace(&seed_l, cur_cw); dmpf64_node_xor_inplace(&seed_r, cur_cw); tl ^= cw_l; tr ^= cw_r; }
            next[2*j]=seed_l; next[2*j+1]=seed_r; ns[2*j]=tl; ns[2*j+1]=tr;
        }
        dmpf64_node *tn=cur; cur=next; next=tn; bool *tb=cs; cs=ns; ns=tb;
    }
    for (size_t i=0;i<n;i++) {
        dmpf64_gf2 y = dmpf64_gf2_from_node(cur[i]);
        if (cs[i]) y = dmpf64_gf2_add(y, k->last_cw);
        if (k->root_bit) y = dmpf64_gf2_neg(y);
        out[i] = dmpf64_gf2_add(out[i], y);
    }
    return 0;
}

int dmpf64_dpf_eval_all(const dmpf64_dpf_key *k, dmpf64_gf2 *out) {
    size_t n = (size_t)1 << k->input_bits;
    for (size_t i=0;i<n;i++) out[i] = dmpf64_gf2_zero();
    dmpf64_dpf_session sess; if (dmpf64_dpf_session_init(&sess, k->input_bits) != 0) return -1;
    int rc = dmpf64_dpf_eval_all_add(k, &sess, out);
    dmpf64_dpf_session_free(&sess); return rc;
}

void dmpf64_dpf_dmpf_key_init_empty(dmpf64_dpf_dmpf_key *k) { memset(k,0,sizeof(*k)); }
void dmpf64_dpf_dmpf_key_free(dmpf64_dpf_dmpf_key *k) { if (!k) return; for(size_t i=0;i<k->count;i++) dmpf64_dpf_key_free(&k->dpf_keys[i]); free(k->dpf_keys); memset(k,0,sizeof(*k)); }

int dmpf64_dpf_dmpf_gen(dmpf64_dpf_dmpf_key *k0, dmpf64_dpf_dmpf_key *k1, size_t input_bits, const dmpf64_kv *inputs, size_t count, dmpf64_rng *rng) {
    dmpf64_dpf_dmpf_key_init_empty(k0); dmpf64_dpf_dmpf_key_init_empty(k1);
    if (count == 0) return -1;
    dmpf64_kv *copy = (dmpf64_kv*)malloc(count*sizeof(*copy)); if(!copy) return -1;
    memcpy(copy, inputs, count*sizeof(*copy)); qsort(copy,count,sizeof(*copy),kv_cmp);
    k0->dpf_keys=(dmpf64_dpf_key*)calloc(count,sizeof(dmpf64_dpf_key)); k1->dpf_keys=(dmpf64_dpf_key*)calloc(count,sizeof(dmpf64_dpf_key));
    if(!k0->dpf_keys || !k1->dpf_keys){free(copy); dmpf64_dpf_dmpf_key_free(k0); dmpf64_dpf_dmpf_key_free(k1); return -1;}
    k0->count=k1->count=count;
    for(size_t i=0;i<count;i++) {
        if (dmpf64_dpf_gen(&k0->dpf_keys[i], &k1->dpf_keys[i], copy[i].x, input_bits, copy[i].y, rng) != 0) { free(copy); return -1; }
    }
    free(copy); return 0;
}

dmpf64_gf2 dmpf64_dpf_dmpf_eval(const dmpf64_dpf_dmpf_key *k, dmpf64_node x) {
    dmpf64_gf2 out = dmpf64_gf2_zero();
    for(size_t i=0;i<k->count;i++) out = dmpf64_gf2_add(out, dmpf64_dpf_eval(&k->dpf_keys[i], x));
    return out;
}

int dmpf64_dpf_dmpf_eval_all(const dmpf64_dpf_dmpf_key *k, dmpf64_gf2 **outp, size_t *out_len) {
    if (k->count == 0) return -1;
    size_t bits = k->dpf_keys[0].input_bits, n=(size_t)1<<bits;
    dmpf64_gf2 *out=(dmpf64_gf2*)calloc(n,sizeof(*out)); if(!out) return -1;
    dmpf64_dpf_session sess; if (dmpf64_dpf_session_init(&sess,bits)!=0){free(out);return -1;}
    for(size_t i=0;i<k->count;i++) if(dmpf64_dpf_eval_all_add(&k->dpf_keys[i], &sess, out)!=0){dmpf64_dpf_session_free(&sess); free(out); return -1;}
    dmpf64_dpf_session_free(&sess); *outp=out; *out_len=n; return 0;
}
size_t dmpf64_dpf_dmpf_input_length(const dmpf64_dpf_dmpf_key *k){return k->count?k->dpf_keys[0].input_bits:0;}
size_t dmpf64_dpf_dmpf_point_count(const dmpf64_dpf_dmpf_key *k){return k->count;}
