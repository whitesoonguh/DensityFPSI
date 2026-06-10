#include "dmpf64/big_state.h"
#include "dmpf64/prg.h"
#include <stdlib.h>
#include <string.h>

static size_t ceil128(size_t x){ return (x + 127u) >> 7; }
static unsigned __int128 node_to_u128(dmpf64_node n){ return ((unsigned __int128)n.hi << 64) | n.lo; }
static unsigned __int128 kv_word(dmpf64_node x, size_t input_bits){
    if (input_bits == 0) return 0;
    return node_to_u128(x) >> (128u - input_bits);
}
static int kv_cmp_node(const void *a, const void *b){
    const dmpf64_kv *aa=(const dmpf64_kv*)a, *bb=(const dmpf64_kv*)b;
    return dmpf64_node_cmp(aa->x, bb->x);
}

/* ----- Signs ----- */
typedef struct { dmpf64_node *v; size_t t; size_t nodes; } signs;
typedef struct { size_t t; dmpf64_node *v; size_t nodes_per_point; } keygen_signs;

static int signs_init(signs *s, size_t t){
    s->t=t; s->nodes=ceil128(t); s->v=(dmpf64_node*)calloc(s->nodes? s->nodes:1, sizeof(dmpf64_node));
    return s->v?0:-1;
}
static void signs_free(signs *s){ free(s->v); memset(s,0,sizeof(*s)); }
static void signs_zero(signs *s){ memset(s->v,0,s->nodes*sizeof(dmpf64_node)); }
static bool signs_get_bit_nodes(const dmpf64_node *v, size_t idx){ return dmpf64_node_get_bit_lsb(v[idx>>7], idx & 127u); }
static void signs_set_bit_nodes(dmpf64_node *v, size_t idx, bool b){ dmpf64_node_set_bit_lsb(&v[idx>>7], idx & 127u, b); }
static bool signs_get_bit(const signs *s, size_t idx){ return signs_get_bit_nodes(s->v, idx); }
static void signs_set_bit(signs *s, size_t idx, bool b){ signs_set_bit_nodes(s->v, idx, b); }
static void signs_fill_with_seed(signs *s, dmpf64_node seed, size_t direction){
    uint16_t first = (uint16_t)(2u + s->nodes * direction);
    dmpf64_many_prg(seed, first, s->nodes, s->v);
}
static void signs_xor_into(signs *dst, const signs *a, const signs *b){
    for(size_t i=0;i<dst->nodes;i++) dst->v[i]=dmpf64_node_xor(a->v[i], b->v[i]);
}
static void xor_bits(const dmpf64_node *src, dmpf64_node *dst, size_t bit_count){
    size_t n=ceil128(bit_count); for(size_t i=0;i<n;i++) dmpf64_node_xor_inplace(&dst[i], src[i]);
}
static void copy_bits(const dmpf64_node *src, dmpf64_node *dst, size_t bit_count){
    size_t n=ceil128(bit_count); for(size_t i=0;i<n;i++) dst[i]=src[i];
}
static size_t chunk_bits_from_nodes(const dmpf64_node *src, size_t start, size_t count, size_t chunk_bits){
    size_t out=0;
    size_t rem=count > start ? count - start : 0;
    if(rem > chunk_bits) rem=chunk_bits;
    for(size_t i=0;i<rem;i++) if(signs_get_bit_nodes(src,start+i)) out |= ((size_t)1u << i);
    return out;
}

static int keygen_signs_init(keygen_signs *s, size_t t){
    s->t=t; s->nodes_per_point=ceil128(t);
    size_t total=t*s->nodes_per_point;
    s->v=(dmpf64_node*)calloc(total?total:1,sizeof(dmpf64_node));
    return s->v?0:-1;
}
static void keygen_signs_free(keygen_signs *s){ free(s->v); memset(s,0,sizeof(*s)); }
static size_t keygen_coord(const keygen_signs *s, size_t point){ return s->nodes_per_point * point; }
static void keygen_set_bit(keygen_signs *s, size_t point, size_t bit_idx, bool b){ signs_set_bit_nodes(&s->v[keygen_coord(s,point)], bit_idx, b); }
static bool keygen_get_bit(const keygen_signs *s, size_t point, size_t bit_idx){ return signs_get_bit_nodes(&s->v[keygen_coord(s,point)], bit_idx); }
static void keygen_set_signs(keygen_signs *s, size_t point, const signs *src){ copy_bits(src->v, &s->v[keygen_coord(s,point)], s->t); }
static size_t keygen_chunk(const keygen_signs *s, size_t point, size_t start, size_t count, size_t chunk_bits){ return chunk_bits_from_nodes(&s->v[keygen_coord(s,point)], start, count, chunk_bits); }

/* ----- Trie utilities over sorted words, matching src/trie.rs ----- */
typedef struct { unsigned __int128 *w; size_t len; size_t depth; } word_trie;
static int cmp_u128(const void *a,const void *b){ unsigned __int128 x=*(const unsigned __int128*)a,y=*(const unsigned __int128*)b; return x<y?-1:x>y?1:0; }
static int trie_init(word_trie *tr, const dmpf64_kv *inputs, size_t count, size_t depth){
    tr->w=(unsigned __int128*)malloc((count?count:1)*sizeof(*tr->w)); if(!tr->w) return -1;
    tr->len=count; tr->depth=depth;
    for(size_t i=0;i<count;i++) tr->w[i]=kv_word(inputs[i].x, depth);
    qsort(tr->w,count,sizeof(*tr->w),cmp_u128);
    return 0;
}
static void trie_free(word_trie *tr){ free(tr->w); memset(tr,0,sizeof(*tr)); }
static size_t trie_prefixes_at_depth(const word_trie *tr, size_t d, unsigned __int128 *out){
    size_t n=0;
    for(size_t i=0;i<tr->len;i++){
        unsigned __int128 p = d==0 ? 0 : (tr->w[i] >> (tr->depth - d));
        if(n==0 || out[n-1]!=p) out[n++]=p;
    }
    return n;
}
static void trie_has_son(const word_trie *tr, unsigned __int128 prefix, size_t prefix_len, bool *has_left, bool *has_right){
    *has_left=false; *has_right=false;
    for(size_t i=0;i<tr->len;i++){
        unsigned __int128 p = prefix_len==0 ? 0 : (tr->w[i] >> (tr->depth - prefix_len));
        if(p != prefix) continue;
        unsigned bit=(unsigned)((tr->w[i] >> (tr->depth - prefix_len - 1u)) & 1u);
        if(bit) *has_right=true; else *has_left=true;
        if(*has_left && *has_right) return;
    }
}

/* ----- SignsCW/CW ----- */
static void signs_cw_free(dmpf64_signs_cw *s){ free(s->bits); memset(s,0,sizeof(*s)); }
static int signs_cw_init_random(dmpf64_signs_cw *s, size_t t, size_t depth, dmpf64_rng *rng){
    s->t=t; s->nodes_per_point_per_direction=ceil128(t);
    size_t min=t < ((size_t)1 << depth) ? t : ((size_t)1 << depth);
    size_t total=2*min*s->nodes_per_point_per_direction;
    s->bits=(dmpf64_node*)malloc((total?total:1)*sizeof(dmpf64_node)); if(!s->bits) return -1;
    for(size_t i=0;i<total;i++) s->bits[i]=dmpf64_rng_node(rng);
    return 0;
}
static size_t signs_cw_coord(const dmpf64_signs_cw *s, bool direction, size_t point_idx){
    return ((point_idx << 1u) + (direction?1u:0u)) * s->nodes_per_point_per_direction;
}
static void signs_cw_put(const signs *src, dmpf64_signs_cw *dst, bool direction, size_t point_idx){ copy_bits(src->v, &dst->bits[signs_cw_coord(dst,direction,point_idx)], dst->t); }
static void signs_cw_flip_bit(dmpf64_signs_cw *s, bool direction, size_t point_idx, size_t bit_idx){
    dmpf64_node *n=&s->bits[signs_cw_coord(s,direction,point_idx)+(bit_idx>>7)];
    n->lo ^= (bit_idx & 127u) < 64 ? (UINT64_C(1) << (bit_idx & 63u)) : 0;
    if((bit_idx & 127u) >= 64) n->hi ^= (UINT64_C(1) << ((bit_idx & 127u)-64));
}
static void signs_cw_xor_point_into(const dmpf64_signs_cw *s, dmpf64_node *output, size_t point_idx, bool direction){ xor_bits(&s->bits[signs_cw_coord(s,direction,point_idx)], output, s->t); }
static int cw_clone(dmpf64_big_cw *dst, const dmpf64_big_cw *src){
    memset(dst,0,sizeof(*dst)); dst->seed_count=src->seed_count; dst->batch_size=src->batch_size;
    dst->seeds=(dmpf64_node*)malloc((src->seed_count?src->seed_count:1)*sizeof(dmpf64_node)); if(!dst->seeds) return -1;
    memcpy(dst->seeds,src->seeds,src->seed_count*sizeof(dmpf64_node));
    dst->signs.t=src->signs.t; dst->signs.nodes_per_point_per_direction=src->signs.nodes_per_point_per_direction;
    size_t total=2*src->seed_count*src->signs.nodes_per_point_per_direction;
    dst->signs.bits=(dmpf64_node*)malloc((total?total:1)*sizeof(dmpf64_node)); if(!dst->signs.bits){free(dst->seeds);return -1;}
    memcpy(dst->signs.bits,src->signs.bits,total*sizeof(dmpf64_node)); return 0;
}
static void cw_free(dmpf64_big_cw *cw){ if(!cw)return; free(cw->seeds); signs_cw_free(&cw->signs); memset(cw,0,sizeof(*cw)); }
static dmpf64_node cw_correct_chunks(const dmpf64_big_cw *cw, const dmpf64_node *sign_nodes, size_t count, bool has_left, bool has_right, dmpf64_node *out_l, dmpf64_node *out_r){
    dmpf64_node correct=dmpf64_node_zero();
    size_t chunk=0, start=0;
    while(start < count && chunk*cw->batch_size < cw->seed_count){
        size_t bits=chunk_bits_from_nodes(sign_nodes,start,count,cw->batch_size);
        for(size_t b=0;b<cw->batch_size;b++){
            size_t idx=chunk*cw->batch_size+b;
            if(idx>=cw->seed_count) break;
            if((bits>>b)&1u){
                dmpf64_node_xor_inplace(&correct,cw->seeds[idx]);
                if(has_left) signs_cw_xor_point_into(&cw->signs,out_l,idx,false);
                if(has_right) signs_cw_xor_point_into(&cw->signs,out_r,idx,true);
            }
        }
        start += cw->batch_size; chunk++;
    }
    return correct;
}
static dmpf64_node cw_correct_keygen(const dmpf64_big_cw *cw, const keygen_signs *kg, size_t point, size_t count, bool has_left, bool has_right, signs *left, signs *right){
    dmpf64_node correct=dmpf64_node_zero();
    size_t chunk=0, start=0;
    while(start<count && chunk*cw->batch_size < cw->seed_count){
        size_t bits=keygen_chunk(kg,point,start,count,cw->batch_size);
        for(size_t b=0;b<cw->batch_size;b++){
            size_t idx=chunk*cw->batch_size+b; if(idx>=cw->seed_count) break;
            if((bits>>b)&1u){
                dmpf64_node_xor_inplace(&correct,cw->seeds[idx]);
                if(has_left) signs_cw_xor_point_into(&cw->signs,left->v,idx,false);
                if(has_right) signs_cw_xor_point_into(&cw->signs,right->v,idx,true);
            }
        }
        start+=cw->batch_size; chunk++;
    }
    return correct;
}
static dmpf64_node cw_correct_single(const dmpf64_big_cw *cw, const signs *in_signs, bool direction, signs *output){
    dmpf64_node correct=dmpf64_node_zero();
    for(size_t idx=0; idx<cw->seed_count && idx<in_signs->t; idx++){
        if(!signs_get_bit(in_signs,idx)) continue;
        dmpf64_node_xor_inplace(&correct,cw->seeds[idx]);
        signs_cw_xor_point_into(&cw->signs, output->v, idx, direction);
    }
    return correct;
}

static int cw_gen(dmpf64_big_cw *cw, size_t depth, const word_trie *tr, dmpf64_signs_cw *signs_in, dmpf64_node *seeds_in, size_t seeds_len, dmpf64_rng *rng, size_t batch_size){
    memset(cw,0,sizeof(*cw));
    size_t t=signs_in->t; size_t min=t < ((size_t)1<<depth) ? t : ((size_t)1<<depth);
    unsigned __int128 *prefixes=(unsigned __int128*)malloc((min?min:1)*sizeof(*prefixes)); if(!prefixes) return -1;
    size_t pn=trie_prefixes_at_depth(tr,depth,prefixes);
    size_t total_next=0;
    for(size_t idx=0;idx<pn;idx++){
        bool hl,hr; trie_has_son(tr,prefixes[idx],depth,&hl,&hr);
        if(hl){ signs_cw_flip_bit(signs_in,false,idx,total_next); total_next++; }
        if(hr){ signs_cw_flip_bit(signs_in,true,idx,total_next); total_next++; }
    }
    free(prefixes);
    cw->seed_count=min; cw->batch_size=batch_size;
    cw->seeds=(dmpf64_node*)malloc((min?min:1)*sizeof(dmpf64_node)); if(!cw->seeds) return -1;
    for(size_t i=0;i<min;i++) cw->seeds[i]= i<seeds_len ? seeds_in[i] : dmpf64_rng_node(rng);
    cw->signs=*signs_in; memset(signs_in,0,sizeof(*signs_in));
    return 0;
}

static void conv_free(dmpf64_conv_cw *c){ free(c->o); memset(c,0,sizeof(*c)); }
static int conv_clone(dmpf64_conv_cw *dst, const dmpf64_conv_cw *src){
    dst->t=src->t; dst->batch_size=src->batch_size; dst->o=(dmpf64_gf2*)malloc((src->t?src->t:1)*sizeof(dmpf64_gf2)); if(!dst->o) return -1;
    memcpy(dst->o,src->o,src->t*sizeof(dmpf64_gf2)); return 0;
}
static dmpf64_gf2 conv_correct(const dmpf64_conv_cw *cw, const signs *s){
    dmpf64_gf2 out=dmpf64_gf2_zero();
    for(size_t i=0;i<cw->t;i++) if(signs_get_bit(s,i)) out=dmpf64_gf2_add(out,cw->o[i]);
    return out;
}


/* ----- Fast eval_all state, direct port of EvalAllState in src/big_state.rs ----- */
typedef struct {
    size_t t;
    size_t nodes_per_path;
    dmpf64_node *bits;
    size_t path_capacity;
} eval_all_state;
static int eval_state_init(eval_all_state *st, size_t points, size_t input_len){
    st->t=points; st->nodes_per_path=ceil128(points); st->path_capacity=(size_t)1 << input_len;
    size_t total=st->nodes_per_path*2*st->path_capacity;
    st->bits=(dmpf64_node*)calloc(total?total:1,sizeof(dmpf64_node));
    return st->bits?0:-1;
}
static void eval_state_free(eval_all_state *st){ free(st->bits); memset(st,0,sizeof(*st)); }
static size_t eval_state_coord(const eval_all_state *st, size_t point){ return point*st->nodes_per_path; }
static void eval_state_put_sign(eval_all_state *st, size_t point, const signs *sgn){ copy_bits(sgn->v,&st->bits[eval_state_coord(st,point)],st->t); }
static void eval_state_get_sign(const eval_all_state *st, size_t point, signs *sgn){ copy_bits(&st->bits[eval_state_coord(st,point)],sgn->v,st->t); }
static dmpf64_node *eval_state_after_expand(eval_all_state *st, size_t point, bool direction){
    size_t start=(2*st->nodes_per_path*point)+st->nodes_per_path*(direction?1u:0u);
    return &st->bits[start];
}
static dmpf64_node *eval_state_sign(eval_all_state *st, size_t point){ return &st->bits[eval_state_coord(st,point)]; }
static int eval_state_fill_from_seeds(eval_all_state *st, const dmpf64_node *seeds, size_t seed_count){
    size_t per=st->nodes_per_path*2;
    if(per > 256) return -2;
    dmpf64_many_many_prg(seeds,seed_count,2,(uint16_t)per,st->bits);
    return 0;
}


/* ----- Small-state eval_all fast path (t <= 128) ----- */
static inline uint64_t low_mask64(size_t bits) {
    return bits >= 64 ? UINT64_MAX : (bits == 0 ? 0 : ((UINT64_C(1) << bits) - 1));
}
static inline bool node_bit_lsb_fast(dmpf64_node v, size_t idx) {
    return idx < 64 ? ((v.lo >> idx) & 1u) : ((v.hi >> (idx - 64u)) & 1u);
}
static dmpf64_gf2 conv_correct_node_fast(const dmpf64_conv_cw *cw, dmpf64_node s) {
    dmpf64_gf2 out = dmpf64_gf2_zero();
    uint64_t lo = s.lo & low_mask64(cw->t < 64 ? cw->t : 64);
    while (lo) {
        unsigned b = (unsigned)__builtin_ctzll(lo);
        out.v[0] ^= cw->o[b].v[0];
        out.v[1] ^= cw->o[b].v[1];
        lo &= lo - 1;
    }
    if (cw->t > 64) {
        uint64_t hi = s.hi & low_mask64(cw->t - 64);
        while (hi) {
            unsigned b = (unsigned)__builtin_ctzll(hi);
            size_t idx = 64u + b;
            out.v[0] ^= cw->o[idx].v[0];
            out.v[1] ^= cw->o[idx].v[1];
            hi &= hi - 1;
        }
    }
    return out;
}
static dmpf64_node cw_correct_node_fast(const dmpf64_big_cw *cw, dmpf64_node sign,
                                        size_t count, dmpf64_node *left, dmpf64_node *right) {
    dmpf64_node correct = dmpf64_node_zero();
    size_t limit = cw->seed_count < count ? cw->seed_count : count;
    uint64_t lo = sign.lo & low_mask64(limit < 64 ? limit : 64);
    while (lo) {
        unsigned b = (unsigned)__builtin_ctzll(lo);
        dmpf64_node_xor_inplace(&correct, cw->seeds[b]);
        dmpf64_node_xor_inplace(left, cw->signs.bits[signs_cw_coord(&cw->signs, false, b)]);
        dmpf64_node_xor_inplace(right, cw->signs.bits[signs_cw_coord(&cw->signs, true, b)]);
        lo &= lo - 1;
    }
    if (limit > 64) {
        uint64_t hi = sign.hi & low_mask64(limit - 64);
        while (hi) {
            unsigned b = (unsigned)__builtin_ctzll(hi);
            size_t idx = 64u + b;
            dmpf64_node_xor_inplace(&correct, cw->seeds[idx]);
            dmpf64_node_xor_inplace(left, cw->signs.bits[signs_cw_coord(&cw->signs, false, idx)]);
            dmpf64_node_xor_inplace(right, cw->signs.bits[signs_cw_coord(&cw->signs, true, idx)]);
            hi &= hi - 1;
        }
    }
    return correct;
}


/* Chunk-table accelerated correction for the Ring-LPN-128 path.  For t<=64
 * the sign state is a single 64-bit mask.  Scanning set bits costs about
 * t/2 branches per path; the 8-bit tables below reduce that to 6-8 straight
 * table XORs and are rebuilt once per depth/key. */
typedef struct {
    size_t chunks;
    dmpf64_node corr[8][256];
    dmpf64_node left[8][256];
    dmpf64_node right[8][256];
} cw64_chunk_table;

typedef struct {
    size_t chunks;
    dmpf64_gf2 v[8][256];
} conv64_chunk_table;

static inline dmpf64_node node_xor3(dmpf64_node a, dmpf64_node b, dmpf64_node c) {
    return (dmpf64_node){ a.lo ^ b.lo ^ c.lo, a.hi ^ b.hi ^ c.hi };
}

static void cw64_table_build(cw64_chunk_table *tab, const dmpf64_big_cw *cw, size_t count) {
    size_t limit = cw->seed_count < count ? cw->seed_count : count;
    if (limit > 64) limit = 64;
    tab->chunks = (limit + 7u) >> 3;
    if (tab->chunks == 0) tab->chunks = 1;
    for (size_t ch = 0; ch < tab->chunks; ch++) {
        for (size_t mask = 0; mask < 256; mask++) {
            dmpf64_node corr = dmpf64_node_zero();
            dmpf64_node left = dmpf64_node_zero();
            dmpf64_node right = dmpf64_node_zero();
            uint32_t m = (uint32_t)mask;
            while (m) {
                unsigned bit = (unsigned)__builtin_ctz(m);
                size_t idx = (ch << 3) + bit;
                if (idx < limit) {
                    dmpf64_node_xor_inplace(&corr, cw->seeds[idx]);
                    dmpf64_node_xor_inplace(&left, cw->signs.bits[signs_cw_coord(&cw->signs, false, idx)]);
                    dmpf64_node_xor_inplace(&right, cw->signs.bits[signs_cw_coord(&cw->signs, true, idx)]);
                }
                m &= m - 1u;
            }
            tab->corr[ch][mask] = corr;
            tab->left[ch][mask] = left;
            tab->right[ch][mask] = right;
        }
    }
}

static inline void cw64_table_apply(const cw64_chunk_table *tab, uint64_t sign,
                                    dmpf64_node *corr, dmpf64_node *left_xor, dmpf64_node *right_xor) {
    dmpf64_node c = dmpf64_node_zero();
    dmpf64_node l = dmpf64_node_zero();
    dmpf64_node r = dmpf64_node_zero();
    for (size_t ch = 0; ch < tab->chunks; ch++) {
        unsigned m = (unsigned)((sign >> (8u * ch)) & 0xffu);
        c.lo ^= tab->corr[ch][m].lo; c.hi ^= tab->corr[ch][m].hi;
        l.lo ^= tab->left[ch][m].lo; l.hi ^= tab->left[ch][m].hi;
        r.lo ^= tab->right[ch][m].lo; r.hi ^= tab->right[ch][m].hi;
    }
    *corr = c; *left_xor = l; *right_xor = r;
}

static void conv64_table_build(conv64_chunk_table *tab, const dmpf64_conv_cw *cw) {
    size_t limit = cw->t < 64 ? cw->t : 64;
    tab->chunks = (limit + 7u) >> 3;
    if (tab->chunks == 0) tab->chunks = 1;
    for (size_t ch = 0; ch < tab->chunks; ch++) {
        for (size_t mask = 0; mask < 256; mask++) {
            dmpf64_gf2 y = dmpf64_gf2_zero();
            uint32_t m = (uint32_t)mask;
            while (m) {
                unsigned bit = (unsigned)__builtin_ctz(m);
                size_t idx = (ch << 3) + bit;
                if (idx < limit) { y.v[0] ^= cw->o[idx].v[0]; y.v[1] ^= cw->o[idx].v[1]; }
                m &= m - 1u;
            }
            tab->v[ch][mask] = y;
        }
    }
}

static inline dmpf64_gf2 conv64_table_apply(const conv64_chunk_table *tab, uint64_t sign) {
    dmpf64_gf2 y = dmpf64_gf2_zero();
    for (size_t ch = 0; ch < tab->chunks; ch++) {
        unsigned m = (unsigned)((sign >> (8u * ch)) & 0xffu);
        y.v[0] ^= tab->v[ch][m].v[0];
        y.v[1] ^= tab->v[ch][m].v[1];
    }
    return y;
}

static dmpf64_gf2 conv_correct_2node_fast(const dmpf64_conv_cw *cw, const dmpf64_node s[2]) {
    dmpf64_gf2 out = dmpf64_gf2_zero();
    uint64_t lo = s[0].lo;
    while (lo) {
        unsigned b = (unsigned)__builtin_ctzll(lo);
        out.v[0] ^= cw->o[b].v[0];
        out.v[1] ^= cw->o[b].v[1];
        lo &= lo - 1;
    }
    uint64_t hi = s[0].hi;
    while (hi) {
        unsigned b = (unsigned)__builtin_ctzll(hi);
        size_t idx = 64u + b;
        if (idx < cw->t) { out.v[0] ^= cw->o[idx].v[0]; out.v[1] ^= cw->o[idx].v[1]; }
        hi &= hi - 1;
    }
    if (cw->t > 128) {
        uint64_t lo2 = s[1].lo & low_mask64(cw->t - 128);
        while (lo2) {
            unsigned b = (unsigned)__builtin_ctzll(lo2);
            size_t idx = 128u + b;
            out.v[0] ^= cw->o[idx].v[0];
            out.v[1] ^= cw->o[idx].v[1];
            lo2 &= lo2 - 1;
        }
        if (cw->t > 192) {
            uint64_t hi2 = s[1].hi & low_mask64(cw->t - 192);
            while (hi2) {
                unsigned b = (unsigned)__builtin_ctzll(hi2);
                size_t idx = 192u + b;
                out.v[0] ^= cw->o[idx].v[0];
                out.v[1] ^= cw->o[idx].v[1];
                hi2 &= hi2 - 1;
            }
        }
    }
    return out;
}

static dmpf64_node cw_correct_2node_fast(const dmpf64_big_cw *cw, const dmpf64_node sign[2],
                                         size_t count, dmpf64_node left[2], dmpf64_node right[2]) {
    dmpf64_node correct = dmpf64_node_zero();
    size_t limit = cw->seed_count < count ? cw->seed_count : count;
    uint64_t lo = sign[0].lo & low_mask64(limit < 64 ? limit : 64);
    while (lo) {
        unsigned b = (unsigned)__builtin_ctzll(lo);
        dmpf64_node_xor_inplace(&correct, cw->seeds[b]);
        const dmpf64_node *lb = &cw->signs.bits[signs_cw_coord(&cw->signs, false, b)];
        const dmpf64_node *rb = &cw->signs.bits[signs_cw_coord(&cw->signs, true, b)];
        dmpf64_node_xor_inplace(&left[0], lb[0]);
        dmpf64_node_xor_inplace(&right[0], rb[0]);
        if (cw->signs.nodes_per_point_per_direction > 1) { dmpf64_node_xor_inplace(&left[1], lb[1]); dmpf64_node_xor_inplace(&right[1], rb[1]); }
        lo &= lo - 1;
    }
    if (limit > 64) {
        uint64_t hi = sign[0].hi & low_mask64(limit - 64);
        while (hi) {
            unsigned b = (unsigned)__builtin_ctzll(hi);
            size_t idx = 64u + b;
            dmpf64_node_xor_inplace(&correct, cw->seeds[idx]);
            const dmpf64_node *lb = &cw->signs.bits[signs_cw_coord(&cw->signs, false, idx)];
            const dmpf64_node *rb = &cw->signs.bits[signs_cw_coord(&cw->signs, true, idx)];
            dmpf64_node_xor_inplace(&left[0], lb[0]);
            dmpf64_node_xor_inplace(&right[0], rb[0]);
            if (cw->signs.nodes_per_point_per_direction > 1) { dmpf64_node_xor_inplace(&left[1], lb[1]); dmpf64_node_xor_inplace(&right[1], rb[1]); }
            hi &= hi - 1;
        }
    }
    if (limit > 128) {
        uint64_t lo2 = sign[1].lo & low_mask64(limit - 128);
        while (lo2) {
            unsigned b = (unsigned)__builtin_ctzll(lo2);
            size_t idx = 128u + b;
            dmpf64_node_xor_inplace(&correct, cw->seeds[idx]);
            const dmpf64_node *lb = &cw->signs.bits[signs_cw_coord(&cw->signs, false, idx)];
            const dmpf64_node *rb = &cw->signs.bits[signs_cw_coord(&cw->signs, true, idx)];
            dmpf64_node_xor_inplace(&left[0], lb[0]);
            dmpf64_node_xor_inplace(&right[0], rb[0]);
            if (cw->signs.nodes_per_point_per_direction > 1) { dmpf64_node_xor_inplace(&left[1], lb[1]); dmpf64_node_xor_inplace(&right[1], rb[1]); }
            lo2 &= lo2 - 1;
        }
        if (limit > 192) {
            uint64_t hi2 = sign[1].hi & low_mask64(limit - 192);
            while (hi2) {
                unsigned b = (unsigned)__builtin_ctzll(hi2);
                size_t idx = 192u + b;
                dmpf64_node_xor_inplace(&correct, cw->seeds[idx]);
                const dmpf64_node *lb = &cw->signs.bits[signs_cw_coord(&cw->signs, false, idx)];
                const dmpf64_node *rb = &cw->signs.bits[signs_cw_coord(&cw->signs, true, idx)];
                dmpf64_node_xor_inplace(&left[0], lb[0]);
                dmpf64_node_xor_inplace(&right[0], rb[0]);
                if (cw->signs.nodes_per_point_per_direction > 1) { dmpf64_node_xor_inplace(&left[1], lb[1]); dmpf64_node_xor_inplace(&right[1], rb[1]); }
                hi2 &= hi2 - 1;
            }
        }
    }
    return correct;
}

static int eval_all_t256(const dmpf64_big_state_key *k, dmpf64_gf2 **outp, size_t *out_len) {
    const size_t input_len = k->input_bits;
    const size_t n = (size_t)1 << input_len;
    dmpf64_gf2 *out = (dmpf64_gf2*)malloc((n ? n : 1) * sizeof(*out));
    dmpf64_node *seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*seeds));
    dmpf64_node *next_seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*next_seeds));
    dmpf64_node *signs_cur = (dmpf64_node*)calloc((2*n ? 2*n : 1), sizeof(*signs_cur));
    dmpf64_node *signs_next = (dmpf64_node*)malloc((2*n ? 2*n : 1) * sizeof(*signs_next));
    dmpf64_node *sign_children = (dmpf64_node*)malloc((4*n ? 4*n : 1) * sizeof(*sign_children));
    if (!out || !seeds || !next_seeds || !signs_cur || !signs_next || !sign_children) {
        free(out); free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children); return -1;
    }
    seeds[0] = k->root;
    if (k->sign) signs_cur[0].lo = 1;
    for (size_t depth = 0; depth < input_len; depth++) {
        const size_t cur_n = (size_t)1 << depth;
        dmpf64_double_prg_many(seeds, cur_n, next_seeds);
        dmpf64_many_many_prg(seeds, cur_n, 2, 4, sign_children);
        const dmpf64_big_cw *cw = &k->cws[depth];
        const size_t min = cw->seed_count;
        for (size_t path = 0; path < cur_n; path++) {
            dmpf64_node left[2] = { sign_children[4 * path + 0], sign_children[4 * path + 1] };
            dmpf64_node right[2] = { sign_children[4 * path + 2], sign_children[4 * path + 3] };
            dmpf64_node corr = cw_correct_2node_fast(cw, &signs_cur[2*path], min, left, right);
            dmpf64_node_xor_inplace(&next_seeds[2 * path], corr);
            dmpf64_node_xor_inplace(&next_seeds[2 * path + 1], corr);
            signs_next[4 * path + 0] = left[0];
            signs_next[4 * path + 1] = left[1];
            signs_next[4 * path + 2] = right[0];
            signs_next[4 * path + 3] = right[1];
        }
        dmpf64_node *tmp = seeds; seeds = next_seeds; next_seeds = tmp;
        tmp = signs_cur; signs_cur = signs_next; signs_next = tmp;
    }
    for (size_t i = 0; i < n; i++) {
        out[i] = dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]), conv_correct_2node_fast(&k->conv_cw, &signs_cur[2*i]));
    }
    free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children);
    *outp = out; *out_len = n; return 0;
}

static int eval_all_t256_into_coeff_mod(const dmpf64_big_state_key *k, dmpf64_gf *out,
                                         size_t coeff_start, size_t coeff_count, size_t mod_degree) {
    const size_t input_len = k->input_bits;
    const size_t n = (size_t)1 << input_len;
    dmpf64_node *seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*seeds));
    dmpf64_node *next_seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*next_seeds));
    dmpf64_node *signs_cur = (dmpf64_node*)calloc((2*n ? 2*n : 1), sizeof(*signs_cur));
    dmpf64_node *signs_next = (dmpf64_node*)malloc((2*n ? 2*n : 1) * sizeof(*signs_next));
    dmpf64_node *sign_children = (dmpf64_node*)malloc((4*n ? 4*n : 1) * sizeof(*sign_children));
    if (!seeds || !next_seeds || !signs_cur || !signs_next || !sign_children) {
        free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children); return -1;
    }
    seeds[0] = k->root;
    if (k->sign) signs_cur[0].lo = 1;
    for (size_t depth = 0; depth < input_len; depth++) {
        const size_t cur_n = (size_t)1 << depth;
        dmpf64_double_prg_many(seeds, cur_n, next_seeds);
        dmpf64_many_many_prg(seeds, cur_n, 2, 4, sign_children);
        const dmpf64_big_cw *cw = &k->cws[depth];
        const size_t min = cw->seed_count;
        for (size_t path = 0; path < cur_n; path++) {
            dmpf64_node left[2] = { sign_children[4 * path + 0], sign_children[4 * path + 1] };
            dmpf64_node right[2] = { sign_children[4 * path + 2], sign_children[4 * path + 3] };
            dmpf64_node corr = cw_correct_2node_fast(cw, &signs_cur[2*path], min, left, right);
            dmpf64_node_xor_inplace(&next_seeds[2 * path], corr);
            dmpf64_node_xor_inplace(&next_seeds[2 * path + 1], corr);
            signs_next[4 * path + 0] = left[0];
            signs_next[4 * path + 1] = left[1];
            signs_next[4 * path + 2] = right[0];
            signs_next[4 * path + 3] = right[1];
        }
        dmpf64_node *tmp = seeds; seeds = next_seeds; next_seeds = tmp;
        tmp = signs_cur; signs_cur = signs_next; signs_next = tmp;
    }
    size_t max_paths = (coeff_count + 1) >> 1;
    if (max_paths > n) max_paths = n;
    for (size_t i = 0; i < max_paths; i++) {
        dmpf64_gf2 y = dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]), conv_correct_2node_fast(&k->conv_cw, &signs_cur[2*i]));
        size_t c0 = 2 * i;
        out[(coeff_start + c0) & (mod_degree - 1)] ^= y.v[0];
        if (c0 + 1 < coeff_count) out[(coeff_start + c0 + 1) & (mod_degree - 1)] ^= y.v[1];
    }
    free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children);
    return 0;
}

static int eval_all_small_t(const dmpf64_big_state_key *k, dmpf64_gf2 **outp, size_t *out_len) {
    const size_t input_len = k->input_bits;
    const size_t n = (size_t)1 << input_len;
    dmpf64_gf2 *out = (dmpf64_gf2*)malloc((n ? n : 1) * sizeof(*out));
    dmpf64_node *seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*seeds));
    dmpf64_node *next_seeds = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*next_seeds));
    dmpf64_node *signs_cur = (dmpf64_node*)calloc((n ? n : 1), sizeof(*signs_cur));
    dmpf64_node *signs_next = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*signs_next));
    dmpf64_node *sign_children = (dmpf64_node*)malloc((n ? n : 1) * sizeof(*sign_children));
    if (!out || !seeds || !next_seeds || !signs_cur || !signs_next || !sign_children) {
        free(out); free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children);
        return -1;
    }
    seeds[0] = k->root;
    if (k->sign) signs_cur[0].lo = 1;
    for (size_t depth = 0; depth < input_len; depth++) {
        const size_t cur_n = (size_t)1 << depth;
        dmpf64_bigstate_prg_many(seeds, cur_n, next_seeds, sign_children);
        const dmpf64_big_cw *cw = &k->cws[depth];
        const size_t min = cw->seed_count;
        for (size_t path = 0; path < cur_n; path++) {
            dmpf64_node left = sign_children[2 * path];
            dmpf64_node right = sign_children[2 * path + 1];
            dmpf64_node corr = cw_correct_node_fast(cw, signs_cur[path], min, &left, &right);
            dmpf64_node_xor_inplace(&next_seeds[2 * path], corr);
            dmpf64_node_xor_inplace(&next_seeds[2 * path + 1], corr);
            signs_next[2 * path] = left;
            signs_next[2 * path + 1] = right;
        }
        dmpf64_node *tmp = seeds; seeds = next_seeds; next_seeds = tmp;
        tmp = signs_cur; signs_cur = signs_next; signs_next = tmp;
    }
    for (size_t i = 0; i < n; i++) {
        out[i] = dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]), conv_correct_node_fast(&k->conv_cw, signs_cur[i]));
    }
    free(seeds); free(next_seeds); free(signs_cur); free(signs_next); free(sign_children);
    *outp = out; *out_len = n;
    return 0;
}


typedef struct {
    size_t cap_n;
    dmpf64_node *seeds;
    dmpf64_node *next_seeds;
    dmpf64_node *signs_cur;
    dmpf64_node *signs_next;
    dmpf64_node *sign_children;
} big_eval128_ws;
static big_eval128_ws g_eval128_ws = {0};

static void *aligned_zalloc64(size_t bytes) {
    if (bytes == 0) bytes = 64;
    void *p = NULL;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
#else
    p = malloc(bytes);
#endif
    if (p) memset(p, 0, bytes);
    return p;
}

static int eval128_ws_reserve(size_t n) {
    if (g_eval128_ws.cap_n >= n) return 0;
    dmpf64_node *seeds = (dmpf64_node*)aligned_zalloc64((n ? n : 1) * sizeof(dmpf64_node));
    dmpf64_node *next = (dmpf64_node*)aligned_zalloc64(((2*n) ? (2*n) : 1) * sizeof(dmpf64_node));
    dmpf64_node *scur = (dmpf64_node*)aligned_zalloc64((n ? n : 1) * sizeof(dmpf64_node));
    dmpf64_node *snext = (dmpf64_node*)aligned_zalloc64(((2*n) ? (2*n) : 1) * sizeof(dmpf64_node));
    dmpf64_node *schild = (dmpf64_node*)aligned_zalloc64(((2*n) ? (2*n) : 1) * sizeof(dmpf64_node));
    if (!seeds || !next || !scur || !snext || !schild) {
        free(seeds); free(next); free(scur); free(snext); free(schild);
        return -1;
    }
    free(g_eval128_ws.seeds);
    free(g_eval128_ws.next_seeds);
    free(g_eval128_ws.signs_cur);
    free(g_eval128_ws.signs_next);
    free(g_eval128_ws.sign_children);
    g_eval128_ws.seeds = seeds;
    g_eval128_ws.next_seeds = next;
    g_eval128_ws.signs_cur = scur;
    g_eval128_ws.signs_next = snext;
    g_eval128_ws.sign_children = schild;
    g_eval128_ws.cap_n = n;
    return 0;
}

static int eval128_into_coeff_ex(const dmpf64_big_state_key *k, dmpf64_gf *out,
                                 size_t coeff_start, size_t coeff_count,
                                 size_t mod_degree, int xor_mode, int wrap_mode,
                                 int force_full_paths) {
    const size_t input_len = k->input_bits;
    const size_t n = (size_t)1 << input_len;
    if (eval128_ws_reserve(n) != 0) return -1;
    dmpf64_node *seeds = g_eval128_ws.seeds;
    dmpf64_node *next_seeds = g_eval128_ws.next_seeds;
    dmpf64_node *signs_cur = g_eval128_ws.signs_cur;
    dmpf64_node *signs_next = g_eval128_ws.signs_next;
    dmpf64_node *sign_children = g_eval128_ws.sign_children;

    size_t max_paths = (coeff_count + 1) >> 1;
    if (max_paths > n) max_paths = n;
    const size_t target_paths = force_full_paths ? n : max_paths;
    if (target_paths == 0) return 0;

    seeds[0] = k->root;
    signs_cur[0] = (dmpf64_node){ k->sign ? 1u : 0u, 0u };

    const int use_chunk_tables = (k->conv_cw.t <= 64);
    cw64_chunk_table cwtab;
    size_t cur_n = 1;
    for (size_t depth = 0; depth < input_len; depth++) {
        const size_t remaining_after = input_len - depth - 1;
        size_t next_n = (target_paths + (((size_t)1 << remaining_after) - 1)) >> remaining_after;
        const size_t full_next = cur_n << 1;
        if (next_n > full_next) next_n = full_next;
        dmpf64_bigstate_prg_many(seeds, cur_n, next_seeds, sign_children);
        const dmpf64_big_cw *cw = &k->cws[depth];
        const size_t min = cw->seed_count;
        if (use_chunk_tables && min >= 12 && cur_n >= 64) {
            cw64_table_build(&cwtab, cw, min);
            for (size_t path = 0; path < cur_n; path++) {
                dmpf64_node corr, lx, rx;
                cw64_table_apply(&cwtab, signs_cur[path].lo, &corr, &lx, &rx);
                dmpf64_node left = dmpf64_node_xor(sign_children[2 * path], lx);
                dmpf64_node right = dmpf64_node_xor(sign_children[2 * path + 1], rx);
                next_seeds[2 * path].lo ^= corr.lo;
                next_seeds[2 * path].hi ^= corr.hi;
                next_seeds[2 * path + 1].lo ^= corr.lo;
                next_seeds[2 * path + 1].hi ^= corr.hi;
                signs_next[2 * path] = left;
                signs_next[2 * path + 1] = right;
            }
        } else {
            for (size_t path = 0; path < cur_n; path++) {
                dmpf64_node left = sign_children[2 * path];
                dmpf64_node right = sign_children[2 * path + 1];
                dmpf64_node corr = cw_correct_node_fast(cw, signs_cur[path], min, &left, &right);
                next_seeds[2 * path].lo ^= corr.lo;
                next_seeds[2 * path].hi ^= corr.hi;
                next_seeds[2 * path + 1].lo ^= corr.lo;
                next_seeds[2 * path + 1].hi ^= corr.hi;
                signs_next[2 * path] = left;
                signs_next[2 * path + 1] = right;
            }
        }
        dmpf64_node *tmp = seeds; seeds = next_seeds; next_seeds = tmp;
        tmp = signs_cur; signs_cur = signs_next; signs_next = tmp;
        cur_n = next_n;
    }
    conv64_chunk_table cvtab;
    if (use_chunk_tables) conv64_table_build(&cvtab, &k->conv_cw);
    if (xor_mode) {
        const size_t mask = mod_degree - 1;
        for (size_t i = 0; i < max_paths; i++) {
            dmpf64_gf2 cc = use_chunk_tables ? conv64_table_apply(&cvtab, signs_cur[i].lo) : conv_correct_node_fast(&k->conv_cw, signs_cur[i]);
            dmpf64_gf2 y = dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]), cc);
            size_t c0 = 2 * i;
            if (wrap_mode) {
                out[(coeff_start + c0) & mask] ^= y.v[0];
                if (c0 + 1 < coeff_count) out[(coeff_start + c0 + 1) & mask] ^= y.v[1];
            } else {
                out[coeff_start + c0] ^= y.v[0];
                if (c0 + 1 < coeff_count) out[coeff_start + c0 + 1] ^= y.v[1];
            }
        }
    } else {
        for (size_t i = 0; i < max_paths; i++) {
            dmpf64_gf2 cc = use_chunk_tables ? conv64_table_apply(&cvtab, signs_cur[i].lo) : conv_correct_node_fast(&k->conv_cw, signs_cur[i]);
            dmpf64_gf2 y = dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]), cc);
            size_t c0 = 2 * i;
            out[coeff_start + c0] = y.v[0];
            if (c0 + 1 < coeff_count) out[coeff_start + c0 + 1] = y.v[1];
        }
    }
    return 0;
}

int dmpf64_big_state_eval_all_into_coeff_mod(const dmpf64_big_state_key *k, dmpf64_gf *out,
                                             size_t coeff_start, size_t coeff_count, size_t mod_degree) {
    if (mod_degree == 0 || (mod_degree & (mod_degree - 1)) != 0) return -1;
    if (k->conv_cw.t > 128) {
        if (k->conv_cw.t <= 256) return eval_all_t256_into_coeff_mod(k, out, coeff_start, coeff_count, mod_degree);
        dmpf64_gf2 *evals = NULL; size_t n = 0;
        if (dmpf64_big_state_eval_all(k, &evals, &n) != 0) return -1;
        size_t coeffs = coeff_count < 2*n ? coeff_count : 2*n;
        for (size_t i = 0; i < coeffs; i++) out[(coeff_start + i) & (mod_degree - 1)] ^= evals[i >> 1].v[i & 1u];
        free(evals);
        return 0;
    }
    return eval128_into_coeff_ex(k, out, coeff_start, coeff_count, mod_degree, 1, 1, 1);
}

int dmpf64_big_state_eval_all_write_coeff(const dmpf64_big_state_key *k, dmpf64_gf *out, size_t coeff_count) {
    if (k->conv_cw.t > 128) {
        dmpf64_gf2 *evals = NULL; size_t n = 0;
        if (dmpf64_big_state_eval_all(k, &evals, &n) != 0) return -1;
        size_t coeffs = coeff_count < 2*n ? coeff_count : 2*n;
        for (size_t i = 0; i < coeffs; i++) out[i] = evals[i >> 1].v[i & 1u];
        if (coeffs < coeff_count) memset(out + coeffs, 0, (coeff_count - coeffs) * sizeof(*out));
        free(evals);
        return 0;
    }
    int rc = eval128_into_coeff_ex(k, out, 0, coeff_count, 1, 0, 0, 0);
    return rc;
}

int dmpf64_big_state_eval_all_xor_coeff(const dmpf64_big_state_key *k, dmpf64_gf *out, size_t coeff_count) {
    if (k->conv_cw.t > 128) {
        dmpf64_gf2 *evals = NULL; size_t n = 0;
        if (dmpf64_big_state_eval_all(k, &evals, &n) != 0) return -1;
        size_t coeffs = coeff_count < 2*n ? coeff_count : 2*n;
        for (size_t i = 0; i < coeffs; i++) out[i] ^= evals[i >> 1].v[i & 1u];
        free(evals);
        return 0;
    }
    return eval128_into_coeff_ex(k, out, 0, coeff_count, 1, 1, 0, 0);
}

void dmpf64_big_state_key_init_empty(dmpf64_big_state_key *k){ memset(k,0,sizeof(*k)); }
void dmpf64_big_state_key_free(dmpf64_big_state_key *k){ if(!k)return; for(size_t i=0;i<k->input_bits;i++) cw_free(&k->cws[i]); free(k->cws); conv_free(&k->conv_cw); memset(k,0,sizeof(*k)); }
size_t dmpf64_big_state_input_length(const dmpf64_big_state_key *k){ return k->input_bits; }
size_t dmpf64_big_state_point_count(const dmpf64_big_state_key *k){ return k->conv_cw.t; }

int dmpf64_big_state_session_init(dmpf64_big_state_session *s, size_t t){
    memset(s,0,sizeof(*s)); s->t=t; s->nodes=ceil128(t);
    s->sign=(dmpf64_node*)calloc(s->nodes? s->nodes:1,sizeof(dmpf64_node));
    s->new_sign=(dmpf64_node*)calloc(s->nodes? s->nodes:1,sizeof(dmpf64_node));
    if(!s->sign||!s->new_sign){dmpf64_big_state_session_free(s);return -1;} return 0;
}
void dmpf64_big_state_session_free(dmpf64_big_state_session *s){ if(!s)return; free(s->sign); free(s->new_sign); memset(s,0,sizeof(*s)); }

int dmpf64_big_state_gen(dmpf64_big_state_key *k0, dmpf64_big_state_key *k1, size_t input_bits, const dmpf64_kv *inputs_in, size_t count, dmpf64_rng *rng, size_t batch_size){
    dmpf64_big_state_key_init_empty(k0); dmpf64_big_state_key_init_empty(k1);
    if(count==0 || input_bits>=63 || batch_size==0 || batch_size >= sizeof(size_t)*8) return -1;
    dmpf64_kv *inputs=(dmpf64_kv*)malloc(count*sizeof(*inputs)); if(!inputs) return -1;
    memcpy(inputs,inputs_in,count*sizeof(*inputs)); qsort(inputs,count,sizeof(*inputs),kv_cmp_node);
    for(size_t i=1;i<count;i++) if(dmpf64_node_eq(inputs[i-1].x,inputs[i].x)){free(inputs);return -1;}
    word_trie trie; if(trie_init(&trie,inputs,count,input_bits)!=0){free(inputs);return -1;}
    keygen_signs signs0,signs1,next_signs0,next_signs1;
    if(keygen_signs_init(&signs0,count)||keygen_signs_init(&signs1,count)||keygen_signs_init(&next_signs0,count)||keygen_signs_init(&next_signs1,count)){free(inputs);trie_free(&trie);return -1;}
    keygen_set_bit(&signs1,0,0,true);
    dmpf64_node *seed0=calloc(count,sizeof(dmpf64_node)),*seed1=calloc(count,sizeof(dmpf64_node)),*next_seed0=calloc(count,sizeof(dmpf64_node)),*next_seed1=calloc(count,sizeof(dmpf64_node));
    if(!seed0||!seed1||!next_seed0||!next_seed1){free(inputs);trie_free(&trie);return -1;}
    dmpf64_node root0=dmpf64_rng_node(rng), root1=dmpf64_rng_node(rng); seed0[0]=root0; seed1[0]=root1;
    signs ns0l,ns0r,ns1l,ns1r,ndeltal,ndeltar;
    if(signs_init(&ns0l,count)||signs_init(&ns0r,count)||signs_init(&ns1l,count)||signs_init(&ns1r,count)||signs_init(&ndeltal,count)||signs_init(&ndeltar,count)){free(inputs);trie_free(&trie);return -1;}
    dmpf64_big_cw *cws=(dmpf64_big_cw*)calloc(input_bits?input_bits:1,sizeof(*cws)); if(!cws){free(inputs);trie_free(&trie);return -1;}
    for(size_t depth=0; depth<input_bits; depth++){
        size_t min=count < ((size_t)1<<depth) ? count : ((size_t)1<<depth);
        unsigned __int128 *prefixes=(unsigned __int128*)malloc((min?min:1)*sizeof(*prefixes)); if(!prefixes){free(inputs);return -1;}
        size_t pn=trie_prefixes_at_depth(&trie,depth,prefixes);
        dmpf64_signs_cw sign_cw={0}; if(signs_cw_init_random(&sign_cw,count,depth,rng)!=0){free(prefixes);free(inputs);return -1;}
        dmpf64_node *cw_seeds=(dmpf64_node*)malloc((min?min:1)*sizeof(dmpf64_node)); if(!cw_seeds){free(prefixes);free(inputs);return -1;}
        size_t cw_seeds_len=0;
        for(size_t idx=0; idx<pn; idx++){
            bool hl,hr; trie_has_son(&trie,prefixes[idx],depth,&hl,&hr);
            dmpf64_node ch0[2],ch1[2]; dmpf64_double_prg(seed0[idx],ch0); dmpf64_double_prg(seed1[idx],ch1);
            dmpf64_node delta_l=dmpf64_node_xor(ch0[0],ch1[0]);
            dmpf64_node delta_r=dmpf64_node_xor(ch0[1],ch1[1]);
            signs_fill_with_seed(&ns0l,seed0[idx],0); signs_fill_with_seed(&ns0r,seed0[idx],1);
            signs_fill_with_seed(&ns1l,seed1[idx],0); signs_fill_with_seed(&ns1r,seed1[idx],1);
            signs_xor_into(&ndeltal,&ns0l,&ns1l); signs_xor_into(&ndeltar,&ns0r,&ns1r);
            cw_seeds[cw_seeds_len++] = (hl&&hr) ? dmpf64_rng_node(rng) : (hl ? delta_r : delta_l);
            signs_cw_put(&ndeltal,&sign_cw,false,idx);
            signs_cw_put(&ndeltar,&sign_cw,true,idx);
        }
        if(cw_gen(&cws[depth],depth,&trie,&sign_cw,cw_seeds,cw_seeds_len,rng,batch_size)!=0){free(cw_seeds);free(prefixes);free(inputs);return -1;}
        free(cw_seeds);
        size_t next_pos=0;
        for(size_t idx=0; idx<pn; idx++){
            bool hl,hr; trie_has_son(&trie,prefixes[idx],depth,&hl,&hr);
            dmpf64_node ch0[2],ch1[2]; dmpf64_double_prg(seed0[idx],ch0); dmpf64_double_prg(seed1[idx],ch1);
            if(hl){ signs_fill_with_seed(&ns0l,seed0[idx],0); signs_fill_with_seed(&ns1l,seed1[idx],0); }
            if(hr){ signs_fill_with_seed(&ns0r,seed0[idx],1); signs_fill_with_seed(&ns1r,seed1[idx],1); }
            dmpf64_node corr0=cw_correct_keygen(&cws[depth],&signs0,idx,min,hl,hr,&ns0l,&ns0r);
            dmpf64_node corr1=cw_correct_keygen(&cws[depth],&signs1,idx,min,hl,hr,&ns1l,&ns1r);
            if(hl){ next_seed0[next_pos]=dmpf64_node_xor(ch0[0],corr0); next_seed1[next_pos]=dmpf64_node_xor(ch1[0],corr1); keygen_set_signs(&next_signs0,next_pos,&ns0l); keygen_set_signs(&next_signs1,next_pos,&ns1l); next_pos++; }
            if(hr){ next_seed0[next_pos]=dmpf64_node_xor(ch0[1],corr0); next_seed1[next_pos]=dmpf64_node_xor(ch1[1],corr1); keygen_set_signs(&next_signs0,next_pos,&ns0r); keygen_set_signs(&next_signs1,next_pos,&ns1r); next_pos++; }
        }
        free(prefixes);
        keygen_signs tmpks=signs0; signs0=next_signs0; next_signs0=tmpks;
        tmpks=signs1; signs1=next_signs1; next_signs1=tmpks;
        dmpf64_node *tmp=seed0; seed0=next_seed0; next_seed0=tmp; tmp=seed1; seed1=next_seed1; next_seed1=tmp;
    }
    dmpf64_conv_cw conv={0}; conv.t=count; conv.batch_size=batch_size; conv.o=(dmpf64_gf2*)malloc((count?count:1)*sizeof(dmpf64_gf2)); if(!conv.o){free(inputs);return -1;}
    for(size_t i=0;i<count;i++){
        dmpf64_gf2 o=dmpf64_gf2_add(dmpf64_gf2_add(dmpf64_gf2_from_node(seed0[i]),dmpf64_gf2_from_node(seed1[i])),inputs[i].y);
        /* Negation/subtraction are identity over GF(2^64)^2. */
        (void)keygen_get_bit(&signs0,i,i);
        conv.o[i]=o;
    }
    k0->sign=false; k0->root=root0; k0->cws=cws; k0->input_bits=input_bits; k0->conv_cw=conv;
    k1->sign=true; k1->root=root1; k1->input_bits=input_bits;
    k1->cws=(dmpf64_big_cw*)calloc(input_bits?input_bits:1,sizeof(*k1->cws)); if(!k1->cws){free(inputs);return -1;}
    for(size_t i=0;i<input_bits;i++) if(cw_clone(&k1->cws[i],&cws[i])!=0){free(inputs);return -1;}
    if(conv_clone(&k1->conv_cw,&conv)!=0){free(inputs);return -1;}
    signs_free(&ns0l); signs_free(&ns0r); signs_free(&ns1l); signs_free(&ns1r); signs_free(&ndeltal); signs_free(&ndeltar);
    keygen_signs_free(&signs0); keygen_signs_free(&signs1); keygen_signs_free(&next_signs0); keygen_signs_free(&next_signs1);
    free(seed0); free(seed1); free(next_seed0); free(next_seed1); trie_free(&trie); free(inputs); return 0;
}

dmpf64_gf2 dmpf64_big_state_eval_with_session(const dmpf64_big_state_key *k, dmpf64_node x, dmpf64_big_state_session *sess){
    signs sign={sess->sign,sess->t,sess->nodes}; signs new_sign={sess->new_sign,sess->t,sess->nodes};
    dmpf64_node seed=k->root; signs_zero(&sign); signs_set_bit(&sign,0,k->sign);
    signs *sign_ref=&sign, *next_ref=&new_sign;
    for(size_t depth=0; depth<k->input_bits; depth++){
        dmpf64_node ch[2]; dmpf64_double_prg(seed,ch);
        bool bit=dmpf64_node_get_bit_msb(x,depth);
        signs_fill_with_seed(next_ref,seed,bit?1:0);
        dmpf64_node corr=cw_correct_single(&k->cws[depth],sign_ref,bit,next_ref);
        seed=dmpf64_node_xor(ch[bit?1:0],corr);
        signs *tmp=sign_ref; sign_ref=next_ref; next_ref=tmp;
    }
    dmpf64_gf2 out=dmpf64_gf2_add(dmpf64_gf2_from_node(seed),conv_correct(&k->conv_cw,sign_ref));
    return out;
}
dmpf64_gf2 dmpf64_big_state_eval(const dmpf64_big_state_key *k, dmpf64_node x){
    dmpf64_big_state_session s; if(dmpf64_big_state_session_init(&s,k->conv_cw.t)!=0) return dmpf64_gf2_zero();
    dmpf64_gf2 out=dmpf64_big_state_eval_with_session(k,x,&s); dmpf64_big_state_session_free(&s); return out;
}
int dmpf64_big_state_eval_all(const dmpf64_big_state_key *k, dmpf64_gf2 **outp, size_t *out_len){
    if (k->conv_cw.t <= 128) return eval_all_small_t(k, outp, out_len);
    if (k->conv_cw.t <= 256) return eval_all_t256(k, outp, out_len);
    size_t input_len=k->input_bits;
    size_t n=(size_t)1 << input_len;
    dmpf64_gf2 *out=(dmpf64_gf2*)malloc((n?n:1)*sizeof(*out)); if(!out) return -1;
    signs cur_sign={0}; if(signs_init(&cur_sign,k->conv_cw.t)!=0){free(out);return -1;}
    eval_all_state st={0}, next={0};
    if(eval_state_init(&st,k->conv_cw.t,input_len)!=0 || eval_state_init(&next,k->conv_cw.t,input_len)!=0){signs_free(&cur_sign);free(out);return -1;}
    dmpf64_node *seeds=(dmpf64_node*)malloc((n?n:1)*sizeof(dmpf64_node));
    dmpf64_node *next_seeds=(dmpf64_node*)malloc((n?n:1)*sizeof(dmpf64_node));
    if(!seeds||!next_seeds){free(seeds);free(next_seeds);eval_state_free(&st);eval_state_free(&next);signs_free(&cur_sign);free(out);return -1;}
    seeds[0]=k->root;
    signs_zero(&cur_sign); signs_set_bit(&cur_sign,0,k->sign); eval_state_put_sign(&st,0,&cur_sign);
    for(size_t depth=0; depth<input_len; depth++){
        size_t cur_n=(size_t)1<<depth;
        dmpf64_double_prg_many(seeds,cur_n,next_seeds);
        int rc=eval_state_fill_from_seeds(&next,seeds,cur_n); if(rc!=0){free(seeds);free(next_seeds);eval_state_free(&st);eval_state_free(&next);signs_free(&cur_sign);free(out);return rc;}
        dmpf64_node *tmp=seeds; seeds=next_seeds; next_seeds=tmp;
        const dmpf64_big_cw *cw=&k->cws[depth]; size_t min=cw->seed_count;
        for(size_t path=0; path<cur_n; path++){
            dmpf64_node *left=eval_state_after_expand(&next,path,false);
            dmpf64_node *right=eval_state_after_expand(&next,path,true);
            dmpf64_node corr=cw_correct_chunks(cw,eval_state_sign(&st,path),min,true,true,left,right);
            dmpf64_node_xor_inplace(&seeds[2*path],corr);
            dmpf64_node_xor_inplace(&seeds[2*path+1],corr);
        }
        eval_all_state tmpst=st; st=next; next=tmpst;
    }
    for(size_t i=0;i<n;i++){
        eval_state_get_sign(&st,i,&cur_sign);
        out[i]=dmpf64_gf2_add(dmpf64_gf2_from_node(seeds[i]),conv_correct(&k->conv_cw,&cur_sign));
    }
    free(seeds); free(next_seeds); eval_state_free(&st); eval_state_free(&next); signs_free(&cur_sign); *outp=out; *out_len=n; return 0;
}
