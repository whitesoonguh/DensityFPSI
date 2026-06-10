#include "dmpf64/dmpf_any.h"
#include <string.h>

const char *dmpf64_backend_name(dmpf64_backend b){
    switch(b){
        case DMPF64_BACKEND_DPF: return "dpf";
        case DMPF64_BACKEND_BIG_STATE: return "bigstate";
        case DMPF64_BACKEND_BATCH_CODE: return "batchcode-not-ported";
        default: return "unknown";
    }
}
static dmpf64_backend dmpf64_force_backend(dmpf64_backend b) {
    (void)b;
    return DMPF64_BACKEND_BIG_STATE;
}

void dmpf64_any_key_init_empty(dmpf64_any_key *k, dmpf64_backend b){
    memset(k,0,sizeof(*k));
    k->backend=dmpf64_force_backend(b);
}
void dmpf64_any_key_free(dmpf64_any_key *k){
    if(!k) return;
    if(k->backend==DMPF64_BACKEND_DPF) dmpf64_dpf_dmpf_key_free(&k->u.dpf);
    else if(k->backend==DMPF64_BACKEND_BIG_STATE) dmpf64_big_state_key_free(&k->u.big);
    memset(k,0,sizeof(*k));
}
int dmpf64_any_gen(dmpf64_any_key *k0, dmpf64_any_key *k1, dmpf64_backend backend, size_t input_bits, const dmpf64_kv *inputs, size_t count, dmpf64_rng *rng, size_t batch_size){
    backend = dmpf64_force_backend(backend);
    dmpf64_any_key_init_empty(k0,backend);
    dmpf64_any_key_init_empty(k1,backend);
    return dmpf64_big_state_gen(&k0->u.big,&k1->u.big,input_bits,inputs,count,rng,batch_size?batch_size:8);
}
int dmpf64_any_eval_all(const dmpf64_any_key *k, dmpf64_gf2 **out, size_t *out_len){
    if(k->backend==DMPF64_BACKEND_DPF) return dmpf64_dpf_dmpf_eval_all(&k->u.dpf,out,out_len);
    if(k->backend==DMPF64_BACKEND_BIG_STATE) return dmpf64_big_state_eval_all(&k->u.big,out,out_len);
    return -2;
}
size_t dmpf64_any_input_length(const dmpf64_any_key *k){
    if(k->backend==DMPF64_BACKEND_DPF) return dmpf64_dpf_dmpf_input_length(&k->u.dpf);
    if(k->backend==DMPF64_BACKEND_BIG_STATE) return dmpf64_big_state_input_length(&k->u.big);
    return 0;
}
size_t dmpf64_any_point_count(const dmpf64_any_key *k){
    if(k->backend==DMPF64_BACKEND_DPF) return dmpf64_dpf_dmpf_point_count(&k->u.dpf);
    if(k->backend==DMPF64_BACKEND_BIG_STATE) return dmpf64_big_state_point_count(&k->u.big);
    return 0;
}
