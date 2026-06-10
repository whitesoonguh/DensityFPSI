#include "dmpf64/prg.h"
#include <string.h>
#include <immintrin.h>

#if defined(__AES__)
static __m128i aes_rk[11];
static int aes_ready = 0;
static inline __m128i aes_keygen_assist(__m128i key, const int rcon) {
    __m128i tmp = _mm_aeskeygenassist_si128(key, rcon);
    tmp = _mm_shuffle_epi32(tmp, 0xff);
    __m128i x = key;
    x = _mm_xor_si128(x, _mm_slli_si128(x, 4));
    x = _mm_xor_si128(x, _mm_slli_si128(x, 4));
    x = _mm_xor_si128(x, _mm_slli_si128(x, 4));
    return _mm_xor_si128(x, tmp);
}
void dmpf64_prg_init(void) {
    if (aes_ready) return;
    const uint8_t key_bytes[16] = {3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2};
    aes_rk[0] = _mm_loadu_si128((const __m128i*)key_bytes);
    aes_rk[1] = aes_keygen_assist(aes_rk[0], 0x01);
    aes_rk[2] = aes_keygen_assist(aes_rk[1], 0x02);
    aes_rk[3] = aes_keygen_assist(aes_rk[2], 0x04);
    aes_rk[4] = aes_keygen_assist(aes_rk[3], 0x08);
    aes_rk[5] = aes_keygen_assist(aes_rk[4], 0x10);
    aes_rk[6] = aes_keygen_assist(aes_rk[5], 0x20);
    aes_rk[7] = aes_keygen_assist(aes_rk[6], 0x40);
    aes_rk[8] = aes_keygen_assist(aes_rk[7], 0x80);
    aes_rk[9] = aes_keygen_assist(aes_rk[8], 0x1B);
    aes_rk[10] = aes_keygen_assist(aes_rk[9], 0x36);
    aes_ready = 1;
}
static inline __m128i aes_encrypt_block(__m128i x) {
    x = _mm_xor_si128(x, aes_rk[0]);
    for (int i=1;i<10;i++) x = _mm_aesenc_si128(x, aes_rk[i]);
    return _mm_aesenclast_si128(x, aes_rk[10]);
}

#if defined(__VAES__) && defined(__AVX512F__)
static inline __m512i aes_encrypt4_blocks(__m512i x) {
    x = _mm512_xor_si512(x, _mm512_broadcast_i64x2(aes_rk[0]));
    for (int i = 1; i < 10; i++) x = _mm512_aesenc_epi128(x, _mm512_broadcast_i64x2(aes_rk[i]));
    return _mm512_aesenclast_epi128(x, _mm512_broadcast_i64x2(aes_rk[10]));
}
#endif
static inline __m128i node_to_m128(dmpf64_node n) { return _mm_set_epi64x((long long)n.hi, (long long)n.lo); }
static inline dmpf64_node m128_to_node(__m128i x) { dmpf64_node n; _mm_storeu_si128((__m128i*)&n, x); return n; }
#else
void dmpf64_prg_init(void) {}
static uint64_t smix(uint64_t x){x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31);} 
#endif

const char *dmpf64_prg_backend(void) {
#if defined(__AES__) && defined(__VAES__) && defined(__AVX512F__)
    return "vaes512-fixed-key";
#elif defined(__AES__)
    return "aesni-fixed-key";
#else
    return "splitmix-fallback";
#endif
}

void dmpf64_double_prg(dmpf64_node in, dmpf64_node out[2]) {
#if defined(__AES__)
    dmpf64_prg_init();
    dmpf64_node b0 = in, b1 = in;
    ((uint8_t*)&b0)[0] ^= 0;
    ((uint8_t*)&b1)[0] ^= 1;
    __m128i x0 = aes_encrypt_block(node_to_m128(b0));
    __m128i x1 = aes_encrypt_block(node_to_m128(b1));
    dmpf64_node o0 = m128_to_node(x0);
    dmpf64_node o1 = m128_to_node(x1);
    dmpf64_node_xor_inplace(&o0, in);
    dmpf64_node_xor_inplace(&o1, in);
    ((uint8_t*)&o0)[0] ^= 0;
    ((uint8_t*)&o1)[0] ^= 1;
    out[0] = o0; out[1] = o1;
#else
    uint64_t a = smix(in.lo ^ in.hi); uint64_t b = smix(a^1);
    out[0]=(dmpf64_node){a,smix(a)}; out[1]=(dmpf64_node){b^1,smix(b)};
#endif
}

void dmpf64_double_prg_many(const dmpf64_node *in, size_t n, dmpf64_node *out) {
    for (size_t i=0;i<n;i++) dmpf64_double_prg(in[i], &out[2*i]);
}

void dmpf64_many_prg(dmpf64_node in, uint16_t first_child, size_t count, dmpf64_node *out) {
#if defined(__AES__)
    dmpf64_prg_init();
    for (size_t i=0;i<count;i++) {
        dmpf64_node b = in;
        uint16_t c = (uint16_t)(first_child + i);
        ((uint8_t*)&b)[0] ^= (uint8_t)(c >> 8);
        ((uint8_t*)&b)[1] ^= (uint8_t)c;
        __m128i x = aes_encrypt_block(node_to_m128(b));
        dmpf64_node o = m128_to_node(x);
        ((uint8_t*)&o)[0] ^= (uint8_t)(c >> 8);
        ((uint8_t*)&o)[1] ^= (uint8_t)c;
        dmpf64_node_xor_inplace(&o, in);
        out[i] = o;
    }
#else
    for(size_t i=0;i<count;i++){uint64_t x=smix(in.lo ^ ((uint64_t)(first_child+i)<<32) ^ in.hi); out[i]=(dmpf64_node){x,smix(x)};}
#endif
}

void dmpf64_many_many_prg(const dmpf64_node *in, size_t n, uint16_t first_child, size_t count_per_input, dmpf64_node *out) {
    for (size_t i=0;i<n;i++) dmpf64_many_prg(in[i], first_child, count_per_input, &out[i*count_per_input]);
}


void dmpf64_bigstate_prg_many(const dmpf64_node *in, size_t n, dmpf64_node *seed_out, dmpf64_node *sign_out) {
#if defined(__AES__)
    dmpf64_prg_init();
#if defined(__VAES__) && defined(__AVX512F__)
    const __m512i tag1 = _mm512_set_epi64(0,1,0,1,0,1,0,1);
    const __m512i tag2 = _mm512_set_epi64(0,((uint64_t)2<<8),0,((uint64_t)2<<8),0,((uint64_t)2<<8),0,((uint64_t)2<<8));
    const __m512i tag3 = _mm512_set_epi64(0,((uint64_t)3<<8),0,((uint64_t)3<<8),0,((uint64_t)3<<8),0,((uint64_t)3<<8));
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m512i vin = _mm512_loadu_si512((const void *)(in + i));
        __m512i x0 = aes_encrypt4_blocks(vin);
        __m512i x1 = aes_encrypt4_blocks(_mm512_xor_si512(vin, tag1));
        __m512i x2 = aes_encrypt4_blocks(_mm512_xor_si512(vin, tag2));
        __m512i x3 = aes_encrypt4_blocks(_mm512_xor_si512(vin, tag3));
        x0 = _mm512_xor_si512(x0, vin);
        x1 = _mm512_xor_si512(_mm512_xor_si512(x1, vin), tag1);
        x2 = _mm512_xor_si512(_mm512_xor_si512(x2, vin), tag2);
        x3 = _mm512_xor_si512(_mm512_xor_si512(x3, vin), tag3);
        dmpf64_node o0[4], o1[4], o2[4], o3[4];
        _mm512_storeu_si512((void*)o0, x0);
        _mm512_storeu_si512((void*)o1, x1);
        _mm512_storeu_si512((void*)o2, x2);
        _mm512_storeu_si512((void*)o3, x3);
        for (int k = 0; k < 4; k++) {
            size_t j = i + (size_t)k;
            seed_out[2*j] = o0[k];
            seed_out[2*j+1] = o1[k];
            sign_out[2*j] = o2[k];
            sign_out[2*j+1] = o3[k];
        }
    }
    for (; i < n; i++) {
        dmpf64_node blk[4] = { in[i], in[i], in[i], in[i] };
        ((uint8_t*)&blk[1])[0] ^= 1;
        ((uint8_t*)&blk[2])[1] ^= 2;
        ((uint8_t*)&blk[3])[1] ^= 3;
        __m512i x = _mm512_loadu_si512((const void*)blk);
        x = aes_encrypt4_blocks(x);
        dmpf64_node out[4];
        _mm512_storeu_si512((void*)out, x);
        for (int j = 0; j < 4; j++) dmpf64_node_xor_inplace(&out[j], in[i]);
        ((uint8_t*)&out[1])[0] ^= 1;
        ((uint8_t*)&out[2])[1] ^= 2;
        ((uint8_t*)&out[3])[1] ^= 3;
        seed_out[2*i] = out[0];
        seed_out[2*i+1] = out[1];
        sign_out[2*i] = out[2];
        sign_out[2*i+1] = out[3];
    }
#else
    for (size_t i = 0; i < n; i++) {
        dmpf64_node b0 = in[i], b1 = in[i], b2 = in[i], b3 = in[i];
        ((uint8_t*)&b1)[0] ^= 1;
        ((uint8_t*)&b2)[1] ^= 2;
        ((uint8_t*)&b3)[1] ^= 3;
        __m128i x0 = aes_encrypt_block(node_to_m128(b0));
        __m128i x1 = aes_encrypt_block(node_to_m128(b1));
        __m128i x2 = aes_encrypt_block(node_to_m128(b2));
        __m128i x3 = aes_encrypt_block(node_to_m128(b3));
        dmpf64_node o0 = m128_to_node(x0);
        dmpf64_node o1 = m128_to_node(x1);
        dmpf64_node o2 = m128_to_node(x2);
        dmpf64_node o3 = m128_to_node(x3);
        dmpf64_node_xor_inplace(&o0, in[i]);
        dmpf64_node_xor_inplace(&o1, in[i]);
        dmpf64_node_xor_inplace(&o2, in[i]);
        dmpf64_node_xor_inplace(&o3, in[i]);
        ((uint8_t*)&o1)[0] ^= 1;
        ((uint8_t*)&o2)[1] ^= 2;
        ((uint8_t*)&o3)[1] ^= 3;
        seed_out[2*i] = o0;
        seed_out[2*i+1] = o1;
        sign_out[2*i] = o2;
        sign_out[2*i+1] = o3;
    }
#endif
#else
    dmpf64_double_prg_many(in, n, seed_out);
    dmpf64_many_many_prg(in, n, 2, 2, sign_out);
#endif
}

