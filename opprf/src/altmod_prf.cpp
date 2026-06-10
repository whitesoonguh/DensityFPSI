#include "sooprf/altmod_prf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sooprf {
namespace {

inline std::uint8_t byte_at(Block128 b, std::size_t i) noexcept {
    if (i < 8) return static_cast<std::uint8_t>(b.lo >> (8 * i));
    return static_cast<std::uint8_t>(b.hi >> (8 * (i - 8)));
}

inline std::uint8_t get_key_bit(const AltModKey& k, std::size_t i) noexcept {
    return bit(k[i / 128], i % 128) ? 1 : 0;
}

inline void set_lane(Block128& x, std::size_t lane) noexcept {
    if (lane < 64) x.lo |= (std::uint64_t{1} << lane);
    else x.hi |= (std::uint64_t{1} << (lane - 64));
}

inline void set_output_bit(Block128& x, std::size_t bit_idx) noexcept {
    if (bit_idx < 64) x.lo |= (std::uint64_t{1} << bit_idx);
    else x.hi |= (std::uint64_t{1} << (bit_idx - 64));
}

inline std::uint8_t mod3_add_scalar(std::uint8_t a, std::uint8_t b) noexcept {
    auto s = static_cast<unsigned>(a) + static_cast<unsigned>(b);
    return static_cast<std::uint8_t>(s >= 3 ? s - 3 : s);
}

inline void mod3_add_block(Block128& dst_msb, Block128& dst_lsb,
                           Block128 src_msb, Block128 src_lsb) noexcept {
    const Block128 a0 = dst_lsb;
    const Block128 a1 = dst_msb;
    const Block128 b0 = src_lsb;
    const Block128 b1 = src_msb;
    const Block128 a10 = a1 ^ a0;
    dst_msb = ~(b0 ^ a0) & (a10 ^ b1);
    dst_lsb = ~(a1 ^ b1) & (a10 ^ b0);
}

void fisher_yates(std::vector<std::uint32_t>& p, SplitMix64& prng) {
    for (std::size_t i = p.size(); i > 1; --i) {
        const auto j = static_cast<std::size_t>(prng.next_u64() % i);
        std::swap(p[i - 1], p[j]);
    }
}

inline void transpose64x64(std::uint64_t a[64]) noexcept {
    // In-place transpose of a 64x64 bit matrix.  Initially a[row] contains
    // columns in its bit positions.  On return a[col] contains rows in its
    // bit positions.  This removes the old per-set-bit scatter loops from the
    // PRF batch path.
    static constexpr std::uint64_t masks[6] = {
        0x00000000FFFFFFFFULL, 0x0000FFFF0000FFFFULL,
        0x00FF00FF00FF00FFULL, 0x0F0F0F0F0F0F0F0FULL,
        0x3333333333333333ULL, 0x5555555555555555ULL
    };
    static constexpr unsigned shifts[6] = {32, 16, 8, 4, 2, 1};
    for (std::size_t s = 0; s < 6; ++s) {
        const unsigned j = shifts[s];
        const std::uint64_t m = masks[s];
        for (unsigned k = 0; k < 64; k = (k + j + 1) & ~j) {
            const std::uint64_t t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= (t << j);
            a[k + j] ^= t;
        }
    }
}

inline void store_transposed_expansion_half(const std::array<AltModKey, 128>& expanded,
                                            std::size_t valid_lanes,
                                            std::size_t block_idx,
                                            bool high_word,
                                            std::uint64_t key_word,
                                            std::size_t cols,
                                            std::size_t col,
                                            Block128* out) noexcept {
    alignas(64) std::uint64_t tmp[64];
    const std::size_t row_base = block_idx * 128 + (high_word ? 64 : 0);

    for (std::size_t base = 0; base < 128; base += 64) {
        for (std::size_t lane = 0; lane < 64; ++lane) {
            const std::size_t idx = base + lane;
            std::uint64_t w = 0;
            if (idx < valid_lanes) {
                const Block128 b = expanded[idx][block_idx];
                w = (high_word ? b.hi : b.lo) & key_word;
            }
            tmp[lane] = w;
        }
        transpose64x64(tmp);
        for (std::size_t bit_idx = 0; bit_idx < 64; ++bit_idx) {
            Block128& dst = out[(row_base + bit_idx) * cols + col];
            if (base == 0) dst.lo = tmp[bit_idx];
            else dst.hi = tmp[bit_idx];
        }
    }
}

inline void transpose_bitsliced_64_rows_lo(const Block128* rows,
                                           std::size_t cols,
                                           std::size_t col,
                                           bool lane_hi,
                                           std::uint64_t out[64]) noexcept {
    for (std::size_t i = 0; i < 64; ++i) {
        const Block128 b = rows[i * cols + col];
        out[i] = lane_hi ? b.hi : b.lo;
    }
    transpose64x64(out);
}

} // namespace

bool compiled_with_avx512() noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    return true;
#else
    return false;
#endif
}

void AltModPrfWorkspace::resize_for(std::size_t input_count) {
    const std::size_t need_cols = (input_count + 127) / 128;
    const std::size_t ek_size = AltModPrfParams::KeyBits * need_cols;
    const std::size_t u_size = AltModPrfParams::MidBits * need_cols;

    if (need_cols != cols || ek_lsb.size() != ek_size) {
        cols = need_cols;
        ek_lsb.resize(ek_size);
        ek_msb.resize(ek_size);
        u_lsb.resize(u_size);
    }

    // ek_lsb and u_lsb are fully overwritten by the batch transpose/F3 output
    // stages.  Only the high bit-plane of the F3 input must be zeroed because
    // the binary input trits are represented as 0=(0,0), 1=(0,1).
    std::fill(ek_msb.begin(), ek_msb.end(), ZeroBlock);
}

void BinaryLinearCode128::init(std::uint64_t seed) {
    SplitMix64 prng(seed);
    for (auto& r : rows_) r = prng.next_block();

    for (std::size_t byte = 0; byte < 16; ++byte) {
        auto* table = sub_.data() + byte * 256;
        table[0] = ZeroBlock;
        for (std::size_t val = 1; val < 256; ++val) {
            Block128 acc = ZeroBlock;
            for (std::size_t b = 0; b < 8; ++b) {
                if ((val >> b) & 1u) acc ^= rows_[byte * 8 + b];
            }
            table[val] = acc;
        }
    }
}

Block128 BinaryLinearCode128::encode(Block128 x) const noexcept {
    // independent accumulator-permutation codes for a 128->128 binary code.
    // This removes the long XOR dependency chain in the final B encode.
    const Block128* T0 = sub_.data() + 0 * 256;
    const Block128* T1 = sub_.data() + 1 * 256;
    const Block128* T2 = sub_.data() + 2 * 256;
    const Block128* T3 = sub_.data() + 3 * 256;
    const Block128* T4 = sub_.data() + 4 * 256;
    const Block128* T5 = sub_.data() + 5 * 256;
    const Block128* T6 = sub_.data() + 6 * 256;
    const Block128* T7 = sub_.data() + 7 * 256;
    Block128 c0 = T0[byte_at(x, 0)];
    Block128 c1 = T1[byte_at(x, 1)];
    Block128 c2 = T2[byte_at(x, 2)];
    Block128 c3 = T3[byte_at(x, 3)];
    Block128 c4 = T4[byte_at(x, 4)];
    Block128 c5 = T5[byte_at(x, 5)];
    Block128 c6 = T6[byte_at(x, 6)];
    Block128 c7 = T7[byte_at(x, 7)];
    T0 += 8 * 256; T1 += 8 * 256; T2 += 8 * 256; T3 += 8 * 256;
    T4 += 8 * 256; T5 += 8 * 256; T6 += 8 * 256; T7 += 8 * 256;
    c0 ^= T0[byte_at(x, 8)];
    c1 ^= T1[byte_at(x, 9)];
    c2 ^= T2[byte_at(x, 10)];
    c3 ^= T3[byte_at(x, 11)];
    c4 ^= T4[byte_at(x, 12)];
    c5 ^= T5[byte_at(x, 13)];
    c6 ^= T6[byte_at(x, 14)];
    c7 ^= T7[byte_at(x, 15)];
    c0 ^= c4; c1 ^= c5; c2 ^= c6; c3 ^= c7;
    c0 ^= c2; c1 ^= c3;
    return c0 ^ c1;
}

void F3AccPermCode::init(std::size_t k, std::size_t n, std::uint64_t seed, std::size_t rounds) {
    if (k <= n) throw std::invalid_argument("F3AccPermCode requires k > n");
    k_ = k;
    n_ = n;
    perms_.assign(rounds, {});
    SplitMix64 prng(seed);
    for (auto& p : perms_) {
        p.resize(k_);
        for (std::size_t i = 0; i < k_; ++i) p[i] = static_cast<std::uint32_t>(i);
        fisher_yates(p, prng);
    }
}

void F3AccPermCode::encode_scalar(std::span<std::uint8_t> inout, std::span<std::uint8_t> out) const {
    if (inout.size() != k_ || out.size() != n_) throw std::invalid_argument("bad F3AccPermCode dimensions");
    for (std::size_t i = 1; i < k_; ++i) inout[i] = mod3_add_scalar(inout[i], inout[i - 1]);
    for (std::size_t r = 0; r + 1 < perms_.size(); ++r) {
        const auto& p = perms_[r];
        for (std::size_t i = 1; i < k_; ++i) {
            const auto pi = p[i];
            const auto pim1 = p[i - 1];
            inout[pi] = mod3_add_scalar(inout[pi], inout[pim1]);
        }
    }
    const auto& last = perms_.back();
    for (std::size_t i = 0; i < n_; ++i) out[i] = inout[last[i]];
}

void mod3_add_inplace(Block128* dst_msb, Block128* dst_lsb,
                      const Block128* src_msb, const Block128* src_lsb,
                      std::size_t blocks) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    std::size_t i = 0;
    for (; i + 4 <= blocks; i += 4) {
        auto a0 = _mm512_loadu_si512(reinterpret_cast<const void*>(dst_lsb + i));
        auto a1 = _mm512_loadu_si512(reinterpret_cast<const void*>(dst_msb + i));
        auto b0 = _mm512_loadu_si512(reinterpret_cast<const void*>(src_lsb + i));
        auto b1 = _mm512_loadu_si512(reinterpret_cast<const void*>(src_msb + i));

        auto a10 = _mm512_xor_si512(a1, a0);
        auto r1 = _mm512_andnot_si512(_mm512_xor_si512(b0, a0), _mm512_xor_si512(a10, b1));
        auto r0 = _mm512_andnot_si512(_mm512_xor_si512(a1, b1), _mm512_xor_si512(a10, b0));

        _mm512_storeu_si512(reinterpret_cast<void*>(dst_lsb + i), r0);
        _mm512_storeu_si512(reinterpret_cast<void*>(dst_msb + i), r1);
    }
    for (; i < blocks; ++i) {
#else
    for (std::size_t i = 0; i < blocks; ++i) {
#endif
        const Block128 a0 = dst_lsb[i];
        const Block128 a1 = dst_msb[i];
        const Block128 b0 = src_lsb[i];
        const Block128 b1 = src_msb[i];
        const Block128 a10 = a1 ^ a0;
        const Block128 r1 = ~(b0 ^ a0) & (a10 ^ b1);
        const Block128 r0 = ~(a1 ^ b1) & (a10 ^ b0);
        dst_lsb[i] = r0;
        dst_msb[i] = r1;
    }
}

void F3AccPermCode::encode_bitsliced(Block128* msb, Block128* lsb,
                                     Block128* out_msb, Block128* out_lsb,
                                     std::size_t cols) const {
    if (!k_ || !n_) throw std::runtime_error("F3AccPermCode not initialized");
    for (std::size_t i = 1; i < k_; ++i) {
        mod3_add_inplace(msb + i * cols, lsb + i * cols,
                         msb + (i - 1) * cols, lsb + (i - 1) * cols, cols);
    }
    for (std::size_t r = 0; r + 1 < perms_.size(); ++r) {
        const auto& p = perms_[r];
        for (std::size_t i = 1; i < k_; ++i) {
            const auto pi = static_cast<std::size_t>(p[i]);
            const auto pim1 = static_cast<std::size_t>(p[i - 1]);
            mod3_add_inplace(msb + pi * cols, lsb + pi * cols,
                             msb + pim1 * cols, lsb + pim1 * cols, cols);
        }
    }
    const auto& last = perms_.back();
    for (std::size_t i = 0; i < n_; ++i) {
        const auto pi = static_cast<std::size_t>(last[i]);
        std::memcpy(out_lsb + i * cols, lsb + pi * cols, cols * sizeof(Block128));
        std::memcpy(out_msb + i * cols, msb + pi * cols, cols * sizeof(Block128));
    }
}

void F3AccPermCode::encode_bitsliced_lsb(Block128* msb, Block128* lsb,
                                         Block128* out_lsb,
                                         std::size_t cols) const {
    if (!k_ || !n_) throw std::runtime_error("F3AccPermCode not initialized");
    for (std::size_t i = 1; i < k_; ++i) {
        mod3_add_inplace(msb + i * cols, lsb + i * cols,
                         msb + (i - 1) * cols, lsb + (i - 1) * cols, cols);
    }
    for (std::size_t r = 0; r + 1 < perms_.size(); ++r) {
        const auto& p = perms_[r];
        for (std::size_t i = 1; i < k_; ++i) {
            const auto pi = static_cast<std::size_t>(p[i]);
            const auto pim1 = static_cast<std::size_t>(p[i - 1]);
            mod3_add_inplace(msb + pi * cols, lsb + pi * cols,
                             msb + pim1 * cols, lsb + pim1 * cols, cols);
        }
    }
    const auto& last = perms_.back();
    for (std::size_t i = 0; i < n_; ++i) {
        const auto pi = static_cast<std::size_t>(last[i]);
        std::memcpy(out_lsb + i * cols, lsb + pi * cols, cols * sizeof(Block128));
    }
}


AltModPrf::AltModPrf() {
    // Seeds fix the public G/B/A objects while avoiding cryptoTools dependency in the arithmetic core.
    b_code_.init(mix_seed(234532451234512134ULL, 214512345123455437ULL));
    g_codes_[0].init(mix_seed(45245327478243784ULL, 28874799389237822ULL) ^ 0x01ULL);
    g_codes_[1].init(mix_seed(45245327478243784ULL, 28874799389237822ULL) ^ 0x02ULL);
    g_codes_[2].init(mix_seed(45245327478243784ULL, 28874799389237822ULL) ^ 0x03ULL);
    a_code_.init(AltModPrfParams::KeyBits, AltModPrfParams::MidBits,
                 0x414c544d4f445052ULL, AltModPrfParams::AccPermRounds);
    rebuild_fast8();
}

AltModPrf::AltModPrf(const AltModKey& key) : AltModPrf() {
    set_key(key);
}

void AltModPrf::set_key(const AltModKey& key) {
    key_ = key;
    rebuild_fast8();
}


AltModKey AltModPrf::random_key(std::uint64_t seed) {
    SplitMix64 prng(seed);
    AltModKey k{};
    for (auto& b : k) b = prng.next_block();
    return k;
}

AltModKey AltModPrf::expand_input_linear(Block128 x) const noexcept {
    AltModKey e{};
    e[0] = x;
    e[1] = g_codes_[0].encode(x);
    e[2] = g_codes_[1].encode(x);
    e[3] = g_codes_[2].encode(x);
    return e;
}

void AltModPrf::rebuild_fast8() {
    std::array<std::uint8_t, AltModPrfParams::KeyBits> state{};
    std::array<std::uint8_t, AltModPrfParams::MidBits> mid{};

    for (std::size_t chunk = 0; chunk < 16; ++chunk) {
        for (std::uint32_t val = 0; val < 256; ++val) {
            Block128 x{};
            const std::size_t bit_base = chunk * 8;
            if (bit_base < 64) x.lo = static_cast<std::uint64_t>(val) << bit_base;
            else x.hi = static_cast<std::uint64_t>(val) << (bit_base - 64);

            const auto e = expand_input_linear(x);
            state.fill(0);
            mid.fill(0);
            for (std::size_t i = 0; i < AltModPrfParams::KeyBits; ++i) {
                state[i] = (bit(e[i / 128], i % 128) && get_key_bit(key_, i)) ? 1 : 0;
            }
            a_code_.encode_scalar(state, mid);

            FastChunkEntry ent{};
            for (std::size_t i = 0; i < 128; ++i) {
                if (mid[i] == 1) set_bit(ent.lsb0, i);
                else if (mid[i] == 2) set_bit(ent.msb0, i);
                if (mid[128 + i] == 1) set_bit(ent.lsb1, i);
                else if (mid[128 + i] == 2) set_bit(ent.msb1, i);
            }
            fast8_[chunk][val] = ent;
        }
    }
    fast_ready_ = true;
}

Block128 AltModPrf::eval_fast8(Block128 x) const noexcept {
    Block128 msb0{}, lsb0{}, msb1{}, lsb1{};
#define SOOPRF_ADD8(CH, IDX) do { \
        const auto& ent = fast8_[(CH)][static_cast<std::uint8_t>(IDX)]; \
        mod3_add_block(msb0, lsb0, ent.msb0, ent.lsb0); \
        mod3_add_block(msb1, lsb1, ent.msb1, ent.lsb1); \
    } while (0)
    const std::uint64_t lo = x.lo;
    const std::uint64_t hi = x.hi;
    SOOPRF_ADD8(0,  lo);
    SOOPRF_ADD8(1,  lo >> 8);
    SOOPRF_ADD8(2,  lo >> 16);
    SOOPRF_ADD8(3,  lo >> 24);
    SOOPRF_ADD8(4,  lo >> 32);
    SOOPRF_ADD8(5,  lo >> 40);
    SOOPRF_ADD8(6,  lo >> 48);
    SOOPRF_ADD8(7,  lo >> 56);
    SOOPRF_ADD8(8,  hi);
    SOOPRF_ADD8(9,  hi >> 8);
    SOOPRF_ADD8(10, hi >> 16);
    SOOPRF_ADD8(11, hi >> 24);
    SOOPRF_ADD8(12, hi >> 32);
    SOOPRF_ADD8(13, hi >> 40);
    SOOPRF_ADD8(14, hi >> 48);
    SOOPRF_ADD8(15, hi >> 56);
#undef SOOPRF_ADD8
    return lsb0 ^ b_code_.encode(lsb1);
}

Block128 AltModPrf::eval(Block128 x) const {
    Block128 y{};
    eval_batch(std::span<const Block128>(&x, 1), std::span<Block128>(&y, 1));
    return y;
}


void AltModPrf::expand_inputs_bitsliced_unkeyed(std::span<const Block128> x,
                                                std::vector<Block128>& e) const {
    // Same bit-sliced layout as expand_inputs_bitsliced(), but without masking by
    // the PRF key.  This is the receiver-input side of the AltMod weak-PRF 2PC:
    // the key multiplication is performed later from OT-derived correlations.
    const std::size_t cols = (x.size() + 127) / 128;
    e.resize(AltModPrfParams::KeyBits * cols);

    std::array<AltModKey, 128> expanded{};
    for (std::size_t col = 0; col < cols; ++col) {
        const std::size_t base = col * 128;
        const std::size_t take = std::min<std::size_t>(128, x.size() - base);

        for (std::size_t lane = 0; lane < take; ++lane) {
            expanded[lane] = expand_input_linear(x[base + lane]);
        }

        for (std::size_t block_idx = 0; block_idx < AltModPrfParams::KeyBlocks; ++block_idx) {
            store_transposed_expansion_half(expanded, take, block_idx, false,
                                            ~std::uint64_t{0}, cols, col, e.data());
            store_transposed_expansion_half(expanded, take, block_idx, true,
                                            ~std::uint64_t{0}, cols, col, e.data());
        }
    }
}

void AltModPrf::accperm_bitsliced(std::span<Block128> msb,
                                  std::span<Block128> lsb,
                                  std::span<Block128> out_msb,
                                  std::span<Block128> out_lsb,
                                  std::size_t cols) const {
    if (msb.size() != AltModPrfParams::KeyBits * cols ||
        lsb.size() != AltModPrfParams::KeyBits * cols ||
        out_msb.size() != AltModPrfParams::MidBits * cols ||
        out_lsb.size() != AltModPrfParams::MidBits * cols) {
        throw std::invalid_argument("AltModPrf::accperm_bitsliced: size mismatch");
    }
    a_code_.encode_bitsliced(msb.data(), lsb.data(), out_msb.data(), out_lsb.data(), cols);
}

void AltModPrf::compress_b_bitsliced_public(std::span<const Block128> v,
                                            std::span<Block128> y) const {
    if (v.size() != AltModPrfParams::MidBits * ((y.size() + 127) / 128)) {
        throw std::invalid_argument("AltModPrf::compress_b_bitsliced_public: size mismatch");
    }
    compress_b_bitsliced(v, y);
}

void AltModPrf::expand_inputs_bitsliced(std::span<const Block128> x, std::vector<Block128>& e) const {
    // Block transpose path.  For each group of 128 inputs we first compute the
    // four 128-bit words of Gx, mask by the binary PRF key, then transpose the
    // resulting 128 x 512 bit matrix into the row-major bit-sliced layout used
    // by the F3 stage.  This replaces the old CTZ-driven per-set-bit scatter.
    const std::size_t cols = (x.size() + 127) / 128;
    e.resize(AltModPrfParams::KeyBits * cols);

    std::array<AltModKey, 128> expanded{};
    for (std::size_t col = 0; col < cols; ++col) {
        const std::size_t base = col * 128;
        const std::size_t take = std::min<std::size_t>(128, x.size() - base);

        for (std::size_t lane = 0; lane < take; ++lane) {
            expanded[lane] = expand_input_linear(x[base + lane]);
        }
        // No need to clear expanded[take..127]; valid_lanes controls reads.

        for (std::size_t block_idx = 0; block_idx < AltModPrfParams::KeyBlocks; ++block_idx) {
            store_transposed_expansion_half(expanded, take, block_idx, false,
                                            key_[block_idx].lo, cols, col, e.data());
            store_transposed_expansion_half(expanded, take, block_idx, true,
                                            key_[block_idx].hi, cols, col, e.data());
        }
    }
}


void AltModPrf::compress_b_bitsliced(std::span<const Block128> v, std::span<Block128> y) const {
    // Inverse bit-slice with 64x64 block transposes.  The previous version
    // scanned set lanes with CTZ; this version reconstructs the 128 lane-local
    // Block128 values by transposing 64 rows at a time.
    const std::size_t cols = (y.size() + 127) / 128;

    alignas(64) std::array<Block128, 128> low{};
    alignas(64) std::array<Block128, 128> high{};
    alignas(64) std::uint64_t tmp[64];

    for (std::size_t col = 0; col < cols; ++col) {
        // low input half: rows 0..127 -> low[lane]
        transpose_bitsliced_64_rows_lo(v.data() + 0 * cols, cols, col, false, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) low[lane].lo = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 64 * cols, cols, col, false, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) low[lane].hi = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 0 * cols, cols, col, true, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) low[64 + lane].lo = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 64 * cols, cols, col, true, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) low[64 + lane].hi = tmp[lane];

        // high input half: rows 128..255 -> high[lane]
        transpose_bitsliced_64_rows_lo(v.data() + 128 * cols, cols, col, false, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) high[lane].lo = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 192 * cols, cols, col, false, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) high[lane].hi = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 128 * cols, cols, col, true, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) high[64 + lane].lo = tmp[lane];
        transpose_bitsliced_64_rows_lo(v.data() + 192 * cols, cols, col, true, tmp);
        for (std::size_t lane = 0; lane < 64; ++lane) high[64 + lane].hi = tmp[lane];

        const std::size_t base = col * 128;
        const std::size_t take = std::min<std::size_t>(128, y.size() - base);
        for (std::size_t lane = 0; lane < take; ++lane) {
            y[base + lane] = low[lane] ^ b_code_.encode(high[lane]);
        }
    }
}


void AltModPrf::eval_batch(std::span<const Block128> x, std::span<Block128> y) const {
    AltModPrfWorkspace ws;
    eval_batch(x, y, ws);
}

void AltModPrf::eval_batch(std::span<const Block128> x, std::span<Block128> y, AltModPrfWorkspace& ws) const {
    if (x.size() != y.size()) throw std::invalid_argument("input/output size mismatch");
    if (x.empty()) return;

    // Canonical secure-join-compatible AltMod PRF path:
    //   e = G*x, ek = e & key over F3, u = A(ek), v = u mod 2, y = B*v.
    // The older fast8 table path evaluated a different byte-decomposed variant
    // and is deliberately not used by the OPPRF/AltMod-WPRF protocol path.
    ws.resize_for(x.size());
    expand_inputs_bitsliced(x, ws.ek_lsb);
    a_code_.encode_bitsliced_lsb(ws.ek_msb.data(), ws.ek_lsb.data(),
                                 ws.u_lsb.data(), ws.cols);
    compress_b_bitsliced(ws.u_lsb, y);
}

void AltModPrf::eval_batch_shared_splitmix(std::span<const Block128> x,
                                           std::span<Block128> sender_share,
                                           std::span<Block128> receiver_share,
                                           SplitMix64& mask_rng) const {
    if (sender_share.size() != x.size() || receiver_share.size() != x.size()) {
        throw std::invalid_argument("share/output size mismatch");
    }
    const std::size_t n = x.size();
    std::vector<Block128> yy(n);
    eval_batch(x, yy);
    for (std::size_t i = 0; i < n; ++i) {
        const Block128 r = mask_rng.next_block();
        sender_share[i] = r;
        receiver_share[i] = r ^ yy[i];
    }
}

} // namespace sooprf
