#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include "sooprf/block.h"
#include "sooprf/prng.h"

namespace sooprf {

enum class InputExpansionMode {
    Linear,
    AesLike
};

struct AltModPrfParams {
    static constexpr std::size_t InputBits = 128;
    static constexpr std::size_t KeyBits = 128 * 4;
    static constexpr std::size_t KeyBlocks = 4;
    static constexpr std::size_t MidBits = 256;
    static constexpr std::size_t OutBits = 128;
    static constexpr std::size_t AccPermRounds = 8;
};

using AltModKey = std::array<Block128, AltModPrfParams::KeyBlocks>;

struct AltModPrfWorkspace {
    std::size_t cols = 0;
    std::vector<Block128> ek_lsb;
    std::vector<Block128> ek_msb;
    std::vector<Block128> u_lsb;

    void resize_for(std::size_t input_count);
};

class BinaryLinearCode128 {
public:
    // Matrix is represented by row-subcode lookup tables. The input is a 128-bit
    // row vector. encode(x) returns XOR of row i for every set bit x_i.
    BinaryLinearCode128() = default;
    explicit BinaryLinearCode128(std::uint64_t seed) { init(seed); }
    void init(std::uint64_t seed);
    Block128 encode(Block128 x) const noexcept;
    const std::array<Block128, 128>& rows() const noexcept { return rows_; }

private:
    // rows_[i] is the 128-bit output contribution for input bit i.
    alignas(64) std::array<Block128, 128> rows_{};
    // 16 input bytes * 256 choices. Each entry is a 128-bit partial codeword.
    alignas(64) std::array<Block128, 16 * 256> sub_{};
};

class F3AccPermCode {
public:
    void init(std::size_t k, std::size_t n, std::uint64_t seed,
              std::size_t rounds = AltModPrfParams::AccPermRounds);

    void encode_scalar(std::span<std::uint8_t> inout, std::span<std::uint8_t> out) const;

    // Bit-sliced F3 encoding. lsb/msb are k x cols arrays of Block128, valid trits are
    // 0=(msb,lsb)=(0,0), 1=(0,1), 2=(1,0). out arrays are n x cols.
    void encode_bitsliced(Block128* msb, Block128* lsb,
                          Block128* out_msb, Block128* out_lsb,
                          std::size_t cols) const;

    // Same transform, but only materializes the low/parity plane of the selected
    // n output rows. AltMod PRF immediately reduces F3 to F2, so the high plane
    // is not needed after the accumulator-permutation stage.
    void encode_bitsliced_lsb(Block128* msb, Block128* lsb,
                              Block128* out_lsb,
                              std::size_t cols) const;

private:
    std::size_t k_ = 0;
    std::size_t n_ = 0;
    std::vector<std::vector<std::uint32_t>> perms_;
};

class AltModPrf {
public:
    AltModPrf();
    explicit AltModPrf(const AltModKey& key);

    void set_key(const AltModKey& key);
    const AltModKey& key() const noexcept { return key_; }

    Block128 eval(Block128 x) const;
    void eval_batch(std::span<const Block128> x, std::span<Block128> y) const;
    void eval_batch(std::span<const Block128> x, std::span<Block128> y, AltModPrfWorkspace& ws) const;
    void eval_batch_shared_splitmix(std::span<const Block128> x,
                                    std::span<Block128> sender_share,
                                    std::span<Block128> receiver_share,
                                    SplitMix64& mask_rng) const;

    AltModKey expand_input_linear(Block128 x) const noexcept;

    // Public wrappers for the AltMod weak-PRF 2PC simulator.  These expose the
    // same core arithmetic used by eval_batch(): G input expansion, the public
    // F3 accumulator/permutation A, and the final binary linear compression B.
    // They intentionally do not expose any transport/coproto state.
    void expand_inputs_bitsliced_unkeyed(std::span<const Block128> x,
                                         std::vector<Block128>& e) const;
    void accperm_bitsliced(std::span<Block128> msb,
                           std::span<Block128> lsb,
                           std::span<Block128> out_msb,
                           std::span<Block128> out_lsb,
                           std::size_t cols) const;
    void compress_b_bitsliced_public(std::span<const Block128> v,
                                     std::span<Block128> y) const;
    static AltModKey random_key(std::uint64_t seed);

private:
    AltModKey key_{};

    std::array<BinaryLinearCode128, 3> g_codes_;
    BinaryLinearCode128 b_code_;
    F3AccPermCode a_code_;

    struct alignas(64) FastChunkEntry {
        Block128 msb0, lsb0;
        Block128 msb1, lsb1;
    };
    // Key-specialized table for 8 chunks of 16 input bits. Each entry is
    // A((Gx & key)) in bit-sliced F3 form over the 256 middle trits.
    std::array<std::array<FastChunkEntry, 256>, 16> fast8_;
    bool fast_ready_ = false;

    void rebuild_fast8();
    Block128 eval_fast8(Block128 x) const noexcept;

    void expand_inputs_bitsliced(std::span<const Block128> x, std::vector<Block128>& e) const;
    void compress_b_bitsliced(std::span<const Block128> v, std::span<Block128> y) const;
};

// Bit-sliced mod-3 addition, dst += src mod 3, with valid trit encoding.
void mod3_add_inplace(Block128* dst_msb, Block128* dst_lsb,
                      const Block128* src_msb, const Block128* src_lsb,
                      std::size_t blocks) noexcept;

bool compiled_with_avx512() noexcept;

} // namespace sooprf
