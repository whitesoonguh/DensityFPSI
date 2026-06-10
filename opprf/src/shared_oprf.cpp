#include "sooprf/shared_oprf.h"

#include <array>
#include <stdexcept>

namespace sooprf {
namespace {

static Block128 valid_f3_lsb(Block128 a, Block128 b) noexcept {
    return a & ~b;
}

static Block128 valid_f3_msb(Block128 a, Block128 b) noexcept {
    return b & ~a;
}

static void f3_add_binary_inplace(Block128& msb, Block128& lsb, Block128 y0) noexcept {
    // secure-join mod3Add(z1,z0,x1,x0,y0): add binary y0 to valid F3 x.
    const Block128 x1x0 = msb ^ lsb;
    const Block128 zz1 = msb ^ (x1x0 & y0);
    const Block128 zz0 = lsb ^ ((~msb) & y0);
    msb = zz1;
    lsb = zz0;
}

static Block128 f4_select_valid(Block128 u0, Block128 u1,
                                const std::array<Block128, 4>& ot) noexcept {
    // Valid F3 encoding has u=0:(0,0), 1:(0,1), 2:(1,0).  u=3 is invalid.
    const Block128 sel0 = ~(u0 | u1);
    return (sel0 & ot[0]) ^ (u0 & ot[1]) ^ (u1 & ot[2]);
}

} // namespace

AltModWPrfCorrelationTape AltModWPrfCorrelationTape::make_ideal(std::size_t input_count,
                                                                std::uint64_t seed) {
    AltModWPrfCorrelationTape tape;
    tape.input_count = input_count;
    tape.cols = (input_count + 127) / 128;
    const std::size_t key_blocks = AltModPrfParams::KeyBits * tape.cols;
    const std::size_t f4_blocks = AltModPrfParams::MidBits * tape.cols;
    tape.key_mult_mask_msb.resize(key_blocks);
    tape.key_mult_mask_lsb.resize(key_blocks);
    tape.f4_sender_ots.resize(f4_blocks);

    SplitMix64 rng(seed ^ (static_cast<std::uint64_t>(input_count) * 0x9e3779b97f4a7c15ULL));
    for (std::size_t i = 0; i < key_blocks; ++i) {
        const Block128 a = rng.next_block();
        const Block128 b = rng.next_block();
        tape.key_mult_mask_lsb[i] = valid_f3_lsb(a, b);
        tape.key_mult_mask_msb[i] = valid_f3_msb(a, b);
    }
    for (auto& row : tape.f4_sender_ots) {
        row[0] = rng.next_block();
        row[1] = rng.next_block();
        row[2] = rng.next_block();
        row[3] = rng.next_block();
    }
    return tape;
}

void AltModWPrfCorrelationTape::clear() {
    input_count = 0;
    cols = 0;
    key_mult_mask_msb.clear();
    key_mult_mask_lsb.clear();
    f4_sender_ots.clear();
    consumed = false;
}

void SharedAltModOprf::eval_altmod_wprf_tape_into(std::span<const Block128> x,
                                                  std::span<Block128> sender_share,
                                                  std::span<Block128> receiver_share,
                                                  AltModWPrfCorrelationTape& tape) const {
    if (sender_share.size() != x.size() || receiver_share.size() != x.size()) {
        throw std::invalid_argument("share/output size mismatch");
    }
    if (tape.consumed) {
        throw std::runtime_error("AltModWPrfCorrelationTape already consumed; call reset() or allocate a fresh tape");
    }
    const std::size_t n = x.size();
    const std::size_t cols = (n + 127) / 128;
    if (tape.input_count != n || tape.cols != cols ||
        tape.key_mult_mask_lsb.size() != AltModPrfParams::KeyBits * cols ||
        tape.key_mult_mask_msb.size() != AltModPrfParams::KeyBits * cols ||
        tape.f4_sender_ots.size() != AltModPrfParams::MidBits * cols) {
        throw std::invalid_argument("AltModWPrfCorrelationTape size does not match input batch");
    }

    // Receiver-input side: e = G*x in the same transposed bit-sliced layout as
    // secure-join AltModPrf::expandInput().
    std::vector<Block128> e;
    prf_.expand_inputs_bitsliced_unkeyed(x, e);

    // secure-join AltModKeyMult, SenderOnly-key / ReceiverOnly-input mode.
    std::vector<Block128> sender_ek_msb(AltModPrfParams::KeyBits * cols);
    std::vector<Block128> sender_ek_lsb(AltModPrfParams::KeyBits * cols);
    std::vector<Block128> receiver_ek_msb(AltModPrfParams::KeyBits * cols);
    std::vector<Block128> receiver_ek_lsb(AltModPrfParams::KeyBits * cols);

    const AltModKey& key = prf_.key();
    for (std::size_t row = 0; row < AltModPrfParams::KeyBits; ++row) {
        const bool key_bit = bit(key[row / 128], row % 128);
        for (std::size_t col = 0; col < cols; ++col) {
            const std::size_t idx = row * cols + col;
            const Block128 a_msb = tape.key_mult_mask_msb[idx];
            const Block128 a_lsb = tape.key_mult_mask_lsb[idx];

            // Input holder keeps -a.  In the {0,1,2} F3 bit encoding, negation
            // swaps the lsb/msb planes.  This matches secure-join's swapped
            // xk0/xk1 assignment in AltModKeyMultSender::mult().
            receiver_ek_msb[idx] = a_lsb;
            receiver_ek_lsb[idx] = a_msb;

            // Key holder obtains a if key[row]=0 and a+e[row] if key[row]=1.
            Block128 s_msb = a_msb;
            Block128 s_lsb = a_lsb;
            if (key_bit) {
                f3_add_binary_inplace(s_msb, s_lsb, e[idx]);
            }
            sender_ek_msb[idx] = s_msb;
            sender_ek_lsb[idx] = s_lsb;
        }
    }

    // Public F3 linear layer A applied independently to both additive shares.
    std::vector<Block128> sender_u_msb(AltModPrfParams::MidBits * cols);
    std::vector<Block128> sender_u_lsb(AltModPrfParams::MidBits * cols);
    std::vector<Block128> receiver_u_msb(AltModPrfParams::MidBits * cols);
    std::vector<Block128> receiver_u_lsb(AltModPrfParams::MidBits * cols);
    prf_.accperm_bitsliced(sender_ek_msb, sender_ek_lsb, sender_u_msb, sender_u_lsb, cols);
    prf_.accperm_bitsliced(receiver_ek_msb, receiver_ek_lsb, receiver_u_msb, receiver_u_lsb, cols);

    // secure-join AltModWPrf::{Sender,Receiver}::mod2OtF4() in local form.
    std::vector<Block128> sender_v(AltModPrfParams::MidBits * cols);
    std::vector<Block128> receiver_v(AltModPrfParams::MidBits * cols);
    for (std::size_t idx = 0; idx < sender_v.size(); ++idx) {
        const Block128 v0 = sender_u_lsb[idx];
        const Block128 v1 = sender_u_msb[idx];
        const Block128 u0 = receiver_u_lsb[idx];
        const Block128 u1 = receiver_u_msb[idx];
        const auto& ot = tape.f4_sender_ots[idx];

        const Block128 Tv0 = v0;
        const Block128 Tv1 = ~(v0 ^ v1);
        const Block128 Tv2 = v1;

        sender_v[idx] = Tv0 ^ ot[0];
        const Block128 t1 = Tv1 ^ sender_v[idx] ^ ot[1];
        const Block128 t2 = Tv2 ^ sender_v[idx] ^ ot[2];
        receiver_v[idx] = (u0 & t1) ^ (u1 & t2) ^ f4_select_valid(u0, u1, ot);
    }

    prf_.compress_b_bitsliced_public(sender_v, sender_share);
    prf_.compress_b_bitsliced_public(receiver_v, receiver_share);
    tape.consumed = true;
}

SharedOprfOutput SharedAltModOprf::eval_altmod_wprf_tape(std::span<const Block128> x,
                                                         AltModWPrfCorrelationTape& tape) const {
    SharedOprfOutput out;
    out.sender_share.resize(x.size());
    out.receiver_share.resize(x.size());
    eval_altmod_wprf_tape_into(x, out.sender_share, out.receiver_share, tape);
    return out;
}

} // namespace sooprf
