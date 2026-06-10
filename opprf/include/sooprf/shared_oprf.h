#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include "sooprf/altmod_prf.h"

namespace sooprf {

// Local AltMod weak-PRF preprocessing tape. This is the in-repo ideal-tape
// sanity-check model: correlations are already expanded, and the online call
// consumes them while executing the secure-join AltMod WPRF arithmetic path:
// G expansion -> key multiplication over F3 -> A -> F3-to-F2 mod2 via F4 OT
// masks -> B compression. It does not include socket/libOTe communication.
struct AltModWPrfCorrelationTape {
    std::size_t input_count = 0;
    std::size_t cols = 0;
    std::vector<Block128> key_mult_mask_msb;
    std::vector<Block128> key_mult_mask_lsb;
    std::vector<std::array<Block128, 4>> f4_sender_ots;
    bool consumed = false;

    static AltModWPrfCorrelationTape make_ideal(std::size_t input_count,
                                                std::uint64_t seed = 0x414c544d57505246ULL);

    void clear();
    void reset() noexcept { consumed = false; }
    std::size_t key_mult_blocks() const noexcept { return key_mult_mask_lsb.size(); }
    std::size_t f4_blocks() const noexcept { return f4_sender_ots.size(); }
    std::size_t reusable_key_base_ots() const noexcept { return AltModPrfParams::KeyBits; }
    std::size_t f4_bit_ots() const noexcept { return f4_sender_ots.size() * 128; }
};

struct SharedOprfOutput {
    std::vector<Block128> sender_share;
    std::vector<Block128> receiver_share;
};

class SharedAltModOprf {
public:
    explicit SharedAltModOprf(AltModPrf prf) : prf_(prf) {}

    void eval_altmod_wprf_tape_into(std::span<const Block128> x,
                                    std::span<Block128> sender_share,
                                    std::span<Block128> receiver_share,
                                    AltModWPrfCorrelationTape& tape) const;

    SharedOprfOutput eval_altmod_wprf_tape(std::span<const Block128> x,
                                           AltModWPrfCorrelationTape& tape) const;

private:
    AltModPrf prf_;
};

} // namespace sooprf
