#include "sooprf/shared_oprf.h"

#include <cassert>
#include <stdexcept>
#include <vector>

static std::vector<sooprf::Block128> make_inputs(std::size_t n) {
    std::vector<sooprf::Block128> x;
    x.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        x.push_back(sooprf::Block128{0x1000ULL + i, 0xabcdefULL ^ (i << 7)});
    }
    return x;
}

int main() {
    const auto key = sooprf::AltModPrf::random_key(0x1234567890abcdefULL);
    sooprf::AltModPrf prf(key);
    sooprf::SharedAltModOprf soprf(prf);

    // secure-join-style AltMod WPRF tape path.  This exercises:
    // G expansion -> key multiplication over F3 -> A -> mod2 via F4 OT -> B.
    for (std::size_t n : {1u, 2u, 17u, 128u, 129u, 257u}) {
        auto x = make_inputs(n);
        auto tape = sooprf::AltModWPrfCorrelationTape::make_ideal(n, 0x414c545f74617065ULL + n);
        assert(tape.reusable_key_base_ots() == sooprf::AltModPrfParams::KeyBits);
        assert(tape.key_mult_blocks() == sooprf::AltModPrfParams::KeyBits * ((n + 127) / 128));
        assert(tape.f4_blocks() == sooprf::AltModPrfParams::MidBits * ((n + 127) / 128));

        auto out = soprf.eval_altmod_wprf_tape(x, tape);
        assert(tape.consumed);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto y = prf.eval(x[i]);
            assert((out.sender_share[i] ^ out.receiver_share[i]) == y);
        }

        bool threw = false;
        try {
            (void)soprf.eval_altmod_wprf_tape(x, tape);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    return 0;
}
