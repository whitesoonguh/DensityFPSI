#include "sooprf/altmod_prf.h"
#include "sooprf/shared_oprf.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace sooprf;

static void require(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

int main() {
    auto key = AltModPrf::random_key(0x12345678ULL);
    AltModPrf prf(key);
    SplitMix64 rng(0xabcdefULL);

    for (std::size_t n : {1u, 2u, 17u, 127u, 128u, 129u, 257u, 1024u}) {
        std::vector<Block128> x(n), ys(n), yb(n);
        for (auto& b : x) b = rng.next_block();
        for (std::size_t i = 0; i < n; ++i) ys[i] = prf.eval(x[i]);
        prf.eval_batch(x, yb);
        for (std::size_t i = 0; i < n; ++i) {
            require(ys[i] == yb[i], "scalar and batch PRF differ");
        }
    }

    std::vector<Block128> x(333), y(333);
    for (auto& b : x) b = rng.next_block();
    prf.eval_batch(x, y);

    SharedAltModOprf soprf(prf);
    auto tape = AltModWPrfCorrelationTape::make_ideal(x.size(), 0x9999ULL);
    auto shares = soprf.eval_altmod_wprf_tape(x, tape);
    for (std::size_t i = 0; i < x.size(); ++i) {
        require((shares.sender_share[i] ^ shares.receiver_share[i]) == y[i], "shares do not reconstruct PRF output");
    }

    auto pair = as_gf64_pair(y[0]);
    (void)pair;

    std::cout << "all tests passed\n";
    return 0;
}
