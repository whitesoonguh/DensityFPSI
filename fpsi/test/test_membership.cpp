#include "fpsi/fpsi.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

static void require(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

int main() {
    constexpr std::size_t M = 3;
    constexpr std::size_t N = 4;
    constexpr std::size_t d = 2;
    constexpr std::uint64_t delta = 1;

    // X = {(10,20), (11,19), (100,20)}.
    // The first two receiver items collide in both axes for y=(10,20).
    std::vector<std::uint64_t> X = {
        10, 20,
        11, 19,
        100, 20,
    };

    std::vector<std::uint64_t> Y = {
        10, 20,   // hit: receiver indices {1,2}
        100, 20,  // hit: receiver index {3}
        50, 20,   // miss
        11, 19,   // hit: receiver indices {1,2}
    };

    const auto cmax = fpsi::membership_cmax_collision(X, M, d, delta);
    require(cmax == 3, "bad C-max verifier result");

    fpsi::MembershipEncodingStats enc_stats;
    auto pairs = fpsi::build_membership_pairs(X, M, d, delta, 4, &enc_stats);
    require(!pairs.empty(), "empty membership encoding");
    require(enc_stats.M == M && enc_stats.d == d && enc_stats.delta == delta, "bad encoding stats dimensions");
    require(enc_stats.max_tag_size == 3, "bad encoding max tag size");

    auto q0 = fpsi::build_sender_membership_keys(std::span<const std::uint64_t>(Y.data(), d), 1, d);
    fpsi::OssiParams params;
    params.C = 4;
    params.opprf_params.epsilon = 0.12;
    params.opprf_params.okvs_seed[0] = 123;

    auto full = fpsi::ossi_run_ideal(pairs, q0, params);
    std::sort(full.intersection.begin(), full.intersection.end());
    require(full.intersection.size() == 2, "full OSSI should return two receiver indices");
    require(full.intersection[0] == 1 && full.intersection[1] == 2, "bad full OSSI intersection");

    fpsi::MembershipEncodingStats fps_stats;
    auto bits = fpsi::fpsi_match_ideal_batch_nonempty(X, M, Y, N, d, delta, params, &fps_stats);
    require(bits.N == N, "bad batch N");
    require(bits.matched.size() == N, "bad matched size");
    require(bits.matched[0] == 1, "Y0 should hit");
    require(bits.matched[1] == 1, "Y1 should hit");
    require(bits.matched[2] == 0, "Y2 should miss");
    require(bits.matched[3] == 1, "Y3 should hit");
    require(fps_stats.programmed_pairs == pairs.size(), "bad FPSI encoding programmed pair count");

    bool threw = false;
    try {
        (void)fpsi::build_membership_pairs(X, M, d, delta, 2, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "C_limit violation should throw");

    std::cout << "membership encoding test passed\n";
}
