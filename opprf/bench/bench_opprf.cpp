#include "opprf/opprf.h"
#include "sooprf/prng.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace opprf;

static double since_s(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    std::size_t n = 1u << 16;
    std::size_t q = n;
    int reps = 3;
    if (argc > 1) n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2) q = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (argc > 3) reps = std::atoi(argv[3]);

    Opprf64Params params;
    params.w = 218;
    params.epsilon = 0.05;
    params.okvs_seed[0] = 11;
    Opprf64 op(AltModPrf::random_key(0x4242424242ULL), params);

    sooprf::SplitMix64 rng(0xabcdef1234567890ULL);
    std::vector<Block128> keys(n), queries(q);
    std::vector<std::uint64_t> values(n), receiver(q), sender(q);
    for (std::size_t i = 0; i < n; ++i) {
        keys[i] = rng.next_block();
        values[i] = rng.next_u64();
    }
    for (std::size_t i = 0; i < q; ++i) queries[i] = keys[i % n];

    double best_encode = 1e100;
    EncodedOpprf64 enc;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        enc = op.encode(keys, values);
        best_encode = std::min(best_encode, since_s(t0));
    }

    sooprf::SharedAltModOprf soprf(op.prf());
    double best_tape_gen = 1e100;
    double best_wprf_online = 1e100;
    std::vector<Block128> receiver_blocks(q), sender_blocks(q);
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        auto tape = sooprf::AltModWPrfCorrelationTape::make_ideal(q, 0x5150ULL + static_cast<std::uint64_t>(r));
        best_tape_gen = std::min(best_tape_gen, since_s(t0));
        t0 = std::chrono::steady_clock::now();
        soprf.eval_altmod_wprf_tape_into(queries, sender_blocks, receiver_blocks, tape);
        best_wprf_online = std::min(best_wprf_online, since_s(t0));
    }
    for (std::size_t i = 0; i < q; ++i) sender[i] = sender_blocks[i].lo;

    double best_decode = 1e100;
    std::uint64_t checksum = 0;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        op.receiver_share_into(enc, queries, receiver_blocks, receiver);
        best_decode = std::min(best_decode, since_s(t0));
        checksum ^= receiver[0];
    }
    bool ok = true;
    for (std::size_t i = 0; i < q; ++i) {
        if ((sender[i] ^ receiver[i]) != values[i % n]) { ok = false; break; }
    }
    for (std::size_t i = 0; i < std::min<std::size_t>(q, 16); ++i) checksum ^= sender[i] ^ receiver[i];

    const std::size_t cols = (q + 127) / 128;
    const std::size_t key_mult_blocks = sooprf::AltModPrfParams::KeyBits * cols;
    const std::size_t f4_blocks = sooprf::AltModPrfParams::MidBits * cols;

    std::cout << "OPPRF64 benchmark (AltMod-WPRF ideal tape + OKVS)\n";
    std::cout << "  n                 : " << n << "\n";
    std::cout << "  q                 : " << q << "\n";
    std::cout << "  reps              : " << reps << "\n";
    std::cout << "  okvs_cols         : " << enc.cols << "\n";
    std::cout << "  okvs_w            : " << enc.params.w << "\n";
    std::cout << "  wprf_key_blocks   : " << key_mult_blocks << "\n";
    std::cout << "  wprf_f4_blocks    : " << f4_blocks << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  okvs_encode_s     : " << best_encode << "\n";
    std::cout << "  okvs_encode_items/s: " << (static_cast<double>(n) / best_encode) << "\n";
    std::cout << "  wprf_tape_gen_s   : " << best_tape_gen << "\n";
    std::cout << "  wprf_online_s     : " << best_wprf_online << "\n";
    std::cout << "  wprf_online_q/s   : " << (static_cast<double>(q) / best_wprf_online) << "\n";
    std::cout << "  okvs_decode_s     : " << best_decode << "\n";
    std::cout << "  okvs_decode_q/s   : " << (static_cast<double>(q) / best_decode) << "\n";
    std::cout << "  verified          : " << (ok ? "yes" : "no") << "\n";
    std::cout << "  checksum          : " << checksum << "\n";
    return ok ? 0 : 4;
}
