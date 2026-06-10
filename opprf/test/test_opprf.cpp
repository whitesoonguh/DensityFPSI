#include "opprf/opprf.h"
#include "sooprf/prng.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace opprf;

static void require(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

int main() {
    auto prf_key = AltModPrf::random_key(0x0f0f50505246ULL);
    Opprf64Params params;
    params.w = 218;
    params.epsilon = 0.08;
    params.okvs_seed[0] = 7;

    Opprf64 pp(prf_key, params);
    sooprf::SplitMix64 rng(0x1122334455667788ULL);

    constexpr std::size_t n = 512;
    std::vector<Block128> keys(n), queries(n);
    std::vector<std::uint64_t> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        keys[i] = rng.next_block();
        queries[i] = keys[i];
        values[i] = rng.next_u64();
    }

    auto out = pp.eval_ideal(keys, values, queries, 0x9999888877776666ULL);
    for (std::size_t i = 0; i < n; ++i) {
        require((out.sender_share[i] ^ out.receiver_share[i]) == values[i],
                "programmed OPPRF reconstruction failed");
    }

    auto enc = pp.encode(keys, values);
    auto corr = pp.decode_correction(enc, keys);
    AltModPrf clear_prf(prf_key);
    for (std::size_t i = 0; i < n; ++i) {
        require((corr[i] ^ clear_prf.eval(keys[i]).lo) == values[i],
                "OKVS correction failed");
    }

    // Half-packed OPPRF sanity: two slot keys use one pair OPRF input.
    constexpr std::size_t pair_count = 256;
    std::vector<Block128> slot_keys(2 * pair_count);
    std::vector<Block128> pair_keys(pair_count);
    std::vector<Block128> programmed_pair_keys(2 * pair_count);
    std::vector<std::uint8_t> slot_hi(2 * pair_count);
    std::vector<std::size_t> pair_index(2 * pair_count);
    std::vector<std::uint64_t> packed_values(2 * pair_count);
    for (std::size_t pidx = 0; pidx < pair_count; ++pidx) {
        pair_keys[pidx] = rng.next_block();
        for (std::size_t off = 0; off < 2; ++off) {
            const std::size_t i = 2 * pidx + off;
            slot_keys[i] = rng.next_block();
            programmed_pair_keys[i] = pair_keys[pidx];
            slot_hi[i] = static_cast<std::uint8_t>(off == 0); // first slot -> .hi, second -> .lo
            pair_index[i] = pidx;
            packed_values[i] = rng.next_u64();
        }
    }
    auto packed_tape = sooprf::AltModWPrfCorrelationTape::make_ideal(pair_count, 0x5050434bULL);
    auto packed = pp.eval_altmod_wprf_tape_packed(slot_keys, programmed_pair_keys, slot_hi,
                                                  packed_values, slot_keys, pair_keys,
                                                  pair_index, slot_hi, packed_tape);
    for (std::size_t i = 0; i < packed_values.size(); ++i) {
        require((packed.sender_share[i] ^ packed.receiver_share[i]) == packed_values[i],
                "half-packed OPPRF reconstruction failed");
    }

    std::cout << "opprf tests passed\n";
    return 0;
}
