#include "fpsi/fpsi.h"
#include "sooprf/prng.h"

#include <cstdlib>
#include <iostream>
#include <vector>

static void require(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

int main() {
    constexpr std::size_t C = 4;
    constexpr std::size_t d = 3;
    constexpr std::size_t N = 4;

    sooprf::SplitMix64 rng(12345);
    std::vector<fpsi::Block128> keys(12);
    for (auto& k : keys) k = rng.next_block();

    std::vector<fpsi::KeySetPair> pairs;
    pairs.push_back({keys[0], {111, 222}});
    pairs.push_back({keys[1], {111, 333}});
    pairs.push_back({keys[2], {111, 444}});
    pairs.push_back({keys[3], {777, 888}});
    pairs.push_back({keys[4], {777, 999}});
    pairs.push_back({keys[5], {1000, 1001}});
    pairs.push_back({keys[6], {42}});
    pairs.push_back({keys[7], {42}});
    pairs.push_back({keys[8], {42}});

    std::vector<fpsi::Block128> flat;
    flat.reserve(N * d);
    // hit: intersection {111}
    flat.insert(flat.end(), {keys[0], keys[1], keys[2]});
    // miss: first two share 777, third does not
    flat.insert(flat.end(), {keys[3], keys[4], keys[5]});
    // hit: intersection {42}
    flat.insert(flat.end(), {keys[6], keys[7], keys[8]});
    // miss: unregistered random keys
    flat.insert(flat.end(), {keys[9], keys[10], keys[11]});

    fpsi::OssiParams p;
    p.C = C;
    p.opprf_params.epsilon = 0.12;
    p.opprf_params.okvs_seed[0] = 99;

    auto out = fpsi::ossi_run_ideal_batch_nonempty(pairs, flat, N, d, p);
    require(out.N == N, "bad N");
    require(out.matched.size() == N, "bad matched size");
    require(out.matched[0] == 1, "query 0 should hit");
    require(out.matched[1] == 0, "query 1 should miss");
    require(out.matched[2] == 1, "query 2 should hit");
    require(out.matched[3] == 0, "query 3 should miss");
    require(out.stats.query_pairs == N * d * (C / 2), "bad query pair count");

    std::cout << "ossi batch nonempty test passed\n";
}
