#include "fpsi/fpsi.h"
#include "sooprf/prng.h"

#include <algorithm>
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
    sooprf::SplitMix64 rng(999);
    std::vector<fpsi::KeySetPair> pairs;
    std::vector<fpsi::Block128> keys(5);
    for (auto& k : keys) k = rng.next_block();

    pairs.push_back({keys[0], {10,20,30}});
    pairs.push_back({keys[1], {20,30,40}});
    pairs.push_back({keys[2], {20,50,60}});
    pairs.push_back({keys[3], {70,80}});
    pairs.push_back({keys[4], {90}});

    fpsi::OssiParams p;
    p.C = 4;
    p.opprf_params.epsilon = 0.12;
    p.opprf_params.okvs_seed[0] = 42;

    std::vector<fpsi::Block128> query = {keys[0], keys[1], keys[2]};
    auto out = fpsi::ossi_run_ideal(pairs, query, p);
    std::sort(out.intersection.begin(), out.intersection.end());
    require(out.intersection == std::vector<std::uint64_t>{20}, "OSSI programmed intersection mismatch");
    require(out.stats.programmed_pairs == 12, "bad programmed pair count");
    require(out.stats.query_pairs == 6, "bad query pair count");

    std::vector<fpsi::Block128> query2 = {keys[0], keys[3], keys[4]};
    auto out2 = fpsi::ossi_run_ideal(pairs, query2, p);
    require(out2.intersection.empty(), "OSSI empty intersection mismatch");

    std::cout << "ossi test passed\n";
}
