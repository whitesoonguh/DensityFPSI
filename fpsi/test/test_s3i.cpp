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
    constexpr std::size_t d = 4;
    constexpr std::size_t C = 8;
    std::vector<std::uint64_t> sets[d] = {
        {11,22,33,44,55,66,77,88},
        {99,22,33,101,102,103,104,105},
        {22,33,201,202,203,204,205,206},
        {7,8,9,22,33,10,12,13},
    };
    sooprf::SplitMix64 rng(123);
    std::vector<std::uint64_t> r(d*C), s(d*C);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t l = 0; l < C; ++l) {
            r[j*C+l] = rng.next_u64();
            s[j*C+l] = r[j*C+l] ^ sets[j][l];
        }
    }
    auto out = fpsi::s3i_intersect_shared(r, s, d, C, 111, 222, 333);
    std::sort(out.begin(), out.end());
    std::vector<std::uint64_t> exp = {22,33};
    require(out == exp, "S3I intersection mismatch");
    std::cout << "s3i test passed\n";
}
